#include <metal_stdlib>

using namespace metal;

// =============================================================================
// SECTION 1: COMPILER DIRECTIVES & CONSTANTS
// =============================================================================

/**
 * BLOCK_SIZE (b): The column-width of each Householder block.
 * To maintain peak occupancy with float32 (staying under the 56-variable limit),
 * we utilize b=16. This allows for four 8x8 AMX tiles while keeping the
 * register-to-threadgroup memory ratio optimal for the L1 cache.
 */
#define BLOCK_SIZE 16

/**
 * SIMD_GROUP_SIZE: The native hardware execution width of Apple Silicon (AGX).
 * All matrix coprocessor (AMX) operations and intra-threadgroup reductions
 * are relative to this strictly enforced 32-thread execution width.
 */
#define SIMD_GROUP_SIZE 32

/**
 * TILE_DIM: The hardware-accelerated dimension for float32 AMX tiles.
 * Apple’s simdgroup_matrix instructions physically map to 8x8 tensor core
 * registers for float32 multiply-accumulate operations.
 */
#define TILE_DIM 8

/**
 * EPSILON: Used for numerical stability in Householder reflections.
 * Defined as a minimally small representable float32 value to prevent
 * catastrophic division-by-zero or NaN propagation during vector normalization.
 */
#define EPSILON 1e-7f

// Max rows safely cached in Threadgroup memory (24KB limit to leave room for AMX staging)
#define MAX_PANEL_ROWS 384

// =============================================================================
// DYNAMIC FUNCTION CONSTANTS (Injected by MLX at compile time)
// =============================================================================
// By guaranteeing M is padded to a multiple of SIMD_GROUP_SIZE (32)
// and N is padded to a multiple of BLOCK_SIZE (16), we completely eliminate
// in-kernel boundary branching.
constant uint M [[function_constant(0)]];
constant uint N [[function_constant(1)]];

constant uint M_pad [[function_constant(2)]];
constant uint N_pad [[function_constant(3)]];

// =============================================================================
// SECTION 2: THE TIER-2 MEMORY STRUCTURE (MANUAL SPILLING)
// =============================================================================

/**
 * MAX_SIMD_GROUPS: The maximum number of SIMD groups per threadgroup.
 * Apple Silicon allows up to 1024 threads per threadgroup.
 * 1024 threads / 32 threads-per-SIMD-group = 32 SIMD groups maximum.
 */
#define MAX_SIMD_GROUPS 32

/**
 * QRSharedMemory acts as our manual Tier-2 Cache.
 * By explicitly staging intermediate values here, we prevent the LLVM compiler
 * from spilling float32 variables to the global device memory, preserving our
 * 112-register (56 float32 variables) occupancy limit.
 *
 * Maximum Capacity: 32 KiB (8,192 float32 values).
 * Current Utilization: ~5.4 KiB (Leaving massive headroom for the hardware).
 */
struct alignas(16) QRSharedMemory {
    float tau_values[BLOCK_SIZE];
    float reduction_space[MAX_SIMD_GROUPS];
    float temp_scalar;

    float compact_T[BLOCK_SIZE * BLOCK_SIZE];
    float clean_Y[BLOCK_SIZE * BLOCK_SIZE];
    float R_diag[BLOCK_SIZE];
};

// =============================================================================
// SECTION 3: SIMD-LEVEL REDUCTION PRIMITIVES (FOR PIVOTING)
// =============================================================================

/**
 * Finds the maximum value across all 32 threads in a SIMD group.
 * Uses a hardware "butterfly" reduction via simd_shuffle_down.
 */
inline float simd_max_reduce(float val) {
    for (uint offset = SIMD_GROUP_SIZE / 2; offset > 0; offset /= 2) {
        float remote_val = simd_shuffle_down(val, offset);
        // fmax ensures NaN protection (prevents garbage math propagation)
        val = fmax(val, remote_val);
    }
    return simd_broadcast(val, 0);
}

/**
 * Finds the index (argmax) of the maximum value across the SIMD group.
 * This is critical for Column Pivoting to know exactly WHICH column holds
 * the maximum norm.
 */
inline uint simd_argmax_reduce(float val, uint local_idx) {
    for (uint offset = SIMD_GROUP_SIZE / 2; offset > 0; offset /= 2) {
        float remote_val = simd_shuffle_down(val, offset);
        uint  remote_idx = simd_shuffle_down(local_idx, offset);

        // Tie-breaker logic: If the norms are exactly equal, favor the smaller index.
        // This guarantees deterministic pivoting across different GPU runs.
        if (remote_val > val || (remote_val == val && remote_idx < local_idx)) {
            val = remote_val;
            local_idx = remote_idx;
        }
    }
    return simd_broadcast(local_idx, 0);
}

/**
 * Sums a value across the SIMD group.
 * Used for accumulating the partial L2 Norms of the Householder vectors.
 */
inline float simd_sum_reduce(float val) {
    // Replaced manual simd_shuffle_down loop with Apple's highly optimized
    // native MSL intrinsic for SIMD group summation.
    return simd_sum(val);
}

/**
 * threadgroup_sum_reduce
 * ----------------------
 * Sums a local float value across every active thread in the threadgroup.
 * This is the engine that allows 1024 threads to compute Householder
 * dot products simultaneously, saturating the AGX memory bandwidth.
 */
inline float threadgroup_sum_reduce(
    float local_val,
    threadgroup float* red_space, // Pointer to shared_mem->reduction_space
    uint sg_id,                   // simd_group_id
    uint lane_id                  // simd_lane_id
) {
    // Phase 1: Intra-SIMD Reduction
    // Each 32-thread SIMD group sums its own internal values using high-speed
    // register shuffling (zero-latency on M-series).
    float simd_val = simd_sum(local_val);

    // Phase 2: Inter-SIMD Communication
    // The "leader" (lane 0) of each SIMD group writes its partial sum to
    // the Tier-2 threadgroup memory.
    if (lane_id == 0) {
        red_space[sg_id] = simd_val;
    }

    // Sync to ensure all SIMD groups have written to the reduction_space.
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: Final Sweep
    // Only the first SIMD group (sg_id == 0) wakes up to sum the
    // partial results left in threadgroup memory by the other groups.
    float total = 0.0f;
    if (sg_id == 0) {
        // If you have 1024 threads, you have 32 SIMD groups.
        // These 32 partial sums fit perfectly into one last simd_sum.
        float val = (lane_id < 32) ? red_space[lane_id] : 0.0f;
        total = simd_sum(val);

        // Write the absolute total to the first slot for broadcasting.
        if (lane_id == 0) {
            red_space[0] = total;
        }
    }

    // Final sync: everyone waits for SIMD group 0 to finish the final sum.
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Every thread now returns the same unified result.
    return red_space[0];
}

// =============================================================================
// SECTION 4: BLOCK HOUSEHOLDER MATH HELPERS
// =============================================================================

/**
 * compute_reflector
 * Generates a Householder reflector H = I - tau * v * v^T such that
 * H * x = [c, 0, 0, ...]^T.
 *
 * @param x_local   The element of the current column owned by this thread.
 * @param r_idx     The global row index for this thread.
 * @param k         The current iteration index within the block (0 to 15).
 * @param shared_mem Shared memory pointer for inter-SIMD reductions.
 * @param sg_id     SIMD group index (0-31).
 * @param lane_id   Lanes within SIMD group (0-31).
 */
inline float compute_reflector(
    float x_local,
    uint r_idx,
    uint k,
    uint block_start,
    threadgroup QRSharedMemory* shared_mem,
    uint sg_id,
    uint lane_id
) {
    const uint pivot_row = block_start + k;

    // -------------------------------------------------------------------------
    // 1. ISOLATE THE TAIL NORM (Strictly below the diagonal)
    // -------------------------------------------------------------------------
    float tail_x2 = (r_idx > pivot_row && r_idx < M) ? (x_local * x_local) : 0.0f;

    float simd_sum_x2 = simd_sum(tail_x2);
    if (lane_id == 0) {
        shared_mem->reduction_space[sg_id] = simd_sum_x2;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg_id == 0) {
        // Each lane reads its corresponding reduction space slot
        float partial = shared_mem->reduction_space[lane_id];
        float total_tail = simd_sum(partial);
        // ONLY lane 0 writes to shared memory to prevent thread races
        if (lane_id == 0) shared_mem->temp_scalar = total_tail;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float tail_norm_sq = shared_mem->temp_scalar;
    threadgroup_barrier(mem_flags::mem_none);

    // -------------------------------------------------------------------------
    // 2. BROADCAST ALPHA (The Diagonal Element)
    // -------------------------------------------------------------------------
    if (r_idx == pivot_row) {
        shared_mem->temp_scalar = x_local;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float alpha = shared_mem->temp_scalar;

    // -------------------------------------------------------------------------
    // 3. THE LAPACK STABILITY CHECK
    // -------------------------------------------------------------------------
    // If the tail is practically 0, the column is already reduced. No reflection!
    if (tail_norm_sq <= EPSILON) {
        if (sg_id == 0 && lane_id == 0) shared_mem->tau_values[k] = 0.0f;
        return (r_idx == pivot_row) ? 1.0f : 0.0f;
    }

    // -------------------------------------------------------------------------
    // 4. THE SAFE HOUSEHOLDER MATH
    // -------------------------------------------------------------------------
    float norm_x = sqrt(alpha * alpha + tail_norm_sq);

    // Sign choice to prevent catastrophic cancellation
    float mu = (alpha >= 0.0f) ? -norm_x : norm_x;

    float v_pivot = alpha - mu;
    float tau = -v_pivot / mu;

    if (sg_id == 0 && lane_id == 0) {
        shared_mem->tau_values[k] = tau;
        shared_mem->R_diag[k] = mu;
    }

    // -------------------------------------------------------------------------
    // 5. NORMALIZE AND RETURN
    // -------------------------------------------------------------------------
    if (r_idx == pivot_row) {
        return 1.0f; // The pivot of the Householder vector is always exactly 1.0
    } else if (r_idx > pivot_row && r_idx < M) {
        return x_local / v_pivot; // Normalize the tail
    } else {
        return 0.0f; // Strict upper triangle is zeroed out
    }
}

// =============================================================================
// SECTION 5: AMX MATRIX COPROCESSOR WRAPPERS (simdgroup_matrix)
// =============================================================================

/**
 * Loads an 8x8 float32 tile from Global Device Memory into the AMX registers.
 *
 * @param tile Reference to the hardware AMX matrix struct (stored in registers).
 * @param src Pointer to the start of the device memory buffer.
 * @param stride The number of elements per row (or column, depending on layout)
 * in the source matrix. Crucial for dynamic M and N.
 * @param col The starting column index of the 8x8 block.
 * @param row The starting row index of the 8x8 block.
 */
inline void load_tile_device(
    thread simdgroup_float8x8& tile,
    const device float* src,
    uint stride,
    uint col,
    uint row,
    bool transpose = false // Add this
) {
    simdgroup_load(tile, src, stride, ulong2(row, col), !transpose);
}

/**
 * load_tile_threadgroup
 * Loads an 8x8 AMX tile from Threadgroup (Shared) Memory.
 * * @param tile      The destination simdgroup_matrix (AMX registers).
 * @param base_ptr  Pointer to the start of the matrix in shared memory.
 * @param stride    The number of elements between consecutive rows in memory.
 * @param col       The starting column index for the 8x8 block.
 * @param row       The starting row index for the 8x8 block.
 * @param transpose If true, the tile is logically transposed during the load.
 */
template <typename T>
inline void load_tile_threadgroup(
    thread simdgroup_matrix<T, 8, 8>& tile,
    const threadgroup T* base_ptr,
    uint stride,
    uint col,
    uint row,
    bool transpose
) {
    const threadgroup T* tile_ptr = base_ptr + (ulong)col * stride + row;
    simdgroup_load(tile, tile_ptr, (ulong)stride, ulong2(0), !transpose);
}

/**
 * Stores an 8x8 float32 tile from AMX registers back to Global Device Memory.
 * Used at the end of the Outer Product Update to write the newly computed
 * Trailing Matrix values.
 */
inline void store_tile_device(
    thread simdgroup_float8x8 tile,
    device float* dst,
    uint stride,
    uint col,
    uint row
) {
    simdgroup_store(tile, dst, stride, ulong2(row, col), true);
}

/**
 * Executes a massively parallel AMX Matrix Multiply-Accumulate.
 * Computes: D = D + (A * B)
 *
 * In our algorithm, this is used for:
 * 1. Z = Z + (W^T * A_chunk)
 * 2. A_chunk = A_chunk - (Y * Z)
 */
inline void amx_mac_8x8(
    thread simdgroup_float8x8& D,
    const thread simdgroup_float8x8& A,
    const thread simdgroup_float8x8& B
) {
    // simdgroup_multiply_accumulate natively uses the Apple Tensor Cores.
    // Latency is hidden by the GPU scheduler if multiple tiles are in flight.
    simdgroup_multiply_accumulate(D, A, B, D);
}

/**
 * Executes a massively parallel AMX Matrix Multiply (No accumulation).
 * Computes: D = A * B
 * * Used specifically for applying the upper triangular T matrix: Z = T * Z.
 */
inline void amx_mul_8x8(
    thread simdgroup_float8x8& D,
    const thread simdgroup_float8x8& A,
    const thread simdgroup_float8x8& B
) {
    simdgroup_multiply(D, A, B);
}

/**
 * Zeros out an AMX tile.
 * Required for initializing the Z accumulators before Pass 1 of the update.
 */
inline void clear_tile(thread simdgroup_float8x8& tile) {
    // By initializing a tile with 0.0f, the compiler optimally zeros the registers.
    // Standard MSL doesn't have a specific "simdgroup_clear" function.
    tile = simdgroup_float8x8(0.0f);
}

// =============================================================================
// SECTION 6: THE MAIN KERNEL ENTRY POINT
// =============================================================================

// Pre-process: Row-major -> Column-major with padding
kernel void pack_batch_memory(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant uint& original_M [[buffer(2)]],
    constant uint& original_N [[buffer(3)]],
    constant uint& M_pad [[buffer(4)]],
    constant uint& N_pad [[buffer(5)]],
    uint3 tid [[thread_position_in_grid]]
) {
    uint col = tid.x;
    uint row = tid.y;
    uint b = tid.z;

    if (row < original_M && col < original_N) {
        // Valid data from the user
        uint src_idx = b * (original_M * original_N) + row * original_N + col;
        uint dst_idx = b * (M_pad * N_pad) + col * M_pad + row;
        dst[dst_idx] = src[src_idx];
    } else if (row < M_pad && col < N_pad) {
        // Explicitly clear the padded region with an Identity matrix
        // Prevents uninitialized memory from creating NaNs during AMX tile loading
        uint dst_idx = b * (M_pad * N_pad) + col * M_pad + row;
        dst[dst_idx] = (row == col) ? 1.0f : 0.0f;
    }
}

// Post-process: Extract R (Upper Triangular) and convert back to Row-major
kernel void unpack_batch_R(
    device const float* src_A [[buffer(0)]],
    device float* dst_R [[buffer(1)]],
    constant uint& original_M [[buffer(2)]],
    constant uint& original_N [[buffer(3)]],
    constant uint& original_K [[buffer(4)]],
    constant uint& M_pad [[buffer(5)]],
    constant uint& N_pad [[buffer(6)]],
    uint3 tid [[thread_position_in_grid]]
) {
    uint col = tid.x;
    uint row = tid.y;
    uint b = tid.z;

    if (row < original_K && col < original_N) {
        uint src_idx = b * (M_pad * N_pad) + col * M_pad + row;
        uint dst_idx = b * (original_K * original_N) + row * original_N + col;

        // Zero out the strictly lower triangle for R
        dst_R[dst_idx] = (row <= col) ? src_A[src_idx] : 0.0f;
    }
}

// Initializes the padded Q matrix to the Identity matrix
kernel void init_identity_batch(
    device float* Q_dst [[buffer(0)]],
    constant uint& M_pad [[buffer(1)]],
    constant uint& N_pad [[buffer(2)]],
    uint3 tid [[thread_position_in_grid]]
) {
    uint col = tid.x;
    uint row = tid.y;
    uint b = tid.z;

    if (row < M_pad && col < M_pad) {
        uint dst_idx = b * (M_pad * M_pad) + col * M_pad + row;
        Q_dst[dst_idx] = (row == col) ? 1.0f : 0.0f;
    }
}

// Extracts Q and converts back to Row-major
kernel void unpack_batch_Q(
    device const float* src_Q [[buffer(0)]],
    device float* dst_Q [[buffer(1)]],
    constant uint& original_M [[buffer(2)]],
    constant uint& original_K [[buffer(3)]],
    constant uint& M_pad [[buffer(4)]],
    uint3 tid [[thread_position_in_grid]]
) {
    uint col = tid.x; // original_K
    uint row = tid.y; // original_M
    uint b = tid.z;

    if (row < original_M && col < original_K) {
        // FIX: Read natively! memory[col * M_pad + row] is Q_{row, col}
        uint src_idx = b * (M_pad * M_pad) + row * M_pad + col;

        // Write to unpadded row-major
        uint dst_idx = b * (original_M * original_K) + row * original_K + col;

        dst_Q[dst_idx] = src_Q[src_idx];
    }
}

/**
 * @brief Standard Block Householder QR Factorization with Explicit Q Accumulation.
 *
 * This kernel computes the QR decomposition of a batch of matrices using a blocked
 * Householder transformation. Unlike standard implementations that only return
 * the R factor and a set of Householder reflectors, this kernel performs "Explicit
 * Accumulation" to generate the orthogonal Q matrix directly on the GPU.
 *
 * @section math Mathematical Formulation
 * For a given input matrix $A \in \mathbb{R}^{M \times N}$, the kernel computes:
 * 1.  $A = QR$, where $Q \in O(M)$ and $R \in \mathbb{R}^{M \times N}$ is upper triangular.
 * 2.  The kernel specifically produces $Q^T$ by applying the same sequence of Householder
 * reflections $H_n \dots H_2 H_1$ to an Identity matrix $I$.
 * 3.  The algorithm uses a "Left-Looking" panel factorization combined with a
 * "Compact WY" representation ($I - YTY^T$) for block updates.
 *
 * @section performance Architectural Optimizations
 * - **AMX Acceleration:** Trailing matrix updates and Q accumulation are dispatched
 * to the Apple Matrix Coprocessor using 8x8 `simdgroup_matrix` tiles.
 * - **Haar-Uniform Compliance:** All column pivoting logic has been removed to ensure
 * the resulting $Q$ matrix is an unbiased sample from the Haar measure (essential
 * for SO(d) Gaussian sampling).
 * - **Unified Memory Efficiency:** By accumulating $Q$ in-place on the GPU, we
 * eliminate the $O(d^3)$ CPU-side matrix inversion typically required to recover
 * $Q$ from $A$ and $R$.
 *
 * @param A [in/out] Device pointer to the input matrix. On output, the upper triangle
 * contains $R$, and the strictly lower triangle contains the Householder
 * vectors ($Y$).
 * @param Q_trans [in/out] Device pointer to a matrix initialized as Identity ($I$).
 * On output, contains the transposed orthogonal matrix $Q^T$.
 * @param shared_mem [threadgroup] L1 cache workspace for T-matrix construction,
 * reduction operations, and AMX staging.
 * @param grid_pos Threadgroup position used for batch indexing (Z-dimension).
 *
 * @note This kernel requires the input matrices to be pre-padded to multiples of
 * SIMD_GROUP_SIZE (32) for rows and BLOCK_SIZE (16) for columns to maintain
 * SIMD-coalesced memory access.
 */
kernel void standard_householder_qr_float32(
    device float* A          [[buffer(0)]],
    device float* Q_trans    [[buffer(1)]],
    threadgroup QRSharedMemory* shared_mem [[threadgroup(0)]],
    uint3 grid_pos [[threadgroup_position_in_grid]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint3 tg_size [[threads_per_threadgroup]]
) {
    // -------------------------------------------------------------------------
    // BATCHING & INITIALIZATION
    // -------------------------------------------------------------------------
    uint batch_idx = grid_pos.z;

    // FIX: A and Q_trans have different padded memory footprints!
    A += batch_idx * (M_pad * N_pad);
    Q_trans += batch_idx * (M_pad * M_pad);

    if (simd_group_id == 0) {
        if (simd_lane_id < MAX_SIMD_GROUPS) shared_mem->reduction_space[simd_lane_id] = 0.0f;
        for (uint i = simd_lane_id; i < (BLOCK_SIZE * BLOCK_SIZE); i += SIMD_GROUP_SIZE) {
            shared_mem->compact_T[i] = 0.0f;
            shared_mem->clean_Y[i] = 0.0f;
        }
        // Explicitly zero the scalars to prevent NaN propagation <---
        if (simd_lane_id < BLOCK_SIZE) {
             shared_mem->tau_values[simd_lane_id] = 0.0f;
             shared_mem->R_diag[simd_lane_id] = 1.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // -------------------------------------------------------------------------
    // THE MASTER BLOCK LOOP
    // -------------------------------------------------------------------------
    uint total_threads = tg_size.x * tg_size.y;
    uint total_simd_groups = total_threads / 32;
    uint tid = simd_group_id * 32 + simd_lane_id;
    uint min_dim = (M < N) ? M : N;

    for (uint block_start = 0; block_start < min_dim; block_start += BLOCK_SIZE) {

        // =====================================================================
        // STEP 3: INFINITE-M STREAMING LEFT-LOOKING PANEL
        // =====================================================================
        for (uint k = 0; k < BLOCK_SIZE && (block_start + k) < min_dim; ++k) {
            uint current_col = block_start + k;
            uint pivot_row = current_col;

            // --- PASS 1: GLOBAL NORM COMPUTATION ---
            float local_x_norm2 = 0.0f;
            float pivot_val = 0.0f;

            for (uint r = pivot_row + tid; r < M; r += total_threads) {
                float val = A[r + current_col * M_pad]; // FIX: M_pad
                if (r == pivot_row) pivot_val = val;
                else local_x_norm2 += val * val;
            }

            float sg_norm2 = simd_sum(local_x_norm2);
            if (simd_lane_id == 0) shared_mem->reduction_space[simd_group_id] = sg_norm2;

            if (tid == 0) shared_mem->temp_scalar = pivot_val;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float total_norm2 = 0.0f;
            if (simd_group_id == 0) {
                float val = (simd_lane_id < total_simd_groups) ? shared_mem->reduction_space[simd_lane_id] : 0.0f;
                total_norm2 = simd_sum(val);
                if (simd_lane_id == 0) shared_mem->reduction_space[0] = total_norm2;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            total_norm2 = shared_mem->reduction_space[0];
            pivot_val = shared_mem->temp_scalar;

            float tau = 0.0f;
            float beta = pivot_val;

            if (total_norm2 > EPSILON) {
                float norm_x = sqrt(pivot_val * pivot_val + total_norm2);
                beta = (pivot_val <= 0.0f) ? norm_x : -norm_x;
                tau = (beta - pivot_val) / beta;
                float scale = 1.0f / (pivot_val - beta);

                // --- PASS 1.5: SCALE REFLECTOR VECTOR ---
                for (uint r = pivot_row + 1 + tid; r < M; r += total_threads) { // FIX: total_threads
                    A[r + current_col * M_pad] *= scale; // FIX: M_pad
                }
            }

            if (tid == 0) {
                shared_mem->R_diag[k] = beta;
                shared_mem->tau_values[k] = tau;
                A[pivot_row + current_col * M_pad] = 1.0f; // FIX: M_pad
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // --- PASS 2: GLOBAL DOT PRODUCTS ---
            for (uint j = k + 1 + simd_group_id; j < BLOCK_SIZE && (block_start + j) < N; j += total_simd_groups) {
                uint target_col = block_start + j;
                float local_dot = 0.0f;
                for (uint r = pivot_row + simd_lane_id; r < M; r += SIMD_GROUP_SIZE) {
                    local_dot += A[r + current_col * M_pad] * A[r + target_col * M_pad];
                }
                float total_dot = simd_sum(local_dot);
                if (simd_lane_id == 0) shared_mem->reduction_space[j] = total_dot;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // --- PASS 3: APPLY PANEL UPDATES ---
            tau = shared_mem->tau_values[k];
            for (uint j_idx = k + 1; j_idx < BLOCK_SIZE && (block_start + j_idx) < N; ++j_idx) {
                float update_factor = tau * shared_mem->reduction_space[j_idx];
                uint target_col = block_start + j_idx;
                for (uint r = pivot_row + tid; r < M; r += total_threads) { // FIX: total_threads
                    A[r + target_col * M_pad] -= update_factor * A[r + current_col * M_pad]; // FIX: M_pad
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // =====================================================================
        // STEP 4: FORM T-MATRIX
        // =====================================================================
        for (uint k = 0; k < BLOCK_SIZE; ++k) {
            for (uint j_sg = simd_group_id; j_sg < k; j_sg += total_simd_groups) {
                uint v_curr_col = block_start + k;
                uint y_col = block_start + j_sg;
                float local_dot = 0.0f;
                for (uint r = (block_start + k) + simd_lane_id; r < M; r += SIMD_GROUP_SIZE) {
                    local_dot += A[r + y_col * M_pad] * A[r + v_curr_col * M_pad];
                }
                float total_dot = simd_sum(local_dot);
                if (simd_lane_id == 0) shared_mem->compact_T[j_sg + k * BLOCK_SIZE] = total_dot;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (simd_group_id == 0 && simd_lane_id == 0) {
                float tau_k = shared_mem->tau_values[k];
                for (uint i = 0; i < k; ++i) {
                    float sum_tk = 0.0f;
                    for (uint j_idx = i; j_idx < k; ++j_idx) {
                        sum_tk += shared_mem->compact_T[i + j_idx * BLOCK_SIZE] * shared_mem->compact_T[j_idx + k * BLOCK_SIZE];
                    }
                    shared_mem->compact_T[i + k * BLOCK_SIZE] = -tau_k * sum_tk;
                }
                shared_mem->compact_T[k + k * BLOCK_SIZE] = tau_k;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (simd_group_id == 0) {
            #pragma unroll
            for (uint i = simd_lane_id; i < (BLOCK_SIZE * BLOCK_SIZE); i += SIMD_GROUP_SIZE) {
                shared_mem->compact_T[i] = -shared_mem->compact_T[i];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // =====================================================================
        // STEP 4.5: BUILD CLEAN 'Y'
        // =====================================================================
        if (simd_group_id == 0) {
            #pragma unroll
            for (uint i = simd_lane_id; i < (BLOCK_SIZE * BLOCK_SIZE); i += SIMD_GROUP_SIZE) {
                uint r_local = i % BLOCK_SIZE;
                uint c_local = i / BLOCK_SIZE;
                if (r_local > c_local) {
                    shared_mem->clean_Y[i] = A[(block_start + r_local) + (block_start + c_local) * M_pad]; // FIX: M_pad
                } else if (r_local == c_local) {
                    shared_mem->clean_Y[i] = 1.0f;
                } else {
                    shared_mem->clean_Y[i] = 0.0f;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // =====================================================================
        // STEP 5: AMX TRAILING UPDATE
        // =====================================================================
        for (uint trailing_col = block_start + BLOCK_SIZE + (simd_group_id * TILE_DIM);
             trailing_col < N;
             trailing_col += (total_simd_groups * TILE_DIM)) {

            simdgroup_float8x8 Z_top  = make_filled_simdgroup_matrix<float, 8>(0.0f);
            simdgroup_float8x8 Z_bot  = make_filled_simdgroup_matrix<float, 8>(0.0f);
            simdgroup_float8x8 ZP_top = make_filled_simdgroup_matrix<float, 8>(0.0f);
            simdgroup_float8x8 ZP_bot = make_filled_simdgroup_matrix<float, 8>(0.0f);
            simdgroup_float8x8 Op_A, Op_B;

            for (uint row = block_start; row < M; row += TILE_DIM) {
                load_tile_device(Op_B, A, M_pad, trailing_col, row, false); // FIX: M_pad

                if (row < block_start + BLOCK_SIZE) {
                    uint local_row = row - block_start;
                    load_tile_threadgroup(Op_A, shared_mem->clean_Y, BLOCK_SIZE, 0, local_row, true);
                    simdgroup_multiply_accumulate(Z_top, Op_A, Op_B, Z_top);
                    load_tile_threadgroup(Op_A, shared_mem->clean_Y, BLOCK_SIZE, TILE_DIM, local_row, true);
                    simdgroup_multiply_accumulate(Z_bot, Op_A, Op_B, Z_bot);
                } else {
                    load_tile_device(Op_A, A, M_pad, block_start, row, true); // FIX: M_pad
                    simdgroup_multiply_accumulate(Z_top, Op_A, Op_B, Z_top);
                    load_tile_device(Op_A, A, M_pad, block_start + TILE_DIM, row, true); // FIX: M_pad
                    simdgroup_multiply_accumulate(Z_bot, Op_A, Op_B, Z_bot);
                }
            }

            load_tile_threadgroup(Op_A, shared_mem->compact_T, BLOCK_SIZE, 0, 0, true);
            simdgroup_multiply_accumulate(ZP_top, Op_A, Z_top, ZP_top);
            load_tile_threadgroup(Op_A, shared_mem->compact_T, BLOCK_SIZE, 0, TILE_DIM, true);
            simdgroup_multiply_accumulate(ZP_top, Op_A, Z_bot, ZP_top);

            load_tile_threadgroup(Op_A, shared_mem->compact_T, BLOCK_SIZE, TILE_DIM, 0, true);
            simdgroup_multiply_accumulate(ZP_bot, Op_A, Z_top, ZP_bot);
            load_tile_threadgroup(Op_A, shared_mem->compact_T, BLOCK_SIZE, TILE_DIM, TILE_DIM, true);
            simdgroup_multiply_accumulate(ZP_bot, Op_A, Z_bot, ZP_bot);

            for (uint row = block_start; row < M; row += TILE_DIM) {
                load_tile_device(Op_B, A, M_pad, trailing_col, row, false); // FIX: M_pad

                if (row < block_start + BLOCK_SIZE) {
                    uint local_row = row - block_start;
                    load_tile_threadgroup(Op_A, shared_mem->clean_Y, BLOCK_SIZE, 0, local_row, false);
                    simdgroup_multiply_accumulate(Op_B, Op_A, ZP_top, Op_B);
                    load_tile_threadgroup(Op_A, shared_mem->clean_Y, BLOCK_SIZE, TILE_DIM, local_row, false);
                    simdgroup_multiply_accumulate(Op_B, Op_A, ZP_bot, Op_B);
                } else {
                    load_tile_device(Op_A, A, M_pad, block_start, row, false); // FIX: M_pad
                    simdgroup_multiply_accumulate(Op_B, Op_A, ZP_top, Op_B);
                    load_tile_device(Op_A, A, M_pad, block_start + TILE_DIM, row, false); // FIX: M_pad
                    simdgroup_multiply_accumulate(Op_B, Op_A, ZP_bot, Op_B);
                }
                store_tile_device(Op_B, A, M_pad, trailing_col, row); // FIX: M_pad
            }
        }

        // =====================================================================
        // STEP 5.5: EXPLICIT Q ACCUMULATION
        // =====================================================================
        // FIX: Bounded safely to M_pad to prevent reading out of bounds on wide matrices
        for (uint q_col = (simd_group_id * TILE_DIM);
             q_col < M_pad;
             q_col += (total_simd_groups * TILE_DIM)) {

            simdgroup_float8x8 Z_top  = make_filled_simdgroup_matrix<float, 8>(0.0f);
            simdgroup_float8x8 Z_bot  = make_filled_simdgroup_matrix<float, 8>(0.0f);
            simdgroup_float8x8 ZP_top = make_filled_simdgroup_matrix<float, 8>(0.0f);
            simdgroup_float8x8 ZP_bot = make_filled_simdgroup_matrix<float, 8>(0.0f);
            simdgroup_float8x8 Op_A, Op_B;

            for (uint row = block_start; row < M; row += TILE_DIM) {
                load_tile_device(Op_B, Q_trans, M_pad, q_col, row, false); // FIX: M_pad

                if (row < block_start + BLOCK_SIZE) {
                    uint local_row = row - block_start;
                    load_tile_threadgroup(Op_A, shared_mem->clean_Y, BLOCK_SIZE, 0, local_row, true);
                    simdgroup_multiply_accumulate(Z_top, Op_A, Op_B, Z_top);
                    load_tile_threadgroup(Op_A, shared_mem->clean_Y, BLOCK_SIZE, TILE_DIM, local_row, true);
                    simdgroup_multiply_accumulate(Z_bot, Op_A, Op_B, Z_bot);
                } else {
                    load_tile_device(Op_A, A, M_pad, block_start, row, true); // FIX: M_pad
                    simdgroup_multiply_accumulate(Z_top, Op_A, Op_B, Z_top);
                    load_tile_device(Op_A, A, M_pad, block_start + TILE_DIM, row, true); // FIX: M_pad
                    simdgroup_multiply_accumulate(Z_bot, Op_A, Op_B, Z_bot);
                }
            }

            load_tile_threadgroup(Op_A, shared_mem->compact_T, BLOCK_SIZE, 0, 0, true);
            simdgroup_multiply_accumulate(ZP_top, Op_A, Z_top, ZP_top);
            load_tile_threadgroup(Op_A, shared_mem->compact_T, BLOCK_SIZE, 0, TILE_DIM, true);
            simdgroup_multiply_accumulate(ZP_top, Op_A, Z_bot, ZP_top);

            load_tile_threadgroup(Op_A, shared_mem->compact_T, BLOCK_SIZE, TILE_DIM, 0, true);
            simdgroup_multiply_accumulate(ZP_bot, Op_A, Z_top, ZP_bot);
            load_tile_threadgroup(Op_A, shared_mem->compact_T, BLOCK_SIZE, TILE_DIM, TILE_DIM, true);
            simdgroup_multiply_accumulate(ZP_bot, Op_A, Z_bot, ZP_bot);

            for (uint row = block_start; row < M; row += TILE_DIM) {
                load_tile_device(Op_B, Q_trans, M_pad, q_col, row, false); // FIX: M_pad

                if (row < block_start + BLOCK_SIZE) {
                    uint local_row = row - block_start;
                    load_tile_threadgroup(Op_A, shared_mem->clean_Y, BLOCK_SIZE, 0, local_row, false);
                    simdgroup_multiply_accumulate(Op_B, Op_A, ZP_top, Op_B);
                    load_tile_threadgroup(Op_A, shared_mem->clean_Y, BLOCK_SIZE, TILE_DIM, local_row, false);
                    simdgroup_multiply_accumulate(Op_B, Op_A, ZP_bot, Op_B);
                } else {
                    load_tile_device(Op_A, A, M_pad, block_start, row, false); // FIX: M_pad
                    simdgroup_multiply_accumulate(Op_B, Op_A, ZP_top, Op_B);
                    load_tile_device(Op_A, A, M_pad, block_start + TILE_DIM, row, false); // FIX: M_pad
                    simdgroup_multiply_accumulate(Op_B, Op_A, ZP_bot, Op_B);
                }
                store_tile_device(Op_B, Q_trans, M_pad, q_col, row); // FIX: M_pad
            }
        }

        // =====================================================================
        // STEP 6: RESTORE R DIAGONAL
        // =====================================================================
        if (simd_group_id == 0 && simd_lane_id == 0) {
            for (uint k = 0; k < BLOCK_SIZE && (block_start + k) < min_dim; ++k) {
                A[(block_start + k) + (block_start + k) * M_pad] = shared_mem->R_diag[k];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}