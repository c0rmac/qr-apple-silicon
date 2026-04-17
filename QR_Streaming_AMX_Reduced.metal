#include <metal_stdlib>

using namespace metal;

// =============================================================================
// SECTION 1: COMPILER DIRECTIVES & CONSTANTS (HIGH-DIMENSIONAL TUNED)
// =============================================================================

/**
 * BLOCK_SIZE (b): Increased from 16 to 32.
 * For a 5000x5000 matrix, a block size of 16 requires too many iterations (312).
 * 32 is the optimal "sweet spot" for Apple Silicon. It perfectly matches the
 * SIMD group size and allows the AMX engine to process larger 32x32 tiles,
 * drastically reducing memory round-trips.
 */
#define BLOCK_SIZE 32

/**
 * SIMD_GROUP_SIZE: Native hardware execution width of Apple Silicon (AGX).
 * Must remain 32.
 */
#define SIMD_GROUP_SIZE 32

/**
 * TILE_DIM: The hardware-accelerated dimension for float32 AMX tiles.
 * simdgroup_matrix instructions map to 8x8 registers.
 */
#define TILE_DIM 8

/**
 * MAX_SIMD_GROUPS: Defines the maximum threadgroup size.
 * For the Panel kernel, we might still use up to 1024 threads (32 groups).
 * For the Update kernel, we will typically use 128 or 256 threads per tile.
 */
#define MAX_SIMD_GROUPS 32

/**
 * PAD_OFFSET: Used to prevent Shared Memory Bank Conflicts.
 * By padding 2D arrays in threadgroup memory, we force consecutive rows to
 * stagger across different memory banks, allowing the AMX engines to load
 * data at maximum bandwidth without serialization stalls.
 */
#define PAD_OFFSET 4

#define EPSILON 1e-7f

// Dynamic constants injected by MLX at compile time.
// Matrices must be pre-padded to multiples of 32.
constant uint M [[function_constant(0)]];
constant uint N [[function_constant(1)]];

constant uint K_dim = (M < N) ? M : N;

// =============================================================================
// SECTION 2: THE TIER-2 MEMORY STRUCTURES (SPECIALIZED SHARED MEMORY)
// =============================================================================

/**
 * PanelSharedMemory
 * Used EXCLUSIVELY by `kernel void panel_factorization`.
 * Since the Panel kernel focuses on generating Householder vectors (Y) and
 * doesn't do massive matrix multiplication, it only needs space for SIMD
 * reductions and storing the 32 tau values.
 */
struct alignas(16) PanelSharedMemory {
    // Stores the tau scalars for the current 32-column block
    float tau_values[BLOCK_SIZE];

    // Used to recover the R-diagonal without hitting global memory
    float R_diag[BLOCK_SIZE];

    // L1 reduction space for global norm calculations across 1024 threads
    float reduction_space[MAX_SIMD_GROUPS];

    // Shared scalar for broadcasting pivot values across the threadgroup
    float temp_scalar;
};

struct alignas(16) HaarSharedMemory {
    float reduction_space[MAX_SIMD_GROUPS];
};

/**
 * UpdateSharedMemory
 * Used EXCLUSIVELY by `kernel void grid_parallel_update`.
 * This is your "Cache Blocking" structure. It holds the 32x32 T-matrix and
 * the 32x32 Y-matrix tile in ultra-fast L1 cache so the AMX units can sweep
 * over them repeatedly without thrashing the 100MB+ global device memory.
 * * Notice the PAD_OFFSET: The stride is (BLOCK_SIZE + PAD_OFFSET) = 36.
 * This guarantees bank-conflict-free simdgroup_loads.
 */
struct alignas(16) UpdateSharedMemory {
    // The Compact WY triangular matrix (T)
    // Size: 32 rows x 36 stride = 1,152 floats (~4.6 KB)
    float compact_T[BLOCK_SIZE * (BLOCK_SIZE + PAD_OFFSET)];

    // A cached tile of the Y vectors to apply to the Trailing Matrix
    // Size: 32 rows x 36 stride = 1,152 floats (~4.6 KB)
    float cached_Y[BLOCK_SIZE * (BLOCK_SIZE + PAD_OFFSET)];

    // Total Size: ~9.2 KB.
    // This leaves massive headroom under Apple's 32 KB limit, allowing
    // multiple threadgroups to reside on the same GPU core concurrently to
    // hide instruction latency.
};

// =============================================================================
// SECTION 3: SIMD-LEVEL REDUCTION PRIMITIVES (HAAR-UNIFORM OPTIMIZED)
// =============================================================================

/**
 * threadgroup_sum_reduce
 * ----------------------
 * Computes the sum of a float value across all active threads in the threadgroup.
 * This is the workhorse for computing the L2 norm of the 5000-element column
 * tails during the Panel Factorization phase.
 *
 * @param local_val The thread-local partial sum to be reduced.
 * @param red_space Pointer to `shared_mem->reduction_space` (size MAX_SIMD_GROUPS).
 * @param sg_id     The SIMD group index (0 to 31).
 * @param lane_id   The lane index within the SIMD group (0 to 31).
 */
inline float threadgroup_sum_reduce(
    float local_val,
    threadgroup float* red_space,
    uint sg_id,
    uint lane_id
) {
    // Phase 1: Intra-SIMD Reduction
    // Apple's native MSL intrinsic compiles directly to tensor core register
    // shuffles. This is a zero-latency operation.
    float simd_val = simd_sum(local_val);

    // Phase 2: Inter-SIMD Communication
    // The "leader" (lane 0) of each of the 32 SIMD groups writes its partial
    // sum to the Tier-2 threadgroup memory (L1 cache).
    if (lane_id == 0) {
        red_space[sg_id] = simd_val;
    }

    // Sync to ensure all active SIMD groups have written their partial sums.
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: Final Sweep
    // Only the first SIMD group (sg_id == 0) wakes up. It reads the partial
    // results left by the other groups and performs one final register sum.
    if (sg_id == 0) {
        // Pull the partial sum corresponding to this lane's ID.
        // If the threadgroup was launched with fewer than 1024 threads,
        // bounds check against MAX_SIMD_GROUPS to prevent garbage data.
        float val = (lane_id < MAX_SIMD_GROUPS) ? red_space[lane_id] : 0.0f;
        float total = simd_sum(val);

        // Write the absolute total back to slot 0 for broadcasting.
        if (lane_id == 0) {
            red_space[0] = total;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    //  Secure the local copy before anyone leaves the function
    float result = red_space[0];

    // Ensure all threads read the result before the next loop iteration overwrites it
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return result;
}

// =============================================================================
// SECTION 4: BLOCK HOUSEHOLDER MATH HELPERS (HAAR-UNIFORM OPTIMIZED)
// =============================================================================

// =============================================================================
// SECTION 5: AMX MATRIX COPROCESSOR WRAPPERS (GRID-PARALLEL OPTIMIZED)
// =============================================================================

/**
 * clear_tile
 * ----------
 * Zeros out an 8x8 AMX register tile.
 * Crucial for initializing the Z accumulators before the multi-pass
 * multiply-accumulate sweeps.
 */
inline void clear_tile(thread simdgroup_float8x8& tile) {
    tile = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
}

/**
 * load_tile_device
 * ----------------
 * Loads an 8x8 float32 tile directly from Global Device Memory (100MB+ pool).
 * In the high-dimensional grid-parallel kernel, this is primarily used
 * to load the chunks of the A and Q matrices that are actively being updated.
 *
 * @param transpose If true, the hardware transposes the 8x8 block on the fly
 * during the load (zero-cost operation on Apple Silicon).
 */
inline void load_tile_device(
    thread simdgroup_float8x8& tile,
    const device float* src,
    uint stride,
    uint col,
    uint row,
    bool transpose = false
) {
    // ulong2 cast is strictly required to prevent overflow on 5000x5000 matrices
    simdgroup_load(tile, src, (ulong)stride, ulong2((ulong)row, (ulong)col), transpose);
}

/**
 * load_tile_threadgroup (The Cache-Block Loader)
 * ----------------------------------------------
 * Loads an 8x8 AMX tile from the ultra-fast Threadgroup Memory (L1 Cache).
 * * CRITICAL FIX: Notice the `stride` parameter. When you call this from your
 * Update kernel, you MUST pass `(BLOCK_SIZE + PAD_OFFSET)` as the stride to
 * avoid shared memory bank conflicts.
 */
inline void load_tile_threadgroup(
    thread simdgroup_float8x8& tile,
    const threadgroup float* base_ptr,
    uint stride,
    uint col,
    uint row,
    bool transpose = false
) {
    // We compute the linear offset to the start of the tile, then load.
    // The hardware automatically understands the physical 8x8 mapping.

    // Shared memory is Row-Major within the block; compute the linear offset.
    const threadgroup float* tile_ptr = base_ptr + ((ulong)row * (ulong)stride) + (ulong)col;

    // Threadgroup loads require ulong2(0,0) with a pre-offset pointer.
    simdgroup_load(tile, tile_ptr, (ulong)stride, ulong2(0, 0), transpose);
}

/**
 * store_tile_device
 * -----------------
 * Flushes an 8x8 AMX accumulator tile back to Global Device Memory.
 * This is the final step of the Update kernel after all cache-blocked
 * math operations are finished.
 */
inline void store_tile_device(
    thread simdgroup_float8x8 tile,
    device float* dst,
    uint stride,
    uint col,
    uint row,
    bool transpose = false
) {
    simdgroup_store(tile, dst, stride, ulong2((ulong)row, (ulong)col), transpose);
}

/**
 * amx_mac_8x8
 * -----------
 * Executes a massively parallel AMX Matrix Multiply-Accumulate.
 * Computes: D = D + (A * B)
 *
 * In the new Kernel C (Update), this handles:
 * 1. Z = Y^T * A   (Building the intermediate)
 * 2. A = A - Y * Z (Applying the update)
 */
inline void amx_mac_8x8(
    thread simdgroup_float8x8& D,
    const thread simdgroup_float8x8& A,
    const thread simdgroup_float8x8& B
) {
    simdgroup_multiply_accumulate(D, A, B, D);
}

/**
 * amx_mul_8x8
 * -----------
 * Executes a massively parallel AMX Matrix Multiply without accumulation.
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

// =============================================================================
// SECTION 6: THE MULTI-PASS KERNEL PIPELINE (GRID-PARALLEL)
// =============================================================================


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

// Extracts Q and converts back to Row-major
kernel void unpack_batch_Q(
    device const float* src_Q [[buffer(0)]],
    device float* dst_Q [[buffer(1)]],
    constant uint& original_M [[buffer(2)]], // Rows in final Q (e.g., 4608)
    constant uint& original_K [[buffer(3)]], // Columns in final Q (e.g., 512)
    constant uint& M_pad      [[buffer(4)]], // Padded rows in buffer (multiple of 32)
    constant uint& K_pad      [[buffer(5)]], // Padded columns in buffer (multiple of 32)
    uint3 tid [[thread_position_in_grid]]
) {
    uint col = tid.x;
    uint row = tid.y;
    uint b   = tid.z;

    // Boundary check for the final output dimensions
    if (row < original_M && col < original_K) {
        // 1. Calculate Source Index (Padded Column-Major)
        // Stride is M_pad, and we jump by full (M_pad * N_pad) blocks for each batch.
        ulong src_idx = (ulong)b * (M_pad * K_pad) + (ulong)col * M_pad + row;

        // 2. Calculate Destination Index (Row-Major)
        // Stride is original_K, and we jump by (original_M * original_K) for each batch.
        ulong dst_idx = (ulong)b * (original_M * original_K) + (ulong)row * original_K + col;

        // 3. Perform the copy
        dst_Q[dst_idx] = src_Q[src_idx];
    }
}

// -----------------------------------------------------------------------------
// KERNEL 0A: ROW-TO-COLUMN MAJOR TRANSPOSE (GPU DATA PREP)
// -----------------------------------------------------------------------------
// Takes MLX row-major tensor and safely converts it to column-major.
kernel void preprocess_transpose(
    device const float* src_row_major [[buffer(0)]],
    device float* dst_col_major [[buffer(1)]],
    constant uint& original_M [[buffer(2)]],
    constant uint& original_N [[buffer(3)]],
    uint3 pos [[thread_position_in_grid]]
) {
    uint col = pos.x;
    uint row = pos.y;
    uint batch = pos.z;

    if (row < original_M && col < original_N) {
        // Valid data from the user
        dst_col_major[(ulong)batch * M * N + col * M + row] =
            src_row_major[(ulong)batch * original_M * original_N + row * original_N + col];
    } else if (row < M && col < N) {
        // CRITICAL FIX: Pad the excess space with an Identity matrix
        // This stops the Householder reflections from dividing by zero
        // and bleeding padded zeros back into the valid M x N area.
        float val = (row == col) ? 1.0f : 0.0f;
        dst_col_major[(ulong)batch * M * N + col * M + row] = val;
    }
}

// -----------------------------------------------------------------------------
// KERNEL 0B: IDENTITY MATRIX INIT FOR Q
// -----------------------------------------------------------------------------
// Initializes the Q buffer as stacked identity matrices for explicitly forming Q
kernel void init_identity_q(
    device float* Q [[buffer(0)]],
    uint3 pos [[thread_position_in_grid]]
) {
    uint col = pos.x;
    uint row = pos.y;
    uint batch = pos.z;

    if (row < M && col < K_dim) {
        float val = (row == col) ? 1.0f : 0.0f;
        Q[(ulong)batch * M * K_dim + col * M + row] = val; // Stride is M * K_dim
    }
}

// -----------------------------------------------------------------------------
// KERNEL 1: PANEL FACTORIZATION
// -----------------------------------------------------------------------------
// Dispatched with: 1 Threadgroup per matrix in the batch (e.g., 64 groups total).
// Threads per group: 1024 (32 SIMD groups).
kernel void panel_factorization(
    device float* A [[buffer(0)]],
    device float* R_diags [[buffer(1)]],
    device float* tau_global [[buffer(2)]],
    constant uint& block_start [[buffer(3)]],
    uint3 grid_pos [[threadgroup_position_in_grid]],
    uint sg_id [[simdgroup_index_in_threadgroup]],
    uint lane_id [[thread_index_in_simdgroup]]
) {
    // CRITICAL FIX: Statically allocate L1 Cache to avoid C++ API dependency & zero-page traps
    threadgroup PanelSharedMemory shared_mem_inst;
    threadgroup PanelSharedMemory* shared_mem = &shared_mem_inst;

    uint batch_idx = grid_pos.z;

    // Prevent 32-bit overflow on massive matrices
    device float* A_batch = A + ((ulong)batch_idx * M * N);
    device float* R_batch = R_diags + ((ulong)batch_idx * M);
    device float* tau_batch = tau_global + ((ulong)batch_idx * N);

    uint tid = sg_id * SIMD_GROUP_SIZE + lane_id;

    for (uint k = 0; k < BLOCK_SIZE && (block_start + k) < N; ++k) {
        uint current_col = block_start + k;

        // =====================================================================
        // PASS 1: SAFE TAIL NORM ACCUMULATION
        // =====================================================================
        float local_tail_sq = 0.0f;
        float my_alpha = 0.0f;

        for (uint r = block_start + tid; r < M; r += 1024) {
            float x = A_batch[r + current_col * M];
            if (r > current_col) {
                local_tail_sq += x * x;
            } else if (r == current_col) {
                my_alpha = x;
            }
        }

        // Exactly ONE barrier-safe reduction per column
        float tail_norm_sq = threadgroup_sum_reduce(local_tail_sq, shared_mem->reduction_space, sg_id, lane_id);

        // CRITICAL FIX: Prevent Read-After-Write hazard on the shared L1 reduction space
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float alpha = threadgroup_sum_reduce(my_alpha, shared_mem->reduction_space, sg_id, lane_id);

        // Thread 0 computes the Householder scalars
        if (tid == 0) {
            if (tail_norm_sq <= EPSILON) {
                shared_mem->tau_values[k] = 0.0f;
                shared_mem->R_diag[k] = alpha;
                shared_mem->temp_scalar = 1.0f; // Prevent division by zero later
            } else {
                float norm_x = sqrt(alpha * alpha + tail_norm_sq);
                float mu = (alpha >= 0.0f) ? -norm_x : norm_x;
                float v_pivot = alpha - mu;

                shared_mem->tau_values[k] = -v_pivot / mu;
                shared_mem->R_diag[k] = mu;
                shared_mem->temp_scalar = v_pivot;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // =====================================================================
        // PASS 2: NORMALIZE VECTOR & PRESERVE R
        // =====================================================================
        float v_pivot = shared_mem->temp_scalar;
        float tau = shared_mem->tau_values[k];

        for (uint r = block_start + tid; r < M; r += 1024) {
            if (r > current_col) {
                if (tau > 0.0f) {
                    A_batch[r + current_col * M] /= v_pivot;
                } else {
                    A_batch[r + current_col * M] = 0.0f;
                }
            } else if (r == current_col) {
                // The thread holding the pivot writes R (modulo check removed)
                A_batch[current_col + current_col * M] = shared_mem->R_diag[k];
            }
        }
        threadgroup_barrier(mem_flags::mem_device);

        // =====================================================================
        // PASS 3: APPLY REFLECTOR TO THE REST OF THE PANEL
        // =====================================================================
        for (uint j = k + 1; j < BLOCK_SIZE && (block_start + j) < N; ++j) {
            uint target_col = block_start + j;

            float local_dot = 0.0f;
            for (uint r = block_start + tid; r < M; r += 1024) {
                if (r >= current_col) {
                    float v = (r == current_col) ? 1.0f : A_batch[r + current_col * M];
                    local_dot += v * A_batch[r + target_col * M];
                }
            }

            float total_dot = threadgroup_sum_reduce(local_dot, shared_mem->reduction_space, sg_id, lane_id);
            float update_factor = tau * total_dot;

            for (uint r = block_start + tid; r < M; r += 1024) {
                if (r >= current_col) {
                    float v = (r == current_col) ? 1.0f : A_batch[r + current_col * M];
                    A_batch[r + target_col * M] -= update_factor * v;
                }
            }

            // CRITICAL FIX: Loop-carried barrier to prevent reduction_space overwrites in the next column
            //threadgroup_barrier(mem_flags::mem_device);
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        }

        // =====================================================================
        // PASS 4: EXPORT TO GLOBAL MEMORY
        // =====================================================================
        if (tid == 0) {
            R_batch[current_col] = shared_mem->R_diag[k];
            tau_batch[current_col] = shared_mem->tau_values[k];
        }

        // Device-level sync so the Trailing Update kernels don't read stale L2 cache
        threadgroup_barrier(mem_flags::mem_device);
    }
}

// -----------------------------------------------------------------------------
// KERNEL 2: T-MATRIX CONSTRUCTION
// -----------------------------------------------------------------------------
// Dispatched with: 1 Threadgroup per matrix in the batch.
// Threads: 1024 (32 SIMD Groups)
kernel void compute_t_matrix(
    device const float* A [[buffer(0)]],
    device float* T_global [[buffer(1)]],         // Output: 32x32 T-Matrices
    device const float* tau_global [[buffer(2)]], // Input: Tau values from Kernel 1
    constant uint& block_start [[buffer(3)]],
    uint3 grid_pos [[threadgroup_position_in_grid]],
    uint sg_id [[simdgroup_index_in_threadgroup]],
    uint lane_id [[thread_index_in_simdgroup]]
) {
    // -------------------------------------------------------------------------
    // L1 CACHE ALLOCATIONS
    // -------------------------------------------------------------------------
    // Size: 32 rows x 33 columns.
    // The +1 padding is CRITICAL. It prevents Shared Memory Bank Conflicts
    // when threads read across rows during the matrix-vector multiplication.
    threadgroup float T_local[BLOCK_SIZE][BLOCK_SIZE + 1];

    // Holds the w = Y^T * y_j dot products for the current column
    threadgroup float W_local[BLOCK_SIZE];

    // -------------------------------------------------------------------------
    // BATCH INDEXING
    // -------------------------------------------------------------------------
    uint batch_idx = grid_pos.z;
    device const float* A_batch   = A + ((ulong)batch_idx * M * N);

    uint num_blocks = K_dim / BLOCK_SIZE;
    device float* T_batch = T_global + ((ulong)batch_idx * num_blocks * BLOCK_SIZE * BLOCK_SIZE) + (block_start / BLOCK_SIZE) * BLOCK_SIZE * BLOCK_SIZE;

    // Assuming tau_global is structured as a continuous array of size [Batch, N]
    device const float* tau_block = tau_global + ((ulong)batch_idx * N) + block_start;

    uint tid = sg_id * SIMD_GROUP_SIZE + lane_id;

    // -------------------------------------------------------------------------
    // INITIALIZATION
    // -------------------------------------------------------------------------
    // Zero out the T_local matrix using all available threads
    for (uint i = tid; i < BLOCK_SIZE * (BLOCK_SIZE + 1); i += 1024) {
        ((threadgroup float*)T_local)[i] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // -------------------------------------------------------------------------
    // RECURSIVE T-MATRIX CONSTRUCTION (Compact WY)
    // -------------------------------------------------------------------------
    for (uint j = 0; j < BLOCK_SIZE && (block_start + j) < N; ++j) {
        uint col_j = block_start + j;
        float tau_j = tau_block[j];

        // --- STEP 1: Compute w_i = Y_i^T * Y_j (For all i < j) ---
        // We use SIMD-Level Parallelism here.
        // SIMD Group 'i' is assigned to compute the dot product for column 'i'.
            uint col_i = block_start + sg_id;

            // Y vectors are 0 above the pivot.
            // The overlap strictly starts at the pivot of the newer column (col_j).
            uint pivot_row = col_j;
            float local_dot = 0.0f;

            // The 32 threads in this SIMD group sweep down the 5000-element column
            for (uint r = pivot_row + lane_id; r < M; r += SIMD_GROUP_SIZE) {
                // y_i is safe because r > col_i (strictly lower triangle)
                float y_i = A_batch[r + col_i * M];

                // FIX: Mask the diagonal to 1.0f to avoid reading the R matrix!
                float y_j = (r == col_j) ? 1.0f : A_batch[r + col_j * M];

                local_dot += (sg_id < j) ? (y_i * y_j) : 0.0f;
            }

            float total_dot = simd_sum(local_dot);

            // The SIMD leader writes the final dot product to shared memory
            if (lane_id == 0) {
                W_local[sg_id] = total_dot;
            }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- STEP 2: Compute T_{0:j-1, j} = -tau_j * T_{0:j-1, 0:j-1} * w ---
        // We use Thread-Level Parallelism here.
        // Thread 'tid' computes the value for row 'tid'.
        if (tid < j) {
            float sum = 0.0f;
            // T is upper triangular, so we only sum from m = tid to j-1
            for (uint m = tid; m < j; ++m) {
                sum += T_local[tid][m] * W_local[m];
            }
            T_local[tid][j] = -tau_j * sum;
        }

        // --- STEP 3: Set the Diagonal Element ---
        if (tid == 0) {
            T_local[j][j] = tau_j;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // -------------------------------------------------------------------------
    // WRITEBACK TO GLOBAL MEMORY
    // -------------------------------------------------------------------------
    // Since T_local is 32x32 (1024 elements) and we have 1024 threads,
    // each thread writes exactly one element to global memory.
    if (tid < BLOCK_SIZE * BLOCK_SIZE) {
        uint row = tid % BLOCK_SIZE;
        uint col = tid / BLOCK_SIZE;

        // AMX SUBTRACTION TRICK:
        // By negating the T matrix here, Kernel 3 can use a native hardware
        // simdgroup_multiply_accumulate (Addition) instead of having to
        // emulate a Multiply-Subtract, saving precious clock cycles.
        T_batch[row + col * BLOCK_SIZE] = -T_local[row][col];
    }
}

// -----------------------------------------------------------------------------
// KERNEL 3: GRID-PARALLEL TRAILING UPDATE (THE ENGINE)
// -----------------------------------------------------------------------------
// Dispatched with:
// Grid Size: [ (N - block_start) / 32, 1, batch ]
// Threadgroup Size: [ 128, 1, 1 ] -> 4 SIMD Groups
kernel void grid_parallel_update(
    device float* A [[buffer(0)]],
    device float* Q [[buffer(1)]],
    device const float* T_global [[buffer(2)]],
    constant uint& block_start [[buffer(3)]],
    constant uint& update_mode [[buffer(4)]],
    uint3 tg_pos [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],       // Must be thread_index (scalar) to avoid Clang vector mismatch
    uint sg_id [[simdgroup_index_in_threadgroup]]
) {
    // -------------------------------------------------------------------------
    // L1 CACHE ALLOCATION
    // We statically allocate this inside the kernel. Passing it as a pointer in
    // the signature relies on the C++ host to allocate the bytes. If the C++ host
    // fails or allocates 0 bytes, reads/writes silently drop, zeroing out the math.
    // -------------------------------------------------------------------------
    threadgroup UpdateSharedMemory shared_mem_inst;
    threadgroup UpdateSharedMemory* shared_mem = &shared_mem_inst;

    // -------------------------------------------------------------------------
    // WORKGROUP TARGET BOUNDS
    // -------------------------------------------------------------------------
    uint batch_idx = tg_pos.z;
    uint target_col_start = block_start + BLOCK_SIZE + (tg_pos.x * BLOCK_SIZE);

    // If we are building Q, we must apply the transformations to the entire matrix (starting at col 0)
    if (update_mode >= 1) {
        target_col_start = tg_pos.x * BLOCK_SIZE;
    }

    // Safely exit threads that overhang the matrix dimensions
    uint max_cols = (update_mode == 0) ? N : K_dim;
    if (target_col_start >= max_cols) return;

    // -------------------------------------------------------------------------
    // GLOBAL MEMORY POINTERS
    // We strictly use ulong casts for the batch offsets. If the batch dimension
    // and matrix size are large, a 32-bit uint will overflow and cause EXC_BAD_ACCESS.
    // -------------------------------------------------------------------------
    device const float* A_batch = A + ((ulong)batch_idx * M * N);
    device float* Target_Matrix = (update_mode == 0) ? A + ((ulong)batch_idx * M * N) : Q + ((ulong)batch_idx * M * K_dim);

    uint num_blocks = K_dim / BLOCK_SIZE;
    device const float* T_batch = T_global + ((ulong)batch_idx * num_blocks * BLOCK_SIZE * BLOCK_SIZE) + (block_start / BLOCK_SIZE) * BLOCK_SIZE * BLOCK_SIZE;

    // Stride includes PAD_OFFSET to prevent L1 Shared Memory bank conflicts
    const uint STRIDE = BLOCK_SIZE + PAD_OFFSET;

    // -------------------------------------------------------------------------
    // FLAT AMX REGISTERS (AST CRASH PREVENTION)
    // We explicitly unroll Z_acc into 4 separate registers instead of Z_acc[4].
    // Trying to dynamically index a simdgroup_matrix array causes the Clang AST
    // parser to crash with a generic "expecting input declarations" error.
    // -------------------------------------------------------------------------
    simdgroup_float8x8 Z_acc_0, Z_acc_1, Z_acc_2, Z_acc_3;
    clear_tile(Z_acc_0); clear_tile(Z_acc_1); clear_tile(Z_acc_2); clear_tile(Z_acc_3);

    simdgroup_float8x8 Op_Y, Op_Target, Op_T;

    // =========================================================================
    // PHASE 1: Z^T = Target^T * Y
    // Hardware Context: AMX natively loads column-major data as transposed (Row-Major).
    // Instead of fighting the hardware, we mathematically invert the update
    // formula to compute the transpose.
    // =========================================================================
    for (uint global_row_block = block_start; global_row_block < M; global_row_block += BLOCK_SIZE) {

        // 1. COLLABORATIVE L1 CACHE LOAD (Y-Matrix)
        for (uint i = 0; i < 8; ++i) {
            uint linear_idx = tid + i * 128;
            uint r_L1 = linear_idx / BLOCK_SIZE;
            uint c_L1 = linear_idx % BLOCK_SIZE;
            float val = A_batch[(global_row_block + r_L1) + (block_start + c_L1) * M];

            // The Householder vectors are stored in the strictly lower triangle.
            // We forcefully inject the 1.0 pivot and 0.0 upper triangle.
            if (global_row_block == block_start) {
                if (r_L1 < c_L1) val = 0.0f;
                else if (r_L1 == c_L1) val = 1.0f;
            }
            shared_mem->cached_Y[r_L1 * STRIDE + c_L1] = val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 2. AMX SWEEP
        for(int k_tile = 0; k_tile < 4; ++k_tile) {
            // FIXED: col-major + false = loads A^T into the registers
            load_tile_device(Op_Target, Target_Matrix, M, target_col_start + (sg_id * 8), global_row_block + (k_tile * 8), false);

            load_tile_threadgroup(Op_Y, shared_mem->cached_Y, STRIDE, 0 * 8, k_tile * 8, false);
            amx_mac_8x8(Z_acc_0, Op_Target, Op_Y);

            load_tile_threadgroup(Op_Y, shared_mem->cached_Y, STRIDE, 1 * 8, k_tile * 8, false);
            amx_mac_8x8(Z_acc_1, Op_Target, Op_Y);

            load_tile_threadgroup(Op_Y, shared_mem->cached_Y, STRIDE, 2 * 8, k_tile * 8, false);
            amx_mac_8x8(Z_acc_2, Op_Target, Op_Y);

            load_tile_threadgroup(Op_Y, shared_mem->cached_Y, STRIDE, 3 * 8, k_tile * 8, false);
            amx_mac_8x8(Z_acc_3, Op_Target, Op_Y);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // =========================================================================
    // PHASE 2: Z_final^T = Z^T * T
    // =========================================================================
    // Load the 32x32 T-Matrix into L1 Cache
    for (uint i = 0; i < 8; ++i) {
        uint linear_idx = tid + i * 128;
        uint r_L1 = linear_idx / BLOCK_SIZE;
        uint c_L1 = linear_idx % BLOCK_SIZE;

        if (update_mode == 2) {
            // BACKWARD PASS: Needs H = I - Y T Y^T.
            // To natively build Q = H_1 H_2 ... H_n, the backward pass must apply H.
            // By swapping r_L1 and c_L1, we natively transpose -T into -T^T in L1 memory.
            shared_mem->compact_T[c_L1 * STRIDE + r_L1] = T_batch[r_L1 + c_L1 * BLOCK_SIZE];
        } else {
            // FORWARD PASS: Needs H^T = I - Y T^T Y^T.
            // Phase 2 computes Z_final^T = Z^T * Op_T. We need Op_T = -T.
            shared_mem->compact_T[r_L1 * STRIDE + c_L1] = T_batch[r_L1 + c_L1 * BLOCK_SIZE];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_float8x8 Z_final_0, Z_final_1, Z_final_2, Z_final_3;
    clear_tile(Z_final_0); clear_tile(Z_final_1); clear_tile(Z_final_2); clear_tile(Z_final_3);

    for(int k_tile = 0; k_tile < 4; ++k_tile) {
        // Map the flat unrolled registers to the loop iteration
        simdgroup_float8x8 Z_acc_k;
        if (k_tile == 0) Z_acc_k = Z_acc_0;
        else if (k_tile == 1) Z_acc_k = Z_acc_1;
        else if (k_tile == 2) Z_acc_k = Z_acc_2;
        else Z_acc_k = Z_acc_3;

        // Swapped coordinates and transpose=false to load -T^T
        load_tile_threadgroup(Op_T, shared_mem->compact_T, STRIDE, 0 * 8, k_tile * 8, false);
        amx_mac_8x8(Z_final_0, Z_acc_k, Op_T);

        load_tile_threadgroup(Op_T, shared_mem->compact_T, STRIDE, 1 * 8, k_tile * 8, false);
        amx_mac_8x8(Z_final_1, Z_acc_k, Op_T);

        load_tile_threadgroup(Op_T, shared_mem->compact_T, STRIDE, 2 * 8, k_tile * 8, false);
        amx_mac_8x8(Z_final_2, Z_acc_k, Op_T);

        load_tile_threadgroup(Op_T, shared_mem->compact_T, STRIDE, 3 * 8, k_tile * 8, false);
        amx_mac_8x8(Z_final_3, Z_acc_k, Op_T);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // =========================================================================
    // PHASE 3: Target^T += Z_final^T * Y^T
    // =========================================================================
    // Stream through the column blocks one last time to apply the update matrix
    for (uint global_row_block = block_start; global_row_block < M; global_row_block += BLOCK_SIZE) {

        // 1. COLLABORATIVE L1 CACHE LOAD (Y-Matrix) - Reloaded for Phase 3
        for (uint i = 0; i < 8; ++i) {
            uint linear_idx = tid + i * 128;
            uint r_L1 = linear_idx / BLOCK_SIZE;
            uint c_L1 = linear_idx % BLOCK_SIZE;
            float val = A_batch[(global_row_block + r_L1) + (block_start + c_L1) * M];

            if (global_row_block == block_start) {
                if (r_L1 < c_L1) val = 0.0f;
                else if (r_L1 == c_L1) val = 1.0f;
            }
            shared_mem->cached_Y[r_L1 * STRIDE + c_L1] = val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 2. AMX UPDATE SWEEP
        for(int r_tile = 0; r_tile < 4; ++r_tile) {
            // FIXED: col-major + false = loads A^T
            load_tile_device(Op_Target, Target_Matrix, M, target_col_start + (sg_id * 8), global_row_block + (r_tile * 8), false);

            load_tile_threadgroup(Op_Y, shared_mem->cached_Y, STRIDE, 0 * 8, r_tile * 8, true);
            amx_mac_8x8(Op_Target, Z_final_0, Op_Y);

            load_tile_threadgroup(Op_Y, shared_mem->cached_Y, STRIDE, 1 * 8, r_tile * 8, true);
            amx_mac_8x8(Op_Target, Z_final_1, Op_Y);

            load_tile_threadgroup(Op_Y, shared_mem->cached_Y, STRIDE, 2 * 8, r_tile * 8, true);
            amx_mac_8x8(Op_Target, Z_final_2, Op_Y);

            load_tile_threadgroup(Op_Y, shared_mem->cached_Y, STRIDE, 3 * 8, r_tile * 8, true);
            amx_mac_8x8(Op_Target, Z_final_3, Op_Y);

            // FIXED: col-major + false = writes (A^T)^T = A back to memory
            store_tile_device(Op_Target, Target_Matrix, M, target_col_start + (sg_id * 8), global_row_block + (r_tile * 8), false);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

/**
 * simd_prod
 * ---------
 * Metal lacks a native `simd_prod` intrinsic. This performs a high-speed
 * register shuffle to multiply a value across all 32 threads in a SIMD group.
 */
inline float simd_prod(float val) {
    for (uint offset = SIMD_GROUP_SIZE / 2; offset > 0; offset /= 2) {
        val *= simd_shuffle_down(val, offset);
    }
    // Broadcast the final product from lane 0 to all lanes
    return simd_broadcast(val, 0);
}

/**
 * threadgroup_prod_reduce
 * -----------------------
 * Multiplies a value across the entire threadgroup (up to 1024 threads).
 */
inline float threadgroup_prod_reduce(
    float local_val,
    threadgroup float* red_space,
    uint sg_id,
    uint lane_id
) {
    float simd_val = simd_prod(local_val);

    if (lane_id == 0) {
        red_space[sg_id] = simd_val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg_id == 0) {
        float val = (lane_id < MAX_SIMD_GROUPS) ? red_space[lane_id] : 1.0f;
        float total = simd_prod(val);

        if (lane_id == 0) {
            red_space[0] = total;
        }
    }

    // CRITICAL FIX: Barrier
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float result = red_space[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    return result;
}

// -----------------------------------------------------------------------------
// KERNEL 4: POST-PROCESS HAAR FIX
// -----------------------------------------------------------------------------
// Dispatched with: 1 Threadgroup per matrix in the batch.
// Threads: 1024 (32 SIMD Groups)
kernel void postprocess_haar_fix(
    device float* A [[buffer(0)]],
    device float* Q [[buffer(1)]],
    device const float* R_diags [[buffer(2)]],
    constant uint& original_M [[buffer(3)]],
    constant uint& original_K [[buffer(4)]],
    uint3 grid_pos [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]],
    uint lane_id [[thread_index_in_simdgroup]]
) {
    threadgroup HaarSharedMemory shared_mem_inst;
    threadgroup HaarSharedMemory* shared_mem = &shared_mem_inst;

    uint batch_idx = grid_pos.z;

    // M and N are constants injected at compile time
    device float* A_batch = A + ((ulong)batch_idx * M * N);

    // Q is now strictly M x N stride
    device float* Q_batch = Q + ((ulong)batch_idx * M * K_dim);
    device const float* R_batch = R_diags + ((ulong)batch_idx * M);

    // =========================================================================
    // PHASE 1: COLUMN SIGN CORRECTION
    // =========================================================================
    for (uint k = tid; k < original_K; k += 1024) {
        if (R_batch[k] < 0.0f) {
            // Q is built Column-Major natively via Backward Accumulation.
            // To flip column 'k' of Q, we iterate through its rows 'r'.
            for (uint r = 0; r < original_M; ++r) {
                Q_batch[r + k * M] = -Q_batch[r + k * M];
            }

            // A holds the upper triangular R. We flip row 'k' of A.
            for (uint c = k; c < N; ++c) {
                A_batch[k + c * M] = -A_batch[k + c * M];
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_device);

    // =========================================================================
    // PHASE 2: DETERMINANT FORCE TO SO(d)
    // =========================================================================
    float local_sign = 1.0f;
    for (uint k = tid; k < original_K; k += 1024) {
        if (R_batch[k] < 0.0f) {
            local_sign = -local_sign;
        }
    }

    // Relying on the `threadgroup_prod_reduce` helper function you defined earlier
    float total_sign = threadgroup_prod_reduce(local_sign, shared_mem->reduction_space, sg_id, lane_id);

    if (tid == 0) {
        float parity = (original_M % 2 == 0) ? -1.0f : 1.0f;
        float det_Q = total_sign * parity;

        if (det_Q < 0.0f) {
            uint last_col = original_K - 1;

            // Flip the last valid column of Q
            for (uint r = 0; r < original_M; ++r) {
                Q_batch[r + last_col * M] = -Q_batch[r + last_col * M];
            }

            // Flip the last valid row of A (which represents the bottom of R)
            for (uint c = last_col; c < N; ++c) {
                A_batch[last_col + c * M] = -A_batch[last_col + c * M];
            }
        }
    }
}