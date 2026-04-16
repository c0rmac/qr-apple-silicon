#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "qr_detail.h" // Adjust path if needed
#include <mlx/mlx.h>
#include <mlx/linalg.h>

#include <stdexcept>
#include <vector>
#include <algorithm>
#include <map>
#include <tuple>

using namespace mlx::core;

namespace custom_math::detail {

static constexpr size_t kQRSharedMemBytes = 5120;

static uint pad_up(uint v, uint multiple) {
    return ((v + multiple - 1) / multiple) * multiple;
}

static id<MTLLibrary> load_library(id<MTLDevice> device) {
#ifndef QR_UNBLOCKED_METALLIB_PATH
#  error "QR_UNBLOCKED_METALLIB_PATH must be set by CMake"
#endif
    NSString* path = @(QR_UNBLOCKED_METALLIB_PATH);
    NSError* err = nil;
    id<MTLLibrary> lib = [device newLibraryWithURL:[NSURL fileURLWithPath:path] error:&err];
    if (!lib) {
        throw std::runtime_error(std::string("[qr_unblocked] Cannot load metallib: ") +
                                 (err ? err.localizedDescription.UTF8String : "file missing"));
    }
    return lib;
}

// Compiles the monolithic QR shader with padded dimension constants
static id<MTLComputePipelineState> make_pipeline(id<MTLDevice> device, id<MTLLibrary> lib, uint M, uint N, uint M_pad, uint N_pad) {
    MTLFunctionConstantValues* cv = [[MTLFunctionConstantValues alloc] init];
    [cv setConstantValue:&M type:MTLDataTypeUInt atIndex:0]; // 0 is original M
    [cv setConstantValue:&N type:MTLDataTypeUInt atIndex:1]; // 1 is original N
    [cv setConstantValue:&M_pad type:MTLDataTypeUInt atIndex:2]; // 2 is padded M
    [cv setConstantValue:&N_pad type:MTLDataTypeUInt atIndex:3]; // 3 is padded N

    NSError* err = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:@"standard_householder_qr_float32" constantValues:cv error:&err];
    if (!fn) throw std::runtime_error("[qr_unblocked] Cannot specialise QR function.");

    id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) throw std::runtime_error("[qr_unblocked] Cannot create QR pipeline.");

    return pso;
}

// Fast compiler helper for our memory-shuffling helper kernels
static id<MTLComputePipelineState> get_pso(id<MTLDevice> dev, id<MTLLibrary> lib, NSString* name) {
    NSError* err = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:name];
    if (!fn) throw std::runtime_error("Function not found in library: " + std::string(name.UTF8String));

    id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) throw std::runtime_error("Failed to compile PSO: " + std::string(name.UTF8String));

    return pso;
}

// =============================================================================
// CACHING STRUCTURES
// =============================================================================

// Holds compiled pipelines for a specific (M, N) size
struct QRPipelines {
    id<MTLComputePipelineState> pso_qr;
    id<MTLComputePipelineState> pso_pack;
    id<MTLComputePipelineState> pso_init_q;
    id<MTLComputePipelineState> pso_unpk_R;
    id<MTLComputePipelineState> pso_unpk_Q;
};

// Holds recycled GPU memory allocations to prevent OS-level allocation overhead
struct Workspace {
    id<MTLBuffer> buf_A_pad;
    id<MTLBuffer> buf_Q_pad;
    id<MTLBuffer> buf_R_out;
    id<MTLBuffer> buf_Q_out;
};

// The Singleton Cache Context
struct MetalContext {
    id<MTLDevice> dev;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> lib;

    // Cache pipelines using (M, N)
    std::map<std::pair<uint, uint>, QRPipelines> pipeline_cache;
    // Cache memory workspaces using (Batch, M, N)
    std::map<std::tuple<uint, uint, uint>, Workspace> workspace_cache;

    MetalContext() {
        // Runs exactly ONCE for the lifetime of the program
        dev = MTLCreateSystemDefaultDevice();
        queue = [dev newCommandQueue];
        lib = load_library(dev);
    }

    QRPipelines get_pipelines(uint M, uint N, uint M_pad, uint N_pad) {
        auto key = std::make_pair(M, N);
        auto it = pipeline_cache.find(key);
        if (it != pipeline_cache.end()) {
            return it->second;
        }

        QRPipelines p;
        p.pso_qr     = make_pipeline(dev, lib, M, N, M_pad, N_pad);
        p.pso_pack   = get_pso(dev, lib, @"pack_batch_memory");
        p.pso_init_q = get_pso(dev, lib, @"init_identity_batch");
        p.pso_unpk_R = get_pso(dev, lib, @"unpack_batch_R");
        p.pso_unpk_Q = get_pso(dev, lib, @"unpack_batch_Q");

        pipeline_cache[key] = p;
        return p;
    }

    Workspace get_workspace(uint batch, uint M, uint N, uint M_pad, uint N_pad, uint K) {
        auto key = std::make_tuple(batch, M, N);
        auto it = workspace_cache.find(key);
        if (it != workspace_cache.end()) {
            return it->second;
        }

        Workspace w;
        MTLResourceOptions opt = MTLResourceStorageModeShared;
        w.buf_A_pad = [dev newBufferWithLength:(batch * M_pad * N_pad * sizeof(float)) options:opt];
        w.buf_Q_pad = [dev newBufferWithLength:(batch * M_pad * M_pad * sizeof(float)) options:opt];
        w.buf_R_out = [dev newBufferWithLength:(batch * K * N * sizeof(float)) options:opt];
        w.buf_Q_out = [dev newBufferWithLength:(batch * M * K * sizeof(float)) options:opt];

        workspace_cache[key] = w;
        return w;
    }
};

// =============================================================================
// MAIN ENTRY POINT
// =============================================================================

std::pair<array, array> qr_unblocked(const array& a) {
    if (a.ndim() < 2) {
        throw std::invalid_argument("[qr_unblocked] Input must be at least a 2D matrix.");
    }

    const Shape& shape = a.shape();
    const uint M = static_cast<uint>(shape[shape.size() - 2]);
    const uint N = static_cast<uint>(shape[shape.size() - 1]);
    const uint K = std::min(M, N);

    uint batch = 1;
    for (size_t i = 0; i + 2 < shape.size(); ++i) {
        batch *= static_cast<uint>(shape[i]);
    }

    const uint M_pad = pad_up(M, 32);
    const uint N_pad = pad_up(N, 16);

    // 1. MLX Array Prep
    array a_f32 = astype(a, float32);
    if (!a_f32.flags().row_contiguous) {
        a_f32 = contiguous(a_f32);
    }
    eval({a_f32});

    // 2. Retrieve Cached State & Workspaces
    static MetalContext ctx;
    QRPipelines p = ctx.get_pipelines(M, N, M_pad, N_pad);
    Workspace w   = ctx.get_workspace(batch, M, N, M_pad, N_pad, K);

    // 3. Map Input Data (Zero-Copy)
    id<MTLBuffer> buf_src = [ctx.dev newBufferWithBytesNoCopy:(void*)a_f32.data<float>()
                                                   length:a_f32.nbytes()
                                                  options:MTLResourceStorageModeShared
                                              deallocator:nil];

    // 4. Encode Command Sequence
    id<MTLCommandBuffer> cmd = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

    MTLSize memory_threads = MTLSizeMake(16, 16, 1);

    // --- Pass 1: Pack Memory ---
    [enc setComputePipelineState:p.pso_pack];
    [enc setBuffer:buf_src offset:0 atIndex:0];
    [enc setBuffer:w.buf_A_pad offset:0 atIndex:1];
    [enc setBytes:&M length:sizeof(uint) atIndex:2];
    [enc setBytes:&N length:sizeof(uint) atIndex:3];
    [enc setBytes:&M_pad length:sizeof(uint) atIndex:4];
    [enc setBytes:&N_pad length:sizeof(uint) atIndex:5];
    [enc dispatchThreads:MTLSizeMake(N, M, batch) threadsPerThreadgroup:memory_threads];

    // --- Pass 2: Init Q to Identity ---
    [enc setComputePipelineState:p.pso_init_q];
    [enc setBuffer:w.buf_Q_pad offset:0 atIndex:0];
    [enc setBytes:&M_pad length:sizeof(uint) atIndex:1];
    [enc setBytes:&N_pad length:sizeof(uint) atIndex:2];
    [enc dispatchThreads:MTLSizeMake(M_pad, M_pad, batch) threadsPerThreadgroup:memory_threads];

    // --- Pass 3: The Monolithic QR Factorization ---
    uint max_simd_groups = std::min((N_pad + 7) / 8, 32u);
    max_simd_groups = std::max(max_simd_groups, 1u);

    [enc setComputePipelineState:p.pso_qr];
    [enc setBuffer:w.buf_A_pad offset:0 atIndex:0];
    [enc setBuffer:w.buf_Q_pad offset:0 atIndex:1];
    [enc setThreadgroupMemoryLength:kQRSharedMemBytes atIndex:0];

    [enc dispatchThreadgroups:MTLSizeMake(1, 1, batch)
        threadsPerThreadgroup:MTLSizeMake(32, max_simd_groups, 1)];

    // --- Pass 4: Unpack R ---
    [enc setComputePipelineState:p.pso_unpk_R];
    [enc setBuffer:w.buf_A_pad offset:0 atIndex:0];
    [enc setBuffer:w.buf_R_out offset:0 atIndex:1];
    [enc setBytes:&M length:sizeof(uint) atIndex:2];
    [enc setBytes:&N length:sizeof(uint) atIndex:3];
    [enc setBytes:&K length:sizeof(uint) atIndex:4];
    [enc setBytes:&M_pad length:sizeof(uint) atIndex:5];
    [enc setBytes:&N_pad length:sizeof(uint) atIndex:6];
    [enc dispatchThreads:MTLSizeMake(N, K, batch) threadsPerThreadgroup:memory_threads];

    // --- Pass 5: Unpack Q ---
    [enc setComputePipelineState:p.pso_unpk_Q];
    [enc setBuffer:w.buf_Q_pad offset:0 atIndex:0];
    [enc setBuffer:w.buf_Q_out offset:0 atIndex:1];
    [enc setBytes:&M length:sizeof(uint) atIndex:2];
    [enc setBytes:&K length:sizeof(uint) atIndex:3];
    [enc setBytes:&M_pad length:sizeof(uint) atIndex:4];
    [enc dispatchThreads:MTLSizeMake(K, M, batch) threadsPerThreadgroup:memory_threads];

    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    if (cmd.error) {
        throw std::runtime_error(std::string("[qr_unblocked] GPU Kernel Error: ") +
                                 cmd.error.localizedDescription.UTF8String);
    }

    // 5. Flat Array Handoff to MLX
    Shape R_shape(shape.begin(), shape.end());
    R_shape[R_shape.size() - 2] = K;

    Shape Q_shape(shape.begin(), shape.end());
    Q_shape[Q_shape.size() - 1] = K;

    const float* r_ptr = static_cast<const float*>([w.buf_R_out contents]);
    const float* q_ptr = static_cast<const float*>([w.buf_Q_out contents]);

    // Note: Passing the pointer like this forces MLX to deep copy the result,
    // which protects our recycled `Workspace` buffers from being overwritten by MLX later.
    array final_R = array(r_ptr, R_shape, float32);
    array final_Q = array(q_ptr, Q_shape, float32);

    return {final_Q, final_R};
}

} // namespace custom_math::detail