#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "qr_detail.h"
#include <mlx/mlx.h>
#include <mlx/linalg.h>

#include <stdexcept>
#include <vector>
#include <numeric>
#include <map>
#include <tuple>

using namespace mlx::core;

namespace custom_math::detail {

static uint pad_up_amx(uint v, uint multiple) {
    return ((v + multiple - 1) / multiple) * multiple;
}

static id<MTLLibrary> load_library_amx(id<MTLDevice> device) {
#ifndef QR_STREAMING_AMX_METALLIB_PATH
#  error "QR_STREAMING_AMX_METALLIB_PATH must be set by CMake"
#endif
    NSString* path = @(QR_STREAMING_AMX_METALLIB_PATH);
    NSError* err = nil;
    id<MTLLibrary> lib = [device newLibraryWithURL:[NSURL fileURLWithPath:path] error:&err];
    if (!lib) {
        throw std::runtime_error("[qr_streaming_amx] Cannot load metallib");
    }
    return lib;
}

static id<MTLComputePipelineState>
make_pipeline_amx(id<MTLDevice> device, id<MTLLibrary> lib,
                  NSString* name, MTLFunctionConstantValues* cv) {
    NSError* err = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:name constantValues:cv error:&err];
    if (!fn) throw std::runtime_error(std::string("Function not found: ") + name.UTF8String);
    return [device newComputePipelineStateWithFunction:fn error:&err];
}

// Fast compiler helper for non-constant helper kernels
static id<MTLComputePipelineState> get_pso_amx(id<MTLDevice> dev, id<MTLLibrary> lib, NSString* name) {
    NSError* err = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:name];
    if (!fn) throw std::runtime_error("Function not found: " + std::string(name.UTF8String));
    return [dev newComputePipelineStateWithFunction:fn error:&err];
}

// =============================================================================
// CACHING STRUCTURES
// =============================================================================

struct StreamingAMXPipelines {
    id<MTLComputePipelineState> pso_transpose;
    id<MTLComputePipelineState> pso_init_q;
    id<MTLComputePipelineState> pso_panel;
    id<MTLComputePipelineState> pso_t_mat;
    id<MTLComputePipelineState> pso_update;
    id<MTLComputePipelineState> pso_haar;
    id<MTLComputePipelineState> pso_unpk_R; // Added Unpack Shader
    id<MTLComputePipelineState> pso_unpk_Q; // Added Unpack Shader
};

struct StreamingWorkspace {
    id<MTLBuffer> buf_A;
    id<MTLBuffer> buf_Q;
    id<MTLBuffer> buf_R_diag; // Renamed to avoid confusion
    id<MTLBuffer> buf_tau;
    id<MTLBuffer> buf_T;
    id<MTLBuffer> buf_R_out;  // Added exact-size output buffer
    id<MTLBuffer> buf_Q_out;  // Added exact-size output buffer
};

struct MetalAMXContext {
    id<MTLDevice> dev;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> lib;

    std::map<std::pair<uint, uint>, StreamingAMXPipelines> pso_cache;
    std::map<std::tuple<uint, uint, uint>, StreamingWorkspace> workspace_cache;

    MetalAMXContext() {
        dev = MTLCreateSystemDefaultDevice();
        queue = [dev newCommandQueue];
        lib = load_library_amx(dev);
    }

    StreamingAMXPipelines get_pipelines(uint M_pad, uint N_pad) {
        auto key = std::make_pair(M_pad, N_pad);
        if (pso_cache.count(key)) return pso_cache[key];

        MTLFunctionConstantValues* cv = [[MTLFunctionConstantValues alloc] init];
        [cv setConstantValue:&M_pad type:MTLDataTypeUInt atIndex:0];
        [cv setConstantValue:&N_pad type:MTLDataTypeUInt atIndex:1];

        StreamingAMXPipelines p;
        p.pso_transpose = make_pipeline_amx(dev, lib, @"preprocess_transpose", cv);
        p.pso_init_q    = make_pipeline_amx(dev, lib, @"init_identity_q",    cv);
        p.pso_panel     = make_pipeline_amx(dev, lib, @"panel_factorization", cv);
        p.pso_t_mat     = make_pipeline_amx(dev, lib, @"compute_t_matrix",    cv);
        p.pso_update    = make_pipeline_amx(dev, lib, @"grid_parallel_update", cv);
        p.pso_haar      = make_pipeline_amx(dev, lib, @"postprocess_haar_fix", cv);
        p.pso_unpk_R    = get_pso_amx(dev, lib, @"unpack_batch_R");
        p.pso_unpk_Q    = get_pso_amx(dev, lib, @"unpack_batch_Q");

        return pso_cache[key] = p;
    }

    StreamingWorkspace get_workspace(uint batch, uint M_pad, uint N_pad, uint M, uint N, uint K) {
        auto key = std::make_tuple(batch, M_pad, N_pad);
        if (workspace_cache.count(key)) return workspace_cache[key];

        StreamingWorkspace w;
        MTLResourceOptions opt = MTLResourceStorageModeShared;
        w.buf_A      = [dev newBufferWithLength:((size_t)batch * M_pad * N_pad * sizeof(float)) options:opt];
        w.buf_Q      = [dev newBufferWithLength:((size_t)batch * M_pad * M_pad * sizeof(float)) options:opt];
        w.buf_R_diag = [dev newBufferWithLength:((size_t)batch * M_pad * sizeof(float))         options:opt];
        w.buf_tau    = [dev newBufferWithLength:((size_t)batch * N_pad * sizeof(float))         options:opt];
        w.buf_T      = [dev newBufferWithLength:((size_t)batch * 32 * 32 * sizeof(float))       options:opt];

        // Exact size output buffers
        w.buf_R_out  = [dev newBufferWithLength:((size_t)batch * K * N * sizeof(float)) options:opt];
        w.buf_Q_out  = [dev newBufferWithLength:((size_t)batch * M * K * sizeof(float)) options:opt];

        return workspace_cache[key] = w;
    }
};

// =============================================================================
// MAIN ENTRY POINT
// =============================================================================

std::pair<array, array> qr_streaming_amx(const array& a) {
    if (a.ndim() < 2)
        throw std::invalid_argument("[qr_streaming_amx] Input must be at least a 2D matrix.");

    const Shape& shape       = a.shape();
    const uint   original_M  = static_cast<uint>(shape[shape.size() - 2]);
    const uint   original_N  = static_cast<uint>(shape[shape.size() - 1]);
    const uint   original_K  = std::min(original_M, original_N);

    uint batch = 1;
    for (size_t i = 0; i + 2 < shape.size(); ++i)
        batch *= static_cast<uint>(shape[i]);

    array a_f32 = astype(a, float32);
    if (!a_f32.flags().row_contiguous)
        a_f32 = contiguous(a_f32);
    eval({a_f32});

    const uint M_pad = pad_up_amx(original_M, 32);
    const uint N_pad = pad_up_amx(original_N, 32);

    static MetalAMXContext ctx;
    StreamingAMXPipelines p = ctx.get_pipelines(M_pad, N_pad);
    StreamingWorkspace w    = ctx.get_workspace(batch, M_pad, N_pad, original_M, original_N, original_K);

    id<MTLBuffer> buf_src = [ctx.dev newBufferWithBytesNoCopy:(void*)a_f32.data<float>()
                                                       length:a_f32.nbytes()
                                                      options:MTLResourceStorageModeShared
                                                  deallocator:nil];

    id<MTLCommandBuffer>         cmd = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

    [enc setComputePipelineState:p.pso_transpose];
    [enc setBuffer:buf_src  offset:0 atIndex:0];
    [enc setBuffer:w.buf_A  offset:0 atIndex:1];
    [enc setBytes:&original_M length:sizeof(uint) atIndex:2];
    [enc setBytes:&original_N length:sizeof(uint) atIndex:3];
    [enc dispatchThreads:MTLSizeMake(N_pad, M_pad, batch) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];

    [enc setComputePipelineState:p.pso_init_q];
    [enc setBuffer:w.buf_Q offset:0 atIndex:0];
    [enc dispatchThreads:MTLSizeMake(M_pad, M_pad, batch) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];

    for (uint block_start = 0; block_start < original_N; block_start += 32) {
        [enc setComputePipelineState:p.pso_panel];
        [enc setBuffer:w.buf_A      offset:0 atIndex:0];
        [enc setBuffer:w.buf_R_diag offset:0 atIndex:1];
        [enc setBuffer:w.buf_tau    offset:0 atIndex:2];
        [enc setBytes:&block_start length:sizeof(uint) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, batch) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];

        [enc setComputePipelineState:p.pso_t_mat];
        [enc setBuffer:w.buf_A   offset:0 atIndex:0];
        [enc setBuffer:w.buf_T   offset:0 atIndex:1];
        [enc setBuffer:w.buf_tau offset:0 atIndex:2];
        [enc setBytes:&block_start length:sizeof(uint) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, batch) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];

        uint update_cols_A = (N_pad - block_start);
        if (update_cols_A > 32) {
            uint groups_x    = (update_cols_A - 32 + 31) / 32;
            uint is_Q_update = 0;
            [enc setComputePipelineState:p.pso_update];
            [enc setBuffer:w.buf_A offset:0 atIndex:0];
            [enc setBuffer:w.buf_Q offset:0 atIndex:1];
            [enc setBuffer:w.buf_T offset:0 atIndex:2];
            [enc setBytes:&block_start length:sizeof(uint) atIndex:3];
            [enc setBytes:&is_Q_update  length:sizeof(uint) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(groups_x, 1, batch) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        }

        uint is_Q_update = 1;
        uint groups_Q_x  = (M_pad + 31) / 32;
        [enc setComputePipelineState:p.pso_update];
        [enc setBuffer:w.buf_A offset:0 atIndex:0];
        [enc setBuffer:w.buf_Q offset:0 atIndex:1];
        [enc setBuffer:w.buf_T offset:0 atIndex:2];
        [enc setBytes:&block_start length:sizeof(uint) atIndex:3];
        [enc setBytes:&is_Q_update  length:sizeof(uint) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(groups_Q_x, 1, batch) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    }

    [enc setComputePipelineState:p.pso_haar];
    [enc setBuffer:w.buf_A offset:0 atIndex:0];
    [enc setBuffer:w.buf_Q offset:0 atIndex:1];
    [enc setBuffer:w.buf_R_diag offset:0 atIndex:2];
    [enc setBytes:&original_M length:sizeof(uint) atIndex:3];
    [enc setBytes:&original_K length:sizeof(uint) atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, batch) threadsPerThreadgroup:MTLSizeMake(1024, 1, 1)];

    // --- NEW: Unpack R (Padded Col-major -> Exact Row-major + Triu) ---
    MTLSize memory_threads = MTLSizeMake(16, 16, 1);
    [enc setComputePipelineState:p.pso_unpk_R];
    [enc setBuffer:w.buf_A offset:0 atIndex:0];
    [enc setBuffer:w.buf_R_out offset:0 atIndex:1];
    [enc setBytes:&original_M length:sizeof(uint) atIndex:2];
    [enc setBytes:&original_N length:sizeof(uint) atIndex:3];
    [enc setBytes:&original_K length:sizeof(uint) atIndex:4];
    [enc setBytes:&M_pad length:sizeof(uint) atIndex:5];
    [enc setBytes:&N_pad length:sizeof(uint) atIndex:6];
    [enc dispatchThreads:MTLSizeMake(original_N, original_K, batch) threadsPerThreadgroup:memory_threads];

    // --- NEW: Unpack Q (Padded Col-major -> Exact Row-major) ---
    [enc setComputePipelineState:p.pso_unpk_Q];
    [enc setBuffer:w.buf_Q offset:0 atIndex:0];
    [enc setBuffer:w.buf_Q_out offset:0 atIndex:1];
    [enc setBytes:&original_M length:sizeof(uint) atIndex:2];
    [enc setBytes:&original_K length:sizeof(uint) atIndex:3];
    [enc setBytes:&M_pad length:sizeof(uint) atIndex:4];
    [enc dispatchThreads:MTLSizeMake(original_K, original_M, batch) threadsPerThreadgroup:memory_threads];

    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    if (cmd.error) {
        throw std::runtime_error(std::string("[qr_streaming_amx] GPU error: ")
                                 + cmd.error.localizedDescription.UTF8String);
    }

    // 4. Instant MLX Handoff
    Shape R_shape(shape.begin(), shape.end());
    R_shape[R_shape.size() - 2] = original_K;

    Shape Q_shape(shape.begin(), shape.end());
    Q_shape[Q_shape.size() - 1] = original_K;

    const float* r_ptr = static_cast<const float*>([w.buf_R_out contents]);
    const float* q_ptr = static_cast<const float*>([w.buf_Q_out contents]);

    array R = array(r_ptr, R_shape, float32);
    array Q = array(q_ptr, Q_shape, float32);

    return {Q, R};
}

} // namespace custom_math::detail