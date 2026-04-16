#include "qr.h"
#include "qr_detail.h"
#include <algorithm>

namespace custom_math {

std::pair<mlx::core::array, mlx::core::array>
qr_accelerated(const mlx::core::array& a) {
    const auto& shape = a.shape();
    const auto M = static_cast<uint>(shape[shape.size() - 2]);
    const auto N = static_cast<uint>(shape[shape.size() - 1]);

    // 1. Calculate total batch size
    uint batch = 1;
    for (size_t i = 0; i + 2 < shape.size(); ++i) {
        batch *= static_cast<uint>(shape[i]);
    }

    uint max_dim = std::max(M, N);

    // 2. The Single-Core Capacity Limit
    // Matrices this large will choke a single SM doing O(M^2) Q-accumulation.
    // We MUST use the grid-parallel streaming shader.
    if (max_dim >= 512) {
        return detail::qr_streaming_amx(a);
    }

    // 3. The High-Occupancy Batch Limit
    // If the matrices are < 512, and we have enough of them to fill the GPU cores,
    // the single-kernel unblocked shader avoids host-loop launch overhead.
    if (batch >= 16) {
        return detail::qr_unblocked(a);
    }

    // 4. The Starvation Zone vs. Latency Floor (Batch < 16)
    // If the batch is small, we want to parallelize intra-matrix to utilize the GPU.
    // However, if the matrix is < 128, the multi-kernel launch overhead of AMX
    // is slower than just running it on a single core.
    if (max_dim >= 128) {
        return detail::qr_streaming_amx(a);
    } else {
        return detail::qr_unblocked(a);
    }
}

} // namespace custom_math