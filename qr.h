#pragma once
#include <mlx/mlx.h>
#include <utility>

namespace custom_math {

    // The user-facing function. Takes a batched matrix A [B, M, N].
    // Returns a pair of MLX arrays: {Q, R}.
    std::pair<mlx::core::array, mlx::core::array> qr_accelerated(const mlx::core::array& a);

} // namespace custom_math