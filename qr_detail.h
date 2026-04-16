#pragma once
#include <mlx/mlx.h>
#include <utility>

namespace custom_math::detail {

// Dispatched when M > 512. Standard Householder QR, single-kernel dispatch.
std::pair<mlx::core::array, mlx::core::array>
qr_unblocked(const mlx::core::array& a);

// Dispatched when M <= 512. Multi-pass Streaming AMX panel factorization.
std::pair<mlx::core::array, mlx::core::array>
qr_streaming_amx(const mlx::core::array& a);

} // namespace custom_math::detail
