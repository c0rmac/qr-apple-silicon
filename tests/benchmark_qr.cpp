#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <cmath>
#include <random>
#include <cstdlib>

#include <mlx/mlx.h>
#include <mlx/linalg.h>

#include "qr.h"

using namespace mlx::core;

// =============================================================================
// Helpers
// =============================================================================

static array random_matrix(int batch, int M, int N, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> data(batch * M * N);
    for (auto& v : data) v = dist(rng);
    if (batch == 1)
        return array(data.begin(), {M, N}, float32);
    return array(data.begin(), {batch, M, N}, float32);
}

// Extract slice A[b, :, :] from a [batch, M, N] array
static array slice_batch(const array& A, int b, int M, int N) {
    return reshape(slice(A, {b, 0, 0}, {b + 1, M, N}), {M, N});
}

static float reconstruction_error(const array& Q, const array& R, const array& A) {
    array QR       = matmul(Q, R);
    array diff     = subtract(QR, A);
    array sq       = multiply(diff, diff);
    array mean_err = mean(sqrt(sum(sq, {-1, -2})));
    eval({mean_err});
    return mean_err.item<float>();
}

template<typename Fn>
static double time_ms(Fn fn, int warmup = 2, int reps = 5) {
    for (int i = 0; i < warmup; ++i) fn();
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < reps; ++i) fn();
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
}

static void print_separator() {
    std::cout << std::string(68, '-') << "\n";
}

// =============================================================================
// Benchmark one (batch, M, N) configuration
// =============================================================================

static void benchmark(int batch, int M, int N) {
    array A = random_matrix(batch, M, N);
    eval({A});

    print_separator();
    if (batch == 1)
        std::cout << "Matrix : " << M << " x " << N << "  (no batch)\n";
    else
        std::cout << "Matrix : " << batch << " x " << M << " x " << N
                  << "  (batch=" << batch << ")\n";
    print_separator();

    // --- GPU ---
    set_default_device(Device::gpu);
    double gpu_ms = time_ms([&] {
        auto [Q, R] = custom_math::qr_accelerated(A);
        eval({Q, R});
    });
    auto [Q_gpu, R_gpu] = custom_math::qr_accelerated(A);
    eval({Q_gpu, R_gpu});

    // --- CPU (MLX / LAPACK) ---
    // linalg::qr does not support batched input — loop over batch dimension
    set_default_device(Device::cpu);
    double cpu_ms = time_ms([&] {
        if (batch == 1) {
            auto [Q, R] = linalg::qr(A, Device::cpu);
            eval({Q, R});
        } else {
            for (int b = 0; b < batch; ++b) {
                auto [Q, R] = linalg::qr(slice_batch(A, b, M, N), Device::cpu);
                eval({Q, R});
            }
        }
    });

    // Collect CPU result for error check
    std::pair<array, array> cpu_result = (batch == 1)
        ? linalg::qr(A, Device::cpu)
        : [&] {
            std::vector<array> Qs, Rs;
            for (int b = 0; b < batch; ++b) {
                auto [Q, R] = linalg::qr(slice_batch(A, b, M, N), Device::cpu);
                eval({Q, R});
                Qs.push_back(Q);
                Rs.push_back(R);
            }
            return std::pair{stack(Qs, 0), stack(Rs, 0)};
        }();

    auto [Q_cpu, R_cpu] = cpu_result;
    eval({Q_cpu, R_cpu});

    // --- Errors ---
    set_default_device(Device::gpu);
    float err_gpu = reconstruction_error(Q_gpu, R_gpu, A);
    set_default_device(Device::cpu);
    float err_cpu = reconstruction_error(Q_cpu, R_cpu, A);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "  GPU (Metal shader)  : " << std::setw(9) << gpu_ms << " ms"
              << "   mean ||QR-A||_F = " << err_gpu << "\n";
    std::cout << "  CPU (MLX / LAPACK)  : " << std::setw(9) << cpu_ms << " ms"
              << "   mean ||QR-A||_F = " << err_cpu << "\n";
    std::cout << "  Speedup             : " << std::setw(9) << (cpu_ms / gpu_ms) << "x\n";
}

// =============================================================================
// main
// =============================================================================

int main(int argc, char* argv[]) {
    int batch_small = 10000;
    int batch_large = 8;

    if (argc >= 2) batch_small = std::atoi(argv[1]);
    if (argc >= 3) batch_large = std::atoi(argv[2]);

    if (batch_small < 1 || batch_large < 1) {
        std::cerr << "Usage: " << argv[0] << " [batch_small] [batch_large]\n"
                  << "  batch_small : matrices per run for M < 512  (default 15000)\n"
                  << "  batch_large : matrices per run for M >= 512 (default 32)\n";
        return 1;
    }

    std::cout << "\n";
    std::cout << "================================================================\n";
    std::cout << "          QR Decomposition: GPU (Metal) vs CPU (MLX)\n";
    std::cout << "================================================================\n\n";
    std::cout << "Timing     : average of 5 runs (2 warmup discarded)\n";
    std::cout << "Error      : mean ||Q*R - A||_F across batch\n\n";

    // Small dimensions (M < 512) → QR_Unblocked
    std::cout << "[ Small dimensions — M < 512 — batch=" << batch_small << " ]\n\n";
    //benchmark(batch_small,  64,  64);
    //benchmark(batch_small, 128,  64);
    //benchmark(batch_small, 256, 128);
    //benchmark(batch_small, 512, 256);

    // Large dimensions (M >= 512) → QR_Streaming_AMX
    std::cout << "\n[ Large dimensions — M >= 512 — batch=" << batch_large << " ]\n\n";
    //benchmark(batch_large,  512,  512);
    //benchmark(batch_large, 1024,  512);
    benchmark(batch_large, 5000, 5000);

    print_separator();
    std::cout << "\n";
    return 0;
}
