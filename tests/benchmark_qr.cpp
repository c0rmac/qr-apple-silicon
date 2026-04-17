#include <iostream>
#include <iomanip>
#include <sstream>
#include <chrono>
#include <vector>
#include <cmath>
#include <random>
#include <cstdlib>
#include <string>

#include <mlx/mlx.h>
#include <mlx/linalg.h>

#include "qr.h"

using namespace mlx::core;

// =============================================================================
// Config
// =============================================================================

struct BenchConfig {
    int M, N;
    int max_batch = 0;  // 0 = no limit; set to skip batches above this value
};

static const std::vector<BenchConfig> SMALL_CONFIGS = {
    {8, 8},
    {16, 16},
    {32, 32},
    { 64,  64},
    {128,  64},
    {256, 128, 1000},   // skipped for batch > 1000
    {512, 256, 500},   // skipped for batch > 500
};

static const std::vector<BenchConfig> LARGE_CONFIGS = {
    { 512,  512},
    {1024,  512, 17},
    {5000, 5000, 9},
};

static const std::vector<int> SMALL_BATCHES = {10, 50, 100, 500, 1000, 5000, 10000, 15000};
static const std::vector<int> LARGE_BATCHES = {1, 8, 16, 32};

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

static bool exceeds_batch_limit(const BenchConfig& cfg, int batch) {
    return cfg.max_batch > 0 && batch > cfg.max_batch;
}

// =============================================================================
// Result collection
// =============================================================================

struct BenchResult {
    double gpu_ms = 0;
    double cpu_ms = 0;
    float  err_gpu = 0;
    bool   skipped = false;
};

static BenchResult run_benchmark(int batch, int M, int N) {
    BenchResult r;

    array A = random_matrix(batch, M, N);
    eval({A});

    // --- GPU ---
    set_default_device(Device::gpu);
    r.gpu_ms = time_ms([&] {
        auto [Q, R] = custom_math::qr_accelerated(A);
        eval({Q, R});
    });
    auto [Q_gpu, R_gpu] = custom_math::qr_accelerated(A);
    eval({Q_gpu, R_gpu});

    // --- CPU (MLX / LAPACK) ---
    set_default_device(Device::cpu);
    r.cpu_ms = time_ms([&] {
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

    // --- Error ---
    set_default_device(Device::gpu);
    r.err_gpu = reconstruction_error(Q_gpu, R_gpu, A);

    return r;
}

// =============================================================================
// Table rendering
// =============================================================================

static const int SHAPE_W = 14;   // width of the "Shape" column
static const int CELL_W  = 17;   // width of each batch-size column

static std::string pad(const std::string& s, int width, bool left = false) {
    if ((int)s.size() >= width) return s.substr(0, width);
    std::string out = left ? s : std::string(width - s.size(), ' ') + s;
    if (left) out += std::string(width - s.size(), ' ');
    return out;
}

static std::string format_shape(int M, int N) {
    std::ostringstream ss;
    ss << std::setw(4) << M << " x " << std::setw(4) << N;
    return ss.str();
}

static std::string format_ms(double ms) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(2) << ms << " ms";
    return ss.str();
}

static std::string format_speedup(double speedup) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(2) << speedup << "x";
    return ss.str();
}

static std::string make_divider(int n_batches) {
    std::string s = std::string(SHAPE_W, '-') + "+";
    for (int i = 0; i < n_batches; ++i)
        s += std::string(CELL_W, '-') + "+";
    return s;
}

static void print_table(
    const std::string& title,
    const std::vector<BenchConfig>& configs,
    const std::vector<int>& batches,
    const std::vector<std::vector<BenchResult>>& results  // [config_idx][batch_idx]
) {
    const int nb = (int)batches.size();

    std::cout << "\n[ " << title << " ]\n\n";

    // Header
    std::cout << pad("Shape", SHAPE_W, true) << "|";
    for (int b : batches)
        std::cout << pad("batch=" + std::to_string(b), CELL_W) << "|";
    std::cout << "\n";

    std::cout << make_divider(nb) << "\n";

    // Four sub-rows per config: shape header, GPU time, CPU time, speedup
    for (int ci = 0; ci < (int)configs.size(); ++ci) {
        const auto& cfg = configs[ci];

        // Sub-row 1: matrix shape (no data — acts as group header)
        std::cout << pad(format_shape(cfg.M, cfg.N), SHAPE_W, true) << "|";
        for (int bi = 0; bi < nb; ++bi)
            std::cout << std::string(CELL_W, ' ') << "|";
        std::cout << "\n";

        // Sub-row 2: GPU time
        std::cout << pad("  GPU", SHAPE_W, true) << "|";
        for (int bi = 0; bi < nb; ++bi) {
            const auto& r = results[ci][bi];
            std::cout << pad(r.skipped ? "--" : format_ms(r.gpu_ms), CELL_W) << "|";
        }
        std::cout << "\n";

        // Sub-row 3: CPU time
        std::cout << pad("  CPU", SHAPE_W, true) << "|";
        for (int bi = 0; bi < nb; ++bi) {
            const auto& r = results[ci][bi];
            std::cout << pad(r.skipped ? "--" : format_ms(r.cpu_ms), CELL_W) << "|";
        }
        std::cout << "\n";

        // Sub-row 4: speedup
        std::cout << pad("  Speedup", SHAPE_W, true) << "|";
        for (int bi = 0; bi < nb; ++bi) {
            const auto& r = results[ci][bi];
            std::cout << pad(r.skipped ? "--" : format_speedup(r.cpu_ms / r.gpu_ms), CELL_W) << "|";
        }
        std::cout << "\n";

        std::cout << make_divider(nb) << "\n";
    }

    std::cout << "  Skipped cells (--) exceeded the per-config batch limit.\n";
}

// =============================================================================
// main
// =============================================================================

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << "\n\n"
              << "  Per-config batch limits are set in SMALL_CONFIGS / LARGE_CONFIGS\n"
              << "  at the top of benchmark_qr.cpp via the max_batch field.\n"
              << "  Set max_batch = 0 on any config to run it at all batch sizes.\n";
}

int main(int argc, char* argv[]) {
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--help" || std::string(argv[i]) == "-h") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown argument: " << argv[i] << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    std::cout << "\n";
    std::cout << "================================================================\n";
    std::cout << "          QR Decomposition: GPU (Metal) vs CPU (MLX)\n";
    std::cout << "================================================================\n\n";
    std::cout << "Timing     : average of 5 runs (2 warmup discarded)\n";
    std::cout << "Error      : mean ||Q*R - A||_F across batch\n";

    // -------------------------------------------------------------------------
    // Collect results
    // -------------------------------------------------------------------------
    auto collect = [&](const std::vector<BenchConfig>& configs,
                       const std::vector<int>& batches) {
        std::vector<std::vector<BenchResult>> results(
            configs.size(), std::vector<BenchResult>(batches.size()));
        mlx::core::set_cache_limit(0);

        for (int ci = 0; ci < (int)configs.size(); ++ci) {
            for (int bi = 0; bi < (int)batches.size(); ++bi) {
                int M = configs[ci].M, N = configs[ci].N, B = batches[bi];
                if (exceeds_batch_limit(configs[ci], B)) {
                    results[ci][bi].skipped = true;
                } else {
                    std::cout << "  running " << B << " x " << M << " x " << N << " ...\r" << std::flush;
                    results[ci][bi] = run_benchmark(B, M, N);
                    mlx::core::clear_cache();
                }
            }
        }
        std::cout << std::string(50, ' ') << "\r";  // clear progress line
        return results;
    };

    std::cout << "Running large benchmarks...\n";
    auto large_results = collect(LARGE_CONFIGS, LARGE_BATCHES);
    print_table("Large dimensions  —  M >= 512", LARGE_CONFIGS, LARGE_BATCHES, large_results);

    std::cout << "\nRunning small benchmarks...\n";
    auto small_results = collect(SMALL_CONFIGS, SMALL_BATCHES);
    print_table("Small dimensions  —  M < 512",  SMALL_CONFIGS, SMALL_BATCHES, small_results);

    std::cout << "\n";
    return 0;
}
