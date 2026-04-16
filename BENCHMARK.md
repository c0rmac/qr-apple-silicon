# Benchmark Results

GPU (Metal shader) vs CPU (MLX / LAPACK) on Apple Silicon.  
Timing is the average of 5 runs with 2 warmup runs discarded.  
Skipped cells (—) exceeded the per-config batch limit.

---

## Small dimensions — M < 512

| Shape | | batch=10 | batch=50 | batch=100 | batch=500 | batch=1000 | batch=5000 | batch=10000 | batch=15000 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **8 × 8** | GPU | 0.49 ms | 0.59 ms | 0.64 ms | 1.63 ms | 3.17 ms | 8.40 ms | 16.54 ms | 24.64 ms |
| | CPU | 1.10 ms | 1.81 ms | 4.21 ms | 29.51 ms | 47.43 ms | 194.28 ms | 392.69 ms | 592.42 ms |
| | **Speedup** | **2.27x** | **3.10x** | **6.58x** | **18.09x** | **14.94x** | **23.13x** | **23.74x** | **24.04x** |
| **16 × 16** | GPU | 0.54 ms | 0.45 ms | 0.85 ms | 2.40 ms | 3.59 ms | 12.48 ms | 25.01 ms | 39.91 ms |
| | CPU | 0.33 ms | 2.05 ms | 4.05 ms | 20.89 ms | 41.17 ms | 204.96 ms | 411.91 ms | 619.39 ms |
| | **Speedup** | **0.61x** | **4.53x** | **4.77x** | **8.70x** | **11.47x** | **16.42x** | **16.47x** | **15.52x** |
| **32 × 32** | GPU | 0.74 ms | 0.90 ms | 1.47 ms | 4.45 ms | 7.39 ms | 33.84 ms | 71.03 ms | 107.61 ms |
| | CPU | 0.47 ms | 2.32 ms | 4.72 ms | 23.73 ms | 46.94 ms | 233.64 ms | 468.85 ms | 727.55 ms |
| | **Speedup** | **0.64x** | **2.57x** | **3.22x** | **5.34x** | **6.36x** | **6.90x** | **6.60x** | **6.76x** |
| **64 × 64** | GPU | 1.01 ms | 2.21 ms | 3.56 ms | 14.19 ms | 27.47 ms | 135.44 ms | 284.26 ms | 411.40 ms |
| | CPU | 1.00 ms | 4.81 ms | 9.52 ms | 47.38 ms | 93.89 ms | 469.27 ms | 940.34 ms | 1413.26 ms |
| | **Speedup** | **0.99x** | **2.18x** | **2.68x** | **3.34x** | **3.42x** | **3.46x** | **3.31x** | **3.44x** |
| **128 × 64** | GPU | 1.85 ms | 2.39 ms | 6.04 ms | 20.12 ms | 41.24 ms | 207.19 ms | 409.29 ms | 579.60 ms |
| | CPU | 1.60 ms | 6.70 ms | 13.21 ms | 66.34 ms | 132.21 ms | 665.30 ms | 1326.68 ms | 2035.80 ms |
| | **Speedup** | **0.87x** | **2.81x** | **2.19x** | **3.30x** | **3.21x** | **3.21x** | **3.24x** | **3.51x** |
| **256 × 128** | GPU | 4.46 ms | 11.27 ms | 22.12 ms | 97.19 ms | 209.31 ms | — | — | — |
| | CPU | 6.12 ms | 30.15 ms | 65.47 ms | 325.97 ms | 596.31 ms | — | — | — |
| | **Speedup** | **1.37x** | **2.68x** | **2.96x** | **3.35x** | **2.85x** | — | — | — |
| **512 × 256** | GPU | 13.53 ms | 59.71 ms | 112.91 ms | 519.76 ms | — | — | — | — |
| | CPU | 24.12 ms | 119.07 ms | 237.59 ms | 1196.59 ms | — | — | — | — |
| | **Speedup** | **1.78x** | **1.99x** | **2.10x** | **2.30x** | — | — | — | — |

### Observations

- At very small batch sizes (≤ 10), GPU launch overhead can exceed the compute cost, resulting in sub-1x speedup for small matrices. This is expected — the GPU only becomes worthwhile once there is enough work to amortise the dispatch cost.
- Speedup scales strongly with batch size. At batch=15000, an 8×8 workload reaches **24x** over LAPACK.
- The crossover from CPU-faster to GPU-faster occurs around batch=50 for most small configurations.

---

## Large dimensions — M ≥ 512

| Shape | | batch=1 | batch=8 | batch=16 | batch=32 |
|---|---|---:|---:|---:|---:|
| **512 × 512** | GPU | 12.07 ms | 24.07 ms | 35.50 ms | 69.65 ms |
| | CPU | 7.00 ms | 43.75 ms | 85.92 ms | 174.61 ms |
| | **Speedup** | **0.58x** | **1.82x** | **2.42x** | **2.51x** |
| **1024 × 512** | GPU | 17.93 ms | 53.30 ms | 92.24 ms | — |
| | CPU | 12.95 ms | 98.48 ms | 196.12 ms | — |
| | **Speedup** | **0.72x** | **1.85x** | **2.13x** | — |
| **5000 × 5000** | GPU | 1213.09 ms | 7494.11 ms | — | — |
| | CPU | 1970.29 ms | 15769.81 ms | — | — |
| | **Speedup** | **1.62x** | **2.10x** | — | — |

### Observations

- At batch=1, the GPU is slower than LAPACK for 512×512 and 1024×512 matrices. A single large matrix does not generate enough parallelism to saturate the GPU's shader multiprocessors.
- Speedup grows steadily with batch size, reaching **2.51x** at batch=32 for 512×512.
- Even at batch=1, the GPU outperforms LAPACK on 5000×5000 (**1.62x**), where the sheer volume of in-matrix computation saturates the shader cores without needing a large batch.
