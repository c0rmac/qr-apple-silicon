# qr-apple-silicon

Hardware-accelerated QR decomposition for Apple Silicon, built on top of [MLX](https://github.com/ml-explore/mlx) and Metal.

Exposes a single function, `custom_math::qr_accelerated`, that accepts a batched MLX array and returns Q and R — automatically routing to the most efficient Metal kernel for the given matrix dimensions and batch size.


## API

This library depends on the MLX C++ library. On macOS:

```sh
brew install mlx
```

```cpp
#include "qr.h"

// a: MLX array of shape [M, N] or [B, M, N] (float32 or auto-cast)
// Returns: {Q, R} where Q is [M, K] and R is [K, N], K = min(M, N)
auto [Q, R] = custom_math::qr_accelerated(a);
```

The function operates on the default MLX device. Set it to GPU before calling:

```cpp
#include "qr.h"
#include <mlx/mlx.h>

using namespace mlx::core;

// Build a random 4 x 4 matrix
std::vector<float> data = {
     1,  2,  3,  4,
     5,  6,  7,  8,
     9, 10, 11, 12,
    13, 14, 15, 16
};
array A(data.begin(), {4, 4}, float32);

set_default_device(Device::gpu);
auto [Q, R] = custom_math::qr_accelerated(A);
eval({Q, R});

// Q: [4, 4] orthogonal matrix
// R: [4, 4] upper-triangular matrix
// A ≈ Q * R
```

## The QR Decomposition explained

Given a matrix $A \in \mathbb{R}^{M \times N}$, the QR decomposition factors it as:

$$A = QR$$

where $K = \min(M, N)$.

**Q** $\in \mathbb{R}^{M \times K}$ is a matrix with orthonormal columns. That is, for any two columns $q_i$ and $q_j$:

$$q_i^T q_j = \delta_{ij} = \begin{cases} 1 & \text{if } i = j \\ 0 & \text{if } i \neq j \end{cases}$$

which can be stated compactly as $Q^T Q = I_K$. The columns of $Q$ form an orthonormal basis for the column space of $A$.

**R** $\in \mathbb{R}^{K \times N}$ is upper triangular. Every entry strictly below the main diagonal is zero:

$$R_{ij} = 0 \quad \text{for all } i > j$$

This is the *thin* (or *reduced*) QR decomposition. The full decomposition extends $Q$ to a square $M \times M$ orthogonal matrix, but the thin form is sufficient to reconstruct $A$ and is more compact when $M > N$.

For example, given:

$$A = \begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{pmatrix}$$

the thin QR decomposition yields:

$$Q = \begin{pmatrix} -0.123 & 0.904 & 0.408 \\ -0.492 & 0.301 & -0.816 \\ -0.862 & -0.301 & 0.408 \end{pmatrix}, \quad R = \begin{pmatrix} -8.124 & -9.601 & -11.078 \\ 0 & 0.905 & 1.809 \\ 0 & 0 & 0 \end{pmatrix}$$

One can verify $QR = A$ and $Q^T Q = I_3$.

The decomposition is fundamental to solving linear least-squares problems, performing Gram-Schmidt orthogonalisation, and as the core step in the QR algorithm for computing eigenvalues.

### The Householder Reflection

Both shaders build $Q$ and $R$ by successively applying **Householder reflections**. A Householder reflector is an orthogonal matrix of the form:

$$H = I - \tau v v^T, \quad \tau \in \mathbb{R}, \quad v \in \mathbb{R}^M$$

chosen so that $H x = \mu e_k$ — i.e. it zeros out every entry of a column vector $x$ below position $k$, leaving a single scalar $\mu$ on the diagonal. The sign of $\mu$ is chosen to avoid catastrophic cancellation:

$$\mu = -\operatorname{sign}(\alpha)\|x\|_2$$

where $\alpha = x_k$ is the pivot element. The reflector vector $v$ is then:

$$v_k = 1, \quad v_i = \frac{x_i}{\alpha - \mu} \text{ for } i > k, \quad \tau = \frac{\mu - \alpha}{\mu}$$

Applying $K = \min(M, N)$ reflectors in sequence drives $A$ to upper triangular form:

$$H_K \cdots H_2 H_1 A = R \implies A = H_1 H_2 \cdots H_K R = QR$$

Since each $H_i$ is orthogonal, their product $Q = H_1 H_2 \cdots H_K$ is also orthogonal. Rather than forming this product one reflector at a time, both shaders use the **Compact WY representation** to batch the updates.

### The Compact WY Representation

For a block of $b$ consecutive Householder reflectors, the product can be written as:

$$H_1 H_2 \cdots H_b = I - Y T Y^T$$

where $Y \in \mathbb{R}^{M \times b}$ has the $b$ reflector vectors as its columns, and $T \in \mathbb{R}^{b \times b}$ is an upper triangular matrix constructed recursively:

$$T_{jj} = \tau_j, \quad T_{ij} = -\tau_j \sum_{m=i}^{j-1} T_{im} (y_m^T y_j) \quad \text{for } i < j$$

This lets a full block update be expressed as a pair of matrix multiplications:

$$A \leftarrow A - Y \bigl( T^T (Y^T A) \bigr)$$

which maps directly onto the AMX matrix coprocessor's 8×8 `simdgroup_matrix` tiles.

### Algorithm: `qr_unblocked` (single-kernel, block size $b = 16$)

This kernel processes the entire matrix in one GPU dispatch. It operates on matrices stored in **column-major** format to align with AMX load/store strides, and pads dimensions to multiples of 32 (rows) and 16 (columns) to eliminate in-kernel boundary branching.

For each block of $b = 16$ columns starting at column $s$:

**Step 1 — Panel factorisation.** For each column $k = 0, \ldots, b-1$ within the block:

1. All 1024 threads cooperatively compute $\|x_\text{tail}\|^2$ via a two-phase threadgroup reduction (intra-SIMD via `simd_sum`, then inter-SIMD via shared memory).
2. Thread 0 computes $\mu$, $\tau$, and the scale $1/(\alpha - \mu)$ and broadcasts them through threadgroup memory.
3. Each thread normalises its portion of the tail: $A[r, k] \leftarrow A[r, k] / (\alpha - \mu)$ for $r > k$.
4. The reflector is applied to the remaining $b - k - 1$ columns of the panel: for each $j > k$, $a_j \leftarrow a_j - \tau (v_k^T a_j) v_k$, with the dot product accumulated via threadgroup reduction.

**Step 2 — Form T.** The $b \times b$ upper triangular matrix $T$ is built in threadgroup memory using the recursive formula above.

**Step 3 — Trailing matrix update.** Each SIMD group owns a set of 8-column tiles of the trailing submatrix $A[{:}, s+b{:}]$. Using AMX 8×8 tiles, it computes:

$$Z = Y^T A_\text{trail}, \quad Z \leftarrow T Z, \quad A_\text{trail} \leftarrow A_\text{trail} - Y Z$$

**Step 4 — Q accumulation.** The same WY update is applied to $Q$ (initialised to $I$):

$$Q \leftarrow Q - Y \bigl( T (Y^T Q) \bigr)$$

**Step 5 — Restore diagonal.** The stored $\mu$ values are written back to the diagonal of $A$ (overwriting the temporary $v_k = 1$ sentinel placed there during factorisation).

### Algorithm: `qr_streaming_amx` (multi-kernel, block size $b = 32$)

For large matrices ($M$ or $N \geq 512$), a single-kernel dispatch causes Q-accumulation to bottleneck on a single shader multiprocessor. The streaming variant splits the computation across **four separate kernel dispatches** per block, allowing the GPU scheduler to assign the trailing update across all available cores in parallel.

The block size is widened to $b = 32$ to match the SIMD group width, maximising AMX tile utilisation and reducing the number of host-side dispatch iterations.

**Kernel 0 — Preprocess.** The input is transposed from row-major to column-major and padded with identity blocks. $Q$ is initialised to $I_{M \times M}$.

**Kernel 1 — Panel factorisation.** Dispatched with 1 threadgroup per matrix. For each column $k$ in the current block, 1024 threads cooperatively compute $\tau_k$ and the normalised reflector vector, then apply it to the remaining $b - k - 1$ panel columns. The $\tau$ values and diagonal elements of $R$ are written to global memory for use by subsequent kernels.

**Kernel 2 — T-matrix construction.** Dispatched with 1 threadgroup per matrix. Reads the reflector columns from $A$ and $\tau$ from global memory and builds $T \in \mathbb{R}^{32 \times 32}$ in threadgroup memory using the recursive Compact WY formula. The completed $T$ is written to global memory negated (i.e. $-T$ is stored), so that Kernel 3 can use `simdgroup_multiply_accumulate` (which adds) rather than needing a subtract path.

**Kernel 3 — Grid-parallel trailing update.** Dispatched with one threadgroup per 32-column tile of the trailing submatrix. Each threadgroup independently computes:

$$\text{Phase 1:} \quad Z^T = A_\text{trail}^T \cdot Y$$

$$\text{Phase 2:} \quad Z_\text{final}^T = Z^T \cdot T$$

$$\text{Phase 3:} \quad A_\text{trail} \leftarrow A_\text{trail} + Y \cdot Z_\text{final}^T$$

$Y$ and $T$ are loaded into threadgroup memory (L1 cache) once per tile and reused across all AMX sweeps. The same kernel is reused for Q-accumulation by setting a flag that redirects the target pointer from $A$ to $Q$.

**Kernel 4 — Haar fix.** Ensures the output $Q$ is a uniform sample from the Haar measure on $O(M)$ and that $R$ has non-negative diagonal. For each column $k$ where $R_{kk} < 0$, the signs of column $k$ in both $Q$ and $R$ are flipped. If the resulting $\det(Q) < 0$, the final column is negated to enforce $\det(Q) = +1$, placing $Q$ in $SO(M)$.

## How it works

Two Metal kernels handle different regimes, with a dispatcher that selects between them at runtime:

**`qr_unblocked`** — Standard Householder QR in a single kernel dispatch. Used for smaller matrices where the overhead of multi-pass streaming is not worth it.

**`qr_streaming_amx`** — Multi-pass panel factorisation with grid-parallel trailing matrix updates, designed to saturate the GPU for large matrices. Factorises column panels of width 32, computes the T-matrix for each WY representation, then launches a grid of threadgroups for the trailing update.

### Dispatch logic

| Condition | Kernel |
|---|---|
| `max(M, N) >= 512` | `qr_streaming_amx` |
| `max(M, N) < 512` and `batch >= 16` | `qr_unblocked` |
| `max(M, N) >= 128` and `batch < 16` | `qr_streaming_amx` |
| `max(M, N) < 128` and `batch < 16` | `qr_unblocked` |

The crossover points reflect two competing costs: single-SM Q-accumulation becomes a bottleneck above ~512, while multi-kernel launch overhead makes streaming slower than unblocked for small matrices with few batch elements.

Both backends cache compiled `MTLComputePipelineState` objects and recycle GPU memory workspaces on repeated calls with the same shape, so warm invocations avoid both JIT compilation and OS-level buffer allocation.

## Requirements

- Apple Silicon Mac (M1 or later)
- macOS with Xcode command line tools (`xcrun`, `metal`, `metallib`)
- [MLX](https://github.com/ml-explore/mlx) installed and findable by CMake
- CMake 3.25+
- C++20

## Installation

> TODO: exportable library packaging is planned for a future release.

## Running the benchmark

Clone the repository and build in release mode:

```sh
git clone https://github.com/cormaccinnsealach/qr-apple-silicon.git
cd qr-apple-silicon

cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target benchmark_qr
```

Run with defaults (batch of 8 for large matrices):

```sh
./build/benchmark_qr
```

Or pass custom batch sizes:

```sh
# benchmark_qr [batch_small] [batch_large]
# batch_small: used for M < 512  (default: 10000)
# batch_large: used for M >= 512 (default: 8)
./build/benchmark_qr 10000 8
```

The benchmark runs each configuration for 5 timed repetitions (2 warmup discarded) and reports GPU time, CPU time (MLX / LAPACK), speedup, and mean Frobenius reconstruction error `||QR - A||_F` for both.
