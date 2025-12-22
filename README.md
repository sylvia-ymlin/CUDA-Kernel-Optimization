A hands‑on exploration of GPU performance through custom CUDA kernel development, with a focus on understanding optimization principles by implementing and systematically improving kernels from scratch.

# Motivation and Design Rationale
Performance on GPUs is governed by efficient data movement through the memory hierarchy and keeping Streaming Multiprocessors (SMs) fully utilized. This project emphasizes learning these principles through practical kernel development rather than relying on high‑level libraries.

# Enviroment

| Property | Value |
|----------|-------|
| Platform | Ubuntu 22.04.5 LTS (GCP VM) |
| Device | NVIDIA Tesla T4 |
| CUDA Capability | 7.5 |
| Total Global Memory | 14930 MB |
| Multiprocessors | 40 |
| CUDA Cores per SM | 64 |
| Total CUDA Cores | 2560 |
| GPU Max Clock | 1590 MHz |
| Memory Clock | 5001 MHz |
| Memory Bus Width | 256-bit |
| L2 Cache Size | 4 MB |
| Shared Memory per Block | 48 KB |
| Registers per Block | 65536 |
| Warp Size | 32 |
| Max Threads per Block | 1024 |
| Max Block Dimensions | (1024, 1024, 64) |
| Max Grid Dimensions | (2147483647, 65535, 65535) |
| CUDA Driver Version | 12.2 |
| CUDA Runtime Version | 12.5 |
| CPU | 4 vCPUs, Intel Broadwell |
| RAM | 26 GB |

# Part 1: GPU and CUDA Fundamentals

- **Execution model**: GPUs consist of Streaming Multiprocessors (SMs), each executing warps of 32 threads in lockstep. Threads are organized into blocks, and blocks form a grid. Control-flow divergence within a warp reduces execution efficiency.

- **Memory hierarchy**: Registers (per-thread, fastest) → shared memory / L1 (per-block) → L2 cache → global device memory (large but high-latency). Performance is largely determined by how effectively global memory traffic is reduced and data reuse is maximized.

- **CUDA programming model**: Kernels are launched with a grid–block configuration (`<<<grid, block>>>`). Threads within a block can cooperate via shared memory and synchronize using `__syncthreads()`.

- **Streams and concurrency**: CUDA streams enable overlapping of computation and memory transfers to improve hardware utilization.

- **CUDA Graphs**: Kernel launches and dependencies can be captured as a DAG to reduce runtime launch overhead.

- **CUDA libraries**: cuBLAS, cuSPARSE, cuTENSOR, and Thrust provide highly optimized primitives and serve as performance references.

Distributed GPU computingAt scale, multiple GPUs communicate using MPI. GPU-aware MPI allows direct device-to-device transfers without staging through host memory.

# Part 2: Foundational Kernel Techniques

Basic GPU kernels are explored to illustrate core optimization principles. Each kernel is implemented, verified, and profiled for performance insights.

- Element-wise operations – Vectorized loads (e.g., float4) to maximize memory bandwidth and thread utilization.
- Reduction – Warp-level shuffle intrinsics replace global atomics for fast, low-synchronization reductions.
- Softmax – Warp-per-row approach fuses max, sum, and normalization in one pass to minimize memory traffic.
- Matrix transpose – Shared memory tiling with padding / swizzling ensures coalesced accesses and avoids bank conflicts.


# Part 3: Extending Optimization Strategies to Other Kernels

With the foundational techniques established, GEMM serves as a comprehensive case study for performance-critical GPU design. The kernel is progressively optimized:

- Naive implementation – Each thread computes one element of C. Simple but suffers from poor memory access patterns and low arithmetic intensity.
- Shared memory tiling (block-level tiling) – Tiles of A and B are staged into shared memory to reduce redundant global memory loads and improve coalescing.
- Register tiling (thread-level tiling) – Each thread computes a sub-tile (e.g., 8×8) using registers, increasing data reuse and compute efficiency.
- Double buffering (prefetching) – Alternates between two buffers to overlap computation with memory loads, hiding latency.
- Low-level tuning (exploratory) – Investigates register pressure, bank conflicts, and instruction scheduling to extract additional performance gains.

Through profiling and iteration, the GEMM kernel approaches cuBLAS-level efficiency while remaining transparent for learning and experimentation.

# References

- S. Böhm, [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance](https://siboehm.com/articles/22/CUDA-MMM)
- Tongkaio, [CUDA Kernel Samples](https://github.com/Tongkaio/CUDA_Kernel_Samples)
- NVIDIA Developer Talk: https://www.youtube.com/watch?v=86FAWCzIe_4
- Kirk, D. B., & Wen-mei, W. H. (2016). *Programming Massively Parallel Processors: A Hands-on Approach*. Morgan Kaufmann.
- Lei Mao, [CUDA Matrix Multiplication Optimization](https://leimao.github.io/article/CUDA-Matrix-Multiplication-Optimization/)
- Zhihu CUDA Optimization Column: https://www.zhihu.com/column/c_1437330196193640448

