# CUDA Kernel Optimization Playground

A hands‑on exploration of GPU performance through custom CUDA kernel development, with a focus on understanding optimization principles by implementing and systematically improving kernels from scratch.

## Motivation and Design Rationale

Performance on GPUs is governed by efficient data movement through the memory hierarchy and keeping Streaming Multiprocessors (SMs) fully utilized. This project emphasizes learning these principles through practical kernel development rather than relying on high‑level libraries.

## Environment

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
| Peak Memory Bandwidth | 320 GB/s |
| Theoretical Peak GFLOPS | ~8,141 (FP32) |

---

# Part 1: GPU and CUDA Fundamentals

- **Execution model**: GPUs consist of Streaming Multiprocessors (SMs), each executing warps of 32 threads in lockstep. Threads are organized into blocks, and blocks form a grid. Control-flow divergence within a warp reduces execution efficiency.

- **Memory hierarchy**: Registers (per-thread, fastest) → shared memory / L1 (per-block) → L2 cache → global device memory (large but high-latency). Performance is largely determined by how effectively global memory traffic is reduced and data reuse is maximized.

- **CUDA programming model**: Kernels are launched with a grid–block configuration (`<<<grid, block>>>`). Threads within a block can cooperate via shared memory and synchronize using `__syncthreads()`.

- **Streams and concurrency**: CUDA streams enable overlapping of computation and memory transfers to improve hardware utilization.

- **CUDA Graphs**: Kernel launches and dependencies can be captured as a DAG to reduce runtime launch overhead.

- **CUDA libraries**: cuBLAS, cuSPARSE, cuTENSOR, and Thrust provide highly optimized primitives and serve as performance references.

- **Distributed GPU computing**: At scale, multiple GPUs communicate using MPI. GPU-aware MPI allows direct device-to-device transfers without staging through host memory.

---

# Part 2: Foundational Kernel Techniques

Basic GPU kernels are explored to illustrate core optimization principles. Each kernel is implemented, verified, and profiled for performance insights.

- **Element-wise operations** – Vectorized loads (e.g., float4) to maximize memory bandwidth and thread utilization.
- **Reduction** – Warp-level shuffle intrinsics replace global atomics for fast, low-synchronization reductions.
- **Softmax** – Warp-per-row approach fuses max, sum, and normalization in one pass to minimize memory traffic.
- **Matrix transpose** – Shared memory tiling with padding / swizzling ensures coalesced accesses and avoids bank conflicts.

---

# Part 3: SGEMM Optimization Case Study

With the foundational techniques established, GEMM serves as a comprehensive case study for performance-critical GPU design. The kernel is progressively optimized:

- **Naive implementation** – Each thread computes one element of C. Simple but suffers from poor memory access patterns and low arithmetic intensity.
- **Shared memory tiling (block-level tiling)** – Tiles of A and B are staged into shared memory to reduce redundant global memory loads and improve coalescing.
- **Register tiling (thread-level tiling)** – Each thread computes a sub-tile (e.g., 8×8) using registers, increasing data reuse and compute efficiency.
- **Double buffering (prefetching)** – Alternates between two buffers to overlap computation with memory loads, hiding latency.
- **Low-level tuning (exploratory)** – Investigates register pressure, bank conflicts, and instruction scheduling to extract additional performance gains.

Through profiling and iteration, the GEMM kernel approaches cuBLAS-level efficiency while remaining transparent for learning and experimentation.

---

# Part 4: Performance Results

## Roofline Analysis

The **roofline model** classifies kernels by their bottleneck:
- **Memory-bound:** Performance limited by data movement. Arithmetic intensity < ~25 FLOP/byte on T4.
- **Compute-bound:** Performance limited by ALU throughput. Arithmetic intensity > ~25 FLOP/byte.

| Kernel Type | Arithmetic Intensity | Bound | Peak Limiter |
|-------------|---------------------|-------|--------------|
| Elementwise | 0.125 FLOP/byte | **Memory** | Bandwidth |
| Reduction | 0.125 FLOP/byte | **Memory** | Bandwidth |
| Transpose | 0 FLOP/byte | **Memory** | Bandwidth |
| SGEMM | ~170 FLOP/byte | **Compute** | FLOPS |

## Summary

| Kernel | Classification | Achieved | Ceiling | Status |
|--------|---------------|----------|---------|--------|
| Elementwise | Memory-bound | 252 GB/s (79%) | ~280 GB/s | ✅ Near optimal |
| Reduction | Memory-bound | 290 GB/s (91%) | ~300 GB/s | ✅ **Optimal** |
| Transpose | Memory-bound | 199 GB/s (62%) | ~250 GB/s | ✅ Good |
| SGEMM | Compute-bound | 4,209 GFLOPS (52%) | ~6,500 GFLOPS | 🔶 Room to grow |

---

## Elementwise Operations (32M elements)

**Classification:** Memory-bound (AI ≈ 0.125 FLOP/byte)

| Kernel | Time (ms) | Bandwidth (GB/s) | Efficiency |
|--------|-----------|------------------|------------|
| ADD (float4) | 1.597 | 252.1 | **78.8%** |
| SIGMOID (float4) | 1.113 | 241.3 | 75.4% |
| RELU (float4) | 1.149 | 233.6 | 73.0% |

At 73-79% of peak bandwidth, we're near the practical ceiling. The limiting factor is DRAM bandwidth, not compute.

---

## Reduction Sum (32M elements)

**Classification:** Memory-bound (AI ≈ 0.125 FLOP/byte)

| Version | Intent | Time (ms) | BW (GB/s) | Speedup |
|---------|--------|-----------|-----------|---------|
| v2 | Baseline shared memory | 2.39 | 56.1 | 1.0x |
| v3 | Show atomic contention | 1.27 | 105.8 | 1.9x |
| v4 | Warp-synchronous (`__shfl_down_sync`) | 1.20 | 111.6 | 2.0x |
| **v5** | **float4 + warp shuffle** | **0.46** | **289.7** | **5.2x** |

v5 achieves **90.5% bandwidth efficiency**—essentially at the ceiling for a reduction kernel.

---

## Matrix Transpose (4096×4096)

**Classification:** Memory-bound (AI = 0, pure data movement)

| Version | Intent | Time (ms) | BW (GB/s) | Speedup |
|---------|--------|-----------|-----------|---------|
| v0 | Naive baseline | 1.91 | 70.3 | 1.0x |
| v1 | Coalesced writes | 1.81 | 74.2 | 1.1x |
| v2 | `__ldg()` cache | 1.43 | 94.1 | 1.3x |
| v3 | Shared memory tiling | 0.88 | 152.0 | 2.2x |
| **v4** | **Shared + padding** | **0.68** | **198.8** | **2.8x** |
| v5 | Shared + swizzling | 0.68 | 198.8 | 2.8x |

At **62% of peak bandwidth**, transpose is limited by the fundamental read-write asymmetry.

---

## SGEMM (1024×1024×1024)

**Classification:** Compute-bound (AI ≈ 170 FLOP/byte)

| Version | Intent | Time (ms) | GFLOPS | % cuBLAS |
|---------|--------|-----------|--------|----------|
| naive | Baseline | 4.74 | 453 | 7% |
| v2 | Shared memory tiling | 3.34 | 643 | 10% |
| v3 | 1D thread tiling | 1.94 | 1,109 | 17% |
| v4 | 2D thread tiling | 1.52 | 1,408 | 22% |
| v5 | Register caching | 1.45 | 1,477 | 23% |
| v6 | Vectorized float4 | 0.53 | 4,052 | 62% |
| **v7** | **Double buffering** | **0.51** | **4,209** | **65%** |
| cuBLAS | Reference | 0.33 | 6,523 | 100% |

**Scalability:**

| Size | v7 GFLOPS | cuBLAS | v7/cuBLAS |
|------|-----------|--------|-----------|
| 512³ | 2,759 | 5,448 | 51% |
| 1024³ | 4,249 | 6,587 | 65% |
| 2048³ | 4,458 | 5,954 | 75% |
| 4096³ | **4,719** | 5,729 | **82%** |

v7 reaches **82% of cuBLAS at large sizes**—the gap narrows as overhead becomes negligible.

---

## Key Learnings

1. **Memory-bound kernels (elementwise, reduction, transpose):**
   - Optimize for bandwidth: coalescing, vectorization, cache utilization
   - Once at >75% bandwidth, diminishing returns
   - Further gains require algorithmic changes (fusion, compression)

2. **Compute-bound kernels (SGEMM):**
   - Optimize for arithmetic intensity: tiling, register blocking
   - Occupancy vs ILP tradeoff is real
   - Next level requires tensor cores or mixed precision

3. **Universal patterns that work:**
   - float4 vectorization for 4x fewer memory transactions
   - Warp shuffle for intra-warp communication (no shared memory)
   - Shared memory tiling for inter-warp data reuse
   - Double buffering to hide latency

4. **Edge cases:**
   - Kernel launch overhead dominates for <10K elements
   - L2 cache causes artificially high bandwidth for <1M elements
   - Non-power-of-two dimensions: ~1% overhead for transpose, ~22% for SGEMM

---

## Next Steps

1. **SGEMM:** Implement WMMA tensor core path (potential 2x)
2. **All kernels:** FP16/BF16 variants (2x throughput, 2x bandwidth)
3. **Reduction:** Multi-pass for >4B elements
4. **Profiling:** Detailed Nsight Compute analysis for register pressure

---

# Build & Run

```bash
mkdir build && cd build
cmake ..
make

# Run tests
./test_elementwise
./test_reduction
./test_transpose
./test_sgemm
```

# Profiling

```bash
# Nsight Compute (detailed kernel metrics)
ncu --set full -o profile ./build/test_sgemm

# Nsight Systems (timeline)
nsys profile -o trace ./build/test_sgemm
```

---

# References

- S. Böhm, [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance](https://siboehm.com/articles/22/CUDA-MMM)
- Tongkaio, [CUDA Kernel Samples](https://github.com/Tongkaio/CUDA_Kernel_Samples)
- NVIDIA Developer Talk: https://www.youtube.com/watch?v=86FAWCzIe_4
- Kirk, D. B., & Wen-mei, W. H. (2016). *Programming Massively Parallel Processors: A Hands-on Approach*. Morgan Kaufmann.
- Lei Mao, [CUDA Matrix Multiplication Optimization](https://leimao.github.io/article/CUDA-Matrix-Multiplication-Optimization/)
- Zhihu CUDA Optimization Column: https://www.zhihu.com/column/c_1437330196193640448

---

*Baseline checkpoint: `v1.0-baseline` | Tesla T4 | December 2024*
