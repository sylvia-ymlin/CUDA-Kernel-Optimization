# CUDA Kernel Performance Report

**Device:** Tesla T4 (Turing, SM 7.5)  
**Peak Memory Bandwidth:** 320 GB/s  
**Theoretical Peak GFLOPS:** ~8,141 (FP32)  
**L2 Cache:** 4 MB

---

## Roofline Context

The **roofline model** classifies kernels by their bottleneck:
- **Memory-bound:** Performance limited by data movement. Arithmetic intensity < ~25 FLOP/byte on T4.
- **Compute-bound:** Performance limited by ALU throughput. Arithmetic intensity > ~25 FLOP/byte.

| Kernel Type | Arithmetic Intensity | Bound | Peak Limiter |
|-------------|---------------------|-------|--------------|
| Elementwise | 0.125 FLOP/byte (ADD) | **Memory** | Bandwidth |
| Reduction | 0.125 FLOP/byte | **Memory** | Bandwidth |
| Transpose | 0 FLOP/byte | **Memory** | Bandwidth |
| SGEMM | ~170 FLOP/byte (1K³) | **Compute** | FLOPS |

---

## 1. Elementwise Operations (32M elements)

**Classification:** Memory-bound (AI ≈ 0.125 FLOP/byte)

| Kernel | Time (ms) | Bandwidth (GB/s) | Efficiency |
|--------|-----------|------------------|------------|
| ADD (float4) | 1.597 | 252.1 | **78.8%** |
| SIGMOID (float4) | 1.113 | 241.3 | 75.4% |
| RELU (float4) | 1.149 | 233.6 | 73.0% |

**Analysis:**  
These kernels are **firmly memory-bound**. At 73-79% of peak bandwidth, we're near the practical ceiling. Further optimizations (more unrolling, different grid sizes) would yield <5% gains. The limiting factor is DRAM bandwidth, not compute.

**Why ~78% and not 100%?**
- ECC overhead on T4 (~6%)
- TLB misses and page table walks
- Memory controller scheduling inefficiency

**Edge case - small sizes (L2 cache effects):**
| Size | Time (ms) | BW (GB/s) | Note |
|------|-----------|-----------|------|
| 1K | 0.0024 | 5.2 | Kernel launch overhead dominates (~2μs) |
| 64K | 0.0023 | 335 | Apparent BW > peak = L2 cache hits |
| 1M | 0.054 | 233 | Working set exceeds L2, DRAM-bound |

*The >100% "bandwidth" at 64K elements is not real—data fits in L2 cache (256KB < 4MB).*

---

## 2. Reduction Sum (32M elements)

**Classification:** Memory-bound (AI ≈ 0.125 FLOP/byte)

| Version | Intent | Time (ms) | BW (GB/s) | Speedup |
|---------|--------|-----------|-----------|---------|
| v2 | Baseline shared memory | 2.39 | 56.1 | 1.0x |
| v3 | Show atomic contention cost | 1.27 | 105.8 | 1.9x |
| v4 | Warp-synchronous programming | 1.20 | 111.6 | 2.0x |
| **v5** | **Vectorization + shuffle wins** | **0.46** | **289.7** | **5.2x** |

**Why each version exists:**
- **v2:** Demonstrates basic shared memory tiling—the textbook approach
- **v3:** Shows that atomics are surprisingly fast on modern GPUs (but don't scale)
- **v4:** Eliminates shared memory in reduction phase via `__shfl_down_sync`
- **v5:** Combines float4 loads (4x fewer transactions) with warp shuffle—the optimal pattern

**Analysis:**  
v5 achieves **90.5% bandwidth efficiency**—essentially at the ceiling for a reduction kernel. The remaining 10% is irreducible overhead from:
- Final atomic aggregation across blocks
- Partial warps at boundaries
- Instruction issue overhead

**Further optimization is not worthwhile.** Any new reduction kernel should match v5's pattern.

**Scalability (sum_v5):**
| Size | Time (ms) | BW (GB/s) | Efficiency |
|------|-----------|-----------|------------|
| 64K | 0.003 | 87.3 | 27.3% |
| 1M | 0.012 | 352.9 | 110%* |
| 4M | 0.061 | 276.3 | 86.3% |
| 32M | 0.463 | 289.8 | 90.5% |

*>100% at 1M = L2 cache reuse during multi-pass reduction

---

## 3. Softmax (4096×1024 matrix)

**Classification:** Memory-bound with moderate compute

| Kernel | Intent | Time (ms) | BW (GB/s) |
|--------|--------|-----------|-----------|
| Row-wise (shared) | Coalesced access pattern | 0.352 | 95.2 |
| Row-wise (shfl_xor) | Butterfly reduction | 0.584 | 57.4 |
| Column-wise (shfl_xor) | Show strided access penalty | 0.833 | 40.3 |

**Analysis:**  
Row-wise softmax is **2.4x faster** than column-wise due to memory coalescing. Column-wise access causes 32-byte transactions to fetch only 4 useful bytes—an 8x amplification of memory traffic.

**Lesson:** Always prefer row-major iteration for row-major storage.

---

## 4. Matrix Transpose (4096×4096)

**Classification:** Memory-bound (AI = 0, pure data movement)

| Version | Intent | Time (ms) | BW (GB/s) | Speedup |
|---------|--------|-----------|-----------|---------|
| v0 | Naive baseline | 1.91 | 70.3 | 1.0x |
| v1 | Coalesced writes | 1.81 | 74.2 | 1.1x |
| v2 | `__ldg()` read-only cache | 1.43 | 94.1 | 1.3x |
| v3 | Shared memory tiling | 0.88 | 152.0 | 2.2x |
| **v4** | **Padding for bank conflicts** | **0.68** | **198.8** | **2.8x** |
| v5 | Swizzling (alternative) | 0.68 | 198.8 | 2.8x |

**Why each version exists:**
- **v0:** Shows the cost of uncoalesced writes (strided by N)
- **v1:** Coalesced writes via transposed output indexing
- **v2:** Tests whether `__ldg()` helps (modest improvement)
- **v3:** Shared memory converts strided reads to coalesced reads—but has bank conflicts
- **v4:** +1 padding eliminates 32-way bank conflicts (1.3x over v3)
- **v5:** XOR swizzling—same effect as padding, slightly less memory

**Analysis:**  
At **62% of peak bandwidth**, transpose is limited by the fundamental read-write asymmetry: we must either read strided or write strided. Shared memory tiling minimizes this penalty but can't eliminate it.

**Edge case - non-square matrices:**
| Matrix | Time (ms) | BW (GB/s) | Note |
|--------|-----------|-----------|------|
| 4096×1024 | 0.153 | 219.9 | Reference |
| 4096×1023 | 0.153 | 218.7 | **Only 0.5% slower** |

Non-power-of-two dimensions cause minimal overhead due to proper bounds checking.

---

## 5. SGEMM (1024×1024×1024)

**Classification:** Compute-bound (AI ≈ 170 FLOP/byte for large matrices)

| Version | Intent | Time (ms) | GFLOPS | % cuBLAS |
|---------|--------|-----------|--------|----------|
| naive | Baseline (2 loads/FLOP) | 4.74 | 453 | 7% |
| v2 | Shared memory tiling | 3.34 | 643 | 10% |
| v3 | 1D thread tiling (TM) | 1.94 | 1,109 | 17% |
| v4 | 2D thread tiling (TM×TN) | 1.52 | 1,408 | 22% |
| v5 | Register caching (a_frag, b_frag) | 1.45 | 1,477 | 23% |
| v6 | Vectorized float4 loads | 0.53 | 4,052 | 62% |
| **v7** | **Double buffering** | **0.51** | **4,209** | **65%** |
| cuBLAS | Reference (tensor cores?) | 0.33 | 6,523 | 100% |

**Why each version exists:**
- **naive:** Shows the memory wall—2K memory ops per output element
- **v2:** Shared tiling reduces global loads by TILE_SIZE factor
- **v3:** Thread tiling increases register reuse (each thread does more work)
- **v4:** 2D tiling maximizes register file utilization
- **v5:** Explicit fragment caching—often same as v4 due to compiler
- **v6:** float4 loads reduce instruction count 4x, transpose A in shared for coalesced access
- **v7:** Double buffering overlaps global→shared loads with compute

**Analysis:**  
At **4,209 GFLOPS (52% of theoretical peak, 65% of cuBLAS)**, v7 is a respectable handwritten kernel. The gap to cuBLAS comes from:

1. **No tensor cores:** cuBLAS likely uses WMMA on T4 for ~2x throughput
2. **Suboptimal occupancy:** v6/v7 use ~100 registers/thread, limiting occupancy
3. **No async copy:** `cp.async` can further overlap memory with compute
4. **Tuning:** cuBLAS auto-tunes tile sizes per GPU

**Scalability:**
| Size | v2 GFLOPS | v5 GFLOPS | v7 GFLOPS | cuBLAS |
|------|-----------|-----------|-----------|--------|
| 256³ | 652 | 265 | N/A* | 1,752 |
| 512³ | 777 | 948 | 2,759 | 5,448 |
| 1024³ | 808 | 1,759 | 4,249 | 6,587 |
| 2048³ | 800 | 1,911 | 4,458 | 5,954 |
| 4096³ | 791 | 1,941 | **4,719** | 5,729 |

*v7 requires 128×128 tiles, too large for 256³

**Key insight:** v7 reaches **82% of cuBLAS at 4096³**—the gap narrows with larger matrices as overhead becomes negligible.

**Edge case - non-power-of-two (1000×1000×1000):**
| Kernel | GFLOPS | Time (ms) | vs 1024³ |
|--------|--------|-----------|----------|
| v5 | 1,342 | 1.49 | **78%** efficiency |
| cuBLAS | 5,819 | 0.34 | 88% efficiency |

Non-power-of-two dimensions hurt v5 more than cuBLAS due to partial tile waste.

---

## 6. Summary

| Kernel | Classification | Achieved | Ceiling | Status |
|--------|---------------|----------|---------|--------|
| Elementwise | Memory-bound | 252 GB/s (79%) | ~280 GB/s | ✅ Near optimal |
| Reduction | Memory-bound | 290 GB/s (91%) | ~300 GB/s | ✅ **Optimal** |
| Transpose | Memory-bound | 199 GB/s (62%) | ~250 GB/s | ✅ Good |
| SGEMM | Compute-bound | 4,209 GFLOPS (52%) | ~6,500 GFLOPS | 🔶 Room to grow |

---

## 7. Key Learnings

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

## 8. Next Steps

To push beyond current results:
1. **SGEMM:** Implement WMMA tensor core path (potential 2x)
2. **All kernels:** FP16/BF16 variants (2x throughput, 2x bandwidth)
3. **Reduction:** Multi-pass for >4B elements
4. **Profiling:** Detailed Nsight Compute analysis for register pressure

---

*Baseline checkpoint: ready for `git tag v1.0-baseline`*

*Report generated on Tesla T4 | CUDA 11.x | December 2024*
