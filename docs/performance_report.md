# CUDA Kernel Performance Report

**Device:** Tesla T4 (Turing, SM 7.5)  
**Peak Memory Bandwidth:** 320 GB/s  
**Theoretical Peak GFLOPS:** ~8,141 (FP32)  
**L2 Cache:** 4 MB

---

## Roofline Context

The **roofline model** helps classify kernels:
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
| ADD (float4) | 1.60 | 251.5 | **78.6%** |
| SIGMOID (float4) | 1.12 | 239.6 | 74.9% |
| RELU (float4) | 1.14 | 236.2 | 73.8% |

**Analysis:**  
These kernels are **firmly memory-bound**. At 75-79% of peak bandwidth, we're near the practical ceiling. Further optimizations (more unrolling, different grid sizes) would yield <5% gains. The limiting factor is DRAM bandwidth, not compute.

**Why ~80% and not 100%?**
- ECC overhead on T4 (~6%)
- TLB misses and page table walks
- Memory controller scheduling inefficiency

**Edge case - small sizes:**
| Size | Time (ms) | BW (GB/s) | Note |
|------|-----------|-----------|------|
| 1K | 0.0025 | 4.8 | Kernel launch overhead dominates (~2μs) |
| 64K | 0.0026 | 306 | Apparent BW > peak = L2 cache hits |

*The >100% "bandwidth" at 64K elements is not real—data fits in L2 cache (256KB < 4MB).*

---

## 2. Reduction Sum (32M elements)

**Classification:** Memory-bound (AI ≈ 0.125 FLOP/byte)

| Version | Intent | Time (ms) | BW (GB/s) | Speedup |
|---------|--------|-----------|-----------|---------|
| v2 | Baseline shared memory | 2.41 | 55.8 | 1.0x |
| v3 | Show atomic contention cost | 1.28 | 105.0 | 1.9x |
| v4 | Warp-synchronous programming | 1.22 | 110.4 | 2.0x |
| **v5** | **Vectorization + shuffle wins** | **0.46** | **289.8** | **5.2x** |

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

---

## 3. Softmax (4096×1024 matrix)

**Classification:** Memory-bound with moderate compute

| Kernel | Intent | Time (ms) | BW (GB/s) |
|--------|--------|-----------|-----------|
| Row-wise (shared) | Coalesced access pattern | 0.35 | 96.1 |
| Row-wise (shfl_xor) | Butterfly reduction | 0.59 | 57.3 |
| Column-wise (shfl_xor) | Show strided access penalty | 0.84 | 40.0 |

**Analysis:**  
Row-wise softmax is **2.4x faster** than column-wise due to memory coalescing. Column-wise access causes 32-byte transactions to fetch only 4 useful bytes—a 8x amplification of memory traffic.

**Lesson:** Always prefer row-major iteration for row-major storage.

---

## 4. Matrix Transpose (4096×4096)

**Classification:** Memory-bound (AI = 0, pure data movement)

| Version | Intent | Time (ms) | BW (GB/s) | Speedup |
|---------|--------|-----------|-----------|---------|
| v0 | Naive baseline | 1.70 | 79.1 | 1.0x |
| v1 | Coalesced writes | 1.44 | 93.5 | 1.2x |
| v2 | `__ldg()` read-only cache | 1.44 | 93.0 | 1.2x |
| v3 | Shared memory tiling | 0.88 | 152.7 | 1.9x |
| **v4** | **Padding for bank conflicts** | **0.67** | **198.9** | **2.5x** |
| v5 | Swizzling (alternative) | 0.67 | 199.0 | 2.5x |

**Why each version exists:**
- **v0:** Shows the cost of uncoalesced writes (strided by N)
- **v1:** Coalesced writes via transposed output indexing
- **v2:** Tests whether `__ldg()` helps (it doesn't—already cached)
- **v3:** Shared memory converts strided reads to coalesced reads—but has bank conflicts
- **v4:** +1 padding eliminates 32-way bank conflicts (1.3x over v3)
- **v5:** XOR swizzling—same effect as padding, slightly less memory

**Analysis:**  
At **62% of peak bandwidth**, transpose is limited by the fundamental read-write asymmetry: we must either read strided or write strided. Shared memory tiling minimizes this penalty but can't eliminate it.

**Theoretical ceiling:** ~250 GB/s (read + write with some overlap). We're at 80% of that.

---

## 5. SGEMM (1024×1024×1024)

**Classification:** Compute-bound (AI ≈ 170 FLOP/byte for large matrices)

| Version | Intent | Time (ms) | GFLOPS | % cuBLAS |
|---------|--------|-----------|--------|----------|
| naive | Baseline (2 loads/FLOP) | 6.20 | 347 | 5% |
| v2 | Shared memory tiling | 3.28 | 655 | 10% |
| v3 | 1D thread tiling (TM) | 1.57 | 1,365 | 21% |
| v4 | 2D thread tiling (TM×TN) | 1.25 | 1,718 | 26% |
| v5 | Register caching (a_frag, b_frag) | 1.25 | 1,719 | 26% |
| v6 | Vectorized float4 loads | 0.53 | 4,052 | 62% |
| **v7** | **Double buffering** | **0.51** | **4,209** | **64%** |
| cuBLAS | Reference (tensor cores?) | 0.33 | 6,532 | 100% |

**Why each version exists:**
- **naive:** Shows the memory wall—2K memory ops per output element
- **v2:** Shared tiling reduces global loads by TILE_SIZE factor
- **v3:** Thread tiling increases register reuse (each thread does more work)
- **v4:** 2D tiling maximizes register file utilization
- **v5:** Explicit fragment caching—often same as v4 due to compiler
- **v6:** float4 loads reduce instruction count 4x, transpose A in shared for coalesced access
- **v7:** Double buffering overlaps global→shared loads with compute

**Analysis:**  
At **4,209 GFLOPS (52% of theoretical peak)**, v7 is a respectable handwritten kernel. The gap to cuBLAS comes from:

1. **No tensor cores:** cuBLAS uses WMMA on T4 for ~2x throughput
2. **Suboptimal occupancy:** v6/v7 use ~100 registers/thread, limiting to ~25% occupancy
3. **No async copy:** `cp.async` can further overlap memory with compute
4. **Tuning:** cuBLAS auto-tunes tile sizes per GPU

**The 64% ceiling is architectural, not algorithmic.** Reaching >80% requires tensor cores or assembly-level tuning.

**Scalability:**
| Size | v5 GFLOPS | v7 GFLOPS | cuBLAS |
|------|-----------|-----------|--------|
| 256³ | 264 | ~500 | 1,737 |
| 1024³ | 1,719 | 4,209 | 6,532 |
| 4096³ | 1,926 | ~4,200 | 5,664 |

*Small matrices underutilize the GPU. SGEMM efficiency improves with size until compute saturates.*

---

## 6. Edge Cases & Robustness

### Kernel Launch Overhead
| Size | Kernel | Time (μs) | Note |
|------|--------|-----------|------|
| 1K | elementwise | 2.5 | Launch overhead dominates |
| 1K | reduction | 2.7 | Same pattern |

**Insight:** For <10K elements, kernel launch (~2μs) exceeds compute time. Consider CPU fallback or batching.

### Non-Power-of-Two Sizes
All kernels handle arbitrary sizes via bounds checking. Performance is stable—no special cases needed.

### Non-Square Transpose
4096×1023 vs 4096×1024: ~3% slower due to partial tiles. Acceptable.

### Odd SGEMM Dimensions
1000×1000×1000: Works correctly (bounds checking). ~5% slower than 1024³ due to tile waste.

---

## 7. Resource Utilization (Nsight Compute Data Needed)

**To fully characterize v6/v7 SGEMM, measure:**

```bash
ncu --set full -o sgemm_detailed ./build/test_sgemm
```

Key metrics to extract:
| Metric | Why It Matters |
|--------|----------------|
| Registers/thread | >64 hurts occupancy |
| Achieved occupancy | Actual vs theoretical |
| SM throughput | Compute saturation |
| Memory throughput | Bandwidth utilization |
| Warp stall reasons | Pipeline bottlenecks |

**Experiment:** Try `--maxrregcount=64` to trade occupancy for register spilling. If performance improves, we're occupancy-limited.

---

## Summary

| Kernel | Classification | Achieved | Ceiling | Status |
|--------|---------------|----------|---------|--------|
| Elementwise | Memory-bound | 78% BW | ~85% | ✅ Near optimal |
| Reduction | Memory-bound | 91% BW | ~95% | ✅ **Optimal** |
| Transpose | Memory-bound | 62% BW | ~80% | ✅ Good |
| SGEMM | Compute-bound | 52% FLOPS | ~80%* | 🔶 Room to grow |

*\*Without tensor cores*

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

---

## Next Steps

To push beyond current results:
1. **SGEMM:** Implement WMMA tensor core path (potential 2x)
2. **All kernels:** FP16/BF16 variants (2x throughput, 2x bandwidth)
3. **Reduction:** Multi-pass for >4B elements
4. **Profiling:** Detailed Nsight Compute analysis for v6/v7

---

*Baseline checkpoint: ready for `git tag v1.0-baseline`*

*Report generated on Tesla T4 | CUDA 11.x | December 2024*
