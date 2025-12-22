# Key Metrics for CUDA Kernel Analysis

## Memory-Bound Kernels (Elementwise, Reduction, Transpose)

### Primary Metrics
| Metric | Target | Why It Matters |
|--------|--------|----------------|
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | >75% | DRAM bandwidth utilization |
| `l2_cache_throughput.avg.pct_of_peak_sustained_elapsed` | Monitor | L2 hit rate affects effective BW |
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | Low for mem-bound | Should NOT be the bottleneck |

### Memory Access Patterns
| Metric | Target | Why It Matters |
|--------|--------|----------------|
| `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` | Minimize | Global load transactions |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum` | Minimize | Global store transactions |
| `l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio` | ≤4 | Coalescing efficiency (4 = perfect) |
| `l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio` | ≤4 | Coalescing efficiency (4 = perfect) |

### Shared Memory (Transpose, Reduction)
| Metric | Target | Why It Matters |
|--------|--------|----------------|
| `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` | 0 | Bank conflicts kill shared mem BW |
| `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum` | 0 | Bank conflicts kill shared mem BW |

---

## Compute-Bound Kernels (SGEMM)

### Primary Metrics
| Metric | Target | Why It Matters |
|--------|--------|----------------|
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | >70% | SM utilization (compute saturation) |
| `sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_elapsed` | High | FMA pipe utilization |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | Balance | Achieved occupancy |

### Register Pressure
| Metric | Target | Why It Matters |
|--------|--------|----------------|
| `launch__registers_per_thread` | <128 | High regs → low occupancy |
| `sm__warps_active.avg.per_cycle_active` | Monitor | Active warps per cycle |
| `l1tex__data_pipe_lsu_wavefronts_mem_shared_cmd_read.sum` | Monitor | Shared mem pressure |

### Latency Hiding
| Metric | Interpretation |
|--------|----------------|
| `smsp__warp_issue_stalled_long_scoreboard_per_warp_active.ratio` | Memory latency not hidden |
| `smsp__warp_issue_stalled_wait_per_warp_active.ratio` | Dependency stalls |
| `smsp__warp_issue_stalled_mio_throttle_per_warp_active.ratio` | Memory instruction queue full |

---

## Universal Metrics

### Occupancy
| Metric | Notes |
|--------|-------|
| `sm__warps_active.avg.pct_of_peak_sustained_active` | Achieved occupancy |
| `launch__occupancy_limit_blocks` | What limits occupancy (regs/shared/blocks) |
| `launch__occupancy_limit_registers` | Register-limited? |
| `launch__occupancy_limit_shared_mem` | Shared memory-limited? |

### Instruction Mix
| Metric | Notes |
|--------|-------|
| `smsp__inst_executed.sum` | Total instructions |
| `smsp__inst_executed_op_ffma.sum` | FMA instructions (compute) |
| `smsp__inst_executed_op_global_ld.sum` | Global loads |
| `smsp__inst_executed_op_shared_ld.sum` | Shared loads |

---

## Quick Reference Commands

```bash
# Full analysis (all metrics)
ncu --set full -o profile ./build/test_sgemm

# Memory-focused analysis
ncu --set memory -o mem_profile ./build/test_sgemm

# Roofline analysis
ncu --set roofline -o roofline ./build/test_sgemm

# Specific metrics only
ncu --metrics \
  dram__throughput.avg.pct_of_peak_sustained_elapsed,\
  sm__throughput.avg.pct_of_peak_sustained_elapsed,\
  launch__registers_per_thread \
  ./build/test_sgemm

# Compare two kernels
ncu --set full --kernel-name "sgemm_v5" --kernel-name "sgemm_v7" \
  -o comparison ./build/test_sgemm
```

---

## Interpretation Cheat Sheet

| Observation | Likely Cause | Action |
|-------------|--------------|--------|
| Low DRAM throughput, low SM throughput | Kernel launch overhead / small problem | Increase problem size |
| High DRAM throughput, low SM throughput | Memory-bound (expected) | Optimize memory access |
| Low DRAM throughput, high SM throughput | Compute-bound (expected for SGEMM) | Optimize compute |
| High bank conflicts | Poor shared memory indexing | Add padding or swizzle |
| High register count, low occupancy | Over-aggressive register tiling | Use `--maxrregcount` |
| High `stalled_long_scoreboard` | Memory latency not hidden | Add prefetching / double buffer |

