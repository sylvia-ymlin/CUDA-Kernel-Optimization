# Nsight Compute Analysis

## Commands Used

```bash
# Generate profile
ncu --set full --target-processes all -o sgemm_profile ./build/test_sgemm

# Open in GUI
ncu-ui sgemm_profile.ncu-rep
```

---

## SGEMM v7 Analysis

### Summary
| Metric | Value | Notes |
|--------|-------|-------|
| Registers/Thread | | |
| Achieved Occupancy | | |
| SM Throughput | | |
| DRAM Throughput | | |
| Compute Throughput | | |

### Memory Analysis
| Metric | Value | Interpretation |
|--------|-------|----------------|
| Global Load Transactions | | |
| Global Store Transactions | | |
| L2 Hit Rate | | |
| Shared Memory Bank Conflicts | | |

### Compute Analysis
| Metric | Value | Interpretation |
|--------|-------|----------------|
| FMA Pipe Utilization | | |
| Warp Stall Reasons | | |
| IPC | | |

### Bottleneck Identification
- [ ] Memory-bound
- [ ] Compute-bound
- [ ] Latency-bound
- [ ] Occupancy-limited

### Observations
1. 
2. 
3. 

---

## SGEMM v6 vs v7 Comparison

| Metric | v6 | v7 | Delta |
|--------|----|----|-------|
| GFLOPS | 4,052 | 4,209 | +4% |
| Registers/Thread | | | |
| Achieved Occupancy | | | |
| Memory Throughput | | | |

### Key Differences
1. 
2. 

---

## Reduction v5 Analysis

### Summary
| Metric | Value | Notes |
|--------|-------|-------|
| Achieved Bandwidth | 290 GB/s | 91% of peak |
| Registers/Thread | | |
| Achieved Occupancy | | |

### Memory Coalescing
| Access Type | Sectors/Request | Optimal? |
|-------------|-----------------|----------|
| Global Load (float4) | | ≤4 is optimal |
| Shared Store | | |
| Shared Load | | |

### Observations
1. 
2. 

---

## Transpose v4 Analysis

### Summary
| Metric | Value | Notes |
|--------|-------|-------|
| Achieved Bandwidth | 199 GB/s | 62% of peak |
| Bank Conflicts | | Should be 0 with padding |

### Shared Memory Analysis
| Metric | Value | Notes |
|--------|-------|-------|
| Shared Load Bank Conflicts | | |
| Shared Store Bank Conflicts | | |
| Shared Memory Throughput | | |

### Observations
1. 
2. 

---

## Screenshots

<!-- Add annotated screenshots here -->

### SGEMM v7 Roofline
<!-- ![SGEMM v7 Roofline](./images/sgemm_v7_roofline.png) -->

### Memory Chart
<!-- ![Memory Analysis](./images/memory_chart.png) -->

---

## Action Items

- [ ] Profile SGEMM v7 with `--set full`
- [ ] Identify register pressure in v6/v7
- [ ] Verify bank conflict elimination in transpose v4
- [ ] Compare achieved vs theoretical occupancy

