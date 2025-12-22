# Nsight Systems Analysis

## Commands Used

```bash
# Generate timeline trace
nsys profile -o sgemm_trace ./build/test_sgemm

# With CUDA API tracing
nsys profile --trace=cuda,nvtx -o detailed_trace ./build/test_sgemm

# Open in GUI
nsys-ui sgemm_trace.nsys-rep
```

---

## Timeline Overview

### Test Execution Summary
| Phase | Duration | Notes |
|-------|----------|-------|
| Total Execution | | |
| Kernel Time | | |
| Memory Transfers | | |
| API Overhead | | |

---

## Kernel Launch Analysis

### SGEMM Kernels
| Kernel | Grid | Block | Duration (avg) | Occupancy |
|--------|------|-------|----------------|-----------|
| sgemm_naive | 32×32 | 32×32 | | |
| sgemm_v2 | 32×32 | 32×32 | | |
| sgemm_v3 | 16×16 | 512 | | |
| sgemm_v4 | 16×16 | 64 | | |
| sgemm_v5 | 16×16 | 64 | | |
| sgemm_v6 | 8×8 | 256 | | |
| sgemm_v7 | 8×8 | 256 | | |

### Reduction Kernels
| Kernel | Grid | Block | Duration (avg) |
|--------|------|-------|----------------|
| sum_v2 | 131072 | 256 | | |
| sum_v3 | 131072 | 256 | | |
| sum_v4 | 131072 | 256 | | |
| sum_v5 | 32768 | 256 | | |

### Transpose Kernels
| Kernel | Grid | Block | Duration (avg) |
|--------|------|-------|----------------|
| transpose_v0 | 16384 | 32×32 | | |
| transpose_v4 | 16384 | 32×32 | | |
| transpose_v5 | 16384 | 32×32 | | |

---

## Memory Transfer Analysis

### Host-to-Device
| Transfer | Size | Duration | Bandwidth |
|----------|------|----------|-----------|
| Matrix A | | | |
| Matrix B | | | |

### Device-to-Host
| Transfer | Size | Duration | Bandwidth |
|----------|------|----------|-----------|
| Matrix C | | | |

---

## GPU Utilization

### SM Activity
- Peak SM utilization: 
- Average SM utilization: 
- Idle gaps observed: [ ] Yes [ ] No

### Memory Controller Activity
- Peak DRAM utilization: 
- Average DRAM utilization: 

---

## CUDA API Overhead

| API Call | Count | Total Time | Avg Time |
|----------|-------|------------|----------|
| cudaMalloc | | | |
| cudaMemcpy (H2D) | | | |
| cudaMemcpy (D2H) | | | |
| cudaLaunchKernel | | | |
| cudaDeviceSynchronize | | | |

---

## Observations

### Timeline Patterns
1. 
2. 
3. 

### Potential Optimizations
1. 
2. 

---

## Screenshots

<!-- Add timeline screenshots here -->

### Full Timeline
<!-- ![Full Timeline](./images/timeline_full.png) -->

### Kernel Zoom
<!-- ![Kernel Zoom](./images/kernel_zoom.png) -->

---

## Action Items

- [ ] Capture full timeline with all tests
- [ ] Identify any gaps between kernel launches
- [ ] Verify warmup iterations are excluded from measurements
- [ ] Check for unexpected synchronization points

