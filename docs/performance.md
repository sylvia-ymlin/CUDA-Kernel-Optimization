# Element-wise Kernel Profiling

## Device Information

| Property                   | Value                       |
|---------------------------- |---------------------------- |
| Device                      | NVIDIA Tesla T4             |
| Compute Capability           | 7.5                         |
| Streaming Multiprocessors    | 40                          |
| Max Threads per SM           | 1024                        |
| Max Threads per Block        | 1024                        |
| Shared Memory per Block      | 48 KB                       |
| Global Memory                | 14.58 GB                    |
| Memory Clock Rate            | 5.00 GHz                    |
| Memory Bus Width             | 256-bit                     |
| Peak Memory Bandwidth        | 320.06 GB/s                 |

## Correctness Verification

| Kernel            | Result |
|------------------ |--------|
| `elementwise_add` | PASS   |
| `sigmoid`         | PASS   |
| `relu`            | PASS   |

All element-wise kernels produced correct results for the tested input datasets.

## Performance Results (32M elements, GPU float4)

| Kernel           | Time / Iter | Bandwidth (GB/s) | GFLOPS | Occupancy |
|----------------- |------------|----------------|--------|-----------|
| `elementwise_add` | 1.597 ms   | 252.11         | 21.01  | 100%      |
| `sigmoid`         | 1.134 ms   | 236.66         | 118.33 | 100%      |
| `relu`            | 1.136 ms   | 236.25         | 29.53  | 100%      |

**Grid Configuration:** 32,768 blocks × 256 threads = 8,388,608 threads  
**Elements Processed:** 33,554,432  

## Scalability Analysis (`elementwise_add`)

| Input Size | Time (ms) | Bandwidth (GB/s) | Efficiency (%) |
|------------|-----------|----------------|----------------|
| 1,024      | 0.0043    | 2.84           | 0.9            |
| 4,096      | 0.0044    | 11.15          | 3.5            |
| 16,384     | 0.0044    | 44.22          | 13.8           |
| 65,536     | 0.0042    | 186.12         | 58.1           |
| 262,144    | 0.0041    | 763.73         | 238.6          |
| 1,048,576  | 0.0551    | 228.46         | 71.4           |
| 4,194,304  | 0.2084    | 241.56         | 75.5           |
| 16,777,216 | 0.8280    | 243.15         | 76.0           |
| 33,554,432 | 1.6570    | 243.01         | 75.9           |

> Observation: Large input sizes saturate the memory bandwidth of the Tesla T4 (~76% of peak). Small sizes suffer from kernel launch overhead and lower efficiency.

## CPU Baseline (1M elements)

| Kernel           | Time / Iter | Bandwidth (GB/s) |
|----------------- |------------|----------------|
| `elementwise_add` | 3.611 ms   | 3.48           |
| `sigmoid`         | 9.381 ms   | 0.89           |
| `relu`            | 3.576 ms   | 2.35           |

> GPU acceleration achieves 60–130× speedup over single-threaded CPU execution.

<!-- ## Profiling Recommendations

- **Nsight Systems (`nsys`)**: Full kernel timeline, memory traffic, and occupancy analysis. Generated report: `elementwise_trace.nsys-rep`.  
- **Nsight Compute (`ncu`)**: Instruction-level metrics (requires special permissions on cloud VMs).  
- Vary **block size** and **grid dimensions** to explore further performance improvements.  
- Record **GFLOPS and memory bandwidth** for different input sizes to document scalability trends. -->