# MNIST in CUDA

This project implements a simple 2-layer MLP for MNIST digit classification, progressively optimizing from high-level PyTorch to low-level CUDA implementations.

- **Architecture:** 784 → 1024 → 10 (input → hidden → output)
- **Dataset:** 10,000 MNIST training samples, batch size 32, 10 epochs
- **Activation:** ReLU | **Loss:** Cross-entropy | **Optimizer:** SGD (lr=0.01)

![MNIST CUDA](image.png)
*Image source: [Infatoshi/mnist-cuda](https://github.com/Infatoshi/mnist-cuda)*

## Environment

| Property | Value |
|----------|-------|
| Platform | Ubuntu 22.04.5 LTS (GCP) |
| GPU | NVIDIA Tesla T4 |
| CUDA Capability | 7.5 |
| Global Memory | 14930 MB |
| CUDA Version | 12.1 |

## Prerequisites

```bash
# Check Python version
python3 --version

# Check NumPy version
python3 -c "import numpy; print(f'NumPy: {numpy.__version__}')"

# Check PyTorch and CUDA availability
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# Check CUDA installation
nvcc --version

# Check GPU info
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv
```

## Usage

```bash
cd MNIST-in-CUDA/src

# Download MNIST data (run once)
python3 downloader.py

# v1: PyTorch baseline (with timing instrumentation)
python3 v1.py

# v1_fast: Optimized PyTorch (for fair speed comparison)
python3 v1_fast.py

# v2: NumPy implementation
python3 v2.py

# v3: C implementation
gcc -O2 -o v3 v3.c -lm && ./v3

# v4: Naive CUDA kernels
nvcc -O2 -o v4 v4.cu && ./v4
```

## Version Progression

### v1.py - PyTorch Baseline
- **Framework:** PyTorch with CUDA tensors. "What should happen" (reference)
- **Features:**
  - High-level PyTorch operations (`nn.Linear`, `nn.ReLU`, `nn.CrossEntropyLoss`)
  - Data pre-loaded to GPU (no per-batch transfers)
  - Custom He initialization for weights (matching NumPy/C implementations)
  - MNIST normalization (mean=0.1307, std=0.3081)
  - `torch.set_float32_matmul_precision("high")` for optimized matmul
  - Detailed timing instrumentation per operation
- **Purpose:** Establishes baseline performance and correctness reference

### v1_fast.py - Optimized PyTorch
- **Framework:** PyTorch with speed optimizations
- **Optimizations vs v1.py:**
  - GPU warmup (10 iterations before timing)
  - `torch.compile()` for kernel fusion (PyTorch 2.0+)
  - `cudnn.benchmark = True` for auto-tuning
  - `zero_grad(set_to_none=True)` for faster gradient clearing
  - Removed per-operation timing overhead
- **Purpose:** Fair speed comparison with CUDA implementations
- **Note:** T4 lacks TF32 Tensor Cores (Ampere+), so still slower than RTX 3090 benchmarks

### v2.py - NumPy Implementation
- **Framework:** Pure NumPy (CPU-only). "How does it work mathematically" (understanding)
- **Features:**
  - Manual forward/backward pass implementation
  - Custom gradient computation and weight updates
  - He initialization for weights
  - MNIST normalization (mean=0.1307, std=0.3081)
  - Manual softmax with numerical stability (`x - max(x)`)
  - Manual cross-entropy loss computation
  - Detailed timing instrumentation per operation
- **Purpose:** Demonstrates the underlying math without GPU acceleration

### v3.c - C/CPU Implementation
- **Framework:** Pure C with timing breakdown. "How to implement without Python" (low-level)
- **Features:**
  - Manual memory management (`malloc`/`free`)
  - Naive triple-nested loop matrix multiplication (no BLAS)
  - Three matmul variants: `A @ B`, `A @ B.T`, `A.T @ B`
  - He initialization for weights
  - MNIST normalization (mean=0.1307, std=0.3081)
  - Softmax with numerical stability (`x - max(x)`)
  - Binary file I/O for data loading (`fread`)
  - Granular timing breakdown per operation (`clock_gettime`)
- **Purpose:** Shows CPU performance baseline and prepares for GPU porting
- **Note:** Intentionally slow (~90s) — no BLAS/SIMD optimization; same loop structure will parallelize on GPU

### v4.cu - Naive CUDA Kernels
- **Framework:** CUDA C with custom kernels. "How to parallelize on GPU" (optimization)
- **Features:**
  - Custom naive matrix multiplication kernels (no shared memory tiling)
  - Three matmul kernel variants: `A @ B`, `A @ B.T`, `A.T @ B`
  - Element-wise kernels: ReLU, bias, softmax, weight update
  - GPU memory management (`cudaMalloc`/`cudaFree`)
  - Per-batch `cudaMemcpy` transfers (H2D and D2H)
  - Loss computation still on CPU (requires D2H copy)
  - `cudaDeviceSynchronize()` after every kernel
  - CUDA error checking macro (`CUDA_CHECK`)
  - Granular timing breakdown per operation
- **Purpose:** First GPU implementation — direct port of v3.c with parallelization
- **Note:** ~100x faster than v3 due to parallel execution; still inefficient (no cuBLAS, no shared memory)

### v5.cu - cuBLAS Optimized
- **Framework:** CUDA with cuBLAS library
- **Features:**
  - cuBLAS optimized matrix operations (SGEMM, SAXPY)
  - Persistent memory buffers to reduce allocations
  - Minimal host-device synchronization points
  - Optimized memory access patterns
- **Purpose:** Production-quality implementation with maximum performance

### v6.cu - Fully Optimized
- **Framework:** CUDA with cuBLAS + advanced optimizations
- **Features:**
  - GPU-side loss computation (eliminates 2 memory transfers per batch)
  - CUDA Streams for overlapping data transfer with compute
  - TF32 Tensor Core math on Ampere+ GPUs
  - Fused kernels (bias + ReLU combined)
  - Pinned host memory for faster transfers
  - Double-buffered inputs for pipelining
- **Purpose:** Maximum performance with all applicable GPU optimizations

### v7.cu - Custom Fused GEMM (Educational)
- **Framework:** CUDA with custom kernels + cuBLAS hybrid
- **Features:**
  - Custom fused GEMM + bias + ReLU kernel (forward pass)
  - Tiled shared memory GEMM (32x32 tiles)
  - cuBLAS for backward pass (reliable gradients)
  - All v6 optimizations (streams, pinned memory, GPU-side loss)
- **Purpose:** Demonstrates how kernel fusion works at the CUDA level
- **Note:** Slower than v6 — shows why cuBLAS/CUTLASS exist (writing efficient GEMM is hard)

### v8.cu - Pure FP16 Implementation
- **Framework:** CUDA with cuBLAS GemmEx + native FP16
- **Features:**
  - Pure FP16 throughout: weights, activations, gradients, accumulation
  - `cublasGemmEx` with `CUBLAS_COMPUTE_16F` for native half-precision math
  - No FP32 master weights — true 16-bit training
  - Tensor Core acceleration (FP16 mode)
  - CUDA Streams for overlapped transfer + compute
  - Double-buffered inputs for pipelining
  - Fused bias + ReLU kernels using half intrinsics (`__hadd`, `__hmul`, `__hsub`)
  - GPU-side softmax/cross-entropy/gradient computation
  - Pre-converted FP16 training data for minimal transfer overhead
- **Purpose:** Maximum performance through native half-precision computation
- **Note:** Same speed as v6 TF32 but half the memory footprint; slight precision loss (final loss 0.145 vs 0.142) acceptable for most applications

## Performance Comparison

| Version | Implementation | Time | Speedup vs v3 | Final Loss |
|---------|---------------|------|---------------|------------|
| v1.py   | PyTorch CUDA  | 3.4s  | ~112x        | 0.141      |
| v2.py   | NumPy CPU     | 21.0s | ~18x         | 0.142      |
| v3.c    | C CPU         | 379.7s| 1x (baseline)| 0.139      |
| v4.cu   | Naive CUDA    | 1.7s  | ~223x        | 0.144      |
| v5.cu   | cuBLAS        | 0.4s  | 225x         | 0.142      |
| v6.cu   | TF32 Optimized| 0.3s  | 300x         | 0.142      |
| v7.cu   | Fused GEMM    | 0.6s  | 150x         | 0.143      |
| v8.cu   | Pure FP16     | 0.3s  | 300x         | 0.145      |

## Timing Breakdown Analysis

Each implementation provides detailed timing breakdowns:

### v1 (PyTorch)
Baseline PyTorch implementation with cuBLAS backend (TF32 precision enabled)
- Data loading: 1.9%
- Forward pass: 18.8%
- Loss computation: 9.5%
- Backward pass: 44.5%
- Weight updates: 21.8%

### v2 (NumPy)
Pure NumPy implementation on CPU
- Data loading: 0.1%
- Forward pass: 25.8%
- Loss computation: 2.6%
- Backward pass: 47.0%
- Weight updates: 24.5%

### v3 (C CPU)
Pure C implementation with naive loops (no BLAS)
- Data loading: 0.0%
- Forward pass: 70.9%
- Loss computation: 0.0%
- Backward pass: 27.7%
- Weight updates: 0.8%

### v4 (Naive CUDA)
First GPU implementation with custom kernels
- Data loading: 7.5% (per-batch cudaMemcpy)
- Forward pass: 50.6%
- Loss computation: 0.1%
- Backward pass: 25.7%
- Weight updates: 10.0%

### v5 (cuBLAS Optimized)
Production-level performance
- GPU compute (forward + backward + updates): ~60% of time
- Memory transfers (H2D + D2H): ~30% of time
- Host computation (loss calculation): ~10% of time

### v6 (Fully Optimized)
Maximum GPU utilization
- GPU compute: ~95% of time (loss computation moved to GPU)
- Memory transfers: ~4% of time (overlapped with compute via streams)
- Overhead: ~1% of time

### v7 (Custom Fused GEMM)
Educational implementation
- GPU compute: ~97% of time (custom tiled GEMM slower than cuBLAS)
- Memory transfers: ~1% of time
- Shows the performance gap between naive and optimized GEMM implementations

### v8 (Pure FP16)
Maximum performance implementation
- GPU compute: ~95% of time (native FP16 Tensor Cores)
- Memory transfers: ~3% of time (half the bandwidth of FP32)
- Fastest version due to native half-precision throughout

## Performance Insights

- **Memory Management:** Persistent buffers (v5) vs per-batch allocation (v4) significantly impacts performance
- **Library Optimization:** cuBLAS provides highly optimized GEMM operations that outperform naive kernels
- **CPU-GPU Transfer:** Minimizing host-device synchronization is crucial for GPU performance
- **Numerical Stability:** Proper softmax implementation with max subtraction prevents overflow
- **Hardware Utilization:** v5 achieves the best performance by maximizing GPU compute utilization

## Advanced Optimizations

### Implemented in v6
- **CUDA Streams:** Overlapping computation with data transfer
- **TF32 Tensor Cores:** Hardware acceleration on Ampere+ GPUs
- **Kernel Fusion:** Combined bias + ReLU operations

### Implemented in v7
- **Custom Fused GEMM:** Tiled shared memory GEMM with fused bias + ReLU epilogue
- Educational demonstration of why optimized libraries (cuBLAS/CUTLASS) are valuable

### Implemented in v8
- **Pure FP16:** Native half-precision for weights, activations, gradients, and GEMM accumulation
- **cublasGemmEx:** Flexible GEMM API supporting `CUBLAS_COMPUTE_16F` for true FP16 math
- **Half Intrinsics:** Direct use of `__hadd`, `__hmul`, `__hsub`, `__hgt` for element-wise ops
- **Pre-converted Data:** Training data converted to FP16 once at startup, eliminating per-batch conversion

### Potential Future Improvements
- **Unified Memory:** Simplified memory management
- **CUDA Graphs:** Capture entire training step to reduce launch overhead
- **CUTLASS:** Template-based GEMM with true epilogue fusion (requires larger batch sizes)

## Reference

- [Infatoshi/mnist-cuda](https://github.com/Infatoshi/mnist-cuda)
