# GPU Computing

A hands-on exploration of GPU programming, from low-level CUDA kernel optimization to end-to-end deep learning training, with PyTorch extension and Triton examples.

**Highlights:**
- **Tested on:** Ubuntu 22.04 (GCP), NVIDIA Tesla T4 (SM 7.5), CUDA 12.1
- **82% of cuBLAS** performance for SGEMM through progressive optimization (at 4096³)
- **91% of peak bandwidth** for memory-bound kernels (reduction)
- **1012× speedup** for MNIST training (v8 FP16 vs C baseline)

## Repository Structure

```
cuda-kernels-from-scratch/
├── kernels/                 # Core CUDA kernels
├── mnist-cuda/              # End-to-end MLP training
├── pytorch-extension/       # Custom PyTorch CUDA extension
├── triton/                  # Triton kernel examples
├── profiling/               # Nsight Systems/Compute guides
└── lectures/                # GPU programming lecture notes
```

## Projects

| Project | Description | Highlights |
|---------|-------------|------------|
| [kernels/](kernels/) | CUDA kernels | 91% peak BW, 82% cuBLAS |
| [mnist-cuda/](mnist-cuda/) | MLP training | PyTorch → C → CUDA → FP16, 1012× speedup |
| [pytorch-extension/](pytorch-extension/) | PyTorch extension | Polynomial activation kernel |
| [triton/](triton/) | Triton kernels | vec_add, softmax |
| [profiling/](profiling/) | Profiling | Nsight Systems/Compute |
| [lectures/](lectures/) | Lecture notes | - |

## Build & Run

### kernels/
```bash
cd kernels && mkdir build && cd build
cmake .. && make
./test_elementwise
./test_sgemm
```

### mnist-cuda/
```bash
cd mnist-cuda
python3 src/downloader.py   # Download MNIST data
mkdir build && cd build && cmake .. && make
./bin/v8                    # FP16 + Tensor Cores
```

### pytorch-extension/
```bash
cd pytorch-extension
pip install -e .
python polynomial_activation.py
```

### triton/
```bash
cd triton
pip install -r requirements.txt
python vec_add.py
```

## References

- S. Böhm, [CUDA Matmul Optimization](https://siboehm.com/articles/22/CUDA-MMM)
- Lei Mao, [CUDA Matrix Multiplication](https://leimao.github.io/article/CUDA-Matrix-Multiplication-Optimization/)
- Kirk & Hwu, *Programming Massively Parallel Processors* (Morgan Kaufmann, 2016)
- [NVIDIA Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html)
- [MNIST-CUDA tutorial](https://github.com/Infatoshi/mnist-cuda) — original inspiration for this project

