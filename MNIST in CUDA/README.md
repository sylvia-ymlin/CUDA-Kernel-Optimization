# MNIST in CUDA

This project implements a simple 2-layer MLP for MNIST digit classification, progressively optimizing from high-level PyTorch to low-level CUDA implementations.

- **Architecture:** 784 → 1024 → 10 (input → hidden → output)
- **Dataset:** 10,000 MNIST training samples, batch size 32, 10 epochs
- **Activation:** ReLU | **Loss:** Cross-entropy | **Optimizer:** SGD (lr=0.01)

## Prerequisites

```bash
# Check CUDA installation
nvcc --version

# Check your GPU's compute capability (5.0 or greater required)
nvidia-smi --query-gpu=compute_cap --format=csv,noheader
```

## Reference

- [Infatoshi/mnist-cuda](https://github.com/Infatoshi/mnist-cuda)