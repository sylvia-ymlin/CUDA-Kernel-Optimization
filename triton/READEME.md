# Triton

Triton is an abstraction on top of CUDA.

- CUDA -> scalar program + blocked threads
- Triton -> blocked program + scalar threads

cuda is a scalar program with blocked threads because we write a kernel to operate at the level of threads (scalars), whereas triton is abstracted up to thread blocks (compiler takes care of thread level operations for us).

Besides, cuda has blocked threads in the context of "worrying" about inter-thread at the level of blocks, whereas triton has scalar threads in the context of "not worrying" about inter-thread at the level of threads (compiler also takes care of this).

Why does this actually mean on an intuitive level?

- higher level of abstract for deep learning operations (activations functions, convolutions, matmul, etc)
- the compiler will take care of boilerplate complexities of load and store instructions, tiling, SRAM caching, etc
- python programmers can write kernels comparable to cuBLAS, cuDNN (which is difficult for most CUDA/GPU programmers)

