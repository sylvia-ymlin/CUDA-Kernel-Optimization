#pragma once // Prevent multiple inclusions of this header file

#include <cstdio> // for printf
#include <cstdlib> // for exit
#include <cublas_v2.h> // cuBLAS header
#include <cuda_runtime.h> // CUDA runtime

/*

Matrix sizes:
MxK * KxN = MxN

*/

// inputs: dims of matrices, alpha, A, B, beta, C
// C = α*(A@B)+β*C
// const and uint: explicitly state that these pointers will not be modified, for code readability and robustness
__global__ __launch_bounds__(1024) // max number of threads per block, help the compiler to optimize the resource allocation
void sgemm_naive(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C) {
    
    const uint col = blockIdx.x * blockDim.x + threadIdx.x;
    const uint row = blockIdx.y * blockDim.y + threadIdx.y;

    // safe check, avoid out-of-bounds access
    if (x < M && y < N) {
        float sum = 0.0;
        // take the row of A and column of B to compute the dot product
        for (int i = 0; i < K; ++i) {
            sum += A[row * K + i] * B[i * N + col];
        }
        // the result is written to C at (row, col)
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

/**
 * 访存比低：每次迭代需要一次 FMA 和 两次全局内存读取，计算访存比 1/2
 * 访存延迟高：全局访存
 * 较低的访存比无法有效隐藏访存延迟
 * 访存量：每个元素的计算需要访问 2K 个元素，全部计算完成需要 2KMN
 * 相同位置元素被重复读取
 */