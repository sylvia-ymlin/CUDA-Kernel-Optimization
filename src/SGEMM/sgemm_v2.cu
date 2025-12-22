// use shared memory, tile the matrix
// faster access, and reduce the global memory access depending on the Tile Size
// here we tile the matrix based on the block size

#include<cuda_runtime.h>
#include<stdio.h>
#include<stdlib.h>

template<const int TILE_SIZE>
__global__ void sgemm_v2(int M, int N, int K, 
    float alpha, float*A, float*B, float beta, float*C){

    const int BM = BLOCK_SIZE;
    const int BN = BLOCK_SIZE;
    const int BK = BLOCK_SIZE;

    // block id and thread id
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // allocate shared memory for the tile
    __shared__ float s_A[BM][BK];
    __shared__ float s_B[BK][BN];

    // move the pointer to the start of the tile
    A = &A[by * BM * K];
    B = &B[bx * BN * K];
    C = &C[by * BM * N + bx * BN * N];

    float temp = 0.0;
    // 滑动窗口
    for(int k = 0; k < K, k += BK){
        As[ty * BK + tx] = A[ty * BK + tx];
        Bs[ty * BK + tx] = B[ty * BK + tx];
        __syncthreads();

        // move to next block
        A += BK;
        B += BK * N;
        // calculate the dot product
        for(int i = 0; i < BK; i++){
            tmp += As[ty * BK + i] * Bs[i * BN + tx];
        }
        __syncthreads(); // make sure all threads finish the calculation
    }

    // 遍历完所有窗口，结果写回全局内存
    C[ty * N + tx] = alpha * temp + beta * C[ty * N + tx];
}

// template for different parameters, block size
template __global__ void sgemm_v2<16>(int M, int N, int K, 
    float alpha, float*A, float*B, float beta, float*C);

template __global__ void sgemm_v2<32>(int M, int N, int K, 
    float alpha, float*A, float*B, float beta, float*C);

template __global__ void sgemm_v2<64>(int M, int N, int K, 
    float alpha, float*A, float*B, float beta, float*C);

/**
 * 读取全局内存的次数会随 block 的尺寸成倍缩小
 * 但是每个block 能分配的 shared memory 空间也会成倍缩小
 * 需要权衡 block 尺寸和 shared memory 空间
 */