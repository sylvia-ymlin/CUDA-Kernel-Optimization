// thread tile and registers
// 一个线程计算 block 中多个元素
// 定义 tmp 缓存用于计算的元素

#include<cuda_runtime.h>
#include<stdio.h>
#include<stdlib.h>

template<const int BM,
        const int BN,
        const int BK,
        const int TM>
__global__ void sgemm_v3(int M, int N, int K,
    float alpha, float*A, float*B, float beta, float*C){
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int thread_num = BM * BN / TM; // TM defines the number of elements a thread computes
    
    int tx = threadIdx.x % BN;
    int ty = threadIdx.x / BN * TM;

    // allocate shared memory for the tile
    __shared__ float s_A[BM][BK];
    __shared__ float s_B[BK][BN];

    // move the pointer to the start of the tile
    A = &A[by * BM * K];
    B = &B[bx * BN * K];
    C = &C[by * BM * N + bx * BN * N];

    int a_tile_row = threadIdx.x / BK;
    int b_tile_col = threadIdx.x % BK;
    int a_tile_stride = threadIdx.x / BK;

    int b_tile_row = threadIdx.x / BN;
    int b_tile_col = threadIdx.x % BN;
    int b_tile_stride = threadIdx.x / BN;

    // allocate the registers for the tile
    // each thread is responsible for TM elements, one for case
    float tmp[TM +1] = {0.}
    // load data to shared memory
    #pragma unroll
    for(int k = 0; k < K; k += BK){ // loop over columns or rows, since one element is still calculated by the same thread
        #pragma unroll
        for(int i = 0; i < BM; i += a_tile_stride){
            As[(a_tile_row + i) * BK + a_tile_col] = A[(a_tile_row + i) * K + a_tile_col];
        }
        #pragma unroll
        for(int i = 0; i < BN; i += b_tile_stride){
            Bs[(b_tile_row + i) * BN + b_tile_col] = B[(b_tile_row + i) * N + b_tile_col];
        }
        __syncthreads();

        // move the pointer to the next tile
        A += BK;
        B += BK * N;

        // case to register
        #pragma unroll
        for(int i = 0; i < BK; i++){
            tmp[TM] = Bs[tx + i * BN];
            #pragma unroll
            for(int i = 0l j < TM; j++){
                tmp[j] += As[(ty + j) * BK + i] * tmp[TM];
            }
        }
        __syncthreads(); // block level synchronization
    }


    # write the TM elements to the global memory
    #pragma unroll
    for(int j = 0; j < TM; j++){
        C[(ty + j) * N + tx] = alpha * tmp[i] + beta * C[(ty + j) * N + tx];
    }
}

template __global__ void sgemm_v3<64, 64, 8, 8>(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C);