// 2D thread tile and registers

#include<cuda_runtime.h>
#include<stdio.h>
#include<stdlib.h>

template<const int BM,
        const int BN,
        const int BK,
        const int TM,
        const int TN>
__global__ void sgemm_v5(int M, int N, int K,
    float alpha, float*A, float*B, float beta, float*C){
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    int block_row_thread = BN / TN;
    int block_col_thread = BM / TM;
    int thread_num = block_row_thread * block_col_thread;
    
    int tx = (threadIdx.x % block_col_thread) * TN;
    int ty = (threadIdx.x / block_col_thread) * TM;

    // shared memory for the tile
    __shared__ float As[BM*BK];
    __shared__ float Bs[BK*BN];

    // move the pointer to the start of the tile
    A = &A[by * BM * K];
    B = &B[bx * BN];
    C = &C[by * BM * N + bx * BN];

    int a_tile_row = threadIdx.x / BK;
    int a_tile_col = threadIdx.x % BK;
    int a_tile_stride = threadIdx.x / BK;

    int b_tile_row = threadIdx.x / BN;
    int b_tile_col = threadIdx.x % BN;
    int b_tile_stride = threadIdx.x / BN;

    float tmp[TM][TN] = {0.};
    float a_frag[TM] = {0.};
    float b_frag[TN] = {0.};

    #pragma unroll
    for(int k = 0; k < K; k += BK){
        #pragma unroll
        for(int i = 0; i < BM; i += a_tile_stride){
            As[(a_tile_row + i) * BK + a_tile_col] = A[(a_tile_row + i) * K + a_tile_col];
        }
        #pragma unroll
        for(int i = 0; i < BN; i += b_tile_stride){
            Bs[(b_tile_row + i) * BN + b_tile_col] = B[(b_tile_row + i) * N + b_tile_col];
        }
        __syncthreads();

        A += BK;
        B += BK * N;

        // cache to registers
        #pragma unroll
        for(int i = 0; i < BK; i++){
            #pragma unroll
            for(int j = 0; k < TM; j++){
                a_frag[j] = As[(ty + j) * BK + i];
            }
            #pragma unroll
            for(int j = 0; k < TN; j++){
                b_frag[j] = Bs[i * BN + tx + j];
            }
            // calculate for the 2D tile
            for(int i = 0; i < TM; i++){
                for(int j = 0; j < TN; j++){
                    tmp[i][j] += a_frag[i] * b_frag[j];
                }
            }
            __syncthreads();
        }
    }

    // write the result to the global memory
    #pragma unroll
    for(int j = 0; j < TM; j++){
        for(int l = 0; l < TN; l++){
            C[(ty + j) * N + tx + l] = alpha * tmp[j][l] + beta * C[(ty + j) * N + tx + l];
        }
    }
}

template __global__ void sgemm_v5<128, 128, 8, 8, 8>(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C);