// 向量化访存

#include<cuda_runtime.h>
#include<stdio.h>
#include<stdlib.h>

template<const int BM,
        const int BN,
        const int BK,
        const int TM,
        const int TN>
__global__ void sgemm_v6(int M, int N, int K,
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

    // 每个线程一次搬运 4 个浮点数，搬运完 As 需要 ldg_a_num, ldg_b_num 次
    const int ldg_a_num = BK * BM / thread_num / 4;
    const int ldg_b_num = BK * BN / thread_num / 4;

    int a_tile_row = threadIdx.x / (BK /4);
    int a_tile_col = threadIdx.x % (BK /4) * 4;
    int a_tile_stride = BM / ldg_a_num;

    int b_tile_row = threadIdx.x / (BN /4);
    int b_tile_col = threadIdx.x % (BN /4) * 4;
    int b_tile_stride = BN / ldg_b_num;

    float accum[TM][TN] = {0.};

    float ldg_a_frag[4 * ldg_a_num] = {0.}; // 元素缓存，用于转置 As

    float a_frag[TM];
    float b_frag[TN];

    // MOVE the pointer to the start of the tile
    A = &A[by * BM * K];
    B = &B[bx * BN];
    C = &C[by * BM * N + bx * BN];

    #pragma unroll
    for(int k = 0; k < K; k += BK){
        #pragma unroll
        for(int i = 0; i < BM; i += a_tile_stride){
            int ldg_index = i / a_tile_stride * 4; 
            FETCH_FLOAT4(ldg_a_reg[ldg_index]) = 
            FETCH_FLOAT4(A[OFFSET(a_tile_row + i, a_tile_col, K)]);
            // As转置存，其中ldg_a_reg做中间缓存，目的是读取时可以按FLOAT4读取
            As[OFFSET(a_tile_col, i + a_tile_row, BM)] = ldg_a_reg[ldg_index];
            As[OFFSET(a_tile_col + 1, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 1];
            As[OFFSET(a_tile_col + 2, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 2];
            As[OFFSET(a_tile_col + 3, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 3];
        }
    
        // 搬运 Bs 到 shared memory, 不需要转置，直接搬运
        #pragma unroll
        for(int i = 0; i < BK; i += b_tile_stride){
            FETCH_FLOAT4(ldg_b_reg[ldg_index]) = 
            FETCH_FLOAT4(B[OFFSET(i + b_tile_row, b_tile_col, N)]);
        }
        __syncthreads();

        A += BK;
        B += BK * N;

        #pragma unroll
        for(int i = 0; i < BK; i){
            #pragma unroll
            for(int m = 0; m < TM; m++){
                FETCH_FLOAT4(a_frag[m]) = As[OFFSET(i, ty + m BM)];
            }
            #pragma unroll
            for(int n = 0; n < TN; n++){
                FETCH_FLOAT4(b_frag[n]) = Bs[OFFSET(i, tx + n, BN)];
            }
            __syncthreads();

            #pragma unroll
            for(int m = 0; m < TM; m++){
                #pragma unroll
                for(int n = 0; n < TN; n++){
                    accum[m][n] += a_frag[m] * b_frag[n];
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for(int m = 0; m < TM; m++){
        #pragma unroll
        for(int n = 0; n < TN; n++){
            float ctmp = FETCH_FLOAT4(C[OFFSET(ty + m, tx + n, N)]);
            // unroll by hand
            ctmp.x = alpha * accum[m][n] + beta * ctmp.x;
            ctmp.y = alpha * accum[m][n] + beta * ctmp.y;
            ctmp.z = alpha * accum[m][n] + beta * ctmp.z;
            ctmp.w = alpha * accum[m][n] + beta * ctmp.w;
            FETCH_FLOAT4(C[OFFSET(ty + m, tx + n, N)]) = ctmp;
        }
    }
}

// a template with recommended parameters
template __global__ void sgemm_v6<128, 128, 8, 8, 8>(int M, int N, int K,
    float alpha, float*A, float*B, float beta, float*C);