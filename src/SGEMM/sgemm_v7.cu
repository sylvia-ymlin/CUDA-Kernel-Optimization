# double buffering
# prefetching the data to shared memory

#include<cuda_runtime.h>
#include<stdio.h>
#include<stdlib.h>

template<const int BM,
        const int BN,
        const int BK,
        const int TM,
        const int TN>
__global__ void sgemm_v7(int M, int N, int K,
    float alpha, float*A, float*B, float beta, float*C){

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int block_row_thread = BN / TN;
    int block_col_thread = BM / TM;
    int thread_num = block_row_thread * block_col_thread;
    
    // index of the top-left corner of the tile
    int tx = (threadIdx.x % block_col_thread) * TN;
    int ty = (threadIdx.x / block_col_thread) * TM;

    __shared__ float As[2][BM][BK];
    __shared__ float Bs[2][BK][BN];
    
    const int ldg_a_num = BK * BM / thread_num / 4;
    const int ldg_b_num = BK * BN / thread_num / 4;

    int a_tile_row = threadIdx.x / (BK /4);
    int a_tile_col = threadIdx.x % (BK /4) * 4;
    int a_tile_stride = BM / ldg_a_num;

    int b_tile_row = threadIdx.x / (BN /4);
    int b_tile_col = threadIdx.x % (BN /4) * 4;
    int b_tile_stride = BN / ldg_b_num;

    float accum[TM][TN] = {0.};

    float ldg_a_frag[4 * ldg_a_num] = {0.};
    float ldg_b_frag[4 * ldg_b_num] = {0.};

    float a_frag[2][TM];
    float b_frag[2][TN];

    // MOVE the pointer to the start of the tile
    A = &A[by * BM * K];
    B = &B[bx * BN];
    C = &C[by * BM * N + bx * BN];

    // first global to shared
    #pragma unroll
    for (int i = 0; i < BM; i += a_tile_stride) {
        int ldg_index = i / a_tile_stride * 4;  // 第ldg_index轮
        FETCH_FLOAT4(ldg_a_reg[ldg_index]) =
                FETCH_FLOAT4(A[OFFSET(a_tile_row + i, a_tile_col, K)]);
        // As转置存，其中ldg_a_reg做中间缓存，目的是读取时可以按FLOAT4读取
        As[0][OFFSET(a_tile_col, i + a_tile_row, BM)] = ldg_a_reg[ldg_index];
        As[0][OFFSET(a_tile_col + 1, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 1];
        As[0][OFFSET(a_tile_col + 2, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 2];
        As[0][OFFSET(a_tile_col + 3, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 3];
    }
    #pragma unroll
    for (int i = 0; i < BK; i += b_tile_stride) {
        FETCH_FLOAT4(Bs[0][OFFSET(b_tile_row + i, b_tile_col, BN)]) =
                FETCH_FLOAT4(B[OFFSET(b_tile_row + i, b_tile_col, N)]); // 不需要转置
    }

    int write_idx = 1;
    int load_idx = 0;
    int k = 0;
    do{
        __syncthreads();
        k += BK;
        if(k < K){
            #pragma unroll
            for (int i = 0; i < BM; i += a_tile_stride) {
                int ldg_index = i / a_tile_stride * 4;  // 第ldg_index轮
                FETCH_FLOAT4(ldg_a_reg[ldg_index]) =
                        FETCH_FLOAT4(A[OFFSET(a_tile_row + i, k + a_tile_col, K)]);
            }
            #pragma unroll
            for (int i = 0; i < BK; i += b_tile_stride) {
                int ldg_index = i / b_tile_stride * 4;  // 第ldg_index轮
                FETCH_FLOAT4(ldg_b_reg[ldg_index]) =
                        FETCH_FLOAT4(B[OFFSET(k + b_tile_row + i, b_tile_col, N)]);
            } 
        }
        load_index = write_index ^ 1; // 交替使用两个缓冲区
        #pragma unroll
        for(int i = 0; i < BK; i++){
            #pragma unroll
            for(int m = 0; m < TM; m++){
                FETCH_FLOAT4(a_frag[load_index][m]) = As[load_index][OFFSET(i, ty + m, BM)];
            }
            #pragma unroll
            for(int n = 0; n < TN; n++){
                FETCH_FLOAT4(b_frag[load_index][n]) = Bs[load_index][OFFSET(i, tx + n, BN)];
            }
        }
        
        #pragma unroll
        for (int bk = 0; bk < BK - 1; bk++) {  // 计算了BK-1次，因为是加载下一次迭代的数据，所以可以隐藏“从shared memory加载到寄存器的访存延迟”。
            for (int m = 0; m < TM; m += 4) {
                FETCH_FLOAT4(a_frag[(bk + 1) % 2][m]) = FETCH_FLOAT4(
                        As[load_index][OFFSET(bk + 1, ty + m, BM)]); // 偏移到当前thread tile
            }
            #pragma unroll
            for (int n = 0; n < TN; n += 4) {
                FETCH_FLOAT4(b_frag[(bk + 1) % 2][n]) = FETCH_FLOAT4(
                        Bs[load_index][OFFSET(bk + 1, tx + n, BN)]); // 偏移到当前thread tile
            }
            #pragma unroll
            for (int m = 0; m < TM; m++) {
                for (int n = 0; n < TN; n++) {
                    accum[m][n] += a_frag[bk % 2][m] * b_frag[bk % 2][n];
                }
            }
        }

        #pragma unroll
        for(int m = 0; m < TM; m++){
            #pragma unroll
            for(int n = 0; n < TN; n++){
                accum[m][n] += a_frag[(BK -1) % 2][m] * b_frag[(BK-1) % 2][n];
            }
        }

        if (k < K) {
            // load reg to shared
            #pragma unroll
            for (int i = 0; i < BM; i += a_tile_stride) {
                int ldg_index = i / a_tile_stride * 4;
                As[write_index][OFFSET(a_tile_col, i + a_tile_row, BM)] = ldg_a_reg[ldg_index];
                As[write_index][OFFSET(a_tile_col + 1, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 1];
                As[write_index][OFFSET(a_tile_col + 2, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 2];
                As[write_index][OFFSET(a_tile_col + 3, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 3];
            }
            #pragma unroll
            for (int i = 0; i < BK; i += b_tile_stride) {
                int ldg_index = i / b_tile_stride * 4;
                FETCH_FLOAT4(Bs[write_index][OFFSET(b_tile_row + i, b_tile_col, BN)]) =
                        FETCH_FLOAT4(ldg_b_reg[ldg_index]);
            }
            write_index ^= 1;
        }
    }while(k < K);

    // final calculation
    #pragma unroll
    for(int m = 0; m < TM; m++){
        #pragma unroll
        for(int n = 0; n < TN; n++){
           float ctmp = FETCH_FLOAT4(C[OFFSET(ty + m, tx + n, N)]);
           ctmp.x = alpha * accum[m][n] + beta * ctmp.x;
           ctmp.y = alpha * accum[m][n] + beta * ctmp.y;
           ctmp.z = alpha * accum[m][n] + beta * ctmp.z;
           ctmp.w = alpha * accum[m][n] + beta * ctmp.w;
           FETCH_FLOAT4(C[OFFSET(ty + m, tx + n, N)]) = ctmp;
        }
    }
}

template __global__ void sgemm_v7<128, 128, 8, 8, 8>(int M, int N, int K,
    float alpha, float*A, float*B, float beta, float*C);