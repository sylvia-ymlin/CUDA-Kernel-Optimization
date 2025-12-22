// consider the coalesced memory access
// bank conflicts
/**
 * 尽量合并访问，即连续线程读取连续内存，且访问的全局内存的首地址是 32 字节
 * （一次数据传输处理的数据量）的倍数 -> cudaMalloc 分配的内存至少是 256 字节的整数倍
 */

 /**
  * 如果不能同时合并读取和写入，应该尽量合并写入
  * 编译器如果判断一个全局内存变量在和函数内只读，会自动调用 __ldg() 函数
  * 读取全局内存，从而对数据进行缓存，缓解非合并访问带来的影响，但这只对只读变量有效
  * 
  * 对于 Kepler 和 Maxwell 架构，需要手动调用 __ldg() 函数
  * B[ny * N + nx] = __ldg(&A[nx * N + ny])
  */

  /**
    [device_transpose_v0] Average time: (6.859354) ms
    [device_transpose_v1] Average time: (4.310410) ms
    [device_transpose_v2] Average time: (2.117488) ms
    [device_transpose_v3] Average time: (3.805533) ms
    [device_transpose_v4] Average time: (2.035469) ms
    [device_transpose_v5] Average time: (2.023494) ms
   */

#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>

void cpu_transpose(float* input, int M, int N, float* output){
    /**
     * inputs:
     * - input: M x N matrix, the matrix to be transposed
     * - output: N x M matrix, the transposed matrix
     * - M: number of rows in the matrix
     * - N: number of columns
     */
    for(int i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            output[j * M + i] = input[i * N + j];
        }
    }
}

// the naive implementation
__global__ void device_transpose_v0(float* input, int M, int N, float* output){
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    if(row < M && col < N){
        output[col * M + row] = input[row * N + col];
    }
}

// coalesced memory access
__global__ void device_transpose_v1(float* input, int M, int N, float* output){
    // we need the increase of thread index along the row
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if(row < M && col < N){
        // traverse the matrix
        output[col * M + row] = input[row * N + col];
    }
}

// explicitly call __ldg() to read the global memory
__global__ void device_transpose_v2(float* input, int M, int N, float* output){
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if(row < M && col < N){
        // traverse the matrix
        output[col * M + row] = __ldg(&input[row * N + col]);
    }
}

// use shared memory to store the transposed matrix
// coalesced memory access and write
// but bank conflicts exist
template<const int TILE-DIM>
__global__ void device_transpose_v3(float* input, int M, int N, float* output){
    // the tile size is TILE_DIM x TILE_DIM
    __shared__ float s_output[TILE_DIM][TILE_DIM]; 
    const int bx = blockIdx.x * TILE_DIM;
    const int by = blockIdx.y * TILE_DIM;
    const int x1 = bx + threadIdx.x;
    const int y1 = by + threadIdx.y;

    if(x1 < N && y1 < M){
        // traverse the matrix
        s_output[x1][y1] = input[y1 * N + x1];
    }
    __syncthreads();

    // write the result to the output
    const int x2 = by + threadIdx.x;
    const int y2 = bx + threadIdx.y;
    if(x2 < M && y2 < N){
        // 同一个 warp 中 的 32 个线程恰好访问对应内存中跨度为 32 的内存空间
        // 将导致 32 路 bank conflicts
        output[y2 * M + x2] = s_output[threadIdx.x][threadIdx.y];
    }
}

// padding to avoid bank conflicts
template<const int TILE_DIM>
__global__ void device_transpose_v4(float* input, int M, int N, float* output){
    __shared__ float s_output[TILE_DIM][TILE_DIM + 1];
    const int bx = blockIdx.x * TILE_DIM;
    const int by = blockIdx.y * TILE_DIM;
    const int x1 = bx + threadIdx.x;
    const int y1 = by + threadIdx.y;
    if(x1 < N && y1 < M){
        s_output[x1][y1] = input[y1 * N + x1];
    }
    __syncthreads();

    const int x2 = by + threadIdx.x;
    const int y2 = bx + threadIdx.y;
    if(x2 < M && y2 < N){
        output[y2 * M + x2] = s_output[threadIdx.x][threadIdx.y];
    }
}

// 使用 swizzling 避免 bank conflicts
// swizzling 是一种优化技巧，通过将内存地址进行重新排列，从而避免 bank conflicts
// 例如，将内存地址从 [0, 1, 2, 3, 4, 5, 6, 7] 重新排列为 [0, 2, 4, 6, 1, 3, 5, 7]
// 这样，同一个 warp 中的 32 个线程可以访问连续的内存空间
// 只适用用于矩阵维度为 2 的幂次方的情况，通过 运算的封闭性，或者异运算
// x1 ^y != x2^y 当且仅当 x1 != x2
__global__ void device_transpose_v5(float* input, int M, int N, float* output){
    __shared__ float s_output[TILE_DIM][TILE_DIM];
    const int bx = blockIdx.x * TILE_DIM;
    const int by = blockIdx.y * TILE_DIM;
    const int x1 = bx + threadIdx.x;
    const int y1 = by + threadIdx.y;
    if(x1 < N && y1 < M){
        s_output[threadIdx.y][threadIdx.x ^ threadIdx.y] = input[y1 * N + x1];
    }
    __syncthreads();

    const int x2 = by + threadIdx.x;
    const int y2 = bx + threadIdx.y;
    if(x2 < M && y2 < N){
        output[y2 * M + x2] = s_output[threadIdx.x][threadIdx.x ^ threadIdx.y];
    }
}

int main() {
    // 输入是M行N列，转置后是N行M列
    size_t M = 12800;
    size_t N = 1280;
    constexpr size_t BLOCK_SIZE = 32;
    const int repeat_times = 10;

    // --------------------host 端计算一遍转置, 输出的结果用于后续验证---------------------- //
    float *h_matrix = (float *)malloc(sizeof(float) * M * N);
    float *h_matrix_tr_ref = (float *)malloc(sizeof(float) * N * M);
    randomize_matrix(h_matrix, M * N);
    host_transpose(h_matrix, M, N, h_matrix_tr_ref);
    // printf("init_matrix:\n");
    // print_matrix(h_matrix, M, N);
    // printf("host_transpose:\n");
    // print_matrix(h_matrix_tr_ref, N, M);

    float *d_matrix;
    cudaMalloc((void **) &d_matrix, sizeof(float) * M * N);
    cudaMemcpy(d_matrix, h_matrix, sizeof(float) * M * N, cudaMemcpyHostToDevice);
    free(h_matrix);

    // --------------------------------call transpose_v0--------------------------------- //
    float *d_output0;
    cudaMalloc((void **) &d_output0, sizeof(float) * N * M);                              // device输出内存
    float *h_output0 = (float *)malloc(sizeof(float) * N * M);                            // host内存, 用于保存device输出的结果

    dim3 block_size0(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size0(CEIL(N, BLOCK_SIZE), CEIL(M, BLOCK_SIZE));                            // 根据input的形状(M行N列)进行切块
    float total_time0 = TIME_RECORD(repeat_times, ([&]{device_transpose_v0<<<grid_size0, block_size0>>>(d_matrix, d_output0, M, N);}));
    cudaMemcpy(h_output0, d_output0, sizeof(float) * N * M, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    verify_matrix(h_output0, h_matrix_tr_ref, M * N);                                     // 检查正确性
    printf("[device_transpose_v0] Average time: (%f) ms\n", total_time0 / repeat_times);  // 输出平均耗时

    cudaFree(d_output0);
    free(h_output0);

    // --------------------------------call transpose_v1--------------------------------- //
    float *d_output1;
    cudaMalloc((void **) &d_output1, sizeof(float) * N * M);                              // device输出内存
    float *h_output1 = (float *)malloc(sizeof(float) * N * M);                            // host内存, 用于保存device输出的结果

    dim3 block_size1(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size1(CEIL(M, BLOCK_SIZE), CEIL(N, BLOCK_SIZE));                            // 根据output的形状(N行M列)进行切块
    float total_time1 = TIME_RECORD(repeat_times, ([&]{device_transpose_v1<<<grid_size1, block_size1>>>(d_matrix, d_output1, M, N);}));
    cudaMemcpy(h_output1, d_output1, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    verify_matrix(h_output1, h_matrix_tr_ref, M * N);                                     // 检查正确性
    printf("[device_transpose_v1] Average time: (%f) ms\n", total_time1 / repeat_times);  // 输出平均耗时

    cudaFree(d_output1);
    free(h_output1);

    // --------------------------------call transpose_v2--------------------------------- //
    float *d_output2;
    cudaMalloc((void **) &d_output2, sizeof(float) * N * M);                              // device输出内存
    float *h_output2 = (float *)malloc(sizeof(float) * N * M);                            // host内存, 用于保存device输出的结果

    dim3 block_size2(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size2(CEIL(M, BLOCK_SIZE), CEIL(N, BLOCK_SIZE));                            // 根据output的形状(N行M列)进行切块
    float total_time2 = TIME_RECORD(repeat_times, ([&]{device_transpose_v2<<<grid_size2, block_size2>>>(d_matrix, d_output2, M, N);}));
    cudaMemcpy(h_output2, d_output2, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    verify_matrix(h_output2, h_matrix_tr_ref, M * N);                                     // 检查正确性
    printf("[device_transpose_v2] Average time: (%f) ms\n", total_time2 / repeat_times);  // 输出平均耗时

    cudaFree(d_output2);
    free(h_output2);

    // --------------------------------call transpose_v3--------------------------------- //
    float *d_output3;
    cudaMalloc((void **) &d_output3, sizeof(float) * N * M);                              // device输出内存
    float *h_output3 = (float *)malloc(sizeof(float) * N * M);                            // host内存, 用于保存device输出的结果

    dim3 block_size3(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size3(CEIL(N, BLOCK_SIZE), CEIL(M, BLOCK_SIZE));                            // 根据input的形状(M行N列)进行切块
    float total_time3 = TIME_RECORD(repeat_times, ([&]{device_transpose_v3<BLOCK_SIZE><<<grid_size3, block_size3>>>(d_matrix, d_output3, M, N);}));
    cudaMemcpy(h_output3, d_output3, sizeof(float) * N * M, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    verify_matrix(h_output3, h_matrix_tr_ref, M * N);                                     // 检查正确性
    printf("[device_transpose_v3] Average time: (%f) ms\n", total_time3 / repeat_times);  // 输出平均耗时

    cudaFree(d_output3);
    free(h_output3);

    // --------------------------------call transpose_v4--------------------------------- //
    float *d_output4;
    cudaMalloc((void **) &d_output4, sizeof(float) * N * M);                              // device输出内存
    float *h_output4 = (float *)malloc(sizeof(float) * N * M);                            // host内存, 用于保存device输出的结果

    dim3 block_size4(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size4(CEIL(N, BLOCK_SIZE), CEIL(M, BLOCK_SIZE));                            // 根据input的形状(M行N列)进行切块
    float total_time4 = TIME_RECORD(repeat_times, ([&]{device_transpose_v4<BLOCK_SIZE><<<grid_size4, block_size4>>>(d_matrix, d_output4, M, N);}));
    cudaMemcpy(h_output4, d_output4, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    verify_matrix(h_output4, h_matrix_tr_ref, M * N);
    printf("[device_transpose_v4] Average time: (%f) ms\n", total_time4 / repeat_times);

    cudaFree(d_output4);
    free(h_output4);

    // --------------------------------call transpose_v5--------------------------------- //
    float *d_output5;
    cudaMalloc((void **) &d_output5, sizeof(float) * N * M);                              // device输出内存
    float *h_output5 = (float *)malloc(sizeof(float) * N * M);                            // host内存, 用于保存device输出的结果

    dim3 block_size5(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size5(CEIL(N, BLOCK_SIZE), CEIL(M, BLOCK_SIZE));                            // 根据input的形状(M行N列)进行切块
    float total_time5 = TIME_RECORD(repeat_times, ([&]{device_transpose_v5<BLOCK_SIZE><<<grid_size5, block_size5>>>(d_matrix, d_output5, M, N);}));
    cudaMemcpy(h_output5, d_output5, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    verify_matrix(h_output5, h_matrix_tr_ref, M * N);
    printf("[device_transpose_v5] Average time: (%f) ms\n", total_time5 / repeat_times);

    cudaFree(d_output5);
    free(h_output5);

    // ---------------------------------------------------------------------------------- //
    free(h_matrix_tr_ref);

    return 0;
}