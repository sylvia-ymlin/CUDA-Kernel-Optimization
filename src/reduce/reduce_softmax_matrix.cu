/**
 * 矩阵 softmax -> 对 M X N 的每一行求 softmax
 * 一个 warp 负责一行或者一列的计算
 */

 // 使用 __shfl_down_sync，warp 归约以后，结果将汇总到第一个线程
 // 需要第一个线程将结果写回 s_mem, 供同一个 warp 中的其他线程使用

 // 可以使用 __shfl_xor_sync，这样每个线程的寄存器的 max_val 和 sum 都是最终的结果，而不用写回共享内存

 /**
    [softmax_row_cpu]: total_time_h = 2.601370 ms
    [softmax_row_gpu]: total_time_d = 0.061936 ms
    [softmax_col_cpu]: total_time_h = 5.155942 ms
    [softmax_col_gpu]: total_time_d = 0.167795 ms
  */
 

#include<cuda_runtime.h>
#include<stdio.h>
#include<stdlib.h>
#inlcude<algorithm>
#include<float.h> // this file defines the float constants

// 按行计算 softmax
// means we calculate the max and sum for each row
void softmax_row_cpu(float* input, float* output, int M, int N){
    /**
     * inputs:
     * - input: M x N matrix
     * - output: M x N matrix
     * - M: number of rows in the matrix
     * - N: number of columns
     */
    
     // for each row
     for(int row = 0; row < M; row++){
        float* input_tmp = input + row * N;
        float* output_tmp = output + row * N;
        
        // the return is a pointer, so we need to dereference
        float max_val = *(std::max_element(intput_tmp, input_tmp + N));

        float sum = 0.0f;
        for(int col = 0; col < N; col++){
            sum += std::exp(input_tmp[col] - max_val);
        }

        for(int col = 0; col < N; col++){
            output_tmp[col] = std::exp(input_tmp[col] - max_val) / sum;
        }
     }
}

// 按列计算 softmax
// means we calculate the max and sum for each column
void softmax_col_cpu(float* inout, float* output, int M, int N){
    /**
     * inputs:
     * - input: M x N matrix
     * - output: M x N matrix
     * - M: number of rows in the matrix
     * - N: number of columns
     */

     // for each column
     for(int col = 0; col < N; col++){
        float* input_tmp = input + col;
        float* output_tmp = output + col;

        // 当前列的最大值
        float max_val = -FLT_MAX;
        for(int row = 0; row < M; row++){
            max_val = fmaxf(max_val, input_tmp[row * N]);
        }

        // 当前列的和
        float sum = 0.0f;
        for(int row = 0; row < M; row++){
            sum += std::exp(input_tmp[row * N] - max_val);
        }
        
        for(int row = 0; row < M; row++){
            output_tmp[row] = std::exp(input_tmp[row * N] - max_val) / sum;
        }
     }

     // gpu: 计算每一行的 softmax
     __global__ void softmax_row_kernel(float* input, float* output, int M, int N){
        __shared__ float s_sum;
        __shared__ float s_max;

        // warp level index
        int laneId = threadIdx.x % warpSize;

        // current row
        int row = blockIdx.x;
        if(row >= M) return;

        int iteration = CEIL(N, warpSize); // 每个线程负责计算的数据个数
        // 一个 warp 计算一行
        float max_val = -FLT_MAX;
        #pragma unroll
        for(int i = 0; i < iteration; i++){
            // the column index
            int col = i * warpSize + laneId;
            max_val = (col < N) ? fmaxf(max_val, input[row * N + col]) : max_val;
        }
        #pragma unroll
        for(int offset = warpSize >> 1; offset > 0; offset >>= 1){
            max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
        }
        if(laneId == 0) s_max = max_val; // warp 的第一个线程负责写入共享内存

        // calculate the sum, just the same as the max_val
        float sum = 0.0f;
        #pragma unroll
        for(int i = 0; i < iteration; i++){
            int col = i * warpSize + laneId;
            if(col < N){
                sum += std::exp(input[row * N + col] - max_val);
            }
        }
        #pragma unroll
        for(int offset = warpSize >> 1; offset > 0; offset >>= 1){
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if(laneId == 0) s_sum = sum; // warp 的第一个线程负责写入共享内存

        // calculate the softmax
        #pragma unroll
        for(int i = 0; i < iteration; i++){
            int col = i * warpSize + laneId;
            if(col < N){
                output[row * N + col] = std::exp(input[row * N + col] - max_val) / s_sum;
            }
        }
    }
}

// use __shfl_xor_sync 
__global__ void softmax_row_kernel_shfl_xor(float* input, float* output, int M, int N){
    // thenn we don't need to use the shared memory
    int laneId = threadIdx.x % warpSize;
    // 当前行
    int row = blockIdx.x;
    if(row >= M) return;

    int iteration = CEIL(N, warpSize); // 每个线程负责计算的数据个数
    // 一个 warp 计算一行
    float max_val = -FLT_MAX;
    #pragma unroll
    for(int i = 0; i < iteration; i++){
        int col = i * warpSize + laneId;
        max_val = (col < N) ? fmaxf(max_val, input[row * N + col]) : max_val;
    }
    #pragma unroll
    for(int offset = warpSize >> 1; offset > 0; offset >>= 1){
        max_val = fmaxf(max_val, __shfl_xor_sync(0xffffffff, max_val, offset));
    }
    

    // calculate the sum, just the same as the max_val
    float sum = 0.0f;
    #pragma unroll
    for(int i = 0; i < iteration; i++){
        int col = i * warpSize + laneId;
        if(col < N){
            sum += std::exp(input[row * N + col] - max_val);
        }
    }
    #pragma unroll
    for(int offset = warpSize >> 1; offset > 0; offset >>= 1){
        sum += __shfl_xor_sync(0xffffffff, sum, offset);
    }
    
    // calculate the softmax
    #pragma unroll
    for(int i = 0; i < iteration; i++){
        int col = i * warpSize + laneId;
        if(col < N){
            output[row * N + col] = std::exp(input[row * N + col] - max_val) / sum;
        }
    }
}

// use __shfl_xor_sync for columnwise softmax
__global__ void softmax_col_kernel_shfl_xor(float* input, float* output, int M, int N){
    int laneId = threadIdx.x % warpSize;
    int col = blockIdx.x;
    if(col >= N) return;

    int iteration = CEIL(M, warpSize); // 每个线程负责计算的数据个数
    float max_val = -FLT_MAX;
    #pragma unroll
    for(int i = 0; i < iteration; i++){
        int row = i * warpSize + laneId;
        max_val = (row < M) ? fmaxf(max_val, input[row * N + col]) : max_val;
    }   
    #pragma unroll
    for(int offset = warpSize >> 1; offset > 0; offset >>= 1){
        max_val = fmaxf(max_val, __shfl_xor_sync(0xffffffff, max_val, offset));
    }

    float sum = 0.0f;
    #pragma unroll
    for(int i = 0; i < iteration; i++){
        int row = i * warpSize + laneId;
        if(row < M){
            sum += std::exp(input[row * N + col] - max_val);
        }
    }
    #pragma unroll
    for(int offset = warpSize >> 1; offset > 0; offset >>= 1){
        sum += __shfl_xor_sync(0xffffffff, sum, offset);
    }

    // calculate the softmax
    #pragma unroll
    for(int i = 0; i < iteration; i++){
        int row = i * warpSize + laneId;
        if(row < M){
            output[row * N + col] = std::exp(input[row * N + col] - max_val) / sum;
        }
    }
}

int main(){
    const int M = 2048;
    const int N = 2048;
    const int repeat_times = 10;

    float* input = (float*) malloc (M * N * sizeof(float));
    float* output = (float*) malloc (M * N * sizeof(float));
    float* output_ref = (float*) malloc (M * N * sizeof(float));

    // initialize the input
    for(int i = 0; i < M * N; i++){
        input[i] = 1 / (float)(M * N);
    }

    // cpu results
    softmax_row_cpu(input, output_ref, M, N);
    softmax_col_cpu(input, output_ref, M, N);

    // gpu results
    float* d_input, *d_output;
    cudaMalloc((void**)&d_input, M * N * sizeof(float));
    cudaMalloc((void**)&d_output, M * N * sizeof(float));
    cudaMemcpy(d_input, input, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, output_ref, M * N * sizeof(float), cudaMemcpyHostToDevice);
    
    softmax_row_kernel<<<M, 256>>>(d_input, d_output, M, N);
    cudaMemcpy(output, d_output, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    float total_time_h = TIME_RECORD(repeat_times, ([&]{softmax_row_cpu(input, output_ref, M, N);}));
    printf("[softmax_row_cpu]: total_time_h = %f ms\n", total_time_h / repeat_times);

    softmax_col_kernel<<<N, 256>>>(d_input, d_output, M, N);
    cudaMemcpy(output, d_output, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    float total_time_h = TIME_RECORD(repeat_times, ([&]{softmax_col_cpu(input, output_ref, M, N);}));
    printf("[softmax_col_cpu]: total_time_h = %f ms\n", total_time_h / repeat_times);

    cudaFree(d_input);
    cudaFree(d_output);

    free(input);
    free(output);
    free(output_ref);
    return 0;
}