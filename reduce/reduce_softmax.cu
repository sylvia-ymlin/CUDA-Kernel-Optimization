#include<stdio.h>
#include<stdlib.h>
#include<algorithm>
#include<float.h>
#include<cuda_runtime.h>

const int N = 2048;
constexpr size_t BLOCK_SIZE = 256;
const int repeat_times = 10;

__global__ void setToNegativeMax(float* d_value){
    *d_value = -FLT_MAX;
}

// atomicMax based on atomicCAS
__device__ static float atomicMax(float* address, float val){
    /**
     * address: the address of the value to be updated
     * val: the value to be updated to the address
     */
     int* address_as_i = (int*)address;
     int old = *address_as_i;
     int assumed;
     do{
        assumed = old; // store the old value
        // use atomicCAS to compare and swap the value
        old = atomicCAS(address_as_i, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
     }while(assumed != old);
     // 返回旧制，最大值存储在 address 里·
     return __int_as_float(old); 
}

// 实现 max_kernel
__global__ void max_kernel(float* input, float* output, int N){
    __shared__ float s_mem[32];
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int warp_id = threadIdx.x / warpSize;
    int laneId = threadIdx.x % warpSize; // the idx in the warp

    // initialize max
    float val = (idx < N) ? input[idx] : -FLT_MAX;
    // recude on warp level
    for(int offset = warpSize >> 1; offset > 0; offset >>= 1){
        // 以 offset 为步长，在 warp 内折半归约
        val = fmaxf(val, __shfl_down_sync(oxFFFFFFFF, val, offset))
    }
    // 将 warp 内的最大值存储到共享内存中
    if(laneId == 0) s_mem[warp_id] = val;
    __syncthreads();

    // reduce in the block
    if(warp_id == 0){
        int warpNum = blockDim.x / warpSize;
        val = (laneId < warpNum) ? s_mem[laneId] : -FLT_MAX;
        for(int offset = warpSize >> 1; offset > 0; offset >>= 1){
            val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
        }
        if(laneId == 0) atomicMax(output, val);
    }
}

// sum_kernel
__global__ void sum_kernel(float* input, float* sum, float* max_val, int N){
    __shared__ float s_mem[32];
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int warp_id = threadIdx.x / warpSize;
    int laneId = threadIdx.x % warpSize; // the idx in the warp

    float val = (idx < N) ? input[idx] : 0.0f;
    #parama unroll
    for(int offset = warpSize >> 1; offset > 0; offset >>= 1){
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    if(laneId == 0) s_mem[warp_id] = val;   
    __syncthreads();

    // reduce in the block
    if(warp_id == 0){
        int warpNum = blockDim.x / warpSize;
        val = (laneId < warpNum) ? s_mem[laneId] : 0.0f;
        for(int offset = warpSize >> 1; offset > 0; offset >>= 1){
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if(laneId == 0) atomicAdd(sum, val);
    }
}

// softmax_kernel: a pointwise operation
__global__ void softmax_kernel(float* input, float* output, float* sum, float* max_val, int N){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N) output[idx] = expf(input[idx] - *max_val) / (*sum)
}

// on cpu
void softmax_cpu(float* input, float* output, int N, float* M, float* sum){
    *M = *(std::max_element(input, input + N));
    *sum = 0.0f;
    for(int i = 0; i < N; i++){
        *sum += expf(input[i] - *M);
    }
    for(int i = 0; i < N; i++){
        output[i] = expf(input[i] - *M) / (*sum);
    }
}

// call the completed kernels
void call_softmax(float* output, float* d_input, float d_output, float* d_total, float* d_max, int N){
    int block_size = BLOCK_SIZE;
    int grid_size = CEIL(N, BLOCK_SIZE);

    // 1. Initialize
    cudaCheck(cudaMemset(d_total, 0.0f, sizeof(float)));
    cudaCheck(cudaMemset(d_max, 0, sizeof(float)));

    // 2. calculate the sum
    sum_kernel<<<grid_size, block_size>>>(d_input, d_total, d_max, N);


    // 3. calculate the max
    max_kernel<<<grid_size, block_size>>>(d_input, d_max, N);

    // 4. calculate the softmax
    softmax_kernel<<<grid_size, block_size>>>(d_input, output, d_total, d_max, N);
}

int main(){
    float* input = (float*) malloc (N * sizeof(float));
    float* output_ref = (float*) malloc (N * sizeof(float));
    float* M = (float*) malloc (sizeof(float));
    float* sum = (float*) malloc (sizeof(float));
    
    // initialize the input
    for(int i = 0; i < N; i++){
        input[i] = 1 / (float)N;
    }
    
    float total_time_h = TIME_RECORD(repeat_times, ([&]{softmax_cpu(input, output_ref, N, M, sum);}));
    printf("[softmax_cpu]: total_time_h = %f ms\n", total_time_h / repeat_times);

    float* d_input, *d_output, *d_total, *d_max;
    cudaMalloc((void**)&d_input, N * sizeof(float));
    cudaMalloc((void**)&d_output, N * sizeof(float));
    cudaMalloc((void**)&d_total, sizeof(float));
    cudaMalloc((void**)&d_max, sizeof(float));

    cudaCheck(cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice));
    float* h_output = (float*) malloc (N * sizeof(float));

    float total_time_d = TIME_RECORD(repeat_times, ([&]{call_softmax(h_output, d_input, d_output, d_total, d_max, N);}));
    printf("[softmax_kernel]: total_time_d = %f ms\n", total_time_d / repeat_times);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_total);
    cudaFree(d_max);
    
    free(input);
    free(output_ref);
    free(M);
    free(sum);
    free(h_output);
    return 0;
}

