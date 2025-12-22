/**
 * Recuction is an aggregation operation that reduces a set of values into a single value.
 * The common reduction operations are sum, max, and softmax.
 */

 #include<cuda_runtime.h>
 #include<stdio.h>
 #include<stdlib.h>

 #define CEIL(a,b) ((a + b -1)/(b))
 #define FLOAT4(a) *(float4*)(&(a))

 #define cudaCheck(err) _cudaCheck(err, __FILE__, __LINE__)
 void _cudaCheck(cudaError_t error, const char *file, int line){
    if (err != cudaSuccess){
        // print the error message
        print("[CUDA ERROR] at file %s(line %d):\n %s\n", file, line, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
    return;
 };

 // Reduction on block level
 __global__ void sum_v0(float* X, float* Y){
    /**
     * Use global memory, and N should be divisible by block size
     */
    const int tid = threadIdx.x;
    // 当前元素块的首地址
    float* x = &X[blockIdx.x * blockDim.x];
    // the threads with id < offset will add their value to the value at id + offset
    // and during each iteration, the offset will be halved
    for(int offset = blockDim.x >> 1; offset > 0; offset >>= 1){
        if (tid < offset){
            x[tid] += x[tid + offset];
        }
        __syncthreads();
    }
    if (tid == 0){ // all the results written to the first thread
        Y[blockIdx.x] = x[0];
    }
 }

 template<const int BLOCK_SIZE>
 void call_sum_v0(float* d_X, float* d_Y, float* h_y, const int N, float* sum){
    /**
     * Use static shared memory, but N should not be divisible by block size
     * The data stored in global memory is not modified and can be reused.
     */
    const int GRID_SIZE = CEIL(N, BLOCK_SIZE);
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size(GRID_SIZE);
    // launch the kernel
    sum_v0<<<grid_size, block_size>>>(d_X, d_Y);
    // copy the result from device to host
    cudaMemcpy(h_y, d_Y, GRID_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    // sync
    cudaDeviceSynchronize();
    // sum the result
    *sum = 0.0; // since we are using pointer, we need to dereference it to get the value
    for(int i = 0; i < GRID_SIZE; i++){
        *sum += h_y[i];
    }
 }

 template<const int BLOCK_SIZE>
 __global__ void sum_v1(float* d_x, float* d_y, const int N){
    /**
     * Use the static shared memory, and N should be divisible by block size
     * Then each block address the same number of elements
     */
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int n = bid * blockDim.x + tid; // the global index of the thread
    // the size of shared memory is the same as the block size, and
    // that is the number of elements in each block
    __shared__ float s_y[BLOCK_SIZE]; 
    // then we need to load the data from global memory to shared memory
    s_y[tid] = (n < N) ? d_x[n] : 0.0; // avoid the out-of-bounds access
    // we need to synchronize before we use the shared memory
    __syncthreads();

    // reduce in the block ( the same logic as v0)
    for(int offset=BLOCK_SIZE>>1; offset>0; offset>>=1){
        if(tid < offset){
            s_y[tid] += s_y[tid + offset];
        }
        // synchronize the threads in this turn of reduction
        __syncthreads();
    }
    // after the reduction in the block
    if(tid == 0){
        d_y[bid] = s_y[0];
    }
 }

 template<const int BLOCK_SIZE>
 void call_sum_v1(float* d_x, float* d_y, float* h_y, const int N, float* sum){
    const int GRID_SIZE = CEIL(N, BLOCK_SIZE);
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size(GRID_SIZE);
    // launch the kernel
    sum_v1<<<grid_size, block_size>>>(d_x, d_y, N);
    // copy the result from device to host
    cudaMemcpy(h_y, d_y, GRID_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    // sync
    cudaDeviceSynchronize();
    // sum the result
    *sum = 0.0;
    for(int i = 0; i < GRID_SIZE; i++){
        *sum += h_y[i];
    }
 }


 __global__ void sum_v2(float* X, float* Y, int n){
    /**
     * Use dynamic shared memory
     */
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int n = bid * blockDim.x + tid; // the global index of the thread
    __shared__ float s_y[]; // determine at the runtime

    s_y[tid] = (n < N) ? d_x[n] : 0.0; // avoid the out-of-bounds access
    __syncthreads();

    for(int offset=blockDim.x>>1; offset>0; offset>>=1){
        if(tid < offset) s_y[tid] += s_y[tid + offset];
        __syncthreads();
    }
    if(tid == 0){
        // copy the result to the global memory
        d_y[bid] = s_y[0];
    }
 }

 template<const int BLOCK_SIZE>
 void call_sum_v2(float* d_x, float* d_y, float* h_y, const int N, float* sum){
    const int GRID_SIZE = CEIL(N, BLOCK_SIZE);
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size(GRID_SIZE);
    // launch the kernel, using dynamic shared memory
    sum_v2<<<grid_size, block_size, BLOCK_SIZE * sizeof(float)>>>(d_x, d_y, N);
    // copy the result from device to host
    cudaMemcpy(h_y, d_y, GRID_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    // sync
    cudaDeviceSynchronize();
    // sum the result
    *sum = 0.0;
    for(int i = 0; i < GRID_SIZE; i++){
        *sum += h_y[i];
    }
 }


 __global__ void sum_v3(float* X, float* Y, int n){
    /**
     * Use atomic functions, but the block size should be exponent of 2
     * then we don;t need to reduce on the host
     */
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int n = bid * blockDim.x + tid; // the global index of the thread
    __shared__ float s_y[];
    s_y[tid] = (n < N) ? d_x[n] : 0.0; // avoid the out-of-bounds access
    __syncthreads();

    for(int offset=BLOCK_SIZE>>1; offset>0; offset>>=1){
        if(tid < offset) s_y[tid] += s_y[tid + offset];
        __syncthreads();
    }

    // then we use the atomic add to sum the result
    if(tid == 0){
        atomicAdd(d_y, s_y[0]);
    }
 }

 template<const int BLOCK_SIZE>
 // d_y and h_y are scalars
 void call_sum_v3(float* d_x, float* d_y, float* h_y, const int N){
    const int GRID_SIZE = CEIL(N, BLOCK_SIZE);
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size(GRID_SIZE);
    // launch the kernel
    sum_v3<<<grid_size, block_size, BLOCK_SIZE * sizeof(float)>>>(d_x, d_y, N);
    // copy the result from device to host
    cudaMemcpy(h_y, d_y, sizeof(float), cudaMemcpyDeviceToHost);
    // cudaDeviceSynchronize();
    // 默认流 + cudaMemcpy() → 已经隐含同步，不再需要 cudaDeviceSynchronize()。
    
 }

 __global__ void sum_v4(float* X, float* Y, int n){
    /**
     * Use warp-level reduction, need the block size to be a multiple of 32
     * otherwise, will access invalid shared memory.
     * don't need to synchronize, because the threads in the same warp will synchronize automatically.
     */
    __shared__ float s_y[32]; // 一个 block 最多 1024 个线程 -> 32 个 warps

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int warp_id = threadIdx.x / warpSize;
    int laneId = threadIdx.x % warpSize; // the idx in the warp

    float val = (idx < n) ? d_x[idx] : 0.0; // store the value in register
    #parama unroll
    for(int offset = warpSize; offset > 0; offset >>= 1){
        val += __shfl_down_sync(0xffffffff, val, offset); // 在 warp 内折半归约
    }

    if(laneId == 0) s_y[warp_id] = val; // warp 里的第一个线程负责写入共享内存
    // synchronize the threads in the warp
    __syncthreads();

    // reduce in the block, to the first warp
    if(warp_id == 0){
        int warpNum = blockDim.x / warpSize;
        val = (laneId < warpNum) ? s_y[laneId] : 0.0f;
        for(int offset = warpSize >> 1; offset > 0; offset >>= 1){
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if(laneId == 0) atomicAdd(d_y, val); // 使用 warp 的第一个线程累加结果
    }
 }

 template<const int BLOCK_SIZE>
 void call_sum_v4(float* d_x, float* d_y, const int N){
    const int GRID_SIZE = CEIL(N, BLOCK_SIZE);
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size(GRID_SIZE);
    *d_y = 0.0f;
    cudaMemset(d_y, h_y, sizeof(float), cudaMemsetHostToDevice);
    // launch the kernel
    sum_v4<<<grid_size, block_size, BLOCK_SIZE * sizeof(float)>>>(d_x, d_y, N);
    cudaMemcpy(h_y, d_y, sizeof(float), cudaMemcpyDeviceToHost);
 }

 __global__ void sum_v5(float* X, float* Y, int n){
    // use float4 vectorized load based on v4
    __shared__ float s_y[32];
    int idx = (blockDim.x * blockIdx.x + threadIdx.x) * 4; // since we are using float4
    int warp_id = threadIdx.x / warpSize;
    int laneId = threadIdx.x % warpSize; // the idx in the warp

    float val = 0.0f;
    if (idx < n){
        float4 tmp_x = FLOAT4(X[idx]);
        val += tmp_x.x;
        val += tmp_x.y;
        val += tmp_x.z;
        val += tmp_x.w;
    }
    #parama unroll
    for(int offset = warpSize >> 1; offset > 0; offset >>= 1){
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    if(laneId == 0) atomicAdd(d_y, val);
    __syncthreads();

    // reduce in the block, to the first warp
    if(warp_id == 0){
        int warpNum = blockDim.x / warpSize;
        val = (laneId < warpNum) ? s_y[laneId] : 0.0f;
        for(int offset = warpSize >> 1; offset > 0; offset >>= 1){
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if(laneId == 0) atomicAdd(d_y, val);
    }
 }

 template<const int BLOCK_SIZE>
 void call_sum_v5(float* d_x, float* d_y, const int N){
    const int GRID_SIZE = CEIL(N, BLOCK_SIZE);
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size(GRID_SIZE);
    *d_y = 0.0f;
    cudaMemset(d_y, h_y, sizeof(float), cudaMemsetHostToDevice);
    // launch the kernel
    sum_v5<<<grid_size, block_size, BLOCK_SIZE * sizeof(float)>>>(d_x, d_y, N);
    cudaMemcpy(h_y, d_y, sizeof(float), cudaMemcpyDeviceToHost);
 }

 int main(){
    size_t N = 100000000;
    constexpr size_t BLOCK_SIZE = 128;
    const int repeat_times = 10;

    // 1. host
    float* h_nums = (floact*) malloc (N * sizeof(float));
    float* sum = (float*) malloc (sizeof(float));
    random_matrix(h_nums, N);

    float total_time_h = TIME_RECORD(repeat_times, ([&]{host_reduce(h_nums, N, sum);}))
    printf("[reduce_host]: sum = %f, total_time_h = %f ms\n", *sum, total_time_h / repeat_times);

    // 2. device
    float *d_nums, *d_rd_nums;
    cudaMalloc((void **) &d_nums, sizeof(float) * N);
    cudaMalloc((void **) &d_rd_nums, sizeof(float) * CEIL(N, BLOCK_SIZE));
    float *h_rd_nums = (float *)malloc(sizeof(float) * CEIL(N, BLOCK_SIZE));
    
    // 2.1 call reduce_v0, 全局内存，因为reduce会把归约结果累加到d_nums（global memory）上，所以重复执行reduce_v0，得到的sum会越来越大
    cudaMemcpy(d_nums, h_nums, sizeof(float) * N, cudaMemcpyHostToDevice);
    float total_time_0 = TIME_RECORD(repeat_times, ([&]{call_reduce_v0<BLOCK_SIZE>(d_nums, d_rd_nums, h_rd_nums, N, sum);}));
    printf("[reduce_v0]: sum = %f, total_time_0 = %f ms\n", *sum, total_time_0 / repeat_times);

    // 2.2 call reduce_v1，使用静态共享内存，重复执行，sum不受影响
    cudaMemcpy(d_nums, h_nums, sizeof(float) * N, cudaMemcpyHostToDevice);
    float total_time_1 = TIME_RECORD(repeat_times, ([&]{call_reduce_v1<BLOCK_SIZE>(d_nums, d_rd_nums, h_rd_nums, N, sum);}));
    printf("[reduce_v1]: sum = %f, total_time_1 = %f ms\n", *sum, total_time_1 / repeat_times);    

    // 2.3 call reduce_v2，在v1基础上改成动态共享内存，性能维持不变
    cudaMemcpy(d_nums, h_nums, sizeof(float) * N, cudaMemcpyHostToDevice);
    float total_time_2 = TIME_RECORD(repeat_times, ([&]{call_reduce_v2<BLOCK_SIZE>(d_nums, d_rd_nums, h_rd_nums, N, sum);}));
    printf("[reduce_v2]: sum = %f, total_time_2 = %f ms\n", *sum, total_time_2 / repeat_times);

    // 2.4 call reduce_v3，在v2基础上引入原子函数，不需要再到CPU进行归约了
    float *d_sum;
    cudaMalloc((void **) &d_sum, sizeof(float));
    cudaMemcpy(d_nums, h_nums, sizeof(float) * N, cudaMemcpyHostToDevice);
    float total_time_3 = TIME_RECORD(repeat_times, ([&]{call_reduce_v3<BLOCK_SIZE>(d_nums, d_sum, sum, N);}));
    printf("[reduce_v3]: sum = %f, total_time_3 = %f ms\n", *sum, total_time_3 / repeat_times);    

    // 2.5 call reduce_v4，使用warp shuffle
    cudaMemcpy(d_nums, h_nums, sizeof(float) * N, cudaMemcpyHostToDevice);
    float total_time_4 = TIME_RECORD(repeat_times, ([&]{call_reduce_v4<BLOCK_SIZE>(d_nums, d_sum, sum, N);}));
    printf("[reduce_v4]: sum = %f, total_time_4 = %f ms\n", *sum, total_time_4 / repeat_times);    

    // 2.6 call reduce_v5，使用warp shuffle + float4
    cudaMemcpy(d_nums, h_nums, sizeof(float) * N, cudaMemcpyHostToDevice);
    float total_time_5 = TIME_RECORD(repeat_times, ([&]{call_reduce_v5<BLOCK_SIZE>(d_nums, d_sum, sum, N);}));
    printf("[reduce_v5]: sum = %f, total_time_5 = %f ms\n", *sum, total_time_5 / repeat_times);    

    // free memory
    free(h_nums);
    free(sum);
    free(h_rd_nums);
    cudaFree(d_nums);
    cudaFree(d_rd_nums);
    cudaFree(d_sum);
    return 0;

}