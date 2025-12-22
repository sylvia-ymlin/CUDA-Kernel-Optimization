// 使用 warp_shuffle 实现
// AtomicMax 不支持 float 类型，需要手动实现

#include<cuda_runtime.h>
#include<stdio.h>
#include<stdlib.h>
#include<float.h>

void max_cpu(float* input, float* output, int n){
   *output = *(std::max_element(input, input + n));
}

__device__ static float atomicMax(float* address, float val){
   int* address_as_i = (int*)address;
   int old = *address_as_i, assumed; // obtain the old value of the address
   do {
      assumed = old;
      // use atomicCAS to compare and swap the value
      old = atomicCAS(address_as_i, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
   } while (assumed != old); // if the value is not changed, the loop will end
   return __int_as_float(old); // return the old value
}

__global__ void max_kernel(float* input, float* output, int N){
   __shared__ float s_mem[32];

   int idx = blockDim.x * blockIdx.x + threadIdx.x;
   int warp_id = threadIdx.x / warpSize;
   int laneId = threadIdx.x % warpSize; // the idx in the warp

   float val = (idx < N) ? input[idx] : -FLT_MAX; // store the value in register
   #parama unroll
   for(int offset = warpSize; offset > 0; offset >>= 1){
      val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
   }
   if(laneId == 0) atomicMax(&s_mem[warp_id], val);
   __syncthreads();

   // reduce in the block, to the first warp
   if(warp_id == 0){
      int warpNum = blockDim.x / warpSize;
      val = (laneId < warpNum) ? s_mem[laneId] : -FLT_MAX;
      for(int offset = warpSize >> 1; offset > 0; offset >>= 1){
         // use the custom atomicMax function
         val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
      }
      if(laneId == 0) atomicMax(output, val);
   }
}


int main(){
   size_t N = 100000000;
   constexpr size_t BLOCK_SIZE = 128;
   const int repeat_times = 10;

   float* input = (float*) malloc (N * sizeof(float));
   for(int i=N; i>=0; i--){
      input[i] = i;
   }
   float* ref_output = (float*) malloc (sizeof(float));
   float total_time_h = TIME_RECORD(repeat_times, ([&]{max_cpu(input, ref_output, N);}));
   printf("[max_cpu]: total_time_h = %f ms\n", total_time_h / repeat_times);

   // max
   float* h_output = (float*) malloc (sizeof(float));
   float* d_input, *d_output;
   cudaMalloc((void**)&d_input, N * sizeof(float));
   cudaMalloc((void**)&d_output, sizeof(float));
   cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);
   float total_time_1 = TIME_RECORD(repeat_times, ([&]{max_kernel<<<grid_size, block_size>>>(d_input, d_output, N);}));
   printf("[max_kernel]: total_time_1 = %f ms\n", total_time_1 / repeat_times);
   cudaMemcpy(h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);
   
   cudaFree(d_input);
   cudaFree(d_output);
   free(h_output);
   free(input);
   free(output);
   return 0;
}

