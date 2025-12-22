#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "add.cuh"

/*
Assumptions:
    - A, B, C are 16-byte aligned (true for cudaMalloc)
    - float4 kernel handles n4 = (n / 4) * 4 elements
    - remainder handled by scalar kernel
*/

__global__ void elementwise_add_float4(
    const float* __restrict__ A, 
    const float* __restrict__ B, 
    float* __restrict__ C, 
    int n4
){
    // get the thread id
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int idx = tid * 4; // since we are using float4
    if (idx >= n4) return;

    const float4* A4 = reinterpret_cast<const float4*>(A);
    const float4* B4 = reinterpret_cast<const float4*>(B);
    float4* C4 = reinterpret_cast<float4*>(C);

    float4 a = A4[tid];
    float4 b = B4[tid];

    float4 c;
    // unroll the loop, for better performance
    c.x = a.x + b.x;
    c.y = a.y + b.y;
    c.z = a.z + b.z;
    c.w = a.w + b.w;

    C4[tid] = c;
}

__global__ void elementwise_add_scalar(
    const float* A, 
    const float* B, 
    float* C, 
    int n, 
    int offset
){
    int idx = blockDim.x * blockIdx.x + threadIdx.x + offset;
    if (idx < n){
        C[idx] = A[idx] + B[idx];
    }
}


__global__ void sigmoid_float4(
    const float4* X, 
    float4* Y, 
    int n4
){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int idx = tid * 4; // since we are using float4
    if (idx >= n4) return;

    const float4* X4 = reinterpret_cast<const float4*>(X);
    float4* Y4 = reinterpret_cast<float4*>(Y);

    float4 x = X4[tid];
    float4 y;

    // unroll the loop, for better performance
    y.x = 1.0f / (1.0f + expf(-x.x));
    y.y = 1.0f / (1.0f + expf(-x.y));
    y.z = 1.0f / (1.0f + expf(-x.z));
    y.w = 1.0f / (1.0f + expf(-x.w));
    Y4[tid] = y;
}

__global__ void relu_float4(
    const float4* X, 
    float4* Y, 
    int n4
){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int idx = tid * 4; // since we are using float4
    if (idx >= n4) return;
    
    const float4* X4 = reinterpret_cast<const float4*>(X);
    float4* Y4 = reinterpret_cast<float4*>(Y);

    float4 x = X4[tid];
    float4 y;
    // unroll the loop, for better performance
    y.x = max(0.0f, x.x);
    y.y = max(0.0f, x.y);
    y.z = max(0.0f, x.z);
    y.w = max(0.0f, x.w);
    Y4[tid] = y;
}
