#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "../src/elementwise/elementwise.cuh"

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))
#define cudaCheck(err) _cudaCheck((err), __FILE__, __LINE__)
inline void _cudaCheck(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        std::printf("[CUDA ERROR] at %s:%d\n%s\n", file, line, cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

bool close_enough(float a, float b, float tol = 1e-5f) {
    return std::fabs(a - b) <= tol * (1.0f + std::fabs(b));
}

void test_elementwise_add() {
    constexpr int n = 10;          // covers vectorized path (8) + tail (2)
    constexpr int n4 = (n / 4) * 4;

    float hA[n], hB[n], hC[n];
    for (int i = 0; i < n; ++i) { hA[i] = float(i); hB[i] = 2.0f * float(i); }

    float *dA, *dB, *dC;
    cudaCheck(cudaMalloc(&dA, n * sizeof(float)));
    cudaCheck(cudaMalloc(&dB, n * sizeof(float)));
    cudaCheck(cudaMalloc(&dC, n * sizeof(float)));
    cudaCheck(cudaMemcpy(dA, hA, n * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dB, hB, n * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(256);
    if (n4 > 0) {
        dim3 grid_vec(CEIL_DIV(n4 / 4, block.x));
        elementwise_add_float4<<<grid_vec, block>>>(dA, dB, dC, n4);
        cudaCheck(cudaGetLastError());
    }
    if (n > n4) {
        dim3 grid_tail(CEIL_DIV(n - n4, block.x));
        elementwise_add_scalar<<<grid_tail, block>>>(dA, dB, dC, n, n4);
        cudaCheck(cudaGetLastError());
    }

    cudaCheck(cudaMemcpy(hC, dC, n * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheck(cudaFree(dA)); cudaCheck(cudaFree(dB)); cudaCheck(cudaFree(dC));

    for (int i = 0; i < n; ++i) {
        float expected = hA[i] + hB[i];
        if (!close_enough(hC[i], expected)) {
            std::printf("elementwise_add failed at %d: got %f expected %f\n", i, hC[i], expected);
            std::exit(EXIT_FAILURE);
        }
    }
    std::printf("elementwise_add: PASS\n");
}

void test_sigmoid() {
    constexpr int n4 = 8; // must be multiple of 4 for float4 path
    float hX[n4] = { -4.f, -2.f, -1.f, 0.f, 1.f, 2.f, 3.f, 4.f };
    float hY[n4];

    float *dX, *dY;
    cudaCheck(cudaMalloc(&dX, n4 * sizeof(float)));
    cudaCheck(cudaMalloc(&dY, n4 * sizeof(float)));
    cudaCheck(cudaMemcpy(dX, hX, n4 * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(256);
    dim3 grid(CEIL_DIV(n4 / 4, block.x));
    sigmoid_float4<<<grid, block>>>(reinterpret_cast<const float4*>(dX),
                                    reinterpret_cast<float4*>(dY),
                                    n4);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaMemcpy(hY, dY, n4 * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheck(cudaFree(dX)); cudaCheck(cudaFree(dY));

    for (int i = 0; i < n4; ++i) {
        float expected = 1.0f / (1.0f + std::exp(-hX[i]));
        if (!close_enough(hY[i], expected, 1e-4f)) {
            std::printf("sigmoid failed at %d: got %f expected %f\n", i, hY[i], expected);
            std::exit(EXIT_FAILURE);
        }
    }
    std::printf("sigmoid: PASS\n");
}

void test_relu() {
    constexpr int n4 = 8;
    float hX[n4] = { -3.f, -1.f, 0.f, 1.f, 2.f, -2.f, 5.f, -5.f };
    float hY[n4];

    float *dX, *dY;
    cudaCheck(cudaMalloc(&dX, n4 * sizeof(float)));
    cudaCheck(cudaMalloc(&dY, n4 * sizeof(float)));
    cudaCheck(cudaMemcpy(dX, hX, n4 * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(256);
    dim3 grid(CEIL_DIV(n4 / 4, block.x));
    relu_float4<<<grid, block>>>(reinterpret_cast<const float4*>(dX),
                                 reinterpret_cast<float4*>(dY),
                                 n4);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaMemcpy(hY, dY, n4 * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheck(cudaFree(dX)); cudaCheck(cudaFree(dY));

    for (int i = 0; i < n4; ++i) {
        float expected = hX[i] > 0.f ? hX[i] : 0.f;
        if (!close_enough(hY[i], expected)) {
            std::printf("relu failed at %d: got %f expected %f\n", i, hY[i], expected);
            std::exit(EXIT_FAILURE);
        }
    }
    std::printf("relu: PASS\n");
}

void perf_elementwise_add() {
    constexpr int n = 32 * 1024 * 1024;  // 32M elements
    constexpr int n4 = (n / 4) * 4;
    constexpr int warmup = 3;
    constexpr int iters = 100;

    float *dA, *dB, *dC;
    cudaCheck(cudaMalloc(&dA, n * sizeof(float)));
    cudaCheck(cudaMalloc(&dB, n * sizeof(float)));
    cudaCheck(cudaMalloc(&dC, n * sizeof(float)));

    dim3 block(256);
    dim3 grid_vec(CEIL_DIV(n4 / 4, block.x));
    dim3 grid_tail(CEIL_DIV(n - n4, block.x));

    // Warmup
    for (int i = 0; i < warmup; ++i) {
        if (n4 > 0) elementwise_add_float4<<<grid_vec, block>>>(dA, dB, dC, n4);
        if (n > n4) elementwise_add_scalar<<<grid_tail, block>>>(dA, dB, dC, n, n4);
    }
    cudaCheck(cudaDeviceSynchronize());

    // Timing
    cudaEvent_t start, stop;
    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&stop));

    cudaCheck(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        if (n4 > 0) elementwise_add_float4<<<grid_vec, block>>>(dA, dB, dC, n4);
        if (n > n4) elementwise_add_scalar<<<grid_tail, block>>>(dA, dB, dC, n, n4);
    }
    cudaCheck(cudaEventRecord(stop));
    cudaCheck(cudaEventSynchronize(stop));

    float ms;
    cudaCheck(cudaEventElapsedTime(&ms, start, stop));
    float avg_ms = ms / iters;
    float bandwidth_gb = (3.0f * n * sizeof(float)) / (avg_ms * 1e6); // 2 reads + 1 write

    std::printf("elementwise_add performance: %.3f ms/iter, %.2f GB/s\n", avg_ms, bandwidth_gb);

    cudaCheck(cudaEventDestroy(start));
    cudaCheck(cudaEventDestroy(stop));
    cudaCheck(cudaFree(dA)); cudaCheck(cudaFree(dB)); cudaCheck(cudaFree(dC));
}

void perf_sigmoid() {
    constexpr int n4 = 32 * 1024 * 1024;  // 32M elements
    constexpr int warmup = 3;
    constexpr int iters = 100;

    float *dX, *dY;
    cudaCheck(cudaMalloc(&dX, n4 * sizeof(float)));
    cudaCheck(cudaMalloc(&dY, n4 * sizeof(float)));

    dim3 block(256);
    dim3 grid(CEIL_DIV(n4 / 4, block.x));

    // Warmup
    for (int i = 0; i < warmup; ++i) {
        sigmoid_float4<<<grid, block>>>(reinterpret_cast<const float4*>(dX),
                                        reinterpret_cast<float4*>(dY), n4);
    }
    cudaCheck(cudaDeviceSynchronize());

    // Timing
    cudaEvent_t start, stop;
    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&stop));

    cudaCheck(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        sigmoid_float4<<<grid, block>>>(reinterpret_cast<const float4*>(dX),
                                        reinterpret_cast<float4*>(dY), n4);
    }
    cudaCheck(cudaEventRecord(stop));
    cudaCheck(cudaEventSynchronize(stop));

    float ms;
    cudaCheck(cudaEventElapsedTime(&ms, start, stop));
    float avg_ms = ms / iters;
    float bandwidth_gb = (2.0f * n4 * sizeof(float)) / (avg_ms * 1e6); // 1 read + 1 write

    std::printf("sigmoid performance: %.3f ms/iter, %.2f GB/s\n", avg_ms, bandwidth_gb);

    cudaCheck(cudaEventDestroy(start));
    cudaCheck(cudaEventDestroy(stop));
    cudaCheck(cudaFree(dX)); cudaCheck(cudaFree(dY));
}

void perf_relu() {
    constexpr int n4 = 32 * 1024 * 1024;  // 32M elements
    constexpr int warmup = 3;
    constexpr int iters = 100;

    float *dX, *dY;
    cudaCheck(cudaMalloc(&dX, n4 * sizeof(float)));
    cudaCheck(cudaMalloc(&dY, n4 * sizeof(float)));

    dim3 block(256);
    dim3 grid(CEIL_DIV(n4 / 4, block.x));

    // Warmup
    for (int i = 0; i < warmup; ++i) {
        relu_float4<<<grid, block>>>(reinterpret_cast<const float4*>(dX),
                                     reinterpret_cast<float4*>(dY), n4);
    }
    cudaCheck(cudaDeviceSynchronize());

    // Timing
    cudaEvent_t start, stop;
    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&stop));

    cudaCheck(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        relu_float4<<<grid, block>>>(reinterpret_cast<const float4*>(dX),
                                     reinterpret_cast<float4*>(dY), n4);
    }
    cudaCheck(cudaEventRecord(stop));
    cudaCheck(cudaEventSynchronize(stop));

    float ms;
    cudaCheck(cudaEventElapsedTime(&ms, start, stop));
    float avg_ms = ms / iters;
    float bandwidth_gb = (2.0f * n4 * sizeof(float)) / (avg_ms * 1e6); // 1 read + 1 write

    std::printf("relu performance: %.3f ms/iter, %.2f GB/s\n", avg_ms, bandwidth_gb);

    cudaCheck(cudaEventDestroy(start));
    cudaCheck(cudaEventDestroy(stop));
    cudaCheck(cudaFree(dX)); cudaCheck(cudaFree(dY));
}

int main() {
    std::printf("=== Correctness Tests ===\n");
    test_elementwise_add();
    test_sigmoid();
    test_relu();
    std::printf("All elementwise tests passed.\n\n");

    std::printf("=== Performance Tests ===\n");
    perf_elementwise_add();
    perf_sigmoid();
    perf_relu();

    return 0;
}