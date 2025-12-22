#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../src/reduce/reduce.cuh"

#define cudaCheck(err) _cudaCheck((err), __FILE__, __LINE__)
inline void _cudaCheck(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        std::printf("[CUDA ERROR] at %s:%d\n%s\n", file, line, cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

// ceiling division helper
#define CEIL(a, b) (((a) + (b) - 1) / (b))

struct PerfMetrics {
    float avg_ms;
    float bandwidth_gb;
    float gflops;
    int blocks;
    int threads_per_block;
};

void print_metrics(const char* name, const PerfMetrics& m, int n) {
    std::printf("%-20s: %.3f ms/iter, %.2f GB/s", name, m.avg_ms, m.bandwidth_gb);
    if (m.gflops > 0) std::printf(", %.2f GFLOPS", m.gflops);
    std::printf("\n");
    std::printf("%-20s  Grid: %d blocks x %d threads\n", "", m.blocks, m.threads_per_block);
    std::printf("%-20s  Elements: %d\n\n", "", n);
}

// Aggregate all sum kernel checks in one place
void test_sum_kernels() {
    // sum_v0: one-block sanity
    {
        constexpr int N0 = 256;
        constexpr int BLOCK0 = 256;
        float h_in[N0];
        for (int i = 0; i < N0; ++i) h_in[i] = static_cast<float>(i);

        float h_out = -1.0f;
        float *d_in = nullptr, *d_out = nullptr;
        cudaCheck(cudaMalloc(&d_in, N0 * sizeof(float)));
        cudaCheck(cudaMalloc(&d_out, sizeof(float)));
        cudaCheck(cudaMemcpy(d_in, h_in, N0 * sizeof(float), cudaMemcpyHostToDevice));

        sum_v0<<<1, BLOCK0>>>(d_in, d_out);
        cudaCheck(cudaGetLastError());
        cudaCheck(cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost));

        const float expected = (N0 - 1) * N0 / 2.0f;
        if (std::fabs(h_out - expected) > 1e-3f) {
            std::printf("sum_v0 failed: got %f expected %f\n", h_out, expected);
            std::exit(EXIT_FAILURE);
        }
        std::printf("sum_v0: PASS (value=%f)\n", h_out);

        cudaCheck(cudaFree(d_in));
        cudaCheck(cudaFree(d_out));
    }

    // Shared setup for the remaining variants
    constexpr int N = 1024;
    constexpr int BLOCK = 256;
    const float expected = (N - 1) * N / 2.0f;

    std::vector<float> h_in(N);
    for (int i = 0; i < N; ++i) h_in[i] = static_cast<float>(i);

    float *d_in = nullptr, *d_out = nullptr;
    const int grid_max = CEIL(N, BLOCK);
    cudaCheck(cudaMalloc(&d_in, N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_out, grid_max * sizeof(float)));
    cudaCheck(cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    std::vector<float> h_out(grid_max, 0.0f);

    auto check_result = [&](const char* name, float value) {
        if (std::fabs(value - expected) > 1e-3f) {
            std::printf("%s failed: got %f expected %f\n", name, value, expected);
            std::exit(EXIT_FAILURE);
        }
        std::printf("%s: PASS (value=%f)\n", name, value);
    };

    // sum_v2: dynamic shared, per-block partials
    {
        const int grid = CEIL(N, BLOCK);
        sum_v2<<<grid, BLOCK, BLOCK * sizeof(float)>>>(d_in, d_out, N);
        cudaCheck(cudaGetLastError());
        cudaCheck(cudaMemcpy(h_out.data(), d_out, grid * sizeof(float), cudaMemcpyDeviceToHost));
        float acc = 0.0f;
        for (int i = 0; i < grid; ++i) acc += h_out[i];
        check_result("sum_v2", acc);
    }

    // sum_v3: dynamic shared + atomic to single output
    {
        const int grid = CEIL(N, BLOCK);
        cudaCheck(cudaMemset(d_out, 0, sizeof(float)));
        sum_v3<<<grid, BLOCK, BLOCK * sizeof(float)>>>(d_in, d_out, N);
        cudaCheck(cudaGetLastError());
        float acc = 0.0f;
        cudaCheck(cudaMemcpy(&acc, d_out, sizeof(float), cudaMemcpyDeviceToHost));
        check_result("sum_v3", acc);
    }

    // sum_v4: warp shuffle producing per-block partials
    {
        const int grid = CEIL(N, BLOCK);
        sum_v4<<<grid, BLOCK>>>(d_in, d_out, N);
        cudaCheck(cudaGetLastError());
        cudaCheck(cudaMemcpy(h_out.data(), d_out, grid * sizeof(float), cudaMemcpyDeviceToHost));
        float acc = 0.0f;
        for (int i = 0; i < grid; ++i) acc += h_out[i];
        check_result("sum_v4", acc);
    }

    // sum_v5: float4 load + warp shuffle, per-block partials
    {
        const int grid = CEIL(N, BLOCK * 4);
        sum_v5<<<grid, BLOCK>>>(d_in, d_out, N);
        cudaCheck(cudaGetLastError());
        cudaCheck(cudaMemcpy(h_out.data(), d_out, grid * sizeof(float), cudaMemcpyDeviceToHost));
        float acc = 0.0f;
        for (int i = 0; i < grid; ++i) acc += h_out[i];
        check_result("sum_v5", acc);
    }

    cudaCheck(cudaFree(d_in));
    cudaCheck(cudaFree(d_out));
}

// Simple sanity test for max_kernel
void test_max_kernel() {
    constexpr int N = 512;
    constexpr int BLOCK = 256;
    const int GRID = (N + BLOCK - 1) / BLOCK;

    float *h_in = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; ++i) h_in[i] = -1000.0f + static_cast<float>(i);
    float h_out = -1.0f;

    float *d_in = nullptr, *d_out = nullptr;
    cudaCheck(cudaMalloc(&d_in, N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_out, sizeof(float)));
    cudaCheck(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));

    max_kernel<<<GRID, BLOCK>>>(d_in, d_out, N);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost));

    const float expected = h_in[N - 1];
    if (std::fabs(h_out - expected) > 1e-3f) {
        std::printf("max_kernel failed: got %f expected %f\n", h_out, expected);
        std::exit(EXIT_FAILURE);
    }
    std::printf("max_kernel: PASS (value=%f)\n", h_out);

    cudaCheck(cudaFree(d_in));
    cudaCheck(cudaFree(d_out));
    free(h_in);
}

void perf_reduce_sum() {
    constexpr int N = 32 * 1024 * 1024;   // 32M elements
    constexpr int BLOCK = 256;
    constexpr int WARMUP = 3;
    constexpr int ITERS = 100;

    float *d_in = nullptr;
    float *d_out_v3 = nullptr;
    float *d_out_v5 = nullptr;
    const int grid_v3 = CEIL(N, BLOCK);
    const int grid_v5 = CEIL(N, BLOCK * 4);

    cudaCheck(cudaMalloc(&d_in, N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_out_v3, sizeof(float)));
    cudaCheck(cudaMalloc(&d_out_v5, grid_v5 * sizeof(float)));

    // Fill input with a simple pattern
    std::vector<float> h_in(N);
    for (int i = 0; i < N; ++i) h_in[i] = static_cast<float>(i % 13);
    cudaCheck(cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    // Warmup sum_v3 (atomic to single output)
    for (int i = 0; i < WARMUP; ++i) {
        cudaCheck(cudaMemset(d_out_v3, 0, sizeof(float)));
        sum_v3<<<grid_v3, BLOCK, BLOCK * sizeof(float)>>>(d_in, d_out_v3, N);
    }
    cudaCheck(cudaDeviceSynchronize());

    // Time sum_v3
    cudaEvent_t start, stop;
    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&stop));

    cudaCheck(cudaEventRecord(start));
    for (int i = 0; i < ITERS; ++i) {
        cudaCheck(cudaMemset(d_out_v3, 0, sizeof(float)));
        sum_v3<<<grid_v3, BLOCK, BLOCK * sizeof(float)>>>(d_in, d_out_v3, N);
    }
    cudaCheck(cudaEventRecord(stop));
    cudaCheck(cudaEventSynchronize(stop));

    float ms_v3 = 0.0f;
    cudaCheck(cudaEventElapsedTime(&ms_v3, start, stop));
    float sum_v3_host = 0.0f;
    cudaCheck(cudaMemcpy(&sum_v3_host, d_out_v3, sizeof(float), cudaMemcpyDeviceToHost));

    PerfMetrics m3;
    m3.avg_ms = ms_v3 / ITERS;
    m3.bandwidth_gb = (N * sizeof(float)) / (m3.avg_ms * 1e6);
    m3.gflops = (N / 1e9f) / (m3.avg_ms / 1000.0f);
    m3.blocks = grid_v3;
    m3.threads_per_block = BLOCK;
    print_metrics("sum_v3 (atomic)", m3, N);

    // Warmup sum_v5 (float4 + warp shuffle producing partials)
    for (int i = 0; i < WARMUP; ++i) {
        sum_v5<<<grid_v5, BLOCK>>>(d_in, d_out_v5, N);
    }
    cudaCheck(cudaDeviceSynchronize());

    // Time sum_v5
    cudaCheck(cudaEventRecord(start));
    for (int i = 0; i < ITERS; ++i) {
        sum_v5<<<grid_v5, BLOCK>>>(d_in, d_out_v5, N);
    }
    cudaCheck(cudaEventRecord(stop));
    cudaCheck(cudaEventSynchronize(stop));

    float ms_v5 = 0.0f;
    cudaCheck(cudaEventElapsedTime(&ms_v5, start, stop));
    std::vector<float> h_out_v5(grid_v5);
    cudaCheck(cudaMemcpy(h_out_v5.data(), d_out_v5, grid_v5 * sizeof(float), cudaMemcpyDeviceToHost));
    float sum_v5_host = 0.0f;
    for (int i = 0; i < grid_v5; ++i) sum_v5_host += h_out_v5[i];

    PerfMetrics m5;
    m5.avg_ms = ms_v5 / ITERS;
    m5.bandwidth_gb = (N * sizeof(float)) / (m5.avg_ms * 1e6);
    m5.gflops = (N / 1e9f) / (m5.avg_ms / 1000.0f);
    m5.blocks = grid_v5;
    m5.threads_per_block = BLOCK;
    print_metrics("sum_v5 (float4)", m5, N);

    const int cycles = N / 13;
    const int rem = N % 13;
    const double expected = static_cast<double>(cycles) * 78.0 + static_cast<double>(rem) * (rem - 1) / 2.0;
    auto validate = [&](const char* name, double got) {
        if (std::fabs(got - expected) > 1e-2 * std::fabs(expected)) {
            std::printf("%s failed validation: got %.4f expected %.4f\n", name, got, expected);
            std::exit(EXIT_FAILURE);
        }
        std::printf("%s validation: PASS (value=%.4f)\n", name, got);
    };

    validate("sum_v3", static_cast<double>(sum_v3_host));
    validate("sum_v5", static_cast<double>(sum_v5_host));

    cudaCheck(cudaEventDestroy(start));
    cudaCheck(cudaEventDestroy(stop));
    cudaCheck(cudaFree(d_in));
    cudaCheck(cudaFree(d_out_v3));
    cudaCheck(cudaFree(d_out_v5));
}

int main() {
    test_sum_kernels();
    test_max_kernel();
    std::printf("\n=== Performance Tests (32M elements) ===\n");
    perf_reduce_sum();
    std::printf("Reduction tests completed.\n");
    return 0;
}
