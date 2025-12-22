#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <algorithm>

#include "../src/reduce/reduce.cuh"

#define cudaCheck(err) _cudaCheck((err), __FILE__, __LINE__)
inline void _cudaCheck(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        std::printf("[CUDA ERROR] at %s:%d\n%s\n", file, line, cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

struct PerfMetrics {
    float avg_ms;
    float bandwidth_gb;
    float gflops;
    int blocks;
    int threads_per_block;
};

// ============================================================================
// CPU Baseline Implementations
// ============================================================================

void softmax_cpu(const float* input, float* output, int N) {
    // Find max for numerical stability
    float max_val = input[0];
    for (int i = 1; i < N; ++i) {
        if (input[i] > max_val) max_val = input[i];
    }
    
    // Compute sum of exp(x - max)
    float sum = 0.0f;
    for (int i = 0; i < N; ++i) {
        sum += std::exp(input[i] - max_val);
    }
    
    // Normalize
    for (int i = 0; i < N; ++i) {
        output[i] = std::exp(input[i] - max_val) / sum;
    }
}

void softmax_row_cpu(const float* input, float* output, int M, int N) {
    for (int row = 0; row < M; ++row) {
        softmax_cpu(input + row * N, output + row * N, N);
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

void print_device_info() {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    std::printf("=== Device Information ===\n");
    std::printf("Device: %s\n", prop.name);
    std::printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    std::printf("SM Count: %d\n", prop.multiProcessorCount);
    std::printf("Peak Memory Bandwidth: %.2f GB/s\n\n", 
                2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
}

void print_metrics(const char* name, const PerfMetrics& m, int n) {
    std::printf("%-30s: %.4f ms/iter, %.2f GB/s", name, m.avg_ms, m.bandwidth_gb);
    if (m.gflops > 0) std::printf(", %.2f GFLOPS", m.gflops);
    std::printf("\n");
    std::printf("%-30s  Grid: %d blocks x %d threads\n", "", m.blocks, m.threads_per_block);
    std::printf("%-30s  Elements: %d\n\n", "", n);
}

bool close_enough(float a, float b, float tol = 1e-5f) {
    return std::fabs(a - b) <= tol * (1.0f + std::fabs(b));
}

// ============================================================================
// GPU Softmax Implementation (3-pass: max, sum, normalize)
// ============================================================================

void gpu_softmax_3pass(float* d_input, float* d_output, float* d_max, float* d_sum, int N, int block_size) {
    int grid_size = CEIL_DIV(N, block_size);
    
    // 1. Initialize max to -FLT_MAX and sum to 0
    float neg_max = -FLT_MAX;
    cudaCheck(cudaMemcpy(d_max, &neg_max, sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemset(d_sum, 0, sizeof(float)));
    
    // 2. Find max using max_kernel from reduce_max.cu
    max_kernel<<<grid_size, block_size>>>(d_input, d_max, N);
    cudaCheck(cudaGetLastError());
    
    // 3. Compute sum of exp(x - max)
    sum_kernel<<<grid_size, block_size>>>(d_input, d_sum, d_max, N);
    cudaCheck(cudaGetLastError());
    
    // 4. Normalize: output = exp(input - max) / sum
    softmax_kernel<<<grid_size, block_size>>>(d_input, d_output, d_sum, d_max, N);
    cudaCheck(cudaGetLastError());
}

// ============================================================================
// Correctness Tests
// ============================================================================

void test_softmax_1d() {
    std::printf("Testing softmax_1d (3-pass: max + sum + normalize)...\n");
    
    constexpr int N = 1024;
    constexpr int BLOCK = 256;
    
    // Allocate host memory
    std::vector<float> h_input(N);
    std::vector<float> h_output_cpu(N);
    std::vector<float> h_output_gpu(N);
    
    // Initialize input with values that test numerical stability
    for (int i = 0; i < N; ++i) {
        h_input[i] = static_cast<float>(i - N/2) * 0.01f;  // Range: -5.12 to 5.11
    }
    
    // CPU reference
    softmax_cpu(h_input.data(), h_output_cpu.data(), N);
    
    // GPU computation
    float *d_input, *d_output, *d_max, *d_sum;
    cudaCheck(cudaMalloc(&d_input, N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_output, N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_max, sizeof(float)));
    cudaCheck(cudaMalloc(&d_sum, sizeof(float)));
    
    cudaCheck(cudaMemcpy(d_input, h_input.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    
    gpu_softmax_3pass(d_input, d_output, d_max, d_sum, N, BLOCK);
    cudaCheck(cudaDeviceSynchronize());
    
    cudaCheck(cudaMemcpy(h_output_gpu.data(), d_output, N * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Verify results
    float max_diff = 0.0f;
    int fail_count = 0;
    for (int i = 0; i < N; ++i) {
        float diff = std::fabs(h_output_gpu[i] - h_output_cpu[i]);
        max_diff = std::max(max_diff, diff);
        if (!close_enough(h_output_gpu[i], h_output_cpu[i], 1e-4f)) {
            if (fail_count < 5) {
                std::printf("  Mismatch at %d: GPU=%f, CPU=%f, diff=%e\n", 
                           i, h_output_gpu[i], h_output_cpu[i], diff);
            }
            fail_count++;
        }
    }
    
    // Verify softmax properties
    float gpu_sum = 0.0f, cpu_sum = 0.0f;
    for (int i = 0; i < N; ++i) {
        gpu_sum += h_output_gpu[i];
        cpu_sum += h_output_cpu[i];
    }
    
    if (fail_count > 0) {
        std::printf("softmax_1d FAILED: %d mismatches, max_diff=%e\n", fail_count, max_diff);
        std::exit(EXIT_FAILURE);
    }
    
    std::printf("softmax_1d: PASS (max_diff=%e, sum=%.6f)\n", max_diff, gpu_sum);
    
    cudaCheck(cudaFree(d_input));
    cudaCheck(cudaFree(d_output));
    cudaCheck(cudaFree(d_max));
    cudaCheck(cudaFree(d_sum));
}

void test_softmax_row() {
    std::printf("Testing softmax_row_kernel (row-wise softmax)...\n");
    
    constexpr int M = 64;   // rows
    constexpr int N = 256;  // cols (must be <= 1024 for single-block row processing)
    
    std::vector<float> h_input(M * N);
    std::vector<float> h_output_cpu(M * N);
    std::vector<float> h_output_gpu(M * N);
    
    // Initialize
    for (int i = 0; i < M * N; ++i) {
        h_input[i] = static_cast<float>((i % N) - N/2) * 0.01f;
    }
    
    // CPU reference
    softmax_row_cpu(h_input.data(), h_output_cpu.data(), M, N);
    
    // GPU computation
    float *d_input, *d_output;
    cudaCheck(cudaMalloc(&d_input, M * N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_output, M * N * sizeof(float)));
    cudaCheck(cudaMemcpy(d_input, h_input.data(), M * N * sizeof(float), cudaMemcpyHostToDevice));
    
    // One block per row
    softmax_row_kernel<<<M, N>>>(d_input, d_output, M, N);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());
    
    cudaCheck(cudaMemcpy(h_output_gpu.data(), d_output, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Verify
    float max_diff = 0.0f;
    int fail_count = 0;
    for (int i = 0; i < M * N; ++i) {
        float diff = std::fabs(h_output_gpu[i] - h_output_cpu[i]);
        max_diff = std::max(max_diff, diff);
        if (!close_enough(h_output_gpu[i], h_output_cpu[i], 1e-4f)) {
            fail_count++;
        }
    }
    
    if (fail_count > 0) {
        std::printf("softmax_row_kernel FAILED: %d mismatches, max_diff=%e\n", fail_count, max_diff);
        std::exit(EXIT_FAILURE);
    }
    
    std::printf("softmax_row_kernel: PASS (max_diff=%e)\n", max_diff);
    
    cudaCheck(cudaFree(d_input));
    cudaCheck(cudaFree(d_output));
}

void test_softmax_row_shfl_xor() {
    std::printf("Testing softmax_row_kernel_shfl_xor (warp shuffle version)...\n");
    
    constexpr int M = 64;
    constexpr int N = 256;
    
    std::vector<float> h_input(M * N);
    std::vector<float> h_output_cpu(M * N);
    std::vector<float> h_output_gpu(M * N);
    
    for (int i = 0; i < M * N; ++i) {
        h_input[i] = static_cast<float>((i % N) - N/2) * 0.01f;
    }
    
    softmax_row_cpu(h_input.data(), h_output_cpu.data(), M, N);
    
    float *d_input, *d_output;
    cudaCheck(cudaMalloc(&d_input, M * N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_output, M * N * sizeof(float)));
    cudaCheck(cudaMemcpy(d_input, h_input.data(), M * N * sizeof(float), cudaMemcpyHostToDevice));
    
    softmax_row_kernel_shfl_xor<<<M, N>>>(d_input, d_output, M, N);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());
    
    cudaCheck(cudaMemcpy(h_output_gpu.data(), d_output, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    
    float max_diff = 0.0f;
    int fail_count = 0;
    for (int i = 0; i < M * N; ++i) {
        float diff = std::fabs(h_output_gpu[i] - h_output_cpu[i]);
        max_diff = std::max(max_diff, diff);
        if (!close_enough(h_output_gpu[i], h_output_cpu[i], 1e-4f)) {
            fail_count++;
        }
    }
    
    if (fail_count > 0) {
        std::printf("softmax_row_kernel_shfl_xor FAILED: %d mismatches, max_diff=%e\n", fail_count, max_diff);
        std::exit(EXIT_FAILURE);
    }
    
    std::printf("softmax_row_kernel_shfl_xor: PASS (max_diff=%e)\n", max_diff);
    
    cudaCheck(cudaFree(d_input));
    cudaCheck(cudaFree(d_output));
}

// ============================================================================
// Performance Tests
// ============================================================================

void perf_softmax_1d() {
    constexpr int N = 32 * 1024 * 1024;  // 32M elements
    constexpr int BLOCK = 256;
    constexpr int WARMUP = 3;
    constexpr int ITERS = 100;
    
    float *d_input, *d_output, *d_max, *d_sum;
    cudaCheck(cudaMalloc(&d_input, N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_output, N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_max, sizeof(float)));
    cudaCheck(cudaMalloc(&d_sum, sizeof(float)));
    
    // Initialize input
    std::vector<float> h_input(N);
    for (int i = 0; i < N; ++i) h_input[i] = static_cast<float>(i % 100) * 0.01f;
    cudaCheck(cudaMemcpy(d_input, h_input.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    
    // Warmup
    for (int i = 0; i < WARMUP; ++i) {
        gpu_softmax_3pass(d_input, d_output, d_max, d_sum, N, BLOCK);
    }
    cudaCheck(cudaDeviceSynchronize());
    
    // Timing
    cudaEvent_t start, stop;
    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&stop));
    
    cudaCheck(cudaEventRecord(start));
    for (int i = 0; i < ITERS; ++i) {
        gpu_softmax_3pass(d_input, d_output, d_max, d_sum, N, BLOCK);
    }
    cudaCheck(cudaEventRecord(stop));
    cudaCheck(cudaEventSynchronize(stop));
    
    float ms;
    cudaCheck(cudaEventElapsedTime(&ms, start, stop));
    
    PerfMetrics m;
    m.avg_ms = ms / ITERS;
    // 3 passes: read N for max, read N for sum, read N + write N for normalize
    m.bandwidth_gb = (4.0f * N * sizeof(float)) / (m.avg_ms * 1e6);
    m.gflops = (3.0f * N / 1e9f) / (m.avg_ms / 1000.0f);  // exp, sub, div per element
    m.blocks = CEIL_DIV(N, BLOCK);
    m.threads_per_block = BLOCK;
    print_metrics("softmax_1d (3-pass)", m, N);
    
    cudaCheck(cudaEventDestroy(start));
    cudaCheck(cudaEventDestroy(stop));
    cudaCheck(cudaFree(d_input));
    cudaCheck(cudaFree(d_output));
    cudaCheck(cudaFree(d_max));
    cudaCheck(cudaFree(d_sum));
}

void perf_softmax_row() {
    constexpr int M = 4096;   // rows
    constexpr int N = 1024;   // cols
    constexpr int WARMUP = 3;
    constexpr int ITERS = 100;
    const int total = M * N;
    
    float *d_input, *d_output;
    cudaCheck(cudaMalloc(&d_input, total * sizeof(float)));
    cudaCheck(cudaMalloc(&d_output, total * sizeof(float)));
    
    // Initialize
    std::vector<float> h_input(total);
    for (int i = 0; i < total; ++i) h_input[i] = static_cast<float>(i % 100) * 0.01f;
    cudaCheck(cudaMemcpy(d_input, h_input.data(), total * sizeof(float), cudaMemcpyHostToDevice));
    
    // Warmup
    for (int i = 0; i < WARMUP; ++i) {
        softmax_row_kernel<<<M, N>>>(d_input, d_output, M, N);
    }
    cudaCheck(cudaDeviceSynchronize());
    
    // Timing
    cudaEvent_t start, stop;
    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&stop));
    
    cudaCheck(cudaEventRecord(start));
    for (int i = 0; i < ITERS; ++i) {
        softmax_row_kernel<<<M, N>>>(d_input, d_output, M, N);
    }
    cudaCheck(cudaEventRecord(stop));
    cudaCheck(cudaEventSynchronize(stop));
    
    float ms;
    cudaCheck(cudaEventElapsedTime(&ms, start, stop));
    
    PerfMetrics m;
    m.avg_ms = ms / ITERS;
    m.bandwidth_gb = (2.0f * total * sizeof(float)) / (m.avg_ms * 1e6);  // read + write
    m.gflops = (3.0f * total / 1e9f) / (m.avg_ms / 1000.0f);
    m.blocks = M;
    m.threads_per_block = N;
    
    std::printf("%-30s: %.4f ms/iter, %.2f GB/s, %.2f GFLOPS\n", 
                "softmax_row_kernel", m.avg_ms, m.bandwidth_gb, m.gflops);
    std::printf("%-30s  Matrix: %d x %d = %d elements\n\n", "", M, N, total);
    
    cudaCheck(cudaEventDestroy(start));
    cudaCheck(cudaEventDestroy(stop));
    cudaCheck(cudaFree(d_input));
    cudaCheck(cudaFree(d_output));
}

void perf_softmax_row_shfl_xor() {
    constexpr int M = 4096;
    constexpr int N = 1024;
    constexpr int WARMUP = 3;
    constexpr int ITERS = 100;
    const int total = M * N;
    
    float *d_input, *d_output;
    cudaCheck(cudaMalloc(&d_input, total * sizeof(float)));
    cudaCheck(cudaMalloc(&d_output, total * sizeof(float)));
    
    std::vector<float> h_input(total);
    for (int i = 0; i < total; ++i) h_input[i] = static_cast<float>(i % 100) * 0.01f;
    cudaCheck(cudaMemcpy(d_input, h_input.data(), total * sizeof(float), cudaMemcpyHostToDevice));
    
    for (int i = 0; i < WARMUP; ++i) {
        softmax_row_kernel_shfl_xor<<<M, N>>>(d_input, d_output, M, N);
    }
    cudaCheck(cudaDeviceSynchronize());
    
    cudaEvent_t start, stop;
    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&stop));
    
    cudaCheck(cudaEventRecord(start));
    for (int i = 0; i < ITERS; ++i) {
        softmax_row_kernel_shfl_xor<<<M, N>>>(d_input, d_output, M, N);
    }
    cudaCheck(cudaEventRecord(stop));
    cudaCheck(cudaEventSynchronize(stop));
    
    float ms;
    cudaCheck(cudaEventElapsedTime(&ms, start, stop));
    
    PerfMetrics m;
    m.avg_ms = ms / ITERS;
    m.bandwidth_gb = (2.0f * total * sizeof(float)) / (m.avg_ms * 1e6);
    m.gflops = (3.0f * total / 1e9f) / (m.avg_ms / 1000.0f);
    m.blocks = M;
    m.threads_per_block = N;
    
    std::printf("%-30s: %.4f ms/iter, %.2f GB/s, %.2f GFLOPS\n", 
                "softmax_row_shfl_xor", m.avg_ms, m.bandwidth_gb, m.gflops);
    std::printf("%-30s  Matrix: %d x %d = %d elements\n\n", "", M, N, total);
    
    cudaCheck(cudaEventDestroy(start));
    cudaCheck(cudaEventDestroy(stop));
    cudaCheck(cudaFree(d_input));
    cudaCheck(cudaFree(d_output));
}

// ============================================================================
// CPU Baseline Comparison
// ============================================================================

void perf_cpu_baseline() {
    constexpr int N = 1024 * 1024;  // 1M elements
    
    std::vector<float> h_input(N);
    std::vector<float> h_output(N);
    
    for (int i = 0; i < N; ++i) {
        h_input[i] = static_cast<float>(i % 100) * 0.01f;
    }
    
    std::printf("\n=== CPU Baseline (1M elements) ===\n");
    
    auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < 10; ++iter) {
        softmax_cpu(h_input.data(), h_output.data(), N);
    }
    auto end = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(end - start).count() / 10.0;
    double cpu_bw = (2.0 * N * sizeof(float)) / (cpu_ms * 1e6);
    std::printf("SOFTMAX (CPU)        : %.3f ms/iter, %.2f GB/s\n", cpu_ms, cpu_bw);
}

// ============================================================================
// Main
// ============================================================================

int main() {
    print_device_info();
    
    std::printf("=== Correctness Tests ===\n");
    test_softmax_1d();
    test_softmax_row();
    test_softmax_row_shfl_xor();
    std::printf("\nAll softmax correctness tests passed.\n\n");
    
    std::printf("=== Performance Tests ===\n");
    perf_softmax_1d();
    perf_softmax_row();
    perf_softmax_row_shfl_xor();
    
    perf_cpu_baseline();
    
    std::printf("\n=== Profiling Commands ===\n");
    std::printf("For detailed kernel metrics, run:\n");
    std::printf("  ncu --set full --target-processes all -o softmax_profile ./build/test_softmax\n");
    std::printf("  nsys profile -o softmax_trace ./build/test_softmax\n\n");

    return 0;
}

