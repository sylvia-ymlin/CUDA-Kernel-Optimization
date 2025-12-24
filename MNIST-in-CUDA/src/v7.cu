/**
 * v7.cu - Custom Fused GEMM (Educational)
 * 
 * Key change from v6:
 * - Custom tiled shared memory GEMM with fused bias + ReLU epilogue
 * - cuBLAS only for backward pass (need reliable gradients)
 * - All v6 optimizations retained (streams, pinned memory, double buffering)
 * 
 * Purpose: Demonstrate how kernel fusion works at the CUDA level
 * Note: Expected to be SLOWER than v6 — shows why cuBLAS/CUTLASS exist
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

typedef struct {
    double memory_transfers;
    double gpu_compute;
    double sync_wait;
    double total_time;
} TimingStats;

#define INPUT_SIZE 784
#define HIDDEN_SIZE 1024
#define OUTPUT_SIZE 10
#define TRAIN_SIZE 10000
#define TEST_SIZE 10000
#define BATCH_SIZE 32
#define EPOCHS 10
#define LEARNING_RATE 0.01
#define NUM_STREAMS 2

// ============================================================
// [NEW in v7] Tile size for custom GEMM
// 32x32 is a common choice: fits in shared memory, good occupancy
// ============================================================
#define TILE_SIZE 32

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s (%d)\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error), error); \
            cudaDeviceReset(); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, status); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

double get_time_diff(struct timespec start, struct timespec end) {
    return (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
}

typedef struct {
    float *d_weights1, *d_weights2, *d_bias1, *d_bias2;
    float *d_grad_weights1, *d_grad_weights2, *d_grad_bias1, *d_grad_bias2;
    
    float *d_fc1_output[NUM_STREAMS];
    float *d_fc2_output[NUM_STREAMS];
    float *d_grad_hidden[NUM_STREAMS];
    float *d_grad_output[NUM_STREAMS];
    float *d_input_batch[NUM_STREAMS];
    int *d_labels[NUM_STREAMS];
    float *d_loss[NUM_STREAMS];

    cudaStream_t streams[NUM_STREAMS];
    cublasHandle_t cublas_handle;
} NeuralNetworkCUDA;

void load_data(const char *filename, float *data, int size) {
    FILE *f = fopen(filename, "rb");
    if (!f) { perror("fopen data"); exit(EXIT_FAILURE); }
    if (fread(data, sizeof(float), size, f) != (size_t)size) {
        perror("fread data"); exit(EXIT_FAILURE);
    }
    fclose(f);
}

void load_labels(const char *filename, int *labels, int size) {
    FILE *f = fopen(filename, "rb");
    if (!f) { perror("fopen labels"); exit(EXIT_FAILURE); }
    if (fread(labels, sizeof(int), size, f) != (size_t)size) {
        perror("fread labels"); exit(EXIT_FAILURE);
    }
    fclose(f);
}

void normalize_data(float *data, int size) {
    const float mean = 0.1307f;
    const float std = 0.3081f;
    for (int i = 0; i < size; i++) {
        data[i] = (data[i] - mean) / std;
    }
}

void initialize_weights_host(float *weights, int rows, int cols) {
    float scale = sqrtf(2.0f / rows);
    for (int i = 0; i < rows * cols; i++) {
        weights[i] = ((float)rand() / RAND_MAX) * 2.0f * scale - scale;
    }
}

void initialize_bias_host(float *bias, int size) {
    memset(bias, 0, size * sizeof(float));
}

// ============================================================
// [NEW in v7] CUSTOM FUSED GEMM + BIAS + RELU KERNEL
// 
// Computes: C[M,N] = ReLU(A[M,K] @ B[K,N] + bias[N])
// 
// Tiled shared memory approach:
// - Each thread block computes a TILE_SIZE x TILE_SIZE output tile
// - Loads tiles of A and B into shared memory
// - Reduces global memory bandwidth by TILE_SIZE factor
// 
// Epilogue fusion:
// - After computing C[i,j], immediately add bias and apply ReLU
// - Saves 2 global memory round-trips (write C, read C for bias, read C for ReLU)
// ============================================================
__global__ void fused_gemm_bias_relu_kernel(
    const float* __restrict__ A,      // [M, K] - input (row-major)
    const float* __restrict__ B,      // [K, N] - weights (column-major for cuBLAS compat)
    const float* __restrict__ bias,   // [N]
    float* __restrict__ C,            // [M, N] - output (column-major)
    int M, int K, int N,
    bool apply_relu                   // false for output layer
) {
    // Shared memory tiles
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    // Output coordinates (column-major: C is stored column-wise like cuBLAS)
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;  // row in output
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;  // col in output
    
    float sum = 0.0f;
    
    // Loop over tiles
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < num_tiles; t++) {
        // Load A tile: A[row, t*TILE + threadIdx.x]
        // A is row-major: A[i,j] = A[i * K + j]
        int a_col = t * TILE_SIZE + threadIdx.x;
        if (row < M && a_col < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + a_col];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Load B tile: B[t*TILE + threadIdx.y, col]
        // B is column-major (like cuBLAS): B[i,j] = B[i + j * K]
        int b_row = t * TILE_SIZE + threadIdx.y;
        if (b_row < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[b_row + col * K];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    // ============================================================
    // EPILOGUE: Fused bias + ReLU (no extra memory traffic!)
    // Without fusion: write sum, load sum, add bias, write, load, ReLU, write
    // With fusion: add bias in register, ReLU in register, single write
    // ============================================================
    if (row < M && col < N) {
        sum += bias[col];  // Add bias
        if (apply_relu) {
            sum = fmaxf(0.0f, sum);  // ReLU
        }
        // C is column-major: C[i,j] = C[i + j * M]
        C[row + col * M] = sum;
    }
}

// ============================================================
// [NEW in v7] Simple GEMM without epilogue (for layers that need it)
// Used for: FC2 output (no ReLU)
// ============================================================
__global__ void fused_gemm_bias_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,
    float* __restrict__ C,
    int M, int K, int N
) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < num_tiles; t++) {
        int a_col = t * TILE_SIZE + threadIdx.x;
        if (row < M && a_col < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + a_col];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        int b_row = t * TILE_SIZE + threadIdx.y;
        if (b_row < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[b_row + col * K];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        sum += bias[col];
        C[row + col * M] = sum;
    }
}

// Backward kernels (same as v6)
__global__ void relu_backward_kernel(float *grad, float *x, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        grad[idx] *= (x[idx] > 0.0f ? 1.0f : 0.0f);
    }
}

__global__ void bias_backward_kernel(float *grad_output, float *grad_bias, int batch, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch * size) {
        int bias_idx = idx % size;
        atomicAdd(&grad_bias[bias_idx], grad_output[idx]);
    }
}

__global__ void softmax_cross_entropy_backward_kernel(
    float *logits, int *labels, float *grad_output, float *loss_per_sample,
    int batch_size, int num_classes
) {
    int b = blockIdx.x;
    if (b >= batch_size) return;

    extern __shared__ float shared[];
    float *sample_logits = shared;
    int tid = threadIdx.x;

    if (tid < num_classes) {
        sample_logits[tid] = logits[b * num_classes + tid];
    }
    __syncthreads();

    __shared__ float max_logit;
    __shared__ float sum_exp;

    if (tid == 0) {
        max_logit = sample_logits[0];
        for (int i = 1; i < num_classes; i++) {
            if (sample_logits[i] > max_logit) max_logit = sample_logits[i];
        }
    }
    __syncthreads();

    if (tid < num_classes) {
        sample_logits[tid] = expf(sample_logits[tid] - max_logit);
    }
    __syncthreads();

    if (tid == 0) {
        sum_exp = 0.0f;
        for (int i = 0; i < num_classes; i++) {
            sum_exp += sample_logits[i];
        }
    }
    __syncthreads();

    if (tid < num_classes) {
        float prob = sample_logits[tid] / sum_exp;
        int label = labels[b];
        float grad = prob;
        if (tid == label) {
            grad -= 1.0f;
        }
        grad /= (float)batch_size;
        grad_output[b * num_classes + tid] = grad;

        if (tid == label) {
            loss_per_sample[b] = -logf(fmaxf(prob, 1e-7f));
        }
    }
}

// ============================================================
// [NEW in v7] Forward pass using CUSTOM FUSED GEMM
// v6: cuBLAS Sgemm + bias_add_relu_kernel (2 ops per layer)
// v7: fused_gemm_bias_relu_kernel (1 op per layer)
// ============================================================
void forward_pass_stream(NeuralNetworkCUDA *nn, int stream_idx, int batch_size) {
    cudaStream_t stream = nn->streams[stream_idx];
    
    // FC1: output = ReLU(input @ W1 + b1)
    // Grid: cover [batch_size, HIDDEN_SIZE] output
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid1((HIDDEN_SIZE + TILE_SIZE - 1) / TILE_SIZE,
               (batch_size + TILE_SIZE - 1) / TILE_SIZE);
    
    fused_gemm_bias_relu_kernel<<<grid1, block, 0, stream>>>(
        nn->d_input_batch[stream_idx],  // A: [batch_size, INPUT_SIZE] row-major
        nn->d_weights1,                  // B: [INPUT_SIZE, HIDDEN_SIZE] col-major
        nn->d_bias1,                     // bias: [HIDDEN_SIZE]
        nn->d_fc1_output[stream_idx],   // C: [batch_size, HIDDEN_SIZE] col-major
        batch_size, INPUT_SIZE, HIDDEN_SIZE,
        true  // apply ReLU
    );

    // FC2: output = hidden @ W2 + b2 (no ReLU)
    dim3 grid2((OUTPUT_SIZE + TILE_SIZE - 1) / TILE_SIZE,
               (batch_size + TILE_SIZE - 1) / TILE_SIZE);
    
    fused_gemm_bias_kernel<<<grid2, block, 0, stream>>>(
        nn->d_fc1_output[stream_idx],   // A: [batch_size, HIDDEN_SIZE] - need row-major view
        nn->d_weights2,                  // B: [HIDDEN_SIZE, OUTPUT_SIZE] col-major
        nn->d_bias2,                     // bias: [OUTPUT_SIZE]
        nn->d_fc2_output[stream_idx],   // C: [batch_size, OUTPUT_SIZE] col-major
        batch_size, HIDDEN_SIZE, OUTPUT_SIZE
    );
}

// Backward pass using cuBLAS (same as v6, need reliable gradients)
void backward_pass_stream(NeuralNetworkCUDA *nn, int stream_idx, int batch_size) {
    const float alpha = 1.0f, beta = 0.0f;
    cudaStream_t stream = nn->streams[stream_idx];
    
    CUBLAS_CHECK(cublasSetStream(nn->cublas_handle, stream));
    
    CUDA_CHECK(cudaMemsetAsync(nn->d_grad_weights1, 0, INPUT_SIZE * HIDDEN_SIZE * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(nn->d_grad_weights2, 0, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(nn->d_grad_bias1, 0, HIDDEN_SIZE * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(nn->d_grad_bias2, 0, OUTPUT_SIZE * sizeof(float), stream));

    // dW2 = hidden.T @ grad_output
    CUBLAS_CHECK(cublasSgemm(nn->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                           OUTPUT_SIZE, HIDDEN_SIZE, batch_size,
                           &alpha, nn->d_grad_output[stream_idx], OUTPUT_SIZE,
                           nn->d_fc1_output[stream_idx], HIDDEN_SIZE, &beta,
                           nn->d_grad_weights2, OUTPUT_SIZE));

    int total_out = batch_size * OUTPUT_SIZE;
    int grid_out = (total_out + 255) / 256;
    bias_backward_kernel<<<grid_out, 256, 0, stream>>>(
        nn->d_grad_output[stream_idx], nn->d_grad_bias2, batch_size, OUTPUT_SIZE);

    // grad_hidden = grad_output @ W2.T
    CUBLAS_CHECK(cublasSgemm(nn->cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                           HIDDEN_SIZE, batch_size, OUTPUT_SIZE,
                           &alpha, nn->d_weights2, OUTPUT_SIZE,
                           nn->d_grad_output[stream_idx], OUTPUT_SIZE, &beta,
                           nn->d_grad_hidden[stream_idx], HIDDEN_SIZE));

    // ReLU backward
    int total_hidden = batch_size * HIDDEN_SIZE;
    int grid_hidden = (total_hidden + 255) / 256;
    relu_backward_kernel<<<grid_hidden, 256, 0, stream>>>(
        nn->d_grad_hidden[stream_idx], nn->d_fc1_output[stream_idx], total_hidden);

    // dW1 = input.T @ grad_hidden
    CUBLAS_CHECK(cublasSgemm(nn->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                           HIDDEN_SIZE, INPUT_SIZE, batch_size,
                           &alpha, nn->d_grad_hidden[stream_idx], HIDDEN_SIZE,
                           nn->d_input_batch[stream_idx], INPUT_SIZE, &beta,
                           nn->d_grad_weights1, HIDDEN_SIZE));

    bias_backward_kernel<<<grid_hidden, 256, 0, stream>>>(
        nn->d_grad_hidden[stream_idx], nn->d_grad_bias1, batch_size, HIDDEN_SIZE);
}

void update_weights_stream(NeuralNetworkCUDA *nn, float lr, int stream_idx) {
    float neg_lr = -lr;
    cudaStream_t stream = nn->streams[stream_idx];
    
    CUBLAS_CHECK(cublasSetStream(nn->cublas_handle, stream));
    
    CUBLAS_CHECK(cublasSaxpy(nn->cublas_handle, INPUT_SIZE * HIDDEN_SIZE,
                           &neg_lr, nn->d_grad_weights1, 1, nn->d_weights1, 1));
    CUBLAS_CHECK(cublasSaxpy(nn->cublas_handle, HIDDEN_SIZE * OUTPUT_SIZE,
                           &neg_lr, nn->d_grad_weights2, 1, nn->d_weights2, 1));
    CUBLAS_CHECK(cublasSaxpy(nn->cublas_handle, HIDDEN_SIZE,
                           &neg_lr, nn->d_grad_bias1, 1, nn->d_bias1, 1));
    CUBLAS_CHECK(cublasSaxpy(nn->cublas_handle, OUTPUT_SIZE,
                           &neg_lr, nn->d_grad_bias2, 1, nn->d_bias2, 1));
}

float compute_loss_on_gpu_stream(NeuralNetworkCUDA *nn, int stream_idx, int batch_size, float *h_loss) {
    cudaStream_t stream = nn->streams[stream_idx];
    int shared_mem = OUTPUT_SIZE * sizeof(float);
    
    softmax_cross_entropy_backward_kernel<<<batch_size, 32, shared_mem, stream>>>(
        nn->d_fc2_output[stream_idx], nn->d_labels[stream_idx], 
        nn->d_grad_output[stream_idx], nn->d_loss[stream_idx],
        batch_size, OUTPUT_SIZE);

    CUDA_CHECK(cudaMemcpyAsync(h_loss, nn->d_loss[stream_idx], 
                               batch_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
    
    return 0.0f;
}

void initialize_random_weights_cuda(NeuralNetworkCUDA *nn) {
    float *h_weights1 = (float *)malloc(INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    initialize_weights_host(h_weights1, INPUT_SIZE, HIDDEN_SIZE);
    CUDA_CHECK(cudaMemcpy(nn->d_weights1, h_weights1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    free(h_weights1);

    float *h_weights2 = (float *)malloc(HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float));
    initialize_weights_host(h_weights2, HIDDEN_SIZE, OUTPUT_SIZE);
    CUDA_CHECK(cudaMemcpy(nn->d_weights2, h_weights2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    free(h_weights2);

    float *h_bias1 = (float *)malloc(HIDDEN_SIZE * sizeof(float));
    initialize_bias_host(h_bias1, HIDDEN_SIZE);
    CUDA_CHECK(cudaMemcpy(nn->d_bias1, h_bias1, HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    free(h_bias1);

    float *h_bias2 = (float *)malloc(OUTPUT_SIZE * sizeof(float));
    initialize_bias_host(h_bias2, OUTPUT_SIZE);
    CUDA_CHECK(cudaMemcpy(nn->d_bias2, h_bias2, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    free(h_bias2);
}

void initialize_nn_cuda(NeuralNetworkCUDA *nn) {
    CUDA_CHECK(cudaMalloc(&nn->d_weights1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_weights2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_bias1, HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_bias2, OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_grad_weights1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_grad_weights2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_grad_bias1, HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nn->d_grad_bias2, OUTPUT_SIZE * sizeof(float)));

    for (int i = 0; i < NUM_STREAMS; i++) {
        CUDA_CHECK(cudaMalloc(&nn->d_fc1_output[i], BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&nn->d_fc2_output[i], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&nn->d_grad_hidden[i], BATCH_SIZE * HIDDEN_SIZE * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&nn->d_grad_output[i], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&nn->d_input_batch[i], BATCH_SIZE * INPUT_SIZE * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&nn->d_labels[i], BATCH_SIZE * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&nn->d_loss[i], BATCH_SIZE * sizeof(float)));
        
        CUDA_CHECK(cudaStreamCreate(&nn->streams[i]));
    }

    CUBLAS_CHECK(cublasCreate(&nn->cublas_handle));
    initialize_random_weights_cuda(nn);
}

void free_nn_cuda(NeuralNetworkCUDA *nn) {
    CUDA_CHECK(cudaFree(nn->d_weights1));
    CUDA_CHECK(cudaFree(nn->d_weights2));
    CUDA_CHECK(cudaFree(nn->d_bias1));
    CUDA_CHECK(cudaFree(nn->d_bias2));
    CUDA_CHECK(cudaFree(nn->d_grad_weights1));
    CUDA_CHECK(cudaFree(nn->d_grad_weights2));
    CUDA_CHECK(cudaFree(nn->d_grad_bias1));
    CUDA_CHECK(cudaFree(nn->d_grad_bias2));

    for (int i = 0; i < NUM_STREAMS; i++) {
        CUDA_CHECK(cudaFree(nn->d_fc1_output[i]));
        CUDA_CHECK(cudaFree(nn->d_fc2_output[i]));
        CUDA_CHECK(cudaFree(nn->d_grad_hidden[i]));
        CUDA_CHECK(cudaFree(nn->d_grad_output[i]));
        CUDA_CHECK(cudaFree(nn->d_input_batch[i]));
        CUDA_CHECK(cudaFree(nn->d_labels[i]));
        CUDA_CHECK(cudaFree(nn->d_loss[i]));
        CUDA_CHECK(cudaStreamDestroy(nn->streams[i]));
    }

    CUBLAS_CHECK(cublasDestroy(nn->cublas_handle));
}

// Evaluate model accuracy on test set (CPU-side, not timed)
void evaluate(NeuralNetworkCUDA *nn, float *X_test, int *y_test) {
    int correct = 0;
    int num_batches = TEST_SIZE / BATCH_SIZE;
    
    float *hidden = (float *)malloc(BATCH_SIZE * HIDDEN_SIZE * sizeof(float));
    float *output = (float *)malloc(BATCH_SIZE * OUTPUT_SIZE * sizeof(float));
    
    float *h_W1 = (float *)malloc(INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    float *h_W2 = (float *)malloc(HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float));
    float *h_b1 = (float *)malloc(HIDDEN_SIZE * sizeof(float));
    float *h_b2 = (float *)malloc(OUTPUT_SIZE * sizeof(float));
    
    CUDA_CHECK(cudaMemcpy(h_W1, nn->d_weights1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_W2, nn->d_weights2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_b1, nn->d_bias1, HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_b2, nn->d_bias2, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    
    for (int batch = 0; batch < num_batches; batch++) {
        float *batch_x = X_test + batch * BATCH_SIZE * INPUT_SIZE;
        int *batch_y = y_test + batch * BATCH_SIZE;
        
        // Forward: hidden = relu(X @ W1 + b1)
        // Note: weights are column-major from GPU, so access pattern differs
        for (int i = 0; i < BATCH_SIZE; i++) {
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                float sum = h_b1[j];
                for (int k = 0; k < INPUT_SIZE; k++) {
                    // W1 is column-major: W1[k,j] = h_W1[k + j * INPUT_SIZE]
                    sum += batch_x[i * INPUT_SIZE + k] * h_W1[k + j * INPUT_SIZE];
                }
                hidden[i * HIDDEN_SIZE + j] = fmaxf(0.0f, sum);
            }
        }
        
        // Forward: output = hidden @ W2 + b2
        for (int i = 0; i < BATCH_SIZE; i++) {
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                float sum = h_b2[j];
                for (int k = 0; k < HIDDEN_SIZE; k++) {
                    // W2 is column-major: W2[k,j] = h_W2[k + j * HIDDEN_SIZE]
                    sum += hidden[i * HIDDEN_SIZE + k] * h_W2[k + j * HIDDEN_SIZE];
                }
                output[i * OUTPUT_SIZE + j] = sum;
            }
        }
        
        // Count correct predictions
        for (int i = 0; i < BATCH_SIZE; i++) {
            int pred = 0;
            float max_val = output[i * OUTPUT_SIZE];
            for (int j = 1; j < OUTPUT_SIZE; j++) {
                if (output[i * OUTPUT_SIZE + j] > max_val) {
                    max_val = output[i * OUTPUT_SIZE + j];
                    pred = j;
                }
            }
            if (pred == batch_y[i]) correct++;
        }
    }
    
    free(hidden); free(output);
    free(h_W1); free(h_W2); free(h_b1); free(h_b2);
    
    float accuracy = 100.0f * correct / (num_batches * BATCH_SIZE);
    printf("Test Accuracy: %.2f%%\n", accuracy);
}

int main() {
    srand(12345);

    float *train_data;
    int *train_labels;
    CUDA_CHECK(cudaMallocHost(&train_data, TRAIN_SIZE * INPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&train_labels, TRAIN_SIZE * sizeof(int)));
    
    float *test_data = (float *)malloc(TEST_SIZE * INPUT_SIZE * sizeof(float));
    int *test_labels = (int *)malloc(TEST_SIZE * sizeof(int));
    
    load_data("./data/X_train.bin", train_data, TRAIN_SIZE * INPUT_SIZE);
    normalize_data(train_data, TRAIN_SIZE * INPUT_SIZE);
    load_labels("./data/y_train.bin", train_labels, TRAIN_SIZE);
    load_data("./data/X_test.bin", test_data, TEST_SIZE * INPUT_SIZE);
    normalize_data(test_data, TEST_SIZE * INPUT_SIZE);
    load_labels("./data/y_test.bin", test_labels, TEST_SIZE);

    float *h_loss[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        CUDA_CHECK(cudaMallocHost(&h_loss[i], BATCH_SIZE * sizeof(float)));
    }

    NeuralNetworkCUDA nn;
    initialize_nn_cuda(&nn);

    int num_batches = TRAIN_SIZE / BATCH_SIZE;
    
    TimingStats stats = {0};
    struct timespec start, end, step_start, step_end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        float total_loss = 0.0f;
        
        for (int batch = 0; batch < num_batches; batch++) {
            int stream_idx = batch % NUM_STREAMS;
            float *batch_input = train_data + batch * BATCH_SIZE * INPUT_SIZE;
            int *batch_labels_ptr = train_labels + batch * BATCH_SIZE;

            if (batch >= NUM_STREAMS) {
                clock_gettime(CLOCK_MONOTONIC, &step_start);
                CUDA_CHECK(cudaStreamSynchronize(nn.streams[stream_idx]));
                clock_gettime(CLOCK_MONOTONIC, &step_end);
                stats.sync_wait += get_time_diff(step_start, step_end);
                
                for (int i = 0; i < BATCH_SIZE; i++) {
                    total_loss += h_loss[stream_idx][i];
                }
            }

            clock_gettime(CLOCK_MONOTONIC, &step_start);
            CUDA_CHECK(cudaMemcpyAsync(nn.d_input_batch[stream_idx], batch_input, 
                                       BATCH_SIZE * INPUT_SIZE * sizeof(float), 
                                       cudaMemcpyHostToDevice, nn.streams[stream_idx]));
            CUDA_CHECK(cudaMemcpyAsync(nn.d_labels[stream_idx], batch_labels_ptr, 
                                       BATCH_SIZE * sizeof(int), 
                                       cudaMemcpyHostToDevice, nn.streams[stream_idx]));
            clock_gettime(CLOCK_MONOTONIC, &step_end);
            stats.memory_transfers += get_time_diff(step_start, step_end);

            clock_gettime(CLOCK_MONOTONIC, &step_start);
            
            forward_pass_stream(&nn, stream_idx, BATCH_SIZE);
            compute_loss_on_gpu_stream(&nn, stream_idx, BATCH_SIZE, h_loss[stream_idx]);
            backward_pass_stream(&nn, stream_idx, BATCH_SIZE);
            update_weights_stream(&nn, LEARNING_RATE, stream_idx);
            
            clock_gettime(CLOCK_MONOTONIC, &step_end);
            stats.gpu_compute += get_time_diff(step_start, step_end);
        }

        for (int s = 0; s < NUM_STREAMS; s++) {
            clock_gettime(CLOCK_MONOTONIC, &step_start);
            CUDA_CHECK(cudaStreamSynchronize(nn.streams[s]));
            clock_gettime(CLOCK_MONOTONIC, &step_end);
            stats.sync_wait += get_time_diff(step_start, step_end);
            
            int remaining_batch = num_batches - NUM_STREAMS + s;
            if (remaining_batch >= 0 && remaining_batch < num_batches) {
                for (int i = 0; i < BATCH_SIZE; i++) {
                    total_loss += h_loss[s][i];
                }
            }
        }

        printf("Epoch %d loss: %.4f\n", epoch, total_loss / (num_batches * BATCH_SIZE));
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    stats.total_time = get_time_diff(start, end);
    
    printf("\n=== CUSTOM FUSED GEMM (EDUCATIONAL) ===\n");
    printf("Total training time: %.3f seconds\n\n", stats.total_time);

    printf("Timing Breakdown:\n");
    printf("  H2D issue:      %6.3fs (%5.1f%%)\n", 
           stats.memory_transfers, 100.0 * stats.memory_transfers / stats.total_time);
    printf("  GPU issue:      %6.3fs (%5.1f%%)\n", 
           stats.gpu_compute, 100.0 * stats.gpu_compute / stats.total_time);
    printf("  Sync wait:      %6.3fs (%5.1f%%)\n", 
           stats.sync_wait, 100.0 * stats.sync_wait / stats.total_time);
    
    double accounted = stats.memory_transfers + stats.gpu_compute + stats.sync_wait;
    printf("  Other:          %6.3fs (%5.1f%%)\n",
           stats.total_time - accounted, 100.0 * (stats.total_time - accounted) / stats.total_time);

    evaluate(&nn, test_data, test_labels);

    free_nn_cuda(&nn);
    
    CUDA_CHECK(cudaFreeHost(train_data));
    CUDA_CHECK(cudaFreeHost(train_labels));
    free(test_data);
    free(test_labels);
    for (int i = 0; i < NUM_STREAMS; i++) {
        CUDA_CHECK(cudaFreeHost(h_loss[i]));
    }

    return 0;
}

