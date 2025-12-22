cmake_minimum_required(VERSION 3.18)
project(cuda_kernels_from_scratch LANGUAGES CUDA CXX)

# Build settings
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# If you want to control GPU arch, uncomment and set accordingly (e.g., 80 for A100)
# set(CMAKE_CUDA_ARCHITECTURES 75)

# Elementwise test binary
add_executable(test_elementwise
    tests/test_elementwise.cu
    src/elementwise/elementwise_add.cu
    src/elementwise/activations.cu
)

target_include_directories(test_elementwise PRIVATE
    src/elementwise
)

# Enable separable compilation for CUDA translation units
set_target_properties(test_elementwise PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# Reduction test binary (includes sum, max, and softmax)
add_executable(test_reduction
    tests/test_reduction.cu
    src/reduce/reduce_sum.cu
    src/reduce/reduce_max.cu
    src/reduce/reduce_softmax.cu
    src/reduce/reduce_softmax_matrix.cu
)

target_include_directories(test_reduction PRIVATE
    src/reduce
)

# Keep files separate with separable compilation
set_target_properties(test_reduction PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
)

# Transpose test binary
add_executable(test_transpose
    tests/test_transpose.cu
    src/transpose/transpose.cu
)

target_include_directories(test_transpose PRIVATE
    src/transpose
)

set_target_properties(test_transpose PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
)

# SGEMM test binary
add_executable(test_sgemm
    tests/test_sgemm.cu
    src/SGEMM/sgemm_v1_naive.cu
    src/SGEMM/sgemm_v2.cu
    src/SGEMM/sgemm_v3.cu
    src/SGEMM/sgemm_v4.cu
    src/SGEMM/sgemm_v5.cu
    src/SGEMM/sgemm_v6.cu
    src/SGEMM/sgemm_v7.cu
)

target_include_directories(test_sgemm PRIVATE
    src/SGEMM
)

# Link cuBLAS for reference comparison
find_library(CUBLAS_LIBRARY cublas HINTS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
target_link_libraries(test_sgemm ${CUBLAS_LIBRARY})

set_target_properties(test_sgemm PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
)
