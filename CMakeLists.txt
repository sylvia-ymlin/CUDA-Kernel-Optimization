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
# Reduction test binary
add_executable(test_reduction
    tests/test_reduction.cu
    src/reduce/reduce_sum.cu
    src/reduce/reduce_max.cu
)

target_include_directories(test_reduction PRIVATE
    src/reduce
)

# Keep files separate with separable compilation
set_target_properties(test_reduction PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
)

# Softmax test binary
add_executable(test_softmax
    tests/test_softmax.cu
    src/reduce/reduce_softmax.cu
    src/reduce/reduce_softmax_matrix.cu
    src/reduce/reduce_max.cu
)

target_include_directories(test_softmax PRIVATE
    src/reduce
)

set_target_properties(test_softmax PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
)
