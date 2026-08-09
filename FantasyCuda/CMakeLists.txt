cmake_minimum_required(VERSION 3.18)
project(AuraKaleidoCuda LANGUAGES CXX)

include(CheckLanguage)
check_language(CUDA)
if(NOT CMAKE_CUDA_COMPILER)
    message(STATUS "CUDA toolkit not found; CUDA examples are skipped on this host")
    return()
endif()

enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(cuda_examples
    hello-cuda:01_hello_cuda.cu
    hello-gpu:01_02_hello-gpu.cu
    hello-malloc:01_03_hello-malloc.cu
    demo_cuda_matadd:01_04_demo_cuda_matadd.cu
    demo_stride:01_05_demo_stride.cu
    demo_error:01_06_demo_error.cu
    demo_matrix_add:01_07_demo_matrix_add.cu
    demo_matrix:01_10_demo_matrix.cu
)

foreach(example IN LISTS cuda_examples)
    string(REPLACE ":" ";" fields "${example}")
    list(GET fields 0 target)
    list(GET fields 1 source)
    add_executable("${target}" "src/${source}")
endforeach()
