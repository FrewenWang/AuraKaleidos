message(STATUS "[AuraKaleidoCuda] ==================AuraKaleidoCuda Build==============================")
# 第一步： 设置CMake版本号
cmake_minimum_required(VERSION 3.10)

# 第二步： 设置工程名称
project(AuraKaleidoCuda)
# 第三步：设置CMake设置的C++的版本
set(CMAKE_CXX_STANDARD 11)

# 项目中集成cuda库。这个指令的意思是从宿主系统上进行寻找cuda
find_package(CUDA REQUIRED)

# 编译连接
cuda_add_executable(hello-cuda src/01_hello_cuda.cu)

# 编译连接
cuda_add_executable(hello-gpu src/01_02_hello-gpu.cu)
target_link_libraries(hello-gpu)

# 编译连接
cuda_add_executable(hello-malloc src/01_03_hello-malloc.cu)
target_link_libraries(hello-malloc)

cuda_add_executable(demo_cuda_matadd src/01_04_demo_cuda_matadd.cu)
target_link_libraries(demo_cuda_matadd)


# 编译连接
cuda_add_executable(demo_stride src/01_05_demo_stride.cu)
target_link_libraries(demo_stride)


# 编译连接
cuda_add_executable(demo_error src/01_06_demo_error.cu)
target_link_libraries(demo_error)


cuda_add_executable(demo_matrix_add src/01_07_demo_matrix_add.cu)
target_link_libraries(demo_matrix_add)


cuda_add_executable(demo_matrix src/01_10_demo_matrix.cu)
target_link_libraries(demo_matrix)



message(STATUS "[AuraKaleidoCuda] ==================AuraKaleidoCuda Build End===========================")
