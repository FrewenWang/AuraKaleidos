//
// Created by Frewen.Wang on 2024/10/24.
//
#include "aura/aura_utils/utils/AuraLog.h"
#include "gtest/gtest.h"
#include <Eigen/Dense>
#include <stdio.h>
#include <stdlib.h>
#ifdef BUILD_MAC
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

const static char *TAG = "TestOpenCLMatrixMul";

// 检查OpenCL错误码的宏
#define CHECK_OPENCL_ERROR(cmd)                                                     \
    do {                                                                            \
        cl_int error = cmd;                                                         \
        if (error != CL_SUCCESS) {                                                  \
            fprintf(stderr, "OpenCL错误 %d: %s:%d\n", error, __FILE__, __LINE__);    \
            exit(EXIT_FAILURE);                                                     \
        }                                                                           \
    } while (0)


#define MATRIX_SIZE 1024  // 矩阵维度为1024x1024


class TestOpenCLMatrixMul : public testing::Test {
public:
    static void SetUpTestSuite() {
        ALOGE(TAG, "SetUpTestSuite");
    }

    static void TearDownTestSuite() {
        ALOGE(TAG, "TearDownTestSuite");
    }
};


TEST_F(TestOpenCLMatrixMul, TestOpenCLMatrixMulHello) {
    ALOGE(TAG, "==============TestOpenCLMatrixMulHello================");

    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;

    // 初始化输入矩阵A、B和输出矩阵C。 开辟出来对象内存地址
    auto *A = (float *) malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    auto *B = (float *) malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(float));
    auto *C = (float *) malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(float));

    // 初始化矩阵数据（示例用随机数填充）
    for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; ++i) {
        A[i] = static_cast<float>(rand()) / RAND_MAX;
        B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // 1. 获取OpenCL平台
    // 通过clGetPlatformIDs和clGetDeviceIDs获取GPU设备，确保计算在异构加速器上执行
    CHECK_OPENCL_ERROR(clGetPlatformIDs(1, &platform, nullptr));

    // 2. 获取GPU设备
    CHECK_OPENCL_ERROR(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr));

    // 3. 创建上下文
    // 创建上下文时选择第一个可用GPU设备，适用于大多数开发环境
    context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);

    // 4. 创建命令队列
    queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, NULL);

    // 5. 创建内存对象
    // 使用clCreateBuffer创建三个内存对象：
    //              bufferA和bufferB为只读内存，通过CL_MEM_COPY_HOST_PTR直接拷贝主机数据
    //              内存标志位（CL_MEM_READ_ONLY等）显式声明访问模式，帮助驱动优化
    //              bufferC为写内存，避免提前初始化开销
    cl_mem bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    MATRIX_SIZE * MATRIX_SIZE * sizeof(float), A, NULL);
    cl_mem bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    MATRIX_SIZE * MATRIX_SIZE * sizeof(float), B, NULL);
    cl_mem bufferC = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                    MATRIX_SIZE * MATRIX_SIZE * sizeof(float), NULL, NULL);

    // 6. 定义OpenCL内核源代码
    const char *kernelSource =
            "__kernel void matrixMul(__global const float* A, \n"
            "                       __global const float* B, \n"
            "                       __global float* C, \n"
            "                       const int N) { \n"
            "    int row = get_global_id(0); \n"
            "    int col = get_global_id(1); \n"
            "    float sum = 0.0f; \n"
            "    for (int k = 0; k < N; ++k) { \n"
            "        sum += A[row * N + k] * B[k * N + col]; \n"
            "    } \n"
            "    C[row * N + col] = sum; \n"
            "} \n";

    // 7. 创建并编译程序
    program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, NULL);
    CHECK_OPENCL_ERROR(clBuildProgram(program, 1, &device, NULL, NULL, NULL));

    // 8. 创建内核对象
    kernel = clCreateKernel(program, "matrixMul", NULL);

    // 9. 设置内核参数
    // 定义临时变量存储矩阵维度
    int matrixSize = MATRIX_SIZE;
    CHECK_OPENCL_ERROR(clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA));
    CHECK_OPENCL_ERROR(clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB));
    CHECK_OPENCL_ERROR(clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferC));
    CHECK_OPENCL_ERROR(clSetKernelArg(kernel, 3, sizeof(int), &matrixSize));

    // 10. 开始执行内核
    size_t globalWorkSize[2] = {MATRIX_SIZE, MATRIX_SIZE}; // 二维工作项分布
    CHECK_OPENCL_ERROR(clEnqueueNDRangeKernel(queue, kernel, 2, nullptr,
        globalWorkSize, nullptr, 0, nullptr, nullptr));

    // 11. 读取结果到主机内存
    CHECK_OPENCL_ERROR(clEnqueueReadBuffer(queue, bufferC, CL_TRUE, 0,
        MATRIX_SIZE * MATRIX_SIZE * sizeof(float),
        C, 0, nullptr, nullptr));


    // 12. 添加结果输出验证
    printf("\n计算结果验证（前3x3元素）：\n");
    for(int i = 0; i < 3; ++i) {
        for(int j = 0; j < 3; ++j) {
            printf("C[%d][%d] = %.2f\t", i, j, C[i * MATRIX_SIZE + j]);
        }
        printf("\n");
    }

    // 13. 释放资源
    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseMemObject(bufferC);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    free(A);
    free(B);
    free(C);
}
