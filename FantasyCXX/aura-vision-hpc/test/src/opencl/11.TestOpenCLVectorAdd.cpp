//
// Created by Frewen.Wang on 2024/10/24.
//
#include "aura/aura_utils/utils/AuraLog.h"
#include "gtest/gtest.h"
#include <Eigen/Dense>

#ifdef BUILD_MAC
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

const static char *TAG = "TestOpenCLVectorAdd";

#define N 1024  // 向量长度

// 用于错误检查的宏
#define CHECK_ERROR(err, msg)                                   \
    if (err != CL_SUCCESS) {                                    \
        printf("%s failed with error code %d\n", msg, err);     \
        exit(EXIT_FAILURE);                                     \
    } else {                                                    \
        printf("%s 执行成功，错误码: %d\n", msg, err);             \
    }


char *readKernelSource(const char *filename) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        perror("Failed to open kernel file");
        exit(EXIT_FAILURE);
    }
    fseek(fp, 0, SEEK_END);
    size_t size = ftell(fp);
    rewind(fp);
    char *src = (char *) malloc(size + 1);
    fread(src, 1, size, fp);
    src[size] = '\0';
    fclose(fp);
    return src;
}

class TestOpenCLVectorAdd : public testing::Test {
public:
    static void SetUpTestSuite() {
        ALOGE(TAG, "SetUpTestSuite");
    }

    static void TearDownTestSuite() {
        ALOGE(TAG, "TearDownTestSuite");
    }
};

TEST_F(TestOpenCLVectorAdd, TestOpenCLVectorAdd) {
    ALOGE(TAG, "==============TestOpenCLVectorAdd================");

    // 进行开辟三个float数据的向量
    float *A = (float*)malloc(sizeof(float) * N);
    float *B = (float*)malloc(sizeof(float) * N);
    float *C = (float*)malloc(sizeof(float) * N);

    // 初始化A向量和B向量数据
    for (int i = 0; i < N; i++) {
        A[i] = i * 1.0f;
        B[i] = (N - i) * 1.0f;
    }



    // 定义的32位int数据
    cl_int err;
    // 获取platform平台的个数
    cl_uint num_platforms;
    // 获取所有的平台的ID
    cl_platform_id platform;
    // 获取device设备的的个数
    cl_uint num_devices;
    // 获取所有的设备的ID
    cl_device_id device;


    // =========================== 第一大步骤： 获取平台和设备信息 =========================================================
    // 第一步：获取平台数量
    err = clGetPlatformIDs(1, &platform, &num_platforms);
    if (err != CL_SUCCESS) {
        printf("无法获取 OpenCL 平台信息\n");
        return;
    }
    printf("获取到的 OpenCL 平台信息 num_platforms：%d \n", num_platforms);

    // 第二步：获取设备数量
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, &num_devices);
    if (err != CL_SUCCESS) {
        printf("无法获取 OpenCL 设备信息\n");
        return ;
    }
    printf("获取到的 OpenCL 设备信息 num_devices：%d \n", num_devices);


    // 输出 OpenCL 版本信息
    char version[1024];
    // 使用 clGetPlatformInfo 函数获取平台的版本信息，并将其存储在 version 变量中，然后输出。
    err = clGetPlatformInfo(platform, CL_PLATFORM_VERSION, sizeof(version), version, nullptr);
    if (err != CL_SUCCESS) {
        printf("无法获取 OpenCL 版本信息\n");
        return;
    }
    printf("获取到的 OpenCL 版本号: %s\n", version);

    // 输出设备信息
    char device_name[1024];
    // 使用 clGetDeviceInfo 函数获取设备的名称信息，并将其存储在 device_name 变量中，然后输出。
    err = clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, nullptr);
    if (err != CL_SUCCESS) {
        printf("无法获取设备名称\n");
        return ;
    }
    printf("获取到的 OpenCL 设备名称: %s\n", device_name);


    // =========================== 第二大步骤： 创建上下文对象和命令队列 ====================================================
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    CHECK_ERROR(err, "创建上下文对象clCreateContext");

    // ✅ 检查版本是否支持 clCreateCommandQueueWithProperties
    // 在较老版本的 OpenCL（特别是 OpenCL 1.2 及以前），是没有 clCreateCommandQueueWithProperties 这个函数的，
    // 这个 API 是从 OpenCL 2.0 开始引入的，用于替代 clCreateCommandQueue。
    // TODO 为什么要进行替代？
    cl_command_queue queue;
#if CL_TARGET_OPENCL_VERSION >= 200
    queue = clCreateCommandQueueWithProperties(context, device, 0, &err);
#else
    queue = clCreateCommandQueue(context, device, 0, &err);
#endif
    CHECK_ERROR(err, "创建命令队列clCreateCommandQueue");





    // =============================== 第三大步骤：创建程序对象和内核对象 ==================================================
    // 加载内核源代码
    char* kernelSource = readKernelSource("./kernel_vector_add.cl");



    // 创建程序对象并编译
    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&kernelSource, NULL, &err);
    CHECK_ERROR(err, "创建程序对象clCreateProgramWithSource");

    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        char buildLog[2048];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buildLog), buildLog, nullptr);
        printf("编译失败:\n%s\n", buildLog);
        exit(EXIT_FAILURE);
    }
    // 找到kernel程序里面的vector_add的方法
    cl_kernel kernel = clCreateKernel(program, "vector_add", &err);
    CHECK_ERROR(err, "创建kernel程序clCreateKernel");

    // 创建缓冲区
    cl_mem bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,sizeof(float) * N, A, &err);
    CHECK_ERROR(err, "创建缓冲区clCreateBuffer A");
    cl_mem bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,sizeof(float) * N, B, &err);
    CHECK_ERROR(err, "创建缓冲区clCreateBuffer B");
    cl_mem bufferC = clCreateBuffer(context, CL_MEM_WRITE_ONLY,sizeof(float) * N, NULL, &err);
    CHECK_ERROR(err, "创建缓冲区clCreateBuffer C");

    // 设置参数. 也就是给我们的kernel函数设置参数
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferC);

    /// 开始执行内核
    size_t global_size = N;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_size, nullptr, 0, nullptr, nullptr);
    CHECK_ERROR(err, "clEnqueueNDRangeKernel");

    // 获取执行结果
    err = clEnqueueReadBuffer(queue, bufferC, CL_TRUE, 0, sizeof(float) * N, C, 0, nullptr, nullptr);
    CHECK_ERROR(err, "clEnqueueReadBuffer");


    // 打印一部分验证
    printf("C[0] = %.2f, A[0] + B[0] = %.2f\n", C[0], A[0] + B[0]);
    printf("C[N-1] = %.2f, A[N-1] + B[N-1] = %.2f\n", C[N-1], A[N-1] + B[N-1]);

    // 清理所有开辟出来的空间
    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseMemObject(bufferC);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    free(A); free(B); free(C); free(kernelSource);
}
