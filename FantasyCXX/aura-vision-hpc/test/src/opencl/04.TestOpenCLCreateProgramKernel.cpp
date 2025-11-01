//
// Created by Frewen.Wang on 2024/10/24.
//
#include "aura/aura_utils/utils/AuraLog.h"
#include "gtest/gtest.h"
#include <Eigen/Dense>

#include "utils/KernelUtils.h"

#ifdef BUILD_MAC
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

const static char *TAG = "TestOpenCLCreateProgramKernel";

#define N 100000  // 向量长度

/**
 * cl_context       context     上下文对象，程序需在此上下文中关联的设备上运行。必须是通过 clCreateContext 创建的有效对象。
 * cl_uint          count       指定源代码字符串的数量（strings 数组的长度）。若为 0 或 count=0，返回 CL_INVALID_VALUE。
 * const char **    strings     指向源代码字符串的指针数组(二级指针)。每个字符串应为有效的 OpenCL C 代码，且可以来自不同文件或代码片段。
 *                              示例：const char *src[] = {kernel_code1, kernel_code2};
 * const size_t*    lengths     每个字符串的长度数组，与 strings 一一对应。若为 NULL，则函数假定每个字符串以 \0 结尾，并自动计算长度。
 *                              若需显式指定长度（例如代码中包含 \0 但不想截断），需提供非 NULL 的长度数组。
 *                              示例：size_t lengths[] = {strlen(kernel_code1), strlen(kernel_code2)};
 * cl_int *         errcode_ret 返回错误码的指针。若为 NULL，忽略错误码；否则返回 CL_SUCCESS 或具体错误。
 * @return
 */
extern CL_API_ENTRY cl_program CL_API_CALL
clCreateProgramWithSource(cl_context /* context */,
                          cl_uint /* count */,
                          const char ** /* strings */,
                          const size_t * /* lengths */,
                          cl_int * /* errcode_ret */) CL_API_SUFFIX__VERSION_1_0 CL_DEPRECATED(10.6, 10.14);


/**
 * cl_context       context     上下文对象，程序需在此上下文中关联的设备上运行。必须是通过 clCreateContext 创建的有效对象。
 * cl_uint          num_devices 指定设备的个数
 * const char **    strings     指向源代码字符串的指针数组(二级指针)。每个字符串应为有效的 OpenCL C 代码，且可以来自不同文件或代码片段。
 *                              示例：const char *src[] = {kernel_code1, kernel_code2};
 * const size_t*    lengths     二进制内容大小
 *                              示例：size_t lengths[] = {strlen(kernel_code1), strlen(kernel_code2)};
 * cl_int *         errcode_ret 返回错误码的指针。若为 NULL，忽略错误码；否则返回 CL_SUCCESS 或具体错误。
 * @return
 */
extern CL_API_ENTRY cl_program CL_API_CALL
clCreateProgramWithBinary(cl_context /* context */,
                          cl_uint /* num_devices */,
                          const cl_device_id * /* device_list */,
                          const size_t * /* lengths */,
                          const unsigned char ** /* binaries */,
                          cl_int * /* binary_status */,
                          cl_int * /* errcode_ret */) CL_API_SUFFIX__VERSION_1_0 CL_DEPRECATED(10.6, 10.14);

/**
 * 编译并链接 OpenCL 程序代码，生成可在设备上运行的内核。
 * cl_program program  要编译的程序对象，必须是通过 clCreateProgramWithSource 或 clCreateProgramWithBinary 创建的有效对象。
 * cl_uint num_devices device_list 参数中指定的设备数量。若 device_list=NULL，则 num_devices 必须为 0，表示对上下文中的所有设备进行编译。
 * const cl_device_id *device_list 目标设备数组，指定编译针对的设备。若为 NULL，则针对 program 关联的上下文中的所有设备编译。
 *                                 设备必须属于创建程序时的上下文，否则返回 CL_INVALID_DEVICE。
 * const char *options  编译选项字符串，控制编译行为。常见选项包括：
 *                      宏定义: -DNAME=value
 *                      头文件路径: -I/path/to/include
 *                      优化级别: -O0（无优化）, -O1, -O2, -O3
 *                      OpenCL 版本: -cl-std=CL2.0（强制使用 OpenCL 2.0 语法）
 *                      警告控制: -w（禁用所有警告）
 *                      内核参数检查: -cl-kernel-arg-info（保留内核参数信息，供 clGetKernelArgInfo 使用）
 *                      若为 NULL，表示无编译选项。
 * pfn_notify           可选的回调函数，在编译完成后异步触发。若为 NULL，则函数同步执行（阻塞直到编译完成）。
 *                      注意: 异步编译时，程序对象必须在回调期间保持有效。
 * void *user_data      传递给回调函数的用户自定义数据（如日志文件指针、状态标志等）。若 pfn_notify=NULL，此参数被忽略。
 *
 *
 *
 * @return
 */
extern CL_API_ENTRY cl_int CL_API_CALL
clBuildProgram(cl_program           /* program */,
               cl_uint              /* num_devices */,
               const cl_device_id * /* device_list */,
               const char *         /* options */,
               void (CL_CALLBACK *  /* pfn_notify */)(cl_program /* program */, void * /* user_data */),
               void *               /* user_data */) CL_API_SUFFIX__VERSION_1_0 CL_DEPRECATED(10.6, 10.14);


/**
 * clCompileProgram 是 OpenCL API 中的一个函数，用于仅编译程序对象（生成中间代码），而不是直接生成可执行代码。
 * 它是 OpenCL 分步编译流程的一部分（需配合 clLinkProgram 完成链接），适用于模块化代码或依赖外部头文件的场景。
 * 参数说明如下：
 * cl_program program 待编译的程序对象，必须通过 clCreateProgramWithSource 或类似函数创建
 * cl_uint num_devices device_list 中的设备数量。若 device_list=NULL，则 num_devices 必须为 0，表示对上下文中的所有设备编译。
 * const cl_device_id *device_list 目标设备数组，指定编译针对的设备。若为 NULL，则针对程序关联的上下文中的所有设备。
 * const char *options  编译选项字符串，控制编译行为（与 clBuildProgram 的选项一致）：
 *                                      -I<dir>: 指定头文件搜索路径。
 *                                      -D<name>=<value>: 定义宏。
 *                                      -cl-opt-disable: 禁用优化。
 *                                      -cl-std=CL2.0: 指定 OpenCL C 语言版本。
 *                                      若为 NULL，表示无编译选项。
 * cl_uint num_input_headers    输入头文件的数量（即 input_headers 和 header_include_names 数组的长度）。若为 0，表示无头文件依赖。
 * const cl_program *input_headers      已编译的头文件程序对象数组。这些程序对象需通过 clCompileProgram 或 clCreateProgramWithSource 创建，表示依赖的头文件内容。
 *                                      例如，若代码中包含 #include "utils.h"，则需将 utils.h 对应的程序对象传入此参数。
 * const char **header_include_names    头文件名称数组，与 input_headers 一一对应。名称需与代码中 #include 的字符串完全匹配。
 *                                      例如，若代码中写 #include "utils.h"，则此处需提供 "utils.h"。
 * pfn_notify 和 user_data              异步回调函数及其用户数据，用法与 clBuildProgram 一致。若 pfn_notify=NULL，函数同步执行。
 * @return
 */
extern CL_API_ENTRY cl_int CL_API_CALL
clCompileProgram(cl_program           /* program */,                // 待编译的程序对象
                 cl_uint              /* num_devices */,            // 目标设备数量
                 const cl_device_id * /* device_list */,            // 目标设备列表
                 const char *         /* options */,                // 编译选项（优化、宏定义等）
                 cl_uint              /* num_input_headers */,      // 输入头文件的数量
                 const cl_program *   /* input_headers */,          // 已编译的头文件程序对象数组
                 const char **        /* header_include_names */,   // 头文件名称数组（与 input_headers 对应）
                 void (CL_CALLBACK *  /* pfn_notify */)(cl_program /* program */, void * /* user_data */), // 编译完成后的回调函数（可选）
                 void *               /* user_data */) CL_API_SUFFIX__VERSION_1_2 CL_DEPRECATED(10.8, 10.14); // 传递给回调函数的数据




// 用于错误检查的宏
#define CHECK_ERROR(err, msg)                                   \
    if (err != CL_SUCCESS) {                                    \
        printf("%s failed with error code %d\n", msg, err);     \
        exit(EXIT_FAILURE);                                     \
    } else {                                                    \
        printf("%s 执行成功，错误码: %d\n", msg, err);             \
    }

class TestOpenCLCreateProgramKernel : public testing::Test {
public:
    static void SetUpTestSuite() {
        ALOGE(TAG, "SetUpTestSuite");
    }

    static void TearDownTestSuite() {
        ALOGE(TAG, "TearDownTestSuite");
    }
};


TEST_F(TestOpenCLCreateProgramKernel, TestGetPlatformIDs) {
    ALOGE(TAG, "==============TestGetPlatformIDs================");

    /// 典型使用场景：
    /// 第一次调用的时候： platforms = NULL，通过 num_platforms 获取实际数量。
    /// clGetPlatformIDs(0, NULL, &num_platforms); // 获取平台总数
    /// 第二次调用的时候：分配内存后再次调用以获取所有平台 ID。
    ///
    /// 第一步：
    cl_uint num_platforms;
    cl_int err = clGetPlatformIDs(0, nullptr, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) {
        // 处理错误或无平台的情况
        printf("获取平台数量的大小异常 size: %d\n", num_platforms);
    }
    /// 第二步：分配内存后再次调用以获取所有平台 ID。
    /// Clang-Tidy: Use auto when initializing with a cast to avoid duplicating the type name
    /// Clang-Tidy：给出这个提示，是为了让代码更加简洁、易读，同时减少因手动指定类型可能引发的错误。你可以依据这个提示，在使用强制类型转换初始化变量时，采用auto关键字。
    // cl_platform_id *platform = static_cast<cl_platform_id *>(malloc(sizeof(cl_platform_id) * num_platforms));
    auto *platform = static_cast<cl_platform_id *>(malloc(sizeof(cl_platform_id) * num_platforms));
    err = clGetPlatformIDs(num_platforms, platform, nullptr);
    if (err != CL_SUCCESS) {
        printf("获取平台的ID列表异常 size: %d\n", num_platforms);
    }

    printf("获取到的 OpenCL 平台数量 num_platforms：%d \n", num_platforms);
}

TEST_F(TestOpenCLCreateProgramKernel, TestGetPlatformInfo) {
    ALOGE(TAG, "==============TestGetPlatformInfo================");


    /// 典型使用场景：
    /// 第一次调用的时候： platforms = NULL，通过 num_platforms 获取实际数量。
    /// clGetPlatformIDs(0, NULL, &num_platforms); // 获取平台总数
    /// 第二次调用的时候：分配内存后再次调用以获取所有平台 ID。
    ///
    /// 第一步：
    cl_uint num_platforms;
    cl_int err = clGetPlatformIDs(0, nullptr, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) {
        // 处理错误或无平台的情况
        printf("获取平台数量的大小异常 size: %d\n", num_platforms);
    }
    /// 第二步：分配内存后再次调用以获取所有平台 ID。
    /// Clang-Tidy: Use auto when initializing with a cast to avoid duplicating the type name
    /// Clang-Tidy：给出这个提示，是为了让代码更加简洁、易读，同时减少因手动指定类型可能引发的错误。你可以依据这个提示，在使用强制类型转换初始化变量时，采用auto关键字。
    // cl_platform_id *platform = static_cast<cl_platform_id *>(malloc(sizeof(cl_platform_id) * num_platforms));
    auto *platform = static_cast<cl_platform_id *>(malloc(sizeof(cl_platform_id) * num_platforms));
    err = clGetPlatformIDs(num_platforms, platform, nullptr);
    if (err != CL_SUCCESS) {
        printf("获取平台的ID列表异常 size: %d\n", num_platforms);
    }

    for (int i = 0; i < num_platforms; i++) {
        size_t size;
        err = clGetPlatformInfo(platform[i], CL_PLATFORM_NAME, 0, NULL, &size);
        char *PName = static_cast<char *>(malloc(size));
        err = clGetPlatformInfo(platform[i], CL_PLATFORM_NAME, size, PName, NULL);
        printf("CL_PLATFORM_NAME 获取平台名称: %s\n", PName);
        err = clGetPlatformInfo(platform[i], CL_PLATFORM_VENDOR, 0, NULL, &size);
        char *PVendor = static_cast<char *>(malloc(size));
        err = clGetPlatformInfo(platform[i], CL_PLATFORM_VENDOR, size, PVendor, NULL);
        printf("CL_PLATFORM_VENDOR 获取平台开发商名称: %s\n", PVendor);
        err = clGetPlatformInfo(platform[i], CL_PLATFORM_VERSION, 0, NULL, &size);
        char *PVersion = static_cast<char *>(malloc(size));
        err = clGetPlatformInfo(platform[i], CL_PLATFORM_VERSION, size, PVersion, NULL);
        printf("CL_PLATFORM_VERSION 获取平台支持的最大的OpenCL版本: %s\n", PVersion);
        err = clGetPlatformInfo(platform[i], CL_PLATFORM_PROFILE, 0,NULL, &size);
        char *PProfile = static_cast<char *>(malloc(size));
        err = clGetPlatformInfo(platform[i], CL_PLATFORM_PROFILE, size, PProfile, NULL);
        printf("CL_PLATFORM_PROFILE 获取平台支持FULL_PROFILE还是EMBBEDED_PROFILE: %s\n", PProfile);
        err = clGetPlatformInfo(platform[i], CL_PLATFORM_EXTENSIONS, 0, NULL, &size);
        char *PExten = static_cast<char *>(malloc(size));
        err = clGetPlatformInfo(platform[i], CL_PLATFORM_EXTENSIONS, size, PExten, NULL);
        printf("CL_PLATFORM_EXTENSIONS 获取平台支持的扩展名列表: %s\n", PExten);
        free(PName);
        free(PVendor);
        free(PVersion);
        free(PProfile);
        free(PExten);
    }
    // 回收platform的开辟的内存
    free(platform);
}

TEST_F(TestOpenCLCreateProgramKernel, TestGetDevice) {
    ALOGE(TAG, "==============TestGetDevice================");
    cl_uint num_platform;
    cl_uint num_device;
    cl_platform_id *platform;
    cl_device_id *devices;
    cl_int err;

    // 第一次调用时，devices参数设置为NULL，num_devices返回指定平台中的设备数；
    // 可以获取num_platform的数量
    err = clGetPlatformIDs(0, 0, &num_platform);

    ///
    platform = static_cast<cl_platform_id *>(malloc(sizeof(cl_platform_id) * num_platform));
    err = clGetPlatformIDs(num_platform, platform, NULL);
    //获得第一个平台上的设备数量
    err = clGetDeviceIDs(platform[0], CL_DEVICE_TYPE_GPU, 0, NULL, &num_device);
    devices = static_cast<cl_device_id *>(malloc(sizeof(cl_device_id) * num_device));
    //初始化可用的设备
    err = clGetDeviceIDs(platform[0], CL_DEVICE_TYPE_GPU, num_device, devices, NULL);
}

TEST_F(TestOpenCLCreateProgramKernel, TestCreateProgram) {
    ALOGE(TAG, "==============TestCreateProgram 创建程序对象================");
    // 进行开辟三个float数据的向量
    float *A = (float *) malloc(sizeof(float) * N);
    float *B = (float *) malloc(sizeof(float) * N);
    float *C = (float *) malloc(sizeof(float) * N);

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

    // =========================== 第一步骤： 获取平台和设备信息 =========================================================
    // 第一步：获取平台数量
    err = clGetPlatformIDs(1, &platform, &num_platforms);
    if (err != CL_SUCCESS) {
        printf("无法获取 OpenCL 平台信息\n");
        return;
    }
    printf("获取到的 OpenCL 平台信息 num_platforms：%d \n", num_platforms);

    // 第一步：获取设备数量
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, &num_devices);
    if (err != CL_SUCCESS) {
        printf("无法获取 OpenCL 设备信息\n");
        return;
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
        return;
    }
    printf("获取到的 OpenCL 设备名称: %s\n", device_name);

    // =========================== 第二步骤： 创建上下文对象 ====================================================
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    CHECK_ERROR(err, "创建上下文对象clCreateContext");


    // =========================== 第三步骤：创建CommandQueue命令队列 ===========================================
    // ✅ 检查版本是否支持 clCreateCommandQueueWithProperties
    // 在较老版本的 OpenCL（特别是 OpenCL 1.2 及以前），是没有 clCreateCommandQueueWithProperties 这个函数的，
    // 这个 API 是从 OpenCL 2.0 开始引入的，用于替代 clCreateCommandQueue。
    cl_command_queue queue;
#if CL_TARGET_OPENCL_VERSION >= 200
    queue = clCreateCommandQueueWithProperties(context, device, 0, &err);
#else
    queue = clCreateCommandQueue(context, device, 0, &err);
#endif
    CHECK_ERROR(err, "创建命令队列clCreateCommandQueue");

    // =========================== 第四步骤：创建Program程序对象 ===========================================
    // 加载内核源代码
    char *kernelSource = KernelUtils::readKernelSource("./kernel_vector_add.cl");
    // 创建程序对象并编译
    cl_program program = clCreateProgramWithSource(context, 1, (const char **) &kernelSource, NULL, &err);
    CHECK_ERROR(err, "创建程序对象clCreateProgramWithSource");



}
