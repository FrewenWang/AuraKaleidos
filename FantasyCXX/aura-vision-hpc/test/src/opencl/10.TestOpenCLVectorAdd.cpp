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

const int ARRAY_SIZE = 1000;

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


class TestOpenCLVectorAdd2 : public testing::Test {
public:
    static void SetUpTestSuite() {
        ALOGE(TAG, "SetUpTestSuite");
    }

    static void TearDownTestSuite() {
        ALOGE(TAG, "TearDownTestSuite");
    }
};


char *ReadKernelSourceFile(const char *filename, size_t *length) {
    FILE *file = NULL;
    size_t sourceLength;
    char *sourceString;
    int ret;
    file = fopen(filename, "rb");
    if (file == NULL) {
        printf("%s at %d :Can't open %s\n", __FILE__, __LINE__ - 2, filename);
        return NULL;
    }
    fseek(file, 0, SEEK_END);
    sourceLength = ftell(file);
    fseek(file, 0, SEEK_SET);
    sourceString = (char *) malloc(sourceLength + 1);
    sourceString[0] = '\0';
    ret = fread(sourceString, sourceLength, 1, file);
    if (ret == 0) {
        printf("%s at %d : Can't read source %s\n", __FILE__, __LINE__ - 2, filename);
        return NULL;
    }
    fclose(file);
    if (length != 0) {
        *length = sourceLength;
    }
    sourceString[sourceLength] = '\0';
    return sourceString;
}


/**
 * 此函数用于获取系统中所有可用的 OpenCL 平台。平台代表不同的 OpenCL 实现，例如：
 *
 * extern :表明该函数定义在外部（如动态链接库或系统库中），由 OpenCL 运行时提供。
 * CL_API_ENTRY: 宏定义，用于指定函数的调用约定（如 __stdcall 或 __cdecl），确保跨编译器的兼容性。
 * CL_API_CALL: 类似 CL_API_ENTRY，通常定义为空或特定调用约定，具体由平台决定。
 * cl_int: 返回值。 函数执行的状态码，例如 CL_SUCCESS 表示成功，其他值如 CL_INVALID_VALUE 表示参数错误。必须检查返回值以确认操作是否成功。
 * 参数列表：
 * num_entries	    cl_uint	            输入	    调用者提供的 platforms 数组大小（如果 platforms 为 NULL，此参数应设为 0）
 *                  num_entries是cl_uint类型，作为输入参数。它的主要作用是指定调用者提供的platforms数组的大小。
 *                  如果platforms参数为NULL，那么num_entries应该设置为0，此时函数会忽略num_entries的值，
 *                  仅通过num_platforms参数返回可用的平台数量。
 *                  反之，如果platforms不为NULL，那么num_entries必须大于0，并且至少等于实际存在的平台数量，否则可能导致缓冲区溢出的错误。
 * platforms	    cl_platform_id *	输出	    存储获取到的平台 ID 的数组（可为 NULL，仅用于查询平台数量）
 * num_platforms	cl_uint *	        输出	    实际可用的平台数量（当 platforms 为 NULL 时，返回总平台数）
 * @return
 */
extern CL_API_ENTRY cl_int CL_API_CALL
clGetPlatformIDs(cl_uint          /* num_entries */,
                 cl_platform_id * /* platforms */,
                 cl_uint *        /* num_platforms */) CL_API_SUFFIX__VERSION_1_0 CL_DEPRECATED(10.6, 10.14);

/**
 * clGetPlatformInfo 是 OpenCL API 中用于查询计算平台（Platform）的详细信息（如厂商名称、支持版本等）的核心函数。以下是逐部分解析和用法说明：
 * platform	            cl_platform_id	    输入	    通过 clGetPlatformIDs 获取的平台句柄。
 * param_name	        cl_platform_info	输入	    指定要查询的信息类型（枚举值，如 CL_PLATFORM_NAME）。
 * param_value_size 	size_t	            输入	    param_value 缓冲区的大小（字节）。若 param_value 为 NULL，此参数被忽略。
 * param_value	        void*	            输出	    存储查询结果的缓冲区。若为 NULL，函数仅返回所需缓冲区大小。
 * param_value_size_ret	size_t*	            输出	    实际写入 param_value 的字节数。若为 NULL，此值不被返回。
 * @return
 */
extern CL_API_ENTRY cl_int CL_API_CALL
clGetPlatformInfo(cl_platform_id   /* platform */,
                  cl_platform_info /* param_name */,
                  size_t           /* param_value_size */,
                  void *           /* param_value */,
                  size_t *         /* param_value_size_ret */) CL_API_SUFFIX__VERSION_1_0 CL_DEPRECATED(10.6, 10.14);


/**
 * clCreateContext 是 OpenCL API 的核心函数之一，用于创建一个执行上下文（Context），该上下文管理一组设备（Devices）及其共享资源（如内存对象、命令队列等）。
 * 以下是该方法的详细解析：
 * const cl_context_properties *properties,     // 上下文属性列表
 * cl_uint num_devices,                      // 设备数量
 * const cl_device_id *devices,              // 设备数组
 *  void (CL_CALLBACK *pfn_notify)(           // 错误回调函数
 *       const char *, const void *, size_t, void *),
 * void *user_data,                          // 回调函数的用户数据
 *  cl_int *errcode_ret                       // 返回错误码
 * @return
 */
extern CL_API_ENTRY cl_context CL_API_CALL
clCreateContext(const cl_context_properties * /* properties */,     // 上下文属性
                cl_uint                 /* num_devices */,
                const cl_device_id *    /* devices */,
                void (CL_CALLBACK * /* pfn_notify */)(const char *, const void *, size_t, void *),
                void *                  /* user_data */,
                cl_int *                /* errcode_ret */) CL_API_SUFFIX__VERSION_1_0 CL_DEPRECATED(10.6, 10.14);


/**
 * clCreateCommandQueue 是 OpenCL API 中的一个关键函数，用于在指定上下文中为特定设备创建命令队列。以下是对该函数的详细介绍：
 * cl_context context   上下文对象，关联一组设备和内存资源。命令队列需属于某个上下文。
 *                      必须是通过 clCreateContext 或 clCreateContextFromType 创建的有效对象。
 *                      cl_device_id device
 * cl_command_queue_properties properties
 *                      命令队列属性，通过位掩码组合以下标志：
 *                      CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE: 启用乱序执行（需用户管理依赖）。
 *                      CL_QUEUE_PROFILING_ENABLE: 启用性能分析，允许通过 clGetEventProfilingInfo 获取时间信息。
 *                      0: 默认属性（顺序执行，不启用性能分析）。
 *                      若设备不支持指定属性（如乱序执行），返回 CL_INVALID_QUEUE_PROPERTIES
 * cl_int *errcode_ret
 *                      用于返回错误码。若为 NULL，则忽略错误码；否则返回 CL_SUCCESS 或具体错误码。
 *                      错误码:
 *                      CL_INVALID_CONTEXT: 上下文无效。
 *                      CL_INVALID_DEVICE: 设备不属于上下文或不支持 OpenCL。
 *                      CL_INVALID_QUEUE_PROPERTIES: 属性不被设备支持。
 *                      CL_INVALID_VALUE: 属性值非法。
 *                      CL_OUT_OF_HOST_MEMORY: 主机内存不足。
 * @return
 */
extern CL_API_ENTRY cl_command_queue CL_API_CALL
clCreateCommandQueue(cl_context                     /* context */,
                     cl_device_id                   /* device */,
                     cl_command_queue_properties    /* properties */,
                     cl_int *                       /* errcode_ret */);


cl_command_queue clCreateCommandQueueWithProperties(
                                    cl_context context,
                                    cl_device_id device,
                                    const cl_command_queue_properties *properties,
                                    cl_int *errcode_ret);


void CL_CALLBACK errorCallback(const char *err_info, const void *, size_t, void *) {
    std::cerr << "Error: " << err_info << std::endl;
}

/***
 *
 *  1.创建平台
 *  2.创建设备
 *  3.根据设备创建上下文
 */
cl_context CreateContext(cl_device_id *device) {
    cl_int errNum;
    cl_uint numPlatforms;
    cl_platform_id firstPlatformId;
    cl_context context = nullptr;
    /// 由于本机只有一个platform。索引我们这个地方就直接传入1
    errNum = clGetPlatformIDs(1, &firstPlatformId, &numPlatforms);
    if (errNum != CL_SUCCESS || numPlatforms <= 0) {
        printf("Failed to find any OpenCL platforms.");
        return nullptr;
    }
    errNum = clGetDeviceIDs(firstPlatformId, CL_DEVICE_TYPE_GPU, 1, device, nullptr);
    if (errNum != CL_SUCCESS) {
        printf("There is no GPU, trying CPU...");
        errNum = clGetDeviceIDs(firstPlatformId,CL_DEVICE_TYPE_CPU, 1, device, nullptr);
    }
    if (errNum != CL_SUCCESS) {
        printf("There is NO GPU or CPU");
        return nullptr;
    }


    // 第三步：创建对应设备的上下文对象
    /// 类型: const cl_context_properties *
    /// 指定上下文的属性列表，以键值对形式传递，以 0 或 NULL 结尾。
    /// 常用属性:
    ///         CL_CONTEXT_PLATFORM: 关联的 OpenCL 平台（cl_platform_id）。
    ///         CL_CONTEXT_INTEROP_USER_SYNC: 与外部 API（如 OpenGL/DirectX）的同步行为。
    cl_context_properties props[] = {
        CL_CONTEXT_PLATFORM, (cl_context_properties) firstPlatformId,
        0
    };
    context = clCreateContext(props, 1, device, errorCallback, nullptr, &errNum);
    if (errNum != CL_SUCCESS) {
        printf(" create context error\n");
        return nullptr;
    }
    return context;
}

/*
 **@在上下文可用的第一个设备中创建命令队列
 */
cl_command_queue CreateCommandQueue(cl_context context, cl_device_id device) {
    cl_int errNum;
    cl_command_queue commandQueue = nullptr;
    // 设置定义是否开启性能分析的开关
    cl_command_queue_properties props = CL_QUEUE_PROFILING_ENABLE;
    // 检查设备是否支持性能分析
    cl_command_queue_properties supported_props;
    clGetDeviceInfo(device, CL_DEVICE_QUEUE_PROPERTIES, sizeof(supported_props), &supported_props, nullptr);
    if (!(supported_props & CL_QUEUE_PROFILING_ENABLE)) {
        // 处理不支持的属性
        printf("当前设备不支持性能分析");
        props = 0;
    }
    /// errNum传入引入，对象数据不做拷贝
    commandQueue = clCreateCommandQueue(context, device, props, &errNum);
    if (errNum != CL_SUCCESS || commandQueue == nullptr) {
        printf("Failed to create commandQueue errNum: %d", errNum);
        return nullptr;
    }
    return commandQueue;
}

/**
 * 读取内核源码创建OpenCL程序
 **/
cl_program CreateProgram(cl_context context, cl_device_id device, const char *fileName) {
    cl_int errNum;
    cl_program program;
    size_t program_length;
    char *const source = ReadKernelSourceFile(fileName, &program_length);
    program = clCreateProgramWithSource(context, 1, (const char **) &source,NULL, NULL);
    if (program == NULL) {
        printf("Failed to create CL program from source.");
        return NULL;
    }
    errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (errNum != CL_SUCCESS) {
        char buildLog[16384];
        clGetProgramBuildInfo(program, device,CL_PROGRAM_BUILD_LOG, sizeof(buildLog), buildLog, NULL);
        printf("Error in kernel:%s ", buildLog);
        clReleaseProgram(program);
        return NULL;
    }
    return program;
}

/***
 *@创建内存对象
*/
bool CreateMemObjects(cl_context context, cl_mem memObjects[3], float *a, float *b) {
    memObjects[0] = clCreateBuffer(context,CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * ARRAY_SIZE, a,
                                   NULL);
    memObjects[1] = clCreateBuffer(context,CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * ARRAY_SIZE, b,
                                   NULL);
    memObjects[2] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * ARRAY_SIZE,NULL, NULL);
    if (memObjects[0] == NULL || memObjects[1] == NULL || memObjects[2] == NULL) {
        printf("Error creating memory objects.");
        return false;
    }
    return true;
}

/**
 * @清除OpenCL资源
 */
void Cleanup(cl_context context, cl_command_queue commandQueue, cl_program program, cl_kernel kernel,
             cl_mem memObjects[3]) {
    for (int i = 0; i < 3; i++) {
        if (memObjects[i] != 0)
            clReleaseMemObject(memObjects[i]);
    }
    if (commandQueue != 0)
        clReleaseCommandQueue(commandQueue);
    if (kernel != 0)
        clReleaseKernel(kernel);
    if (program != 0)
        clReleaseProgram(program);
    if (context != 0)
        clReleaseContext(context);
}


TEST_F(TestOpenCLVectorAdd2, TestOpenCLVectorAdd) {
    ALOGE(TAG, "==============TestOpenCLVectorAdd================");
    cl_context context = nullptr;
    cl_command_queue commandQueue = nullptr;
    cl_program program = nullptr;
    cl_device_id device = nullptr;
    cl_kernel kernel = nullptr;
    cl_mem memObjects[3] = {nullptr, nullptr, nullptr};

    cl_int errNum;
    //第一步： 获取平台和设备，然后进行创建OpenCL上下文
    context = CreateContext(&device);
    if (context == nullptr) {
        printf("Failed to create OpenCL context.");
        return;
    }


    //第二步：获得OpenCL设备,并创建命令队列
    commandQueue = CreateCommandQueue(context, device);
    if (commandQueue == nullptr) {
        Cleanup(context, commandQueue, program, kernel, memObjects);
        return;
    }
    //创建OPenCL程序
    program = CreateProgram(context, device, "kernel_vector_add.cl");
    if (program == NULL) {
        Cleanup(context, commandQueue, program,
                kernel, memObjects);
        return;
    }
    //创建OpenCL内核
    kernel = clCreateKernel(program, "vector_add", NULL);
    if (kernel == NULL) {
        printf("Failed to create kernel");
        Cleanup(context, commandQueue, program,
                kernel, memObjects);
        return;
    }
    //创建OpenCL内存对象
    float result[ARRAY_SIZE];
    float a[ARRAY_SIZE];
    float b[ARRAY_SIZE];
    for (int i = 0; i < ARRAY_SIZE; i++) {
        a[i] = (float) i;
        b[i] = (float) (i * 2);
    }
    if (!CreateMemObjects(context, memObjects, a, b)) {
        Cleanup(context, commandQueue, program,
                kernel, memObjects);
        return;
    }
    //设置内核参数
    errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem),
                            &memObjects[0]);
    errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem),
                             &memObjects[1]);
    errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem),
                             &memObjects[2]);
    if (errNum != CL_SUCCESS) {
        printf("Error setting kernel arguments.");
        Cleanup(context, commandQueue, program,
                kernel, memObjects);
        return;
    }
    size_t globalWorkSize[1] = {ARRAY_SIZE};
    size_t localWorkSize[1] = {1};
    //执行内核
    errNum = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL,NULL);
    if (errNum != CL_SUCCESS) {
        printf("Error queuing kernel for execution.");
        Cleanup(context, commandQueue, program, kernel,
                memObjects);
        return;
    }
    //计算结果拷贝回主机
    errNum = clEnqueueReadBuffer(commandQueue, memObjects[2],CL_TRUE, 0, ARRAY_SIZE * sizeof(float), result, 0, NULL,
                                 NULL);
    if (errNum != CL_SUCCESS) {
        printf("Error reading result buffer.");
        Cleanup(context, commandQueue, program, kernel,
                memObjects);
        return;
    }
    for (int i = 0; i < ARRAY_SIZE; i++) {
        printf("i=%d:%f\n", i, result[i]);
    }
    printf("Executed program succesfully.");
    Cleanup(context, commandQueue, program, kernel, memObjects);
}
