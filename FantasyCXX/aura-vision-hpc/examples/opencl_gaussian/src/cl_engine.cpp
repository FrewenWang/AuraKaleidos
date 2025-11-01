//
// Created by wangzhijiang on 25-5-7.
//
#include "cl_engine.h"

#include <cmath>
#include <cstring>
#include <iostream>

#ifdef QCOM
#include <CL/cl_ext_qcom.h>
#endif

namespace gaussian
{

#define CHECK_ERROR_RET(err)                                                            \
if (err != CL_SUCCESS)                                                                  \
{                                                                                       \
    fprintf(stderr, "OpenCL Error: %d @ %s:%d\n", err, __FILE__, __LINE__);             \
    return err;                                                                         \
}

CLEngine::CLEngine(): context(NULL), platform(), device(), queue(), program(), kernel()
{
}

CLEngine::~CLEngine()
{
    ClRelease();
}


cl_int CLEngine::QueryPlatforms()
{
    cl_int ret = CL_SUCCESS;
    cl_uint num_platforms = 0;
    ret = clGetPlatformIDs(0,NULL, &num_platforms);
    CHECK_ERROR_RET(ret);

    ret = clGetPlatformIDs(1, &(platform.platform_id), &num_platforms);
    CHECK_ERROR_RET(ret);
    return ret;
}

cl_int CLEngine::QueryDevices()
{
    cl_int ret = CL_SUCCESS;
    cl_uint num_devices = 0;

    ret = clGetDeviceIDs(platform.platform_id,CL_DEVICE_TYPE_GPU, 0,NULL, &num_devices);
    CHECK_ERROR_RET(ret);

    ret = clGetDeviceIDs(platform.platform_id,CL_DEVICE_TYPE_GPU, 1, &device.device_id, &num_devices);
    CHECK_ERROR_RET(ret);

    // get device max work group size
    ret = clGetDeviceInfo(device.device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &device.max_work_group_size,
                          NULL);
    CHECK_ERROR_RET(ret);
    // 查询各维度最大尺寸
    ret = clGetDeviceInfo(device.device_id, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t) * 3,
                          &device.max_work_group_size,
                          NULL);
    CHECK_ERROR_RET(ret);

    printf("Max work-group size: %zu \n", device.max_work_group_size);

    cl_device_svm_capabilities caps;
    ret = clGetDeviceInfo(device.device_id, CL_DEVICE_SVM_CAPABILITIES, sizeof(caps), &caps, NULL);
    CHECK_ERROR_RET(ret);
    printf("cl_device_svm_capabilities: %zu \n", caps);

    cl_device_fp_config fp_config;
    clGetDeviceInfo(device.device_id, CL_DEVICE_DOUBLE_FP_CONFIG, sizeof(fp_config), &fp_config, NULL);
    if (fp_config == 0)
    {
        printf("device not support double-precision floating-point arithmetic, changed to float\n");
    }

    // cl_uint maxWidth, maxHeight;
    // clGetDeviceInfo(device.device_id, CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof(maxWidth), &maxWidth, NULL);
    // clGetDeviceInfo(device.device_id, CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof(maxHeight), &maxHeight, NULL);

    ret = clGetDeviceInfo(device.device_id, CL_DEVICE_IMAGE_PITCH_ALIGNMENT, sizeof(cl_uint),
                          &device.image_pitch_alignment, NULL);
    CHECK_ERROR_RET(ret);
    printf("cl_device_image_pitch_alignment pitch_align: %u\n", device.image_pitch_alignment);
    return ret;
}

cl_int CLEngine::CreateContext()
{
    cl_int ret = CL_SUCCESS;
    cl_context_properties properties[16] = {0};
    int idx = 0;
    // create context
    properties[idx++] = CL_CONTEXT_PLATFORM;
    properties[idx++] = (cl_context_properties)platform.platform_id;
#ifdef QCOM
    properties[idx++] = CL_CONTEXT_PERF_HINT_QCOM;
    properties[idx++] = CL_PERF_HINT_HIGH_QCOM;
    properties[idx++] = CL_CONTEXT_PRIORITY_HINT_QCOM;
    properties[idx++] = CL_PRIORITY_HINT_HIGH_QCOM;
#endif
    this->context = clCreateContext(properties, 1, &device.device_id,NULL,NULL, &ret);
    return ret;
}

cl_int CLEngine::CreateCommandQueue()
{
    cl_int ret = CL_SUCCESS;
#if CL_TARGET_OPENCL_VERSION >= 200
    cl_command_queue_properties cmd_properties = 0;
    cmd_properties |= CL_QUEUE_PROFILING_ENABLE;
    cl_queue_properties queue_properties[] = {CL_QUEUE_PROPERTIES, cmd_properties, 0};
    this->queue = clCreateCommandQueueWithProperties(context, device.device_id, queue_properties, &ret);
#else
    this->queue = clCreateCommandQueue(context, device.device_id, 0, &ret);
#endif

    return ret;
}

cl_int CLEngine::CreateProgram(const char *kernel_src)
{
    cl_int ret = CL_SUCCESS;
    program = clCreateProgramWithSource(context, 1, &kernel_src, NULL, &ret);
    CHECK_ERROR_RET(ret);

    ret = clBuildProgram(program, 1, &device.device_id, "-cl-std=CL2.0", NULL, NULL);
    if (ret != CL_SUCCESS)
    {
        size_t log_size;
        clGetProgramBuildInfo(program, device.device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = static_cast<char *>(malloc(log_size));
        clGetProgramBuildInfo(program, device.device_id, CL_PROGRAM_BUILD_LOG,
                              log_size, log, NULL);
        clReleaseProgram(program);
        fprintf(stderr, "clBuildProgram failed. ret=%d\n Log:%s", ret, log);
        free(log);
        return ret;
    }
    return ret;
}

cl_int CLEngine::CreateKernel(const char *kernel_name)
{
    cl_int ret = CL_SUCCESS;
    // return error code：
    // CL_INVALID_PROGRAM(-44)
    // CL_INVALID_PROGRAM_EXECUTABLE(-45)
    // CL_INVALID_KERNEL_NAME(-46)
    // CL_INVALID_KERNEL_DEFINITION(-47)
    // CL_INVALID_VALUE(-30)
    // CL_OUT_OF_RESOURCES(-5)
    // CL_OUT_OF_HOST_MEMORY(-6)
    kernel = clCreateKernel(program, kernel_name, &ret);
    CHECK_ERROR_RET(ret);

    size_t max_work_group_size;
    size_t perferred_work_group_size_multiple;
    ret = clGetKernelWorkGroupInfo(kernel, device.device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t),
                                   &max_work_group_size,
                                   NULL);
    ret |= clGetKernelWorkGroupInfo(kernel, device.device_id, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
                                    sizeof(size_t),
                                    &perferred_work_group_size_multiple, NULL);
    CHECK_ERROR_RET(ret);
    if (ret != CL_SUCCESS)
        printf("Kernel %s max workgroup size=%zu\n", kernel_name, max_work_group_size);
    printf("Kernel %s perferred workgroup size multiple=%zu\n", kernel_name, perferred_work_group_size_multiple);

    return ret;
}

void CLEngine::GetProfilingInfo(cl_event &event)
{
    cl_ulong t_queued;
    cl_ulong t_submitted;
    cl_ulong t_started;
    cl_ulong t_ended;
    cl_ulong t_completed;

    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), &t_queued, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_SUBMIT, sizeof(cl_ulong), &t_submitted, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &t_started, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &t_ended, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_COMPLETE, sizeof(cl_ulong), &t_completed, NULL);

#ifdef ANDROID
    printf("queue -> submit : %fms\n", (t_submitted - t_queued) * 1e-6);
    printf("submit -> start : %fms\n", (t_started - t_submitted) * 1e-6);
    printf("start -> end : %fms\n", (t_ended - t_started) * 1e-6);
    printf("end -> finish : %fms\n", (t_completed - t_ended) * 1e-6);
#endif

}

ClPlatformInfo &CLEngine::GetPlatformInfo()
{
    return platform;
}

ClDeviceInfo &CLEngine::GetDevicesInfo()
{
    return device;
}

cl_command_queue &CLEngine::GetCommandQueue()
{
    return queue;
}

cl_program &CLEngine::GetProgram()
{
    return program;
}

cl_context &CLEngine::GetContext()
{
    return context;
}

cl_kernel &CLEngine::GetKernel()
{
    return kernel;
}

void CLEngine::ClRelease()
{
    if (NULL != kernel)
    {
        clReleaseKernel(kernel);
        kernel = NULL;
    }
    if (NULL != program)
    {
        clReleaseProgram(program);
        program = NULL;
    }
    if (NULL != queue)
    {
        clReleaseCommandQueue(queue);
        queue = NULL;
    }
    if (NULL != context)
    {
        clReleaseContext(context);
        context = NULL;
    }
}
}
