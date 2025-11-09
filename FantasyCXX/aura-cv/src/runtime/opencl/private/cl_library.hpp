#ifndef AURA_RUNTIME_OPENCL_CL_LIBRARY_HPP__
#define AURA_RUNTIME_OPENCL_CL_LIBRARY_HPP__

#include "aura/runtime/core.h"

#include <mutex>

#include "CL/cl.h"
#include "CL/cl_ext.h"

/**
 * @defgroup runtime Runtime
 * @{
 *      @defgroup cl OpenCL
 * @}
 */

namespace aura
{

/**
 * @addtogroup cl
 * @{
 */

/**
 * @brief Represents a manager for loading and unloading OpenCL libraries.
 *
 * This class manages the loading and unloading of OpenCL libraries. It ensures
 * thread safety and provides methods for loading libraries from specified paths.
 */
class CLLibrary final
{
public:
    /**
     * @brief Provides access to the singleton instance of CLLibrary.
     *
     * @return Reference to the CLLibrary singleton.
     */
    static CLLibrary& Get();

    /* https://github.com/KhronosGroup/OpenCL-Headers/releases/tag/v2021.06.30 */
    /* Platform API */
    AURA_API_DEF(clGetPlatformIDs)  = cl_int (CL_API_CALL*)(cl_uint, cl_platform_id*, cl_uint*);
    AURA_API_PTR(clGetPlatformIDs);

    AURA_API_DEF(clGetPlatformInfo) = cl_int (CL_API_CALL*)(cl_platform_id, cl_platform_info, size_t, void*, size_t*);
    AURA_API_PTR(clGetPlatformInfo);

    /* Device APIs */
    AURA_API_DEF(clGetDeviceIDs)    = cl_int (CL_API_CALL*)(cl_platform_id, cl_device_type, cl_uint, cl_device_id*, cl_uint*);
    AURA_API_PTR(clGetDeviceIDs);

    AURA_API_DEF(clGetDeviceInfo)   = cl_int (CL_API_CALL*)(cl_device_id, cl_device_info, size_t, void*, size_t*);
    AURA_API_PTR(clGetDeviceInfo);

#if defined(CL_VERSION_1_2)
    AURA_API_DEF(clCreateSubDevices) = cl_int (CL_API_CALL*)(cl_device_id, const cl_device_partition_property*, cl_uint, cl_device_id*, cl_uint*);
    AURA_API_PTR(clCreateSubDevices);

    AURA_API_DEF(clRetainDevice)     = cl_int (CL_API_CALL*)(cl_device_id);
    AURA_API_PTR(clRetainDevice);

    AURA_API_DEF(clReleaseDevice)    = cl_int (CL_API_CALL*)(cl_device_id);
    AURA_API_PTR(clReleaseDevice);
#endif // CL_VERSION_1_2

#if defined(CL_VERSION_2_1)
    AURA_API_DEF(clSetDefaultDeviceCommandQueue) = cl_int (CL_API_CALL*)(cl_context, cl_device_id, cl_command_queue);
    AURA_API_PTR(clSetDefaultDeviceCommandQueue);

    AURA_API_DEF(clGetDeviceAndHostTimer)        = cl_int (CL_API_CALL*)(cl_device_id, cl_ulong*, cl_ulong*);
    AURA_API_PTR(clGetDeviceAndHostTimer);

    AURA_API_DEF(clGetHostTimer)                 = cl_int (CL_API_CALL*)(cl_device_id, cl_ulong*);
    AURA_API_PTR(clGetHostTimer);
#endif //CL_VERSION_2_1

    /* Context APIs */
    AURA_API_DEF(clCreateContext)         = cl_context (CL_API_CALL*)(const cl_context_properties*, cl_uint, const cl_device_id*, void (CL_CALLBACK*)(const char*, const void*, size_t, void*), void*, cl_int*);
    AURA_API_PTR(clCreateContext);

    AURA_API_DEF(clCreateContextFromType) = cl_context (CL_API_CALL*)(const cl_context_properties*, cl_device_type, void (CL_CALLBACK*)(const char*, const void*, size_t, void*), void*, cl_int*);
    AURA_API_PTR(clCreateContextFromType);

    AURA_API_DEF(clRetainContext)         = cl_int (CL_API_CALL*)(cl_context);
    AURA_API_PTR(clRetainContext);

    AURA_API_DEF(clReleaseContext)        = cl_int (CL_API_CALL*)(cl_context);
    AURA_API_PTR(clReleaseContext);

    AURA_API_DEF(clGetContextInfo)        = cl_int (CL_API_CALL*)(cl_context, cl_context_info, size_t, void*, size_t*);
    AURA_API_PTR(clGetContextInfo);

#if defined(CL_VERSION_3_0)
    AURA_API_DEF(clSetContextDestructorCallback)     = cl_int (CL_API_CALL*)(cl_context, void (CL_CALLBACK*)(cl_context, void*), void*);
    AURA_API_PTR(clSetContextDestructorCallback);
#endif // CL_VERSION_3_0

    /* Command Queue APIs */
#if defined(CL_VERSION_2_0)
    AURA_API_DEF(clCreateCommandQueueWithProperties) = cl_command_queue (CL_API_CALL*)(cl_context, cl_device_id, const cl_queue_properties*, cl_int*);
    AURA_API_PTR(clCreateCommandQueueWithProperties);
#endif

    AURA_API_DEF(clRetainCommandQueue)  = cl_int (CL_API_CALL*)(cl_command_queue);
    AURA_API_PTR(clRetainCommandQueue);

    AURA_API_DEF(clReleaseCommandQueue) = cl_int (CL_API_CALL*)(cl_command_queue);
    AURA_API_PTR(clReleaseCommandQueue);

    AURA_API_DEF(clGetCommandQueueInfo) = cl_int (CL_API_CALL*)(cl_command_queue, cl_command_queue_info, size_t, void*, size_t*);
    AURA_API_PTR(clGetCommandQueueInfo);

    /* Memory Object APIs */
    AURA_API_DEF(clCreateBuffer)    = cl_mem (CL_API_CALL*)(cl_context, cl_mem_flags, size_t, void*, cl_int*);
    AURA_API_PTR(clCreateBuffer);

#if defined(CL_VERSION_1_1)
    AURA_API_DEF(clCreateSubBuffer) = cl_mem (CL_API_CALL*)(cl_mem, cl_mem_flags, cl_buffer_create_type, const void*, cl_int*);
    AURA_API_PTR(clCreateSubBuffer);
#endif //CL_VERSION_1_1

#if defined(CL_VERSION_1_2)
    AURA_API_DEF(clCreateIaura)     = cl_mem (CL_API_CALL*)(cl_context, cl_mem_flags, const cl_iaura_format*, const cl_iaura_desc*, void*, cl_int*);
    AURA_API_PTR(clCreateIaura);
#endif // CL_VERSION_1_2

#if defined(CL_VERSION_2_0)
    AURA_API_DEF(clCreatePipe)      = cl_mem (CL_API_CALL*)(cl_context, cl_mem_flags, cl_uint, cl_uint, const cl_pipe_properties*, cl_int*);
    AURA_API_PTR(clCreatePipe);
#endif //CL_VERSION_2_0

#if defined(CL_VERSION_3_0)
    AURA_API_DEF(clCreateBufferWithProperties) = cl_mem (CL_API_CALL*)(cl_context, const cl_mem_properties*, cl_mem_flags, size_t, void*, cl_int*);
    AURA_API_PTR(clCreateBufferWithProperties);

    AURA_API_DEF(clCreateIauraWithProperties)  = cl_mem (CL_API_CALL*)(cl_context, const cl_mem_properties*, cl_mem_flags, const cl_iaura_format*, const cl_iaura_desc*, void*, cl_int*);
    AURA_API_PTR(clCreateIauraWithProperties);
#endif // CL_VERSION_3_0

    AURA_API_DEF(clRetainMemObject)          = cl_int (CL_API_CALL*)(cl_mem);
    AURA_API_PTR(clRetainMemObject);

    AURA_API_DEF(clReleaseMemObject)         = cl_int (CL_API_CALL*)(cl_mem);
    AURA_API_PTR(clReleaseMemObject);

    AURA_API_DEF(clGetSupportedIauraFormats) = cl_int (CL_API_CALL*)(cl_context, cl_mem_flags, cl_mem_object_type, cl_uint, cl_iaura_format*, cl_uint*);
    AURA_API_PTR(clGetSupportedIauraFormats);

    AURA_API_DEF(clGetMemObjectInfo)         = cl_int (CL_API_CALL*)(cl_mem, cl_mem_info, size_t, void*, size_t*);
    AURA_API_PTR(clGetMemObjectInfo);

    AURA_API_DEF(clGetIauraInfo)             = cl_int (CL_API_CALL*)(cl_mem, cl_iaura_info, size_t, void*, size_t*);
    AURA_API_PTR(clGetIauraInfo);

#if defined(CL_VERSION_2_0)
    AURA_API_DEF(clGetPipeInfo)              = cl_int (CL_API_CALL*)(cl_mem, cl_pipe_info, size_t, void*, size_t*);
    AURA_API_PTR(clGetPipeInfo);
#endif // CL_VERSION_2_0

#if defined(CL_VERSION_1_1)
    AURA_API_DEF(clSetMemObjectDestructorCallback) = cl_int (CL_API_CALL*)(cl_mem, void (CL_CALLBACK*)(cl_mem, void*), void*);
    AURA_API_PTR(clSetMemObjectDestructorCallback);
#endif // CL_VERSION_1_1

    /* SVM Allocation APIs */
#if defined(CL_VERSION_2_0)
    AURA_API_DEF(clSVMAlloc)   = void* (CL_API_CALL*)(cl_context, cl_svm_mem_flags, size_t, cl_uint);
    AURA_API_PTR(clSVMAlloc);

    AURA_API_DEF(clSVMFree) = void (CL_API_CALL*)(cl_context, void*);
    AURA_API_PTR(clSVMFree);
#endif // CL_VERSION_2_0

    /* Sampler APIs */
#if defined(CL_VERSION_2_0)
    AURA_API_DEF(clCreateSamplerWithProperties) = cl_sampler (CL_API_CALL*)(cl_context, const cl_sampler_properties*, cl_int*);
    AURA_API_PTR(clCreateSamplerWithProperties);
#endif // CL_VERSION_2_0

    AURA_API_DEF(clRetainSampler)  = cl_int (CL_API_CALL*)(cl_sampler);
    AURA_API_PTR(clRetainSampler);

    AURA_API_DEF(clReleaseSampler) = cl_int (CL_API_CALL*)(cl_sampler);
    AURA_API_PTR(clReleaseSampler);

    AURA_API_DEF(clGetSamplerInfo) = cl_int (CL_API_CALL*)(cl_sampler, cl_sampler_info, size_t, void*, size_t*);
    AURA_API_PTR(clGetSamplerInfo);

    /* Program Object APIs */
    AURA_API_DEF(clCreateProgramWithSource) = cl_program (CL_API_CALL*)(cl_context, cl_uint, const char**, const size_t*, cl_int*);
    AURA_API_PTR(clCreateProgramWithSource);

    AURA_API_DEF(clCreateProgramWithBinary) = cl_program (CL_API_CALL*)(cl_context, cl_uint, const cl_device_id*, const size_t*, const unsigned char**, cl_int*, cl_int*);
    AURA_API_PTR(clCreateProgramWithBinary);

#if defined(CL_VERSION_1_2)
    AURA_API_DEF(clCreateProgramWithBuiltInKernels) = cl_program (CL_API_CALL*)(cl_context, cl_uint, const cl_device_id*, const char*, cl_int*);
    AURA_API_PTR(clCreateProgramWithBuiltInKernels);
#endif // CL_VERSION_1_2

#if defined(CL_VERSION_2_1)
    AURA_API_DEF(clCreateProgramWithIL) = cl_program (CL_API_CALL*)(cl_context, const void*, size_t, cl_int*);
    AURA_API_PTR(clCreateProgramWithIL);
#endif // CL_VERSION_2_1

    AURA_API_DEF(clRetainProgram)  = cl_int (CL_API_CALL*)(cl_program);
    AURA_API_PTR(clRetainProgram);

    AURA_API_DEF(clReleaseProgram) = cl_int (CL_API_CALL*)(cl_program);
    AURA_API_PTR(clReleaseProgram);

    AURA_API_DEF(clBuildProgram)   = cl_int (CL_API_CALL*)(cl_program, cl_uint, const cl_device_id*, const char*, void (CL_CALLBACK*)(cl_program,void*), void*);
    AURA_API_PTR(clBuildProgram);

#if defined(CL_VERSION_1_2)
    AURA_API_DEF(clCompileProgram) = cl_int (CL_API_CALL*)(cl_program, cl_uint, const cl_device_id*, const char*, cl_uint, const cl_program*, const char**, void (CL_CALLBACK*)(cl_program, void*), void*);
    AURA_API_PTR(clCompileProgram);

    AURA_API_DEF(clLinkProgram)    = cl_program (CL_API_CALL*)(cl_context, cl_uint, const cl_device_id*, const char*, cl_uint, const cl_program*, void (CL_CALLBACK*)(cl_program, void*), void*, cl_int*);
    AURA_API_PTR(clLinkProgram);
#endif // CL_VERSION_1_2

#if defined(CL_VERSION_2_2)
    AURA_API_DEF(clSetProgramReleaseCallback)        = cl_int (CL_API_CALL*)(cl_program, void (CL_CALLBACK*)(cl_program, void*), void*);
    AURA_API_PTR(clSetProgramReleaseCallback);

    AURA_API_DEF(clSetProgramSpecializationConstant) = cl_int (CL_API_CALL*)(cl_program, cl_uint, size_t, const void*);
    AURA_API_PTR(clSetProgramSpecializationConstant);
#endif // CL_VERSION_2_2

#if defined(CL_VERSION_1_2)
    AURA_API_DEF(clUnloadPlatformCompiler) = cl_int (CL_API_CALL*)(cl_platform_id);
    AURA_API_PTR(clUnloadPlatformCompiler);
#endif // CL_VERSION_1_2

    AURA_API_DEF(clGetProgramInfo)         = cl_int (CL_API_CALL*)(cl_program, cl_program_info, size_t, void*, size_t*);
    AURA_API_PTR(clGetProgramInfo);

    AURA_API_DEF(clGetProgramBuildInfo)    = cl_int (CL_API_CALL*)(cl_program, cl_device_id, cl_program_build_info, size_t, void*, size_t*);
    AURA_API_PTR(clGetProgramBuildInfo);

    /* Kernel Object APIs */
    AURA_API_DEF(clCreateKernel)           = cl_kernel (CL_API_CALL*)(cl_program, const char*, cl_int*);
    AURA_API_PTR(clCreateKernel);

    AURA_API_DEF(clCreateKernelsInProgram) = cl_int (CL_API_CALL*)(cl_program, cl_uint, cl_kernel*, cl_uint*);
    AURA_API_PTR(clCreateKernelsInProgram);

#if defined(CL_VERSION_2_1)
    AURA_API_DEF(clCloneKernel)   = cl_kernel (CL_API_CALL*)(cl_kernel, cl_int*);
    AURA_API_PTR(clCloneKernel);
#endif // CL_VERSION_2_1

    AURA_API_DEF(clRetainKernel)  = cl_int (CL_API_CALL*)(cl_kernel);
    AURA_API_PTR(clRetainKernel);

    AURA_API_DEF(clReleaseKernel) = cl_int (CL_API_CALL*)(cl_kernel);
    AURA_API_PTR(clReleaseKernel);

    AURA_API_DEF(clSetKernelArg)  = cl_int (CL_API_CALL*)(cl_kernel, cl_uint, size_t, const void*);
    AURA_API_PTR(clSetKernelArg);

#if defined(CL_VERSION_2_0)
    AURA_API_DEF(clSetKernelArgSVMPointer) = cl_int (CL_API_CALL*)(cl_kernel, cl_uint, const void*);
    AURA_API_PTR(clSetKernelArgSVMPointer);

    AURA_API_DEF(clSetKernelExecInfo)      = cl_int (CL_API_CALL*)(cl_kernel, cl_kernel_exec_info, size_t, const void*);
    AURA_API_PTR(clSetKernelExecInfo);
#endif // CL_VERSION_2_0

    AURA_API_DEF(clGetKernelInfo) = cl_int (CL_API_CALL*)(cl_kernel, cl_kernel_info, size_t, void*, size_t*);
    AURA_API_PTR(clGetKernelInfo);

#if defined(CL_VERSION_1_2)
    AURA_API_DEF(clGetKernelArgInfo) = cl_int (CL_API_CALL*)(cl_kernel, cl_uint, cl_kernel_arg_info, size_t, void*, size_t*);
    AURA_API_PTR(clGetKernelArgInfo);
#endif // CL_VERSION_1_2

    AURA_API_DEF(clGetKernelWorkGroupInfo) = cl_int (CL_API_CALL*)(cl_kernel, cl_device_id, cl_kernel_work_group_info, size_t, void*, size_t*);
    AURA_API_PTR(clGetKernelWorkGroupInfo);

#if defined(CL_VERSION_2_1)
    AURA_API_DEF(clGetKernelSubGroupInfo) = cl_int (CL_API_CALL*)(cl_kernel, cl_device_id, cl_kernel_sub_group_info, size_t, const void*, size_t, void*, size_t*);
    AURA_API_PTR(clGetKernelSubGroupInfo);
#endif // CL_VERSION_2_1

    /* Event Object APIs */
    AURA_API_DEF(clWaitForEvents) = cl_int (CL_API_CALL*)(cl_uint, const cl_event*);
    AURA_API_PTR(clWaitForEvents);

    AURA_API_DEF(clGetEventInfo)  = cl_int (CL_API_CALL*)(cl_event, cl_event_info, size_t, void*, size_t*);
    AURA_API_PTR(clGetEventInfo);

#if defined(CL_VERSION_1_1)
    AURA_API_DEF(clCreateUserEvent) = cl_event (CL_API_CALL*)(cl_context, cl_int*);
    AURA_API_PTR(clCreateUserEvent);
#endif // CL_VERSION_1_1

    AURA_API_DEF(clRetainEvent)  = cl_int (CL_API_CALL*)(cl_event);
    AURA_API_PTR(clRetainEvent);

    AURA_API_DEF(clReleaseEvent) = cl_int (CL_API_CALL*)(cl_event);
    AURA_API_PTR(clReleaseEvent);

#if defined(CL_VERSION_1_1)
    AURA_API_DEF(clSetUserEventStatus) = cl_int (CL_API_CALL*)(cl_event, cl_int);
    AURA_API_PTR(clSetUserEventStatus);

    AURA_API_DEF(clSetEventCallback)   = cl_int (CL_API_CALL*)(cl_event, cl_int, void (CL_CALLBACK*)(cl_event, cl_int, void*), void*);
    AURA_API_PTR(clSetEventCallback);
#endif // CL_VERSION_1_1

    /* Profiling APIs */
    AURA_API_DEF(clGetEventProfilingInfo) = cl_int (CL_API_CALL*)(cl_event, cl_profiling_info, size_t, void*, size_t*);
    AURA_API_PTR(clGetEventProfilingInfo);

    /* Flush and Finish APIs */
    AURA_API_DEF(clFlush)  = cl_int (CL_API_CALL*)(cl_command_queue);
    AURA_API_PTR(clFlush);

    AURA_API_DEF(clFinish) = cl_int (CL_API_CALL*)(cl_command_queue);
    AURA_API_PTR(clFinish);

    /* Enqueued Commands APIs */
    AURA_API_DEF(clEnqueueReadBuffer) = cl_int (CL_API_CALL*)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, void*, cl_uint, const cl_event*, cl_event*);
    AURA_API_PTR(clEnqueueReadBuffer);

#if defined(CL_VERSION_1_1)
    AURA_API_DEF(clEnqueueReadBufferRect) = cl_int (CL_API_CALL*)(cl_command_queue, cl_mem, cl_bool, const size_t*, const size_t*, const size_t*, size_t, size_t, size_t, size_t, void*, cl_uint, const cl_event*, cl_event*);
    AURA_API_PTR(clEnqueueReadBufferRect);
#endif // CL_VERSION_1_1

    AURA_API_DEF(clEnqueueWriteBuffer)    = cl_int (CL_API_CALL*)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void*, cl_uint, const cl_event*, cl_event*);
    AURA_API_PTR(clEnqueueWriteBuffer);

#if defined(CL_VERSION_1_1)
    AURA_API_DEF(clEnqueueWriteBufferRect) = cl_int (CL_API_CALL*)(cl_command_queue, cl_mem, cl_bool, const size_t*, const size_t*, const size_t*, size_t, size_t, size_t, size_t, const void*, cl_uint, const cl_event*, cl_event*);
    AURA_API_PTR(clEnqueueWriteBufferRect);
#endif // CL_VERSION_1_1

#if defined(CL_VERSION_1_2)
    AURA_API_DEF(clEnqueueFillBuffer) = cl_int (CL_API_CALL*)(cl_command_queue, cl_mem, const void *, size_t, size_t, size_t, cl_uint, const cl_event*, cl_event*);
    AURA_API_PTR(clEnqueueFillBuffer);
#endif // CL_VERSION_1_2

    AURA_API_DEF(clEnqueueCopyBuffer) = cl_int  (CL_API_CALL*)(cl_command_queue, cl_mem, cl_mem, size_t, size_t, size_t, cl_uint, const cl_event*, cl_event*);
    AURA_API_PTR(clEnqueueCopyBuffer);

#if defined(CL_VERSION_1_1)
    AURA_API_DEF(clEnqueueCopyBufferRect) = cl_int (CL_API_CALL*)(cl_command_queue, cl_mem, cl_mem, const size_t*, const size_t*, const size_t*, size_t, size_t, size_t, size_t, cl_uint, const cl_event*, cl_event*);
    AURA_API_PTR(clEnqueueCopyBufferRect);
#endif // CL_VERSION_1_1

    AURA_API_DEF(clEnqueueReadIaura)  = cl_int (CL_API_CALL*)(cl_command_queue, cl_mem, cl_bool, const size_t*, const size_t*, size_t, size_t, void*, cl_uint, const cl_event*, cl_event*);
    AURA_API_PTR(clEnqueueReadIaura);

    AURA_API_DEF(clEnqueueWriteIaura) = cl_int (CL_API_CALL*)(cl_command_queue, cl_mem, cl_bool, const size_t*, const size_t*, size_t, size_t, const void*, cl_uint, const cl_event*, cl_event*);
    AURA_API_PTR(clEnqueueWriteIaura);

#if defined(CL_VERSION_1_2)
    AURA_API_DEF(clEnqueueFillIaura) = cl_int (CL_API_CALL*)(cl_command_queue, cl_mem, const void*, const size_t*, const size_t*, cl_uint, const cl_event*, cl_event*);
    AURA_API_PTR(clEnqueueFillIaura);
#endif // CL_VERSION_1_2

    AURA_API_DEF(clEnqueueCopyIaura)         = cl_int (CL_API_CALL*)(cl_command_queue, cl_mem, cl_mem, const size_t*, const size_t*, const size_t*, cl_uint, const cl_event*, cl_event*);
    AURA_API_PTR(clEnqueueCopyIaura);

    AURA_API_DEF(clEnqueueCopyIauraToBuffer) = cl_int (CL_API_CALL*)(cl_command_queue, cl_mem, cl_mem, const size_t*, const size_t*, size_t, cl_uint, const cl_event*, cl_event*);
    AURA_API_PTR(clEnqueueCopyIauraToBuffer);

    AURA_API_DEF(clEnqueueCopyBufferToIaura) = cl_int (CL_API_CALL*)(cl_command_queue, cl_mem, cl_mem, size_t, const size_t*, const size_t*, cl_uint, const cl_event*, cl_event*);
    AURA_API_PTR(clEnqueueCopyBufferToIaura);

    AURA_API_DEF(clEnqueueMapBuffer)         = void* (CL_API_CALL*)(cl_command_queue, cl_mem, cl_bool, cl_map_flags, size_t, size_t, cl_uint, const cl_event*, cl_event*, cl_int*);
    AURA_API_PTR(clEnqueueMapBuffer);

    AURA_API_DEF(clEnqueueMapIaura)          = void* (CL_API_CALL*)(cl_command_queue, cl_mem, cl_bool, cl_map_flags, const size_t*, const size_t*, size_t*, size_t*, cl_uint, const cl_event*, cl_event*, cl_int*);
    AURA_API_PTR(clEnqueueMapIaura);

    AURA_API_DEF(clEnqueueUnmapMemObject)    = cl_int (CL_API_CALL*)(cl_command_queue, cl_mem, void*, cl_uint, const cl_event*, cl_event*);
    AURA_API_PTR(clEnqueueUnmapMemObject);

#if defined(CL_VERSION_1_2)
    AURA_API_DEF(clEnqueueMigrateMemObjects) = cl_int (CL_API_CALL*)(cl_command_queue, cl_uint, const cl_mem*, cl_mem_migration_flags, cl_uint, const cl_event*, cl_event*);
    AURA_API_PTR(clEnqueueMigrateMemObjects);
#endif // CL_VERSION_1_2

    AURA_API_DEF(clEnqueueNDRangeKernel) = cl_int (CL_API_CALL*)(cl_command_queue, cl_kernel, cl_uint, const size_t*, const size_t*, const size_t*, cl_uint, const cl_event*, cl_event*);
    AURA_API_PTR(clEnqueueNDRangeKernel);

    AURA_API_DEF(clEnqueueNativeKernel)  = cl_int (CL_API_CALL*)(cl_command_queue, void (CL_CALLBACK*)(void*), void*, size_t, cl_uint, const cl_mem*, const void**, cl_uint, const cl_event*, cl_event*);
    AURA_API_PTR(clEnqueueNativeKernel);

#if defined(CL_VERSION_1_2)
    AURA_API_DEF(clEnqueueMarkerWithWaitList)  = cl_int (CL_API_CALL*)(cl_command_queue, cl_uint, const cl_event*, cl_event*);
    AURA_API_PTR(clEnqueueMarkerWithWaitList);

    AURA_API_DEF(clEnqueueBarrierWithWaitList) = cl_int (CL_API_CALL*)(cl_command_queue, cl_uint, const cl_event*, cl_event*);
    AURA_API_PTR(clEnqueueBarrierWithWaitList);
#endif // CL_VERSION_1_2

#if defined(CL_VERSION_2_0)
    AURA_API_DEF(clEnqueueSVMFree)    = cl_int (CL_API_CALL*)(cl_command_queue, cl_uint, void* p[], void (CL_CALLBACK*)(cl_command_queue, cl_uint, void* p[], void*), void*, cl_uint, const cl_event*, cl_event*);
    AURA_API_PTR(clEnqueueSVMFree);

    AURA_API_DEF(clEnqueueSVMMemcpy)  = cl_int (CL_API_CALL*)(cl_command_queue, cl_bool, void*, const void*, size_t, cl_uint, const cl_event*, cl_event*);
    AURA_API_PTR(clEnqueueSVMMemcpy);

    AURA_API_DEF(clEnqueueSVMMemFill) = cl_int (CL_API_CALL*)(cl_command_queue, void*, const void*, size_t, size_t, cl_uint, const cl_event*, cl_event*);
    AURA_API_PTR(clEnqueueSVMMemFill);

    AURA_API_DEF(clEnqueueSVMMap)     = cl_int (CL_API_CALL*)(cl_command_queue, cl_bool, cl_map_flags, void*, size_t, cl_uint, const cl_event*, cl_event*);
    AURA_API_PTR(clEnqueueSVMMap);

    AURA_API_DEF(clEnqueueSVMUnmap)   = cl_int (CL_API_CALL*)(cl_command_queue, void*, cl_uint, const cl_event*, cl_event*);
    AURA_API_PTR(clEnqueueSVMUnmap);
#endif //CL_VERSION_2_0

#if defined(CL_VERSION_2_1)
    AURA_API_DEF(clEnqueueSVMMigrateMem) = cl_int (CL_API_CALL*)(cl_command_queue, cl_uint, const void**, const size_t*, cl_mem_migration_flags, cl_uint, const cl_event*, cl_event*);
    AURA_API_PTR(clEnqueueSVMMigrateMem);
#endif // CL_VERSION_2_1

#if defined(CL_VERSION_1_2)
    /* Extension function access
    *
    * Returns the extension function address for the given function name,
    * or NULL if a valid function can not be found.  The client must
    * check to make sure the address is not NULL, before using or
    * calling the returned function address.
    */
    AURA_API_DEF(clGetExtensionFunctionAddressForPlatform) = void* (CL_API_CALL*)(cl_platform_id, const char*);
    AURA_API_PTR(clGetExtensionFunctionAddressForPlatform);
#endif // CL_VERSION_1_2

#if defined(CL_USE_DEPRECATED_OPENCL_1_0_APIS)
    /*
    *  WARNING:
    *     This API introduces mutable state into the OpenCL implementation. It has been REMOVED
    *  to better facilitate thread safety.  The 1.0 API is not thread safe. It is not tested by the
    *  OpenCL 1.1 conformance test, and consequently may not work or may not work dependably.
    *  It is likely to be non-performant. Use of this API is not advised. Use at your own risk.
    *
    *  Software developers previously relying on this API are instructed to set the command queue
    *  properties when creating the queue, instead.
    */
    AURA_API_DEF(clSetCommandQueueProperty) = cl_int (CL_API_CALL*)(cl_command_queue, cl_command_queue_properties, cl_bool, cl_command_queue_properties*);
    AURA_API_PTR(clSetCommandQueueProperty);
#endif // CL_USE_DEPRECATED_OPENCL_1_0_APIS

    /* Deprecated OpenCL 1.1 APIs */
    AURA_API_DEF(clCreateIaura2D) = cl_mem (CL_API_CALL*)(cl_context, cl_mem_flags, const cl_iaura_format*, size_t, size_t, size_t, void*, cl_int*);
    AURA_API_PTR(clCreateIaura2D);

    AURA_API_DEF(clCreateIaura3D) = cl_mem (CL_API_CALL*)(cl_context, cl_mem_flags, const cl_iaura_format*, size_t, size_t, size_t, size_t, size_t, void*, cl_int*);
    AURA_API_PTR(clCreateIaura3D);

    AURA_API_DEF(clEnqueueMarker) = cl_int (CL_API_CALL*)(cl_command_queue, cl_event*);
    AURA_API_PTR(clEnqueueMarker);

    AURA_API_DEF(clEnqueueWaitForEvents) = cl_int (CL_API_CALL*)(cl_command_queue, cl_uint, const cl_event*);
    AURA_API_PTR(clEnqueueWaitForEvents);

    AURA_API_DEF(clEnqueueBarrier) = cl_int (CL_API_CALL*)(cl_command_queue);
    AURA_API_PTR(clEnqueueBarrier);

    AURA_API_DEF(clUnloadCompiler) = cl_int (CL_API_CALL*)(void);
    AURA_API_PTR(clUnloadCompiler);

    AURA_API_DEF(clGetExtensionFunctionAddress) = void* (CL_API_CALL*)(const char*);
    AURA_API_PTR(clGetExtensionFunctionAddress);

    /* Deprecated OpenCL 2.0 APIs */
    AURA_API_DEF(clCreateCommandQueue) = cl_command_queue (CL_API_CALL*)(cl_context, cl_device_id, cl_command_queue_properties, cl_int*);
    AURA_API_PTR(clCreateCommandQueue);

    AURA_API_DEF(clCreateSampler)      = cl_sampler (CL_API_CALL*)(cl_context, cl_bool, cl_addressing_mode, cl_filter_mode, cl_int*);
    AURA_API_PTR(clCreateSampler);

    AURA_API_DEF(clEnqueueTask)        = cl_int (CL_API_CALL*)(cl_command_queue, cl_kernel, cl_uint, const cl_event*, cl_event*);
    AURA_API_PTR(clEnqueueTask);

    AURA_API_DEF(clGetDeviceIauraInfoQCOM) = cl_int (CL_API_CALL*)(cl_device_id, size_t, size_t, const cl_iaura_format *, cl_iaura_pitch_info_qcom, size_t, void *, size_t *);
    AURA_API_PTR(clGetDeviceIauraInfoQCOM);

private:
    CLLibrary();

    ~CLLibrary();

    AURA_DISABLE_COPY_AND_ASSIGN(CLLibrary);

    DT_VOID* LoadSymbols(const std::string &path);

    Status Load();

    Status UnLoad();

private:
    DT_VOID *m_handle;
};

/**
 * @}
 */

} // namespace aura

#endif // AURA_RUNTIME_OPENCL_CL_LIBRARY_HPP__