#include "cl_library.hpp"

#include "aura/runtime/logger.h"

#include <dlfcn.h>
#include <memory>
#include <mutex>

namespace aura
{

CLLibrary& aura::CLLibrary::Get()
{
    static CLLibrary library;
    return library;
}

CLLibrary::CLLibrary() : m_handle(MI_NULL)
{
    Load();
}

CLLibrary::~CLLibrary()
{
    UnLoad();
}

AURA_VOID* CLLibrary::LoadSymbols(const std::string &path)
{
    Status ret = Status::ERROR;

    dlerror();

    AURA_VOID *handle = dlopen(path.c_str(), RTLD_LAZY | RTLD_LOCAL);
    if (MI_NULL == handle)
    {
        std::string info = "dlopen " + path + " failed, err : " + std::string(dlerror());
        AURA_PRINTE(AURA_TAG, "%s\n", info.c_str());
        return handle;
    }

    do
    {
        /* https://github.com/KhronosGroup/OpenCL-Headers/releases/tag/v2021.06.30 */
        /* Platform API */
        AURA_DLSYM_API(handle, clGetPlatformIDs)
        AURA_DLSYM_API(handle, clGetPlatformInfo)

        /* Device APIs */
        AURA_DLSYM_API(handle, clGetDeviceIDs)
        AURA_DLSYM_API(handle, clGetDeviceInfo)

#if defined(CL_VERSION_1_2)
        AURA_DLSYM_API(handle, clCreateSubDevices)
        AURA_DLSYM_API(handle, clRetainDevice)
        AURA_DLSYM_API(handle, clReleaseDevice)
#endif // CL_VERSION_1_2

#if defined(CL_VERSION_2_1)
        AURA_DLSYM_API(handle, clSetDefaultDeviceCommandQueue)
        AURA_DLSYM_API(handle, clGetDeviceAndHostTimer)
        AURA_DLSYM_API(handle, clGetHostTimer)
#endif //CL_VERSION_2_1

        /* Context APIs */
        AURA_DLSYM_API(handle, clCreateContext)
        AURA_DLSYM_API(handle, clCreateContextFromType)
        AURA_DLSYM_API(handle, clRetainContext)
        AURA_DLSYM_API(handle, clReleaseContext)
        AURA_DLSYM_API(handle, clGetContextInfo)

#if defined(CL_VERSION_3_0)
        AURA_DLSYM_API(handle, clSetContextDestructorCallback)
#endif // CL_VERSION_3_0

        /* Command Queue APIs */
#if defined(CL_VERSION_2_0)
        AURA_DLSYM_API(handle, clCreateCommandQueueWithProperties)
#endif

        AURA_DLSYM_API(handle, clRetainCommandQueue)
        AURA_DLSYM_API(handle, clReleaseCommandQueue)
        AURA_DLSYM_API(handle, clGetCommandQueueInfo)

        /* Memory Object APIs */
        AURA_DLSYM_API(handle, clCreateBuffer)

#if defined(CL_VERSION_1_1)
        AURA_DLSYM_API(handle, clCreateSubBuffer)
#endif //CL_VERSION_1_1

#if defined(CL_VERSION_1_2)
        AURA_DLSYM_API(handle, clCreateIaura)
#endif // CL_VERSION_1_2

#if defined(CL_VERSION_2_0)
        AURA_DLSYM_API(handle, clCreatePipe)
#endif //CL_VERSION_2_0

#if defined(CL_VERSION_3_0)
        AURA_DLSYM_API(handle, clCreateBufferWithProperties)
        AURA_DLSYM_API(handle, clCreateIauraWithProperties)
#endif // CL_VERSION_3_0

        AURA_DLSYM_API(handle, clRetainMemObject)
        AURA_DLSYM_API(handle, clReleaseMemObject)
        AURA_DLSYM_API(handle, clGetSupportedIauraFormats)
        AURA_DLSYM_API(handle, clGetMemObjectInfo)
        AURA_DLSYM_API(handle, clGetIauraInfo)

#if defined(CL_VERSION_2_0)
        AURA_DLSYM_API(handle, clGetPipeInfo)
#endif // CL_VERSION_2_0

#if defined(CL_VERSION_1_1)
        AURA_DLSYM_API(handle, clSetMemObjectDestructorCallback)
#endif // CL_VERSION_1_1

        /* SVM Allocation APIs */
#if defined(CL_VERSION_2_0)
        AURA_DLSYM_API(handle, clSVMAlloc)
        AURA_DLSYM_API(handle, clSVMFree)
#endif // CL_VERSION_2_0

        /* Sampler APIs */
#if defined(CL_VERSION_2_0)
        AURA_DLSYM_API(handle, clCreateSamplerWithProperties)
#endif // CL_VERSION_2_0

        AURA_DLSYM_API(handle, clRetainSampler)
        AURA_DLSYM_API(handle, clReleaseSampler)
        AURA_DLSYM_API(handle, clGetSamplerInfo)

        /* Program Object APIs */
        AURA_DLSYM_API(handle, clCreateProgramWithSource)
        AURA_DLSYM_API(handle, clCreateProgramWithBinary)

#if defined(CL_VERSION_1_2)
        AURA_DLSYM_API(handle, clCreateProgramWithBuiltInKernels)
#endif // CL_VERSION_1_2

#if defined(CL_VERSION_2_1)
        AURA_DLSYM_API(handle, clCreateProgramWithIL)
#endif // CL_VERSION_2_1

        AURA_DLSYM_API(handle, clRetainProgram)
        AURA_DLSYM_API(handle, clReleaseProgram)
        AURA_DLSYM_API(handle, clBuildProgram)

#if defined(CL_VERSION_1_2)
        AURA_DLSYM_API(handle, clCompileProgram)
        AURA_DLSYM_API(handle, clLinkProgram)
#endif // CL_VERSION_1_2

#if defined(CL_VERSION_2_2)
        AURA_DLSYM_API(handle, clSetProgramReleaseCallback)
        AURA_DLSYM_API(handle, clSetProgramSpecializationConstant)
#endif // CL_VERSION_2_2

#if defined(CL_VERSION_1_2)
        AURA_DLSYM_API(handle, clUnloadPlatformCompiler)
#endif // CL_VERSION_1_2

        AURA_DLSYM_API(handle, clGetProgramInfo)
        AURA_DLSYM_API(handle, clGetProgramBuildInfo)

        /* Kernel Object APIs */
        AURA_DLSYM_API(handle, clCreateKernel)
        AURA_DLSYM_API(handle, clCreateKernelsInProgram)

#if defined(CL_VERSION_2_1)
        AURA_DLSYM_API(handle, clCloneKernel)
#endif // CL_VERSION_2_1

        AURA_DLSYM_API(handle, clRetainKernel)
        AURA_DLSYM_API(handle, clReleaseKernel)
        AURA_DLSYM_API(handle, clSetKernelArg)

#if defined(CL_VERSION_2_0)
        AURA_DLSYM_API(handle, clSetKernelArgSVMPointer)
        AURA_DLSYM_API(handle, clSetKernelExecInfo)
#endif // CL_VERSION_2_0

        AURA_DLSYM_API(handle, clGetKernelInfo)

#if defined(CL_VERSION_1_2)
        AURA_DLSYM_API(handle, clGetKernelArgInfo)
#endif // CL_VERSION_1_2

        AURA_DLSYM_API(handle, clGetKernelWorkGroupInfo)

#if defined(CL_VERSION_2_1)
        AURA_DLSYM_API(handle, clGetKernelSubGroupInfo)
#endif // CL_VERSION_2_1

        /* Event Object APIs */
        AURA_DLSYM_API(handle, clWaitForEvents)
        AURA_DLSYM_API(handle, clGetEventInfo)

#if defined(CL_VERSION_1_1)
        AURA_DLSYM_API(handle, clCreateUserEvent)
#endif // CL_VERSION_1_1

        AURA_DLSYM_API(handle, clRetainEvent)
        AURA_DLSYM_API(handle, clReleaseEvent)

#if defined(CL_VERSION_1_1)
        AURA_DLSYM_API(handle, clSetUserEventStatus)
        AURA_DLSYM_API(handle, clSetEventCallback)
#endif // CL_VERSION_1_1

        /* Profiling APIs */
        AURA_DLSYM_API(handle, clGetEventProfilingInfo)

        /* Flush and Finish APIs */
        AURA_DLSYM_API(handle, clFlush)
        AURA_DLSYM_API(handle, clFinish)

        /* Enqueued Commands APIs */
        AURA_DLSYM_API(handle, clEnqueueReadBuffer)

#if defined(CL_VERSION_1_1)
        AURA_DLSYM_API(handle, clEnqueueReadBufferRect)
#endif // CL_VERSION_1_1

        AURA_DLSYM_API(handle, clEnqueueWriteBuffer)

#if defined(CL_VERSION_1_1)
        AURA_DLSYM_API(handle, clEnqueueWriteBufferRect)
#endif // CL_VERSION_1_1

#if defined(CL_VERSION_1_2)
        AURA_DLSYM_API(handle, clEnqueueFillBuffer)
#endif // CL_VERSION_1_2

        AURA_DLSYM_API(handle, clEnqueueCopyBuffer)

#if defined(CL_VERSION_1_1)
        AURA_DLSYM_API(handle, clEnqueueCopyBufferRect)
#endif // CL_VERSION_1_1

        AURA_DLSYM_API(handle, clEnqueueReadIaura)
        AURA_DLSYM_API(handle, clEnqueueWriteIaura)

#if defined(CL_VERSION_1_2)
        AURA_DLSYM_API(handle, clEnqueueFillIaura)
#endif // CL_VERSION_1_2

        AURA_DLSYM_API(handle, clEnqueueCopyIaura)
        AURA_DLSYM_API(handle, clEnqueueCopyIauraToBuffer)
        AURA_DLSYM_API(handle, clEnqueueCopyBufferToIaura)
        AURA_DLSYM_API(handle, clEnqueueMapBuffer)
        AURA_DLSYM_API(handle, clEnqueueMapIaura)
        AURA_DLSYM_API(handle, clEnqueueUnmapMemObject)

#if defined(CL_VERSION_1_2)
        AURA_DLSYM_API(handle, clEnqueueMigrateMemObjects)
#endif // CL_VERSION_1_2

        AURA_DLSYM_API(handle, clEnqueueNDRangeKernel)
        AURA_DLSYM_API(handle, clEnqueueNativeKernel)

#if defined(CL_VERSION_1_2)
        AURA_DLSYM_API(handle, clEnqueueMarkerWithWaitList)
        AURA_DLSYM_API(handle, clEnqueueBarrierWithWaitList)
#endif // CL_VERSION_1_2

#if defined(CL_VERSION_2_0)
        AURA_DLSYM_API(handle, clEnqueueSVMFree)
        AURA_DLSYM_API(handle, clEnqueueSVMMemcpy)
        AURA_DLSYM_API(handle, clEnqueueSVMMemFill)
        AURA_DLSYM_API(handle, clEnqueueSVMMap)
        AURA_DLSYM_API(handle, clEnqueueSVMUnmap)
#endif //CL_VERSION_2_0

#if defined(CL_VERSION_2_1)
        AURA_DLSYM_API(handle, clEnqueueSVMMigrateMem)
#endif // CL_VERSION_2_1

#if defined(CL_VERSION_1_2)
        AURA_DLSYM_API(handle, clGetExtensionFunctionAddressForPlatform)
#endif // CL_VERSION_1_2

#if defined(CL_USE_DEPRECATED_OPENCL_1_0_APIS)
        AURA_DLSYM_API(handle, clSetCommandQueueProperty)
#endif // CL_USE_DEPRECATED_OPENCL_1_0_APIS

        /* Deprecated OpenCL 1.1 APIs */
        AURA_DLSYM_API(handle, clCreateIaura2D)
        AURA_DLSYM_API(handle, clCreateIaura3D)
        AURA_DLSYM_API(handle, clEnqueueMarker)
        AURA_DLSYM_API(handle, clEnqueueWaitForEvents)
        AURA_DLSYM_API(handle, clEnqueueBarrier)
        AURA_DLSYM_API(handle, clUnloadCompiler)
        AURA_DLSYM_API(handle, clGetExtensionFunctionAddress)

        /* Deprecated OpenCL 2.0 APIs */
        AURA_DLSYM_API(handle, clCreateCommandQueue)
        AURA_DLSYM_API(handle, clCreateSampler)
        AURA_DLSYM_API(handle, clEnqueueTask)

        ret = Status::OK;
    } while (0);

    do
    {
       AURA_DLSYM_API_ADRENO(handle, clGetDeviceIauraInfoQCOM)
    } while (0);

    if (ret != Status::OK)
    {
        dlclose(handle);
        handle = MI_NULL;
    }

    return handle;
}

Status CLLibrary::Load()
{
    Status ret = Status::ERROR;

    const std::vector<std::string> default_cl_library_paths =
    {
        //default opencl library path
    #if defined(AURA_BUILD_ANDROID)
        "libOpenCL.so",
        "libGLES_mali.so",
        "libmali.so",
    #  if defined(__aarch64__)
        // Qualcomm Adreno
        "/system/vendor/lib64/libOpenCL.so",
        "/system/lib64/libOpenCL.so",
        // Mali
        "/system/vendor/lib64/egl/libGLES_mali.so",
        "/system/lib64/egl/libGLES_mali.so",
    #  else // not __aarch64__
        // Qualcomm Adreno
        "/system/vendor/lib/libOpenCL.so",
        "/system/lib/libOpenCL.so",
        // Mali
        "/system/vendor/lib/egl/libGLES_mali.so",
        "/system/lib/egl/libGLES_mali.so",
    #  endif // __aarch64__
    #endif // AURA_BUILD_ANDROID
    };

    if (MI_NULL == m_handle)
    {
        for (auto path : default_cl_library_paths)
        {
            AURA_VOID *handle = LoadSymbols(path);
            if (handle)
            {
                m_handle = handle;
                ret      = Status::OK;
                break;
            }
        }
    }
    else
    {
        ret = Status::OK;
    }

    return ret;
}

Status CLLibrary::UnLoad()
{
    Status ret = Status::ERROR;
    if (m_handle)
    {
        dlclose(m_handle);
        m_handle = MI_NULL;
        ret = Status::OK;
    }

    return ret;
}

} // namespace aura


/********************************************************************************************************/


/* Platform API */
CL_API_ENTRY cl_int
clGetPlatformIDs(cl_uint        num_entries,
                 cl_platform_id *platforms,
                 cl_uint        *num_platforms) CL_API_SUFFIX__VERSION_1_0
{
    auto func = aura::CLLibrary::Get().clGetPlatformIDs;
    if (func)
    {
        return func(num_entries, platforms, num_platforms);
    }
    return CL_INVALID_PLATFORM;
}

CL_API_ENTRY cl_int
clGetPlatformInfo(cl_platform_id   platform,
                  cl_platform_info param_name,
                  size_t           param_value_size,
                  void             *param_value,
                  size_t           *param_value_size_ret) CL_API_SUFFIX__VERSION_1_0
{
    auto func = aura::CLLibrary::Get().clGetPlatformInfo;
    if (func)
    {
        return func(platform, param_name, param_value_size, param_value, param_value_size_ret);
    }
    return CL_INVALID_PLATFORM;
}

/* Device APIs */
CL_API_ENTRY cl_int
clGetDeviceIDs(cl_platform_id platform,
               cl_device_type device_type,
               cl_uint        num_entries,
               cl_device_id   *devices,
               cl_uint        *num_devices) CL_API_SUFFIX__VERSION_1_0
{
    auto func = aura::CLLibrary::Get().clGetDeviceIDs;
    if (func)
    {
        return func(platform, device_type, num_entries, devices, num_devices);
    }
    return CL_INVALID_PLATFORM;
}

CL_API_ENTRY cl_int
clGetDeviceInfo(cl_device_id   device,
                cl_device_info param_name,
                size_t         param_value_size,
                void           *param_value,
                size_t         *param_value_size_ret) CL_API_SUFFIX__VERSION_1_0
{
    auto func = aura::CLLibrary::Get().clGetDeviceInfo;
    if (func)
    {
        return func(device, param_name, param_value_size, param_value, param_value_size_ret);
    }
    return CL_INVALID_PLATFORM;
}

#if defined(CL_VERSION_1_2)
CL_API_ENTRY cl_int
clCreateSubDevices(cl_device_id                       in_device,
                   const cl_device_partition_property *properties,
                   cl_uint                            num_devices,
                   cl_device_id                       *out_devices,
                   cl_uint                            *num_devices_ret) CL_API_SUFFIX__VERSION_1_2
{
    auto func = aura::CLLibrary::Get().clCreateSubDevices;
    if (func)
    {
        return func(in_device, properties, num_devices, out_devices, num_devices_ret);
    }
    return CL_INVALID_PLATFORM;
}

CL_API_ENTRY cl_int
clRetainDevice(cl_device_id device) CL_API_SUFFIX__VERSION_1_2
{
    auto func = aura::CLLibrary::Get().clRetainDevice;
    if (func)
    {
        return func(device);
    }
    return CL_INVALID_PLATFORM;
}

CL_API_ENTRY cl_int
clReleaseDevice(cl_device_id device) CL_API_SUFFIX__VERSION_1_2
{
    auto func = aura::CLLibrary::Get().clReleaseDevice;
    if (func)
    {
        return func(device);
    }
    return CL_INVALID_PLATFORM;
}
#endif // CL_VERSION_1_2

#if defined(CL_VERSION_2_1)
CL_API_ENTRY cl_int
clSetDefaultDeviceCommandQueue(cl_context       context,
                               cl_device_id     device,
                               cl_command_queue command_queue) CL_API_SUFFIX__VERSION_2_1
{
    auto func = aura::CLLibrary::Get().clSetDefaultDeviceCommandQueue;
    if (func)
    {
        return func(context, device, command_queue);
    }
    return CL_INVALID_PLATFORM;
}

CL_API_ENTRY cl_int
clGetDeviceAndHostTimer(cl_device_id device,
                        cl_ulong     *device_timestamp,
                        cl_ulong     *host_timestamp) CL_API_SUFFIX__VERSION_2_1
{
    auto func = aura::CLLibrary::Get().clGetDeviceAndHostTimer;
    if (func)
    {
        return func(device, device_timestamp, host_timestamp);
    }
    return CL_INVALID_PLATFORM;
}

CL_API_ENTRY cl_int
clGetHostTimer(cl_device_id device,
               cl_ulong     *host_timestamp) CL_API_SUFFIX__VERSION_2_1
{
    auto func = aura::CLLibrary::Get().clGetHostTimer;
    if (func)
    {
        return func(device, host_timestamp);
    }
    return CL_INVALID_PLATFORM;
}
#endif // CL_VERSION_2_1

/* Context APIs */
CL_API_ENTRY cl_context
clCreateContext(const cl_context_properties *properties,
                cl_uint                     num_devices,
                const cl_device_id          *devices,
                void                        (CL_CALLBACK *pfn_notify)(const char *errinfo,
                                                                      const void *private_info,
                                                                      size_t     cb,
                                                                      void       *user_data),
                void                        *user_data,
                cl_int                      *errcode_ret) CL_API_SUFFIX__VERSION_1_0
{
    auto func = aura::CLLibrary::Get().clCreateContext;
    if (func)
    {
        return func(properties, num_devices, devices, pfn_notify, user_data, errcode_ret);
    }
    if (errcode_ret)
    {
        *errcode_ret = CL_INVALID_PLATFORM;
    }
    return MI_NULL;
}

CL_API_ENTRY cl_context
clCreateContextFromType(const cl_context_properties *properties,
                        cl_device_type              device_type,
                        void                        (CL_CALLBACK * pfn_notify)(const char *errinfo,
                                                                               const void *private_info,
                                                                               size_t     cb,
                                                                               void       *user_data),
                        void                        *user_data,
                        cl_int                      *errcode_ret) CL_API_SUFFIX__VERSION_1_0
{
    auto func = aura::CLLibrary::Get().clCreateContextFromType;
    if (func)
    {
        return func(properties, device_type, pfn_notify, user_data, errcode_ret);
    }
    if (errcode_ret)
    {
        *errcode_ret = CL_INVALID_PLATFORM;
    }
    return MI_NULL;
}

CL_API_ENTRY cl_int
clRetainContext(cl_context context) CL_API_SUFFIX__VERSION_1_0
{
    auto func = aura::CLLibrary::Get().clRetainContext;
    if (func)
    {
        return func(context);
    }
    return CL_INVALID_PLATFORM;
}

CL_API_ENTRY cl_int
clReleaseContext(cl_context context) CL_API_SUFFIX__VERSION_1_0
{
    auto func = aura::CLLibrary::Get().clReleaseContext;
    if (func)
    {
        return func(context);
    }
    return CL_INVALID_PLATFORM;
}

CL_API_ENTRY cl_int
clGetContextInfo(cl_context      context,
                 cl_context_info param_name,
                 size_t          param_value_size,
                 void            *param_value,
                 size_t          *param_value_size_ret) CL_API_SUFFIX__VERSION_1_0
{
    auto func = aura::CLLibrary::Get().clGetContextInfo;
    if (func)
    {
        return func(context, param_name, param_value_size, param_value, param_value_size_ret);
    }
    return CL_INVALID_PLATFORM;
}

#if defined(CL_VERSION_3_0)
CL_API_ENTRY cl_int
clSetContextDestructorCallback(cl_context context,
                               void       (CL_CALLBACK *pfn_notify)(cl_context context,
                                                                    void       *user_data),
                               void       *user_data) CL_API_SUFFIX__VERSION_3_0
{
    auto func = aura::CLLibrary::Get().clSetContextDestructorCallback;
    if (func)
    {
        return func(context, pfn_notify, user_data);
    }
    return CL_INVALID_PLATFORM;
}
#endif // CL_VERSION_3_0

/* Command Queue APIs */

#if defined(CL_VERSION_2_0)
CL_API_ENTRY cl_command_queue
clCreateCommandQueueWithProperties(cl_context                context,
                                   cl_device_id              device,
                                   const cl_queue_properties *properties,
                                   cl_int                    *errcode_ret) CL_API_SUFFIX__VERSION_2_0
{
    auto func = aura::CLLibrary::Get().clCreateCommandQueueWithProperties;
    if (func)
    {
        return func(context, device, properties, errcode_ret);
    }
    if (errcode_ret)
    {
        *errcode_ret = CL_INVALID_PLATFORM;
    }
    return MI_NULL;
}
#endif // CL_VERSION_2_0

CL_API_ENTRY cl_int
clRetainCommandQueue(cl_command_queue command_queue) CL_API_SUFFIX__VERSION_1_0
{
    auto func = aura::CLLibrary::Get().clRetainCommandQueue;
    if (func)
    {
        return func(command_queue);
    }
    return CL_INVALID_PLATFORM;
}

CL_API_ENTRY cl_int
clReleaseCommandQueue(cl_command_queue command_queue) CL_API_SUFFIX__VERSION_1_0
{
    auto func = aura::CLLibrary::Get().clReleaseCommandQueue;
    if (func)
    {
        return func(command_queue);
    }
    return CL_INVALID_PLATFORM;
}

CL_API_ENTRY cl_int
clGetCommandQueueInfo(cl_command_queue      command_queue,
                      cl_command_queue_info param_name,
                      size_t                param_value_size,
                      void                  *param_value,
                      size_t                *param_value_size_ret) CL_API_SUFFIX__VERSION_1_0
{
    auto func = aura::CLLibrary::Get().clGetCommandQueueInfo;
    if (func)
    {
        return func(command_queue, param_name, param_value_size, param_value, param_value_size_ret);
    }
    return CL_INVALID_PLATFORM;
}

/* Memory Object APIs */
CL_API_ENTRY cl_mem
clCreateBuffer(cl_context   context,
               cl_mem_flags flags,
               size_t       size,
               void         *host_ptr,
               cl_int       *errcode_ret) CL_API_SUFFIX__VERSION_1_0
{
    auto func = aura::CLLibrary::Get().clCreateBuffer;
    if (func)
    {
        return func(context, flags, size, host_ptr, errcode_ret);
    }
    if (errcode_ret)
    {
        *errcode_ret = CL_INVALID_PLATFORM;
    }
    return MI_NULL;
}

#if defined(CL_VERSION_1_1)
CL_API_ENTRY cl_mem
clCreateSubBuffer(cl_mem                buffer,
                  cl_mem_flags          flags,
                  cl_buffer_create_type buffer_create_type,
                  const void            *buffer_create_info,
                  cl_int                *errcode_ret) CL_API_SUFFIX__VERSION_1_1
{
    auto func = aura::CLLibrary::Get().clCreateSubBuffer;
    if (func)
    {
        return func(buffer, flags, buffer_create_type, buffer_create_info, errcode_ret);
    }
    if (errcode_ret)
    {
        *errcode_ret = CL_INVALID_PLATFORM;
    }
    return MI_NULL;
}
#endif // CL_VERSION_1_1

#if defined(CL_VERSION_1_2)
CL_API_ENTRY cl_mem
clCreateIaura(cl_context            context,
              cl_mem_flags          flags,
              const cl_iaura_format *iaura_format,
              const cl_iaura_desc   *iaura_desc,
              void                  *host_ptr,
              cl_int                *errcode_ret) CL_API_SUFFIX__VERSION_1_2
{
    auto func = aura::CLLibrary::Get().clCreateIaura;
    if (func)
    {
        return func(context, flags, iaura_format, iaura_desc, host_ptr, errcode_ret);
    }
    if (errcode_ret)
    {
        *errcode_ret = CL_INVALID_PLATFORM;
    }
    return MI_NULL;
}
#endif // CL_VERSION_1_2

#if defined(CL_VERSION_2_0)
CL_API_ENTRY cl_mem
clCreatePipe(cl_context               context,
             cl_mem_flags             flags,
             cl_uint                  pipe_packet_size,
             cl_uint                  pipe_max_packets,
             const cl_pipe_properties *properties,
             cl_int                   *errcode_ret) CL_API_SUFFIX__VERSION_2_0
{
    auto func = aura::CLLibrary::Get().clCreatePipe;
    if (func)
    {
        return func(context, flags, pipe_packet_size, pipe_max_packets, properties, errcode_ret);
    }
    if (errcode_ret)
    {
        *errcode_ret = CL_INVALID_PLATFORM;
    }
    return MI_NULL;
}
#endif // CL_VERSION_2_0

#if defined(CL_VERSION_3_0)
CL_API_ENTRY cl_mem
clCreateBufferWithProperties(cl_context              context,
                             const cl_mem_properties *properties,
                             cl_mem_flags            flags,
                             size_t                  size,
                             void                    *host_ptr,
                             cl_int                  *errcode_ret) CL_API_SUFFIX__VERSION_3_0
{
    auto func = aura::CLLibrary::Get().clCreateBufferWithProperties;
    if (func)
    {
        return func(context, properties, flags, size, host_ptr, errcode_ret);
    }
    if (errcode_ret)
    {
        *errcode_ret = CL_INVALID_PLATFORM;
    }
    return MI_NULL;
}

CL_API_ENTRY cl_mem
clCreateIauraWithProperties(cl_context              context,
                            const cl_mem_properties *properties,
                            cl_mem_flags            flags,
                            const cl_iaura_format   *iaura_format,
                            const cl_iaura_desc     *iaura_desc,
                            void                    *host_ptr,
                            cl_int                  *errcode_ret) CL_API_SUFFIX__VERSION_3_0
{
    auto func = aura::CLLibrary::Get().clCreateIauraWithProperties;
    if (func)
    {
        return func(context, properties, flags, iaura_format, iaura_desc, host_ptr, errcode_ret);
    }
    if (errcode_ret)
    {
        *errcode_ret = CL_INVALID_PLATFORM;
    }
    return MI_NULL;
}
#endif // CL_VERSION_3_0

CL_API_ENTRY cl_int
clRetainMemObject(cl_mem memobj) CL_API_SUFFIX__VERSION_1_0
{
    auto func = aura::CLLibrary::Get().clRetainMemObject;
    if (func)
    {
        return func(memobj);
    }
    return CL_INVALID_PLATFORM;
}

CL_API_ENTRY cl_int
clReleaseMemObject(cl_mem memobj) CL_API_SUFFIX__VERSION_1_0
{
    auto func = aura::CLLibrary::Get().clReleaseMemObject;
    if (func)
    {
        return func(memobj);
    }
    return CL_INVALID_PLATFORM;
}

CL_API_ENTRY cl_int
clGetSupportedIauraFormats(cl_context         context,
                           cl_mem_flags       flags,
                           cl_mem_object_type iaura_type,
                           cl_uint            num_entries,
                           cl_iaura_format    *iaura_formats,
                           cl_uint            *num_iaura_formats) CL_API_SUFFIX__VERSION_1_0
{
    auto func = aura::CLLibrary::Get().clGetSupportedIauraFormats;
    if (func)
    {
        return func(context, flags, iaura_type, num_entries, iaura_formats, num_iaura_formats);
    }
    return CL_INVALID_PLATFORM;
}

CL_API_ENTRY cl_int
clGetMemObjectInfo(cl_mem      memobj,
                   cl_mem_info param_name,
                   size_t      param_value_size,
                   void        *param_value,
                   size_t      *param_value_size_ret) CL_API_SUFFIX__VERSION_1_0
{
    auto func = aura::CLLibrary::Get().clGetMemObjectInfo;
    if (func)
    {
        return func(memobj, param_name, param_value_size, param_value, param_value_size_ret);
    }
    return CL_INVALID_PLATFORM;
}

CL_API_ENTRY cl_int
clGetIauraInfo(cl_mem        iaura,
               cl_iaura_info param_name,
               size_t        param_value_size,
               void          *param_value,
               size_t        *param_value_size_ret) CL_API_SUFFIX__VERSION_1_0
{
    auto func = aura::CLLibrary::Get().clGetIauraInfo;
    if (func)
    {
        return func(iaura, param_name, param_value_size, param_value, param_value_size_ret);
    }
    return CL_INVALID_PLATFORM;
}

#if defined(CL_VERSION_2_0)
CL_API_ENTRY cl_int
clGetPipeInfo(cl_mem       pipe,
              cl_pipe_info param_name,
              size_t       param_value_size,
              void         *param_value,
              size_t       *param_value_size_ret) CL_API_SUFFIX__VERSION_2_0
{
    auto func = aura::CLLibrary::Get().clGetPipeInfo;
    if (func)
    {
        return func(pipe, param_name, param_value_size, param_value, param_value_size_ret);
    }
    return CL_INVALID_PLATFORM;
}
#endif // CL_VERSION_2_0

#if defined(CL_VERSION_1_1)
CL_API_ENTRY cl_int
clSetMemObjectDestructorCallback(cl_mem memobj,
                                 void   (CL_CALLBACK *pfn_notify)(cl_mem memobj,
                                                                void   *user_data),
                                 void   *user_data) CL_API_SUFFIX__VERSION_1_1
{
    auto func = aura::CLLibrary::Get().clSetMemObjectDestructorCallback;
    if (func)
    {
        return func(memobj, pfn_notify, user_data);
    }
    return CL_INVALID_PLATFORM;
}
#endif // CL_VERSION_1_1

/* SVM Allocation APIs */
#if defined(CL_VERSION_2_0)
CL_API_ENTRY void*
clSVMAlloc(cl_context       context,
           cl_svm_mem_flags flags,
           size_t           size,
           cl_uint          alignment) CL_API_SUFFIX__VERSION_2_0
{
    auto func = aura::CLLibrary::Get().clSVMAlloc;
    if (func)
    {
        return func(context, flags, size, alignment);
    }
    return MI_NULL;
}

CL_API_ENTRY void
clSVMFree(cl_context context,
          void       *svm_pointer) CL_API_SUFFIX__VERSION_2_0
{
    auto func = aura::CLLibrary::Get().clSVMFree;
    if (func)
    {
        func(context, svm_pointer);
    }
}
#endif // CL_VERSION_2_0

/* Sampler APIs */
#if defined(CL_VERSION_2_0)
CL_API_ENTRY cl_sampler
clCreateSamplerWithProperties(cl_context                  context,
                              const cl_sampler_properties *sampler_properties,
                              cl_int                      *errcode_ret) CL_API_SUFFIX__VERSION_2_0
{
    auto func = aura::CLLibrary::Get().clCreateSamplerWithProperties;
    if (func)
    {
        return func(context, sampler_properties, errcode_ret);
    }
    if (errcode_ret)
    {
        *errcode_ret = CL_INVALID_PLATFORM;
    }
    return MI_NULL;
}
#endif // CL_VERSION_2_0

CL_API_ENTRY cl_int
clRetainSampler(cl_sampler sampler) CL_API_SUFFIX__VERSION_1_0
{
    auto func = aura::CLLibrary::Get().clRetainSampler;
    if (func)
    {
        return func(sampler);
    }
    return CL_INVALID_PLATFORM;
}

CL_API_ENTRY cl_int
clReleaseSampler(cl_sampler sampler) CL_API_SUFFIX__VERSION_1_0
{
    auto func = aura::CLLibrary::Get().clReleaseSampler;
    if (func)
    {
        return func(sampler);
    }
    return CL_INVALID_PLATFORM;
}

CL_API_ENTRY cl_int
clGetSamplerInfo(cl_sampler      sampler,
                 cl_sampler_info param_name,
                 size_t          param_value_size,
                 void            *param_value,
                 size_t          *param_value_size_ret) CL_API_SUFFIX__VERSION_1_0
{
    auto func = aura::CLLibrary::Get().clGetSamplerInfo;
    if (func)
    {
        return func(sampler, param_name, param_value_size, param_value, param_value_size_ret);
    }
    return CL_INVALID_PLATFORM;
}

/* Program Object APIs */
CL_API_ENTRY cl_program
clCreateProgramWithSource(cl_context   context,
                          cl_uint      count,
                          const char   **strings,
                          const size_t *lengths,
                          cl_int       *errcode_ret) CL_API_SUFFIX__VERSION_1_0
{
    auto func = aura::CLLibrary::Get().clCreateProgramWithSource;
    if (func)
    {
        return func(context, count, strings, lengths, errcode_ret);
    }
    if (errcode_ret)
    {
        *errcode_ret = CL_INVALID_PLATFORM;
    }
    return MI_NULL;
}

CL_API_ENTRY cl_program
clCreateProgramWithBinary(cl_context          context,
                          cl_uint             num_devices,
                          const cl_device_id  *device_list,
                          const size_t        *lengths,
                          const unsigned char **binaries,
                          cl_int              *binary_status,
                          cl_int              *errcode_ret) CL_API_SUFFIX__VERSION_1_0
{
    auto func = aura::CLLibrary::Get().clCreateProgramWithBinary;
    if (func)
    {
        return func(context, num_devices, device_list, lengths, binaries, binary_status, errcode_ret);
    }
    if (errcode_ret)
    {
        *errcode_ret = CL_INVALID_PLATFORM;
    }
    return MI_NULL;
}

#if defined(CL_VERSION_1_2)
CL_API_ENTRY cl_program
clCreateProgramWithBuiltInKernels(cl_context         context,
                                  cl_uint            num_devices,
                                  const cl_device_id *device_list,
                                  const char         *kernel_names,
                                  cl_int             *errcode_ret) CL_API_SUFFIX__VERSION_1_2
{
    auto func = aura::CLLibrary::Get().clCreateProgramWithBuiltInKernels;
    if (func)
    {
        return func(context, num_devices, device_list, kernel_names, errcode_ret);
    }
    if (errcode_ret)
    {
        *errcode_ret = CL_INVALID_PLATFORM;
    }
    return MI_NULL;
}
#endif // CL_VERSION_1_2

#if defined(CL_VERSION_2_1)
CL_API_ENTRY cl_program
clCreateProgramWithIL(cl_context context,
                     const void  *il,
                     size_t      length,
                     cl_int      *errcode_ret) CL_API_SUFFIX__VERSION_2_1
{
    auto func = aura::CLLibrary::Get().clCreateProgramWithIL;
    if (func)
    {
        return func(context, il, length, errcode_ret);
    }
    if (errcode_ret)
    {
        *errcode_ret = CL_INVALID_PLATFORM;
    }
    return MI_NULL;
}
#endif // CL_VERSION_2_1

CL_API_ENTRY cl_int
clRetainProgram(cl_program program) CL_API_SUFFIX__VERSION_1_0
{
    auto func = aura::CLLibrary::Get().clRetainProgram;
    if (func)
    {
        return func(program);
    }
    return CL_INVALID_PLATFORM;
}

CL_API_ENTRY cl_int
clReleaseProgram(cl_program program) CL_API_SUFFIX__VERSION_1_0
{
    auto func = aura::CLLibrary::Get().clReleaseProgram;
    if (func)
    {
        return func(program);
    }
    return CL_INVALID_PLATFORM;
}

CL_API_ENTRY cl_int
clBuildProgram(cl_program         program,
               cl_uint            num_devices,
               const cl_device_id *device_list,
               const char         *options,
               void               (CL_CALLBACK *pfn_notify)(cl_program program,
                                                            void       *user_data),
               void               *user_data) CL_API_SUFFIX__VERSION_1_0
{
    auto func = aura::CLLibrary::Get().clBuildProgram;
    if (func)
    {
        return func(program, num_devices, device_list, options, pfn_notify, user_data);
    }
    return CL_INVALID_PLATFORM;
}

#if defined(CL_VERSION_1_2)
CL_API_ENTRY cl_int
clCompileProgram(cl_program         program,
                 cl_uint            num_devices,
                 const cl_device_id *device_list,
                 const char         *options,
                 cl_uint            num_input_headers,
                 const cl_program   *input_headers,
                 const char         **header_include_names,
                 void               (CL_CALLBACK *pfn_notify)(cl_program program,
                                                              void       *user_data),
                 void               *user_data) CL_API_SUFFIX__VERSION_1_2
{
    auto func = aura::CLLibrary::Get().clCompileProgram;
    if (func)
    {
        return func(program, num_devices, device_list, options, num_input_headers,
                    input_headers, header_include_names, pfn_notify, user_data);
    }
    return CL_INVALID_PLATFORM;
}

CL_API_ENTRY cl_program
clLinkProgram(cl_context         context,
              cl_uint            num_devices,
              const cl_device_id *device_list,
              const char         *options,
              cl_uint            num_input_programs,
              const cl_program   *input_programs,
              void               (CL_CALLBACK *pfn_notify)(cl_program program,
                                                           void       *user_data),
              void               *user_data,
              cl_int             *errcode_ret) CL_API_SUFFIX__VERSION_1_2
{
    auto func = aura::CLLibrary::Get().clLinkProgram;
    if (func)
    {
        return func(context, num_devices, device_list, options, num_input_programs,
                    input_programs, pfn_notify, user_data, errcode_ret);
    }
    if (errcode_ret)
    {
        *errcode_ret = CL_INVALID_PLATFORM;
    }
    return MI_NULL;
}
#endif // CL_VERSION_1_2

#if defined(CL_VERSION_2_2)
CL_API_ENTRY cl_int
clSetProgramReleaseCallback(cl_program program,
                            void       (CL_CALLBACK *pfn_notify)(cl_program program,
                                                                 void       *user_data),
                            void       *user_data)
{
    auto func = aura::CLLibrary::Get().clSetProgramReleaseCallback;
    if (func)
    {
        return func(program, pfn_notify, user_data);
    }
    return CL_INVALID_PLATFORM;
}

CL_API_ENTRY cl_int
clSetProgramSpecializationConstant(cl_program program,
                                   cl_uint    spec_id,
                                   size_t     spec_size,
                                   const void *spec_value) CL_API_SUFFIX__VERSION_2_2
{
    auto func = aura::CLLibrary::Get().clSetProgramSpecializationConstant;
    if (func)
    {
        return func(program, spec_id, spec_size, spec_value);
    }
    return CL_INVALID_PLATFORM;
}
#endif // CL_VERSION_2_2

#if defined(CL_VERSION_1_2)
CL_API_ENTRY cl_int
clUnloadPlatformCompiler(cl_platform_id platform) CL_API_SUFFIX__VERSION_1_2
{
    auto func = aura::CLLibrary::Get().clUnloadPlatformCompiler;
    if (func)
    {
        return func(platform);
    }
    return CL_INVALID_PLATFORM;
}
#endif // CL_VERSION_1_2

CL_API_ENTRY cl_int
clGetProgramInfo(cl_program      program,
                 cl_program_info param_name,
                 size_t          param_value_size,
                 void            *param_value,
                 size_t          *param_value_size_ret) CL_API_SUFFIX__VERSION_1_0
{
    auto func = aura::CLLibrary::Get().clGetProgramInfo;
    if (func)
    {
        return func(program, param_name, param_value_size, param_value, param_value_size_ret);
    }
    return CL_INVALID_PLATFORM;
}

CL_API_ENTRY cl_int
clGetProgramBuildInfo(cl_program            program,
                      cl_device_id          device,
                      cl_program_build_info param_name,
                      size_t                param_value_size,
                      void                  *param_value,
                      size_t                *param_value_size_ret) CL_API_SUFFIX__VERSION_1_0
{
    auto func = aura::CLLibrary::Get().clGetProgramBuildInfo;
    if (func)
    {
        return func(program, device, param_name, param_value_size, param_value, param_value_size_ret);
    }
    return CL_INVALID_PLATFORM;
}

/* Kernel Object APIs */
CL_API_ENTRY cl_kernel
clCreateKernel(cl_program program,
               const char *kernel_name,
               cl_int     *errcode_ret) CL_API_SUFFIX__VERSION_1_0
{
    auto func = aura::CLLibrary::Get().clCreateKernel;
    if (func)
    {
        return func(program, kernel_name, errcode_ret);
    }
    if (errcode_ret)
    {
        *errcode_ret = CL_INVALID_PLATFORM;
    }
    return MI_NULL;
}

CL_API_ENTRY cl_int
clCreateKernelsInProgram(cl_program program,
                         cl_uint    num_kernels,
                         cl_kernel  *kernels,
                         cl_uint    *num_kernels_ret) CL_API_SUFFIX__VERSION_1_0
{
    auto func = aura::CLLibrary::Get().clCreateKernelsInProgram;
    if (func)
    {
        return func(program, num_kernels, kernels, num_kernels_ret);
    }
    return CL_INVALID_PLATFORM;
}

#if defined(CL_VERSION_2_1)
CL_API_ENTRY cl_kernel
clCloneKernel(cl_kernel source_kernel,
              cl_int    *errcode_ret) CL_API_SUFFIX__VERSION_2_1
{
    auto func = aura::CLLibrary::Get().clCloneKernel;
    if (func)
    {
        return func(source_kernel, errcode_ret);
    }
    if (errcode_ret)
    {
        *errcode_ret = CL_INVALID_PLATFORM;
    }
    return MI_NULL;
}
#endif // CL_VERSION_2_1

CL_API_ENTRY cl_int
clRetainKernel(cl_kernel kernel) CL_API_SUFFIX__VERSION_1_0
{
    auto func = aura::CLLibrary::Get().clRetainKernel;
    if (func)
    {
        return func(kernel);
    }
    return CL_INVALID_PLATFORM;
}

CL_API_ENTRY cl_int
clReleaseKernel(cl_kernel kernel) CL_API_SUFFIX__VERSION_1_0
{
    auto func = aura::CLLibrary::Get().clReleaseKernel;
    if (func)
    {
        return func(kernel);
    }
    return CL_INVALID_PLATFORM;
}

CL_API_ENTRY cl_int
clSetKernelArg(cl_kernel   kernel,
               cl_uint     arg_index,
               size_t      arg_size,
               const void *arg_value) CL_API_SUFFIX__VERSION_1_0
{
    auto func = aura::CLLibrary::Get().clSetKernelArg;
    if (func)
    {
        return func(kernel, arg_index, arg_size, arg_value);
    }
    return CL_INVALID_PLATFORM;
}

#if defined(CL_VERSION_2_0)
CL_API_ENTRY cl_int
clSetKernelArgSVMPointer(cl_kernel  kernel,
                         cl_uint    arg_index,
                         const void *arg_value) CL_API_SUFFIX__VERSION_2_0
{
    auto func = aura::CLLibrary::Get().clSetKernelArgSVMPointer;
    if (func)
    {
        return func(kernel, arg_index, arg_value);
    }
    return CL_INVALID_PLATFORM;
}

CL_API_ENTRY cl_int
clSetKernelExecInfo(cl_kernel           kernel,
                    cl_kernel_exec_info param_name,
                    size_t              param_value_size,
                    const void          *param_value) CL_API_SUFFIX__VERSION_2_0
{
    auto func = aura::CLLibrary::Get().clSetKernelExecInfo;
    if (func)
    {
        return func(kernel, param_name, param_value_size, param_value);
    }
    return CL_INVALID_PLATFORM;
}
#endif // CL_VERSION_2_0

CL_API_ENTRY cl_int
clGetKernelInfo(cl_kernel      kernel,
                cl_kernel_info param_name,
                size_t         param_value_size,
                void           *param_value,
                size_t         *param_value_size_ret) CL_API_SUFFIX__VERSION_1_0
{
    auto func = aura::CLLibrary::Get().clGetKernelInfo;
    if (func)
    {
        return func(kernel, param_name, param_value_size, param_value, param_value_size_ret);
    }
    return CL_INVALID_PLATFORM;
}

#if defined(CL_VERSION_1_2)
CL_API_ENTRY cl_int
clGetKernelArgInfo(cl_kernel          kernel,
                   cl_uint            arg_indx,
                   cl_kernel_arg_info param_name,
                   size_t             param_value_size,
                   void               *param_value,
                   size_t             *param_value_size_ret) CL_API_SUFFIX__VERSION_1_2
{
    auto func = aura::CLLibrary::Get().clGetKernelArgInfo;
    if (func)
    {
        return func(kernel, arg_indx, param_name, param_value_size, param_value, param_value_size_ret);
    }
    return CL_INVALID_PLATFORM;
}
#endif //CL_VERSION_1_2

CL_API_ENTRY cl_int
clGetKernelWorkGroupInfo(cl_kernel                  kernel,
                         cl_device_id               device,
                         cl_kernel_work_group_info  param_name,
                         size_t                     param_value_size,
                         void                       *param_value,
                         size_t                     *param_value_size_ret) CL_API_SUFFIX__VERSION_1_0
{
    auto func = aura::CLLibrary::Get().clGetKernelWorkGroupInfo;
    if (func)
    {
        return func(kernel, device, param_name, param_value_size, param_value, param_value_size_ret);
    }
    return CL_INVALID_PLATFORM;
}

#if defined(CL_VERSION_2_1)
CL_API_ENTRY cl_int
clGetKernelSubGroupInfo(cl_kernel                kernel,
                        cl_device_id             device,
                        cl_kernel_sub_group_info param_name,
                        size_t                   input_value_size,
                        const void               *input_value,
                        size_t                   param_value_size,
                        void                     *param_value,
                        size_t                   *param_value_size_ret) CL_API_SUFFIX__VERSION_2_1
{
    auto func = aura::CLLibrary::Get().clGetKernelSubGroupInfo;
    if (func)
    {
        return func(kernel, device, param_name, input_value_size, input_value,
                    param_value_size, param_value, param_value_size_ret);
    }
    return CL_INVALID_PLATFORM;
}
#endif // CL_VERSION_2_1

/* Event Object APIs */
CL_API_ENTRY cl_int
clWaitForEvents(cl_uint         num_events,
                const cl_event *event_list) CL_API_SUFFIX__VERSION_1_0
{
    auto func = aura::CLLibrary::Get().clWaitForEvents;
    if (func)
    {
        return func(num_events, event_list);
    }
    return CL_INVALID_PLATFORM;
}

CL_API_ENTRY cl_int
clGetEventInfo(cl_event      event,
               cl_event_info param_name,
               size_t        param_value_size,
               void          *param_value,
               size_t        *param_value_size_ret) CL_API_SUFFIX__VERSION_1_0
{
    auto func = aura::CLLibrary::Get().clGetEventInfo;
    if (func)
    {
        return func(event, param_name, param_value_size, param_value, param_value_size_ret);
    }
    return CL_INVALID_PLATFORM;
}

#if defined(CL_VERSION_1_1)
CL_API_ENTRY cl_event
clCreateUserEvent(cl_context context,
                  cl_int     *errcode_ret) CL_API_SUFFIX__VERSION_1_1
{
    auto func = aura::CLLibrary::Get().clCreateUserEvent;
    if (func)
    {
        return func(context, errcode_ret);
    }
    if (errcode_ret)
    {
        *errcode_ret = CL_INVALID_PLATFORM;
    }
    return MI_NULL;
}
#endif // CL_VERSION_1_1

CL_API_ENTRY cl_int
clRetainEvent(cl_event event) CL_API_SUFFIX__VERSION_1_0
{
    auto func = aura::CLLibrary::Get().clRetainEvent;
    if (func)
    {
        return func(event);
    }
    return CL_INVALID_PLATFORM;
}

CL_API_ENTRY cl_int
clReleaseEvent(cl_event event) CL_API_SUFFIX__VERSION_1_0
{
    auto func = aura::CLLibrary::Get().clReleaseEvent;
    if (func)
    {
        return func(event);
    }
    return CL_INVALID_PLATFORM;
}

#if defined(CL_VERSION_1_1)
CL_API_ENTRY cl_int
clSetUserEventStatus(cl_event event,
                     cl_int   execution_status) CL_API_SUFFIX__VERSION_1_1
{
    auto func = aura::CLLibrary::Get().clSetUserEventStatus;
    if (func)
    {
        return func(event, execution_status);
    }
    return CL_INVALID_PLATFORM;
}

CL_API_ENTRY cl_int
clSetEventCallback(cl_event event,
                   cl_int   command_exec_callback_type,
                   void     (CL_CALLBACK *pfn_notify)(cl_event event,
                                                      cl_int   event_command_status,
                                                      void     *user_data),
                   void     *user_data) CL_API_SUFFIX__VERSION_1_1
{
    auto func = aura::CLLibrary::Get().clSetEventCallback;
    if (func)
    {
        return func(event, command_exec_callback_type, pfn_notify, user_data);
    }
    return CL_INVALID_PLATFORM;
}
#endif // CL_VERSION_1_1

/* Profiling APIs */
CL_API_ENTRY cl_int
clGetEventProfilingInfo(cl_event          event,
                        cl_profiling_info param_name,
                        size_t            param_value_size,
                        void              *param_value,
                        size_t            *param_value_size_ret) CL_API_SUFFIX__VERSION_1_0
{
    auto func = aura::CLLibrary::Get().clGetEventProfilingInfo;
    if (func)
    {
        return func(event, param_name, param_value_size, param_value, param_value_size_ret);
    }
    return CL_INVALID_PLATFORM;
}

/* Flush and Finish APIs */
CL_API_ENTRY cl_int
clFlush(cl_command_queue command_queue) CL_API_SUFFIX__VERSION_1_0
{
    auto func = aura::CLLibrary::Get().clFlush;
    if (func)
    {
        return func(command_queue);
    }
    return CL_INVALID_PLATFORM;
}

CL_API_ENTRY cl_int
clFinish(cl_command_queue command_queue) CL_API_SUFFIX__VERSION_1_0
{
    auto func = aura::CLLibrary::Get().clFinish;
    if (func)
    {
        return func(command_queue);
    }
    return CL_INVALID_PLATFORM;
}

/* Enqueued Commands APIs */
CL_API_ENTRY cl_int
clEnqueueReadBuffer(cl_command_queue command_queue,
                    cl_mem           buffer,
                    cl_bool          blocking_read,
                    size_t           offset,
                    size_t           size,
                    void             *ptr,
                    cl_uint          num_events_in_wait_list,
                    const cl_event   *event_wait_list,
                    cl_event         *event) CL_API_SUFFIX__VERSION_1_0
{
    auto func = aura::CLLibrary::Get().clEnqueueReadBuffer;
    if (func)
    {
        return func(command_queue, buffer, blocking_read, offset, size, ptr,
                    num_events_in_wait_list, event_wait_list, event);
    }
    return CL_INVALID_PLATFORM;
}

#if defined(CL_VERSION_1_1)
CL_API_ENTRY cl_int
clEnqueueReadBufferRect(cl_command_queue command_queue,
                        cl_mem           buffer,
                        cl_bool          blocking_read,
                        const size_t     *buffer_origin,
                        const size_t     *host_origin,
                        const size_t     *region,
                        size_t           buffer_row_pitch,
                        size_t           buffer_slice_pitch,
                        size_t           host_row_pitch,
                        size_t           host_slice_pitch,
                        void             *ptr,
                        cl_uint          num_events_in_wait_list,
                        const cl_event   *event_wait_list,
                        cl_event         *event) CL_API_SUFFIX__VERSION_1_1
{
    auto func = aura::CLLibrary::Get().clEnqueueReadBufferRect;
    if (func)
    {
        return func(command_queue, buffer, blocking_read, buffer_origin, host_origin, region,
                    buffer_row_pitch, buffer_slice_pitch, host_row_pitch, host_slice_pitch,
                    ptr, num_events_in_wait_list, event_wait_list, event);
    }
    return CL_INVALID_PLATFORM;
}
#endif // CL_VERSION_1_1

CL_API_ENTRY cl_int
clEnqueueWriteBuffer(cl_command_queue command_queue,
                     cl_mem           buffer,
                     cl_bool          blocking_write,
                     size_t           offset,
                     size_t           size,
                     const void       *ptr,
                     cl_uint          num_events_in_wait_list,
                     const cl_event   *event_wait_list,
                     cl_event         *event) CL_API_SUFFIX__VERSION_1_0
{
    auto func = aura::CLLibrary::Get().clEnqueueWriteBuffer;
    if (func)
    {
        return func(command_queue, buffer, blocking_write, offset, size, ptr,
                    num_events_in_wait_list, event_wait_list, event);
    }
    return CL_INVALID_PLATFORM;
}

#if defined(CL_VERSION_1_1)

CL_API_ENTRY cl_int
clEnqueueWriteBufferRect(cl_command_queue command_queue,
                         cl_mem           buffer,
                         cl_bool          blocking_write,
                         const size_t     *buffer_origin,
                         const size_t     *host_origin,
                         const size_t     *region,
                         size_t           buffer_row_pitch,
                         size_t           buffer_slice_pitch,
                         size_t           host_row_pitch,
                         size_t           host_slice_pitch,
                         const void       *ptr,
                         cl_uint          num_events_in_wait_list,
                         const cl_event   *event_wait_list,
                         cl_event         *event) CL_API_SUFFIX__VERSION_1_1
{
    auto func = aura::CLLibrary::Get().clEnqueueWriteBufferRect;
    if (func)
    {
        return func(command_queue, buffer, blocking_write, buffer_origin, host_origin, region,
                    buffer_row_pitch, buffer_slice_pitch, host_row_pitch, host_slice_pitch,
                    ptr, num_events_in_wait_list, event_wait_list, event);
    }
    return CL_INVALID_PLATFORM;
}
#endif // CL_VERSION_1_1

#if defined(CL_VERSION_1_2)
CL_API_ENTRY cl_int
clEnqueueFillBuffer(cl_command_queue command_queue,
                    cl_mem           buffer,
                    const void       *pattern,
                    size_t           pattern_size,
                    size_t           offset,
                    size_t           size,
                    cl_uint          num_events_in_wait_list,
                    const cl_event   *event_wait_list,
                    cl_event         *event) CL_API_SUFFIX__VERSION_1_2
{
    auto func = aura::CLLibrary::Get().clEnqueueFillBuffer;
    if (func)
    {
        return func(command_queue, buffer, pattern, pattern_size, offset, size,
                    num_events_in_wait_list, event_wait_list, event);
    }
    return CL_INVALID_PLATFORM;
}
#endif // CL_VERSION_1_2

CL_API_ENTRY cl_int
clEnqueueCopyBuffer(cl_command_queue command_queue,
                    cl_mem           src_buffer,
                    cl_mem           dst_buffer,
                    size_t           src_offset,
                    size_t           dst_offset,
                    size_t           size,
                    cl_uint          num_events_in_wait_list,
                    const cl_event   *event_wait_list,
                    cl_event         *event) CL_API_SUFFIX__VERSION_1_0
{
    auto func = aura::CLLibrary::Get().clEnqueueCopyBuffer;
    if (func)
    {
        return func(command_queue, src_buffer, dst_buffer, src_offset, dst_offset, size,
                    num_events_in_wait_list, event_wait_list, event);
    }
    return CL_INVALID_PLATFORM;
}

#if defined(CL_VERSION_1_1)
CL_API_ENTRY cl_int
clEnqueueCopyBufferRect(cl_command_queue command_queue,
                        cl_mem           src_buffer,
                        cl_mem           dst_buffer,
                        const size_t     *src_origin,
                        const size_t     *dst_origin,
                        const size_t     *region,
                        size_t           src_row_pitch,
                        size_t           src_slice_pitch,
                        size_t           dst_row_pitch,
                        size_t           dst_slice_pitch,
                        cl_uint          num_events_in_wait_list,
                        const cl_event   *event_wait_list,
                        cl_event         *event) CL_API_SUFFIX__VERSION_1_1
{
    auto func = aura::CLLibrary::Get().clEnqueueCopyBufferRect;
    if (func)
    {
        return func(command_queue, src_buffer, dst_buffer, src_origin, dst_origin, region,
                    src_row_pitch, src_slice_pitch, dst_row_pitch, dst_slice_pitch,
                    num_events_in_wait_list, event_wait_list, event);
    }
    return CL_INVALID_PLATFORM;
}
#endif // CL_VERSION_1_1

CL_API_ENTRY cl_int
clEnqueueReadIaura(cl_command_queue command_queue,
                   cl_mem           iaura,
                   cl_bool          blocking_read,
                   const size_t     *origin,
                   const size_t     *region,
                   size_t           row_pitch,
                   size_t           slice_pitch,
                   void             *ptr,
                   cl_uint          num_events_in_wait_list,
                   const cl_event   *event_wait_list,
                   cl_event         *event) CL_API_SUFFIX__VERSION_1_0
{
    auto func = aura::CLLibrary::Get().clEnqueueReadIaura;
    if (func)
    {
        return func(command_queue, iaura, blocking_read, origin, region, row_pitch, slice_pitch,
                    ptr, num_events_in_wait_list, event_wait_list, event);
    }
    return CL_INVALID_PLATFORM;
}

CL_API_ENTRY cl_int
clEnqueueWriteIaura(cl_command_queue command_queue,
                    cl_mem           iaura,
                    cl_bool          blocking_write,
                    const size_t     *origin,
                    const size_t     *region,
                    size_t           input_row_pitch,
                    size_t           input_slice_pitch,
                    const void       *ptr,
                    cl_uint          num_events_in_wait_list,
                    const cl_event   *event_wait_list,
                    cl_event         *event) CL_API_SUFFIX__VERSION_1_0
{
    auto func = aura::CLLibrary::Get().clEnqueueWriteIaura;
    if (func)
    {
        return func(command_queue, iaura, blocking_write, origin, region, input_row_pitch,
                    input_slice_pitch, ptr, num_events_in_wait_list, event_wait_list, event);
    }
    return CL_INVALID_PLATFORM;
}

#if defined(CL_VERSION_1_2)
CL_API_ENTRY cl_int
clEnqueueFillIaura(cl_command_queue command_queue,
                   cl_mem           iaura,
                   const void       *fill_color,
                   const size_t     *origin,
                   const size_t     *region,
                   cl_uint          num_events_in_wait_list,
                   const cl_event   *event_wait_list,
                   cl_event         *event) CL_API_SUFFIX__VERSION_1_2
{
    auto func = aura::CLLibrary::Get().clEnqueueFillIaura;
    if (func)
    {
        return func(command_queue, iaura, fill_color, origin, region,
                    num_events_in_wait_list, event_wait_list, event);
    }
    return CL_INVALID_PLATFORM;
}
#endif // CL_VERSION_1_2

CL_API_ENTRY cl_int
clEnqueueCopyIaura(cl_command_queue command_queue,
                   cl_mem           src_iaura,
                   cl_mem           dst_iaura,
                   const size_t     *src_origin,
                   const size_t     *dst_origin,
                   const size_t     *region,
                   cl_uint          num_events_in_wait_list,
                   const cl_event   *event_wait_list,
                   cl_event         *event) CL_API_SUFFIX__VERSION_1_0
{
    auto func = aura::CLLibrary::Get().clEnqueueCopyIaura;
    if (func)
    {
        return func(command_queue, src_iaura, dst_iaura, src_origin, dst_origin, region,
                    num_events_in_wait_list, event_wait_list, event);
    }
    return CL_INVALID_PLATFORM;
}

CL_API_ENTRY cl_int
clEnqueueCopyIauraToBuffer(cl_command_queue command_queue,
                           cl_mem           src_iaura,
                           cl_mem           dst_buffer,
                           const size_t     *src_origin,
                           const size_t     *region,
                           size_t           dst_offset,
                           cl_uint          num_events_in_wait_list,
                           const cl_event   *event_wait_list,
                           cl_event         *event) CL_API_SUFFIX__VERSION_1_0
{
    auto func = aura::CLLibrary::Get().clEnqueueCopyIauraToBuffer;
    if (func)
    {
        return func(command_queue, src_iaura, dst_buffer, src_origin, region, dst_offset,
                    num_events_in_wait_list, event_wait_list, event);
    }
    return CL_INVALID_PLATFORM;
}

CL_API_ENTRY cl_int
clEnqueueCopyBufferToIaura(cl_command_queue command_queue,
                           cl_mem           src_buffer,
                           cl_mem           dst_iaura,
                           size_t           src_offset,
                           const size_t     *dst_origin,
                           const size_t     *region,
                           cl_uint          num_events_in_wait_list,
                           const cl_event   *event_wait_list,
                           cl_event         *event) CL_API_SUFFIX__VERSION_1_0
{
    auto func = aura::CLLibrary::Get().clEnqueueCopyBufferToIaura;
    if (func)
    {
        return func(command_queue, src_buffer, dst_iaura, src_offset, dst_origin, region,
                    num_events_in_wait_list, event_wait_list, event);
    }
    return CL_INVALID_PLATFORM;
}

CL_API_ENTRY void*
clEnqueueMapBuffer(cl_command_queue command_queue,
                   cl_mem           buffer,
                   cl_bool          blocking_map,
                   cl_map_flags     map_flags,
                   size_t           offset,
                   size_t           size,
                   cl_uint          num_events_in_wait_list,
                   const cl_event   *event_wait_list,
                   cl_event         *event,
                   cl_int           *errcode_ret) CL_API_SUFFIX__VERSION_1_0
{
    auto func = aura::CLLibrary::Get().clEnqueueMapBuffer;
    if (func)
    {
        return func(command_queue, buffer, blocking_map, map_flags, offset, size,
                    num_events_in_wait_list, event_wait_list, event, errcode_ret);
    }
    if (errcode_ret)
    {
        *errcode_ret = CL_INVALID_PLATFORM;
    }
    return MI_NULL;
}

CL_API_ENTRY void*
clEnqueueMapIaura(cl_command_queue command_queue,
                  cl_mem           iaura,
                  cl_bool          blocking_map,
                  cl_map_flags     map_flags,
                  const size_t     *origin,
                  const size_t     *region,
                  size_t           *iaura_row_pitch,
                  size_t           *iaura_slice_pitch,
                  cl_uint          num_events_in_wait_list,
                  const cl_event   *event_wait_list,
                  cl_event         *event,
                  cl_int           *errcode_ret) CL_API_SUFFIX__VERSION_1_0
{
    auto func = aura::CLLibrary::Get().clEnqueueMapIaura;
    if (func)
    {
        return func(command_queue, iaura, blocking_map, map_flags, origin, region, iaura_row_pitch,
                    iaura_slice_pitch, num_events_in_wait_list, event_wait_list, event, errcode_ret);
    }
    if (errcode_ret)
    {
        *errcode_ret = CL_INVALID_PLATFORM;
    }
    return MI_NULL;
}

CL_API_ENTRY cl_int
clEnqueueUnmapMemObject(cl_command_queue command_queue,
                        cl_mem           memobj,
                        void             *mapped_ptr,
                        cl_uint          num_events_in_wait_list,
                        const cl_event   *event_wait_list,
                        cl_event         *event) CL_API_SUFFIX__VERSION_1_0
{
    auto func = aura::CLLibrary::Get().clEnqueueUnmapMemObject;
    if (func)
    {
        return func(command_queue, memobj, mapped_ptr, num_events_in_wait_list, event_wait_list, event);
    }
    return CL_INVALID_PLATFORM;
}

#if defined(CL_VERSION_1_2)
CL_API_ENTRY cl_int
clEnqueueMigrateMemObjects(cl_command_queue       command_queue,
                           cl_uint                num_mem_objects,
                           const cl_mem           *mem_objects,
                           cl_mem_migration_flags flags,
                           cl_uint                num_events_in_wait_list,
                           const cl_event         *event_wait_list,
                           cl_event               *event) CL_API_SUFFIX__VERSION_1_2
{
    auto func = aura::CLLibrary::Get().clEnqueueMigrateMemObjects;
    if (func)
    {
        return func(command_queue, num_mem_objects, mem_objects, flags,
                    num_events_in_wait_list, event_wait_list, event);
    }
    return CL_INVALID_PLATFORM;
}
#endif // CL_VERSION_1_2

CL_API_ENTRY cl_int
clEnqueueNDRangeKernel(cl_command_queue command_queue,
                       cl_kernel        kernel,
                       cl_uint          work_dim,
                       const size_t     *global_work_offset,
                       const size_t     *global_work_size,
                       const size_t     *local_work_size,
                       cl_uint          num_events_in_wait_list,
                       const cl_event   *event_wait_list,
                       cl_event         *event) CL_API_SUFFIX__VERSION_1_0
{
    auto func = aura::CLLibrary::Get().clEnqueueNDRangeKernel;
    if (func)
    {
        return func(command_queue, kernel, work_dim, global_work_offset, global_work_size,
                    local_work_size, num_events_in_wait_list, event_wait_list, event);
    }
    return CL_INVALID_PLATFORM;
}

CL_API_ENTRY cl_int
clEnqueueNativeKernel(cl_command_queue  command_queue,
                      void (CL_CALLBACK *user_func)(void*),
                      void              *args,
                      size_t            cb_args,
                      cl_uint           num_mem_objects,
                      const cl_mem      *mem_list,
                      const void        **args_mem_loc,
                      cl_uint           num_events_in_wait_list,
                      const cl_event    *event_wait_list,
                      cl_event          *event) CL_API_SUFFIX__VERSION_1_0
{
    auto func = aura::CLLibrary::Get().clEnqueueNativeKernel;
    if (func)
    {
        return func(command_queue, user_func, args, cb_args, num_mem_objects, mem_list,
                    args_mem_loc, num_events_in_wait_list, event_wait_list, event);
    }
    return CL_INVALID_PLATFORM;
}

#if defined(CL_VERSION_1_2)
CL_API_ENTRY cl_int
clEnqueueMarkerWithWaitList(cl_command_queue command_queue,
                            cl_uint          num_events_in_wait_list,
                            const cl_event   *event_wait_list,
                            cl_event         *event) CL_API_SUFFIX__VERSION_1_2
{
    auto func = aura::CLLibrary::Get().clEnqueueMarkerWithWaitList;
    if (func)
    {
        return func(command_queue, num_events_in_wait_list, event_wait_list, event);
    }
    return CL_INVALID_PLATFORM;
}

CL_API_ENTRY cl_int
clEnqueueBarrierWithWaitList(cl_command_queue command_queue,
                             cl_uint          num_events_in_wait_list,
                             const cl_event   *event_wait_list,
                             cl_event         *event) CL_API_SUFFIX__VERSION_1_2
{
    auto func = aura::CLLibrary::Get().clEnqueueBarrierWithWaitList;
    if (func)
    {
        return func(command_queue, num_events_in_wait_list, event_wait_list, event);
    }
    return CL_INVALID_PLATFORM;
}
#endif // CL_VERSION_1_2

#if defined(CL_VERSION_2_0)
CL_API_ENTRY cl_int
clEnqueueSVMFree(cl_command_queue  command_queue,
                 cl_uint           num_svm_pointers,
                 void              *svm_pointers[],
                 void (CL_CALLBACK *pfn_free_func)(cl_command_queue queue,
                                                    cl_uint         num_svm_pointers,
                                                    void            *svm_pointers[],
                                                    void            *user_data),
                 void              *user_data,
                 cl_uint           num_events_in_wait_list,
                 const cl_event    *event_wait_list,
                 cl_event          *event) CL_API_SUFFIX__VERSION_2_0
{
    auto func = aura::CLLibrary::Get().clEnqueueSVMFree;
    if (func)
    {
        return func(command_queue, num_svm_pointers, svm_pointers, pfn_free_func, user_data,
                    num_events_in_wait_list, event_wait_list, event);
    }
    return CL_INVALID_PLATFORM;
}

CL_API_ENTRY cl_int
clEnqueueSVMMemcpy(cl_command_queue command_queue,
                   cl_bool          blocking_copy,
                   void             *dst_ptr,
                   const void       *src_ptr,
                   size_t           size,
                   cl_uint          num_events_in_wait_list,
                   const cl_event   *event_wait_list,
                   cl_event         *event) CL_API_SUFFIX__VERSION_2_0
{
    auto func = aura::CLLibrary::Get().clEnqueueSVMMemcpy;
    if (func)
    {
        return func(command_queue, blocking_copy, dst_ptr, src_ptr, size,
                    num_events_in_wait_list, event_wait_list, event);
    }
    return CL_INVALID_PLATFORM;
}

CL_API_ENTRY cl_int
clEnqueueSVMMemFill(cl_command_queue command_queue,
                    void             *svm_ptr,
                    const void       *pattern,
                    size_t           pattern_size,
                    size_t           size,
                    cl_uint          num_events_in_wait_list,
                    const cl_event   *event_wait_list,
                    cl_event         *event) CL_API_SUFFIX__VERSION_2_0
{
    auto func = aura::CLLibrary::Get().clEnqueueSVMMemFill;
    if (func)
    {
        return func(command_queue, svm_ptr, pattern, pattern_size, size,
                    num_events_in_wait_list, event_wait_list, event);
    }
    return CL_INVALID_PLATFORM;
}

CL_API_ENTRY cl_int
clEnqueueSVMMap(cl_command_queue command_queue,
                cl_bool          blocking_map,
                cl_map_flags     flags,
                void             *svm_ptr,
                size_t           size,
                cl_uint          num_events_in_wait_list,
                const cl_event   *event_wait_list,
                cl_event         *event) CL_API_SUFFIX__VERSION_2_0
{
    auto func = aura::CLLibrary::Get().clEnqueueSVMMap;
    if (func)
    {
        return func(command_queue, blocking_map, flags, svm_ptr, size,
                    num_events_in_wait_list, event_wait_list, event);
    }
    return CL_INVALID_PLATFORM;
}

CL_API_ENTRY cl_int
clEnqueueSVMUnmap(cl_command_queue command_queue,
                  void             *svm_ptr,
                  cl_uint          num_events_in_wait_list,
                  const cl_event   *event_wait_list,
                  cl_event         *event) CL_API_SUFFIX__VERSION_2_0
{
    auto func = aura::CLLibrary::Get().clEnqueueSVMUnmap;
    if (func)
    {
        return func(command_queue, svm_ptr,
                    num_events_in_wait_list, event_wait_list, event);
    }
    return CL_INVALID_PLATFORM;
}
#endif // CL_VERSION_2_0

#if defined(CL_VERSION_2_1)
CL_API_ENTRY cl_int
clEnqueueSVMMigrateMem(cl_command_queue       command_queue,
                       cl_uint                num_svm_pointers,
                       const void             **svm_pointers,
                       const size_t           *sizes,
                       cl_mem_migration_flags flags,
                       cl_uint                num_events_in_wait_list,
                       const cl_event         *event_wait_list,
                       cl_event               *event) CL_API_SUFFIX__VERSION_2_1
{
    auto func = aura::CLLibrary::Get().clEnqueueSVMMigrateMem;
    if (func)
    {
        return func(command_queue, num_svm_pointers, svm_pointers, sizes, flags,
                    num_events_in_wait_list, event_wait_list, event);
    }
    return CL_INVALID_PLATFORM;
}
#endif // CL_VERSION_2_1

#if defined(CL_VERSION_1_2)
CL_API_ENTRY void*
clGetExtensionFunctionAddressForPlatform(cl_platform_id platform,
                                         const char     *func_name) CL_API_SUFFIX__VERSION_1_2
{
    auto func = aura::CLLibrary::Get().clGetExtensionFunctionAddressForPlatform;
    if (func)
    {
        return func(platform, func_name);
    }
    return MI_NULL;
}
#endif // CL_VERSION_1_2

#if defined(CL_USE_DEPRECATED_OPENCL_1_0_APIS)
CL_API_ENTRY cl_int
clSetCommandQueueProperty(cl_command_queue              command_queue,
                            cl_command_queue_properties properties,
                            cl_bool                     enable,
                            cl_command_queue_properties *old_properties)
{
    auto func = aura::CLLibrary::Get().clSetCommandQueueProperty;
    if (func)
    {
        return func(command_queue, properties, enable, old_properties);
    }
    return CL_INVALID_PLATFORM;
}
#endif // CL_USE_DEPRECATED_OPENCL_1_0_APIS

/* Deprecated OpenCL 1.1 APIs */
CL_API_ENTRY cl_mem
clCreateIaura2D(cl_context            context,
                cl_mem_flags          flags,
                const cl_iaura_format *iaura_format,
                size_t                iaura_width,
                size_t                iaura_height,
                size_t                iaura_row_pitch,
                void                  *host_ptr,
                cl_int                *errcode_ret)
{
    auto func = aura::CLLibrary::Get().clCreateIaura2D;
    if (func)
    {
        return func(context, flags, iaura_format, iaura_width, iaura_height,
                    iaura_row_pitch, host_ptr, errcode_ret);
    }
    if (errcode_ret)
    {
        *errcode_ret = CL_INVALID_PLATFORM;
    }
    return MI_NULL;
}

CL_API_ENTRY cl_mem
clCreateIaura3D(cl_context             context,
                cl_mem_flags           flags,
                const cl_iaura_format  *iaura_format,
                size_t                 iaura_width,
                size_t                 iaura_height,
                size_t                 iaura_depth,
                size_t                 iaura_row_pitch,
                size_t                 iaura_slice_pitch,
                void                   *host_ptr,
                cl_int                 *errcode_ret)
{
    auto func = aura::CLLibrary::Get().clCreateIaura3D;
    if (func)
    {
        return func(context, flags, iaura_format, iaura_width, iaura_height, iaura_depth,
                    iaura_row_pitch, iaura_slice_pitch, host_ptr, errcode_ret);
    }
    if (errcode_ret)
    {
        *errcode_ret = CL_INVALID_PLATFORM;
    }
    return MI_NULL;
}

CL_API_ENTRY cl_int
clEnqueueMarker(cl_command_queue command_queue,
                cl_event         *event)
{
    auto func = aura::CLLibrary::Get().clEnqueueMarker;
    if (func)
    {
        return func(command_queue, event);
    }
    return CL_INVALID_PLATFORM;
}

CL_API_ENTRY cl_int
clEnqueueWaitForEvents(cl_command_queue command_queue,
                        cl_uint         num_events,
                        const cl_event *event_list)
{
    auto func = aura::CLLibrary::Get().clEnqueueWaitForEvents;
    if (func)
    {
        return func(command_queue, num_events, event_list);
    }
    return CL_INVALID_PLATFORM;
}

CL_API_ENTRY cl_int
clEnqueueBarrier(cl_command_queue command_queue)
{
    auto func = aura::CLLibrary::Get().clEnqueueBarrier;
    if (func)
    {
        return func(command_queue);
    }
    return CL_INVALID_PLATFORM;
}

CL_API_ENTRY cl_int
clUnloadCompiler(void)
{
    auto func = aura::CLLibrary::Get().clUnloadCompiler;
    if (func)
    {
        return func();
    }
    return CL_INVALID_PLATFORM;
}

CL_API_ENTRY void*
clGetExtensionFunctionAddress(const char *func_name)
{
    auto func = aura::CLLibrary::Get().clGetExtensionFunctionAddress;
    if (func)
    {
        return func(func_name);
    }
    return MI_NULL;
}

/* Deprecated OpenCL 2.0 APIs */
CL_API_ENTRY cl_command_queue
clCreateCommandQueue(cl_context                  context,
                     cl_device_id                device,
                     cl_command_queue_properties properties,
                     cl_int                      *errcode_ret)
{
    auto func = aura::CLLibrary::Get().clCreateCommandQueue;
    if (func)
    {
        return func(context, device, properties, errcode_ret);
    }
    if (errcode_ret)
    {
        *errcode_ret = CL_INVALID_PLATFORM;
    }
    return MI_NULL;
}

CL_API_ENTRY cl_sampler
clCreateSampler(cl_context         context,
                cl_bool            normalized_coords,
                cl_addressing_mode addressing_mode,
                cl_filter_mode     filter_mode,
                cl_int             *errcode_ret)
{
    auto func = aura::CLLibrary::Get().clCreateSampler;
    if (func)
    {
        return func(context, normalized_coords, addressing_mode, filter_mode, errcode_ret);
    }
    if (errcode_ret)
    {
        *errcode_ret = CL_INVALID_PLATFORM;
    }
    return MI_NULL;
}

CL_API_ENTRY cl_int
clEnqueueTask(cl_command_queue command_queue,
              cl_kernel        kernel,
              cl_uint          num_events_in_wait_list,
              const cl_event   *event_wait_list,
              cl_event         *event)
{
    auto func = aura::CLLibrary::Get().clEnqueueTask;
    if (func)
    {
        return func(command_queue, kernel, num_events_in_wait_list, event_wait_list, event);
    }
    return CL_INVALID_PLATFORM;
}

CL_API_ENTRY cl_int
clGetDeviceIauraInfoQCOM(cl_device_id device,
                        size_t iaura_width,
                        size_t iaura_height,
                        const cl_iaura_format *iaura_format,
                        cl_iaura_pitch_info_qcom param_name,
                        size_t param_value_size,
                        void *param_value,
                        size_t *param_value_size_ret)
{
    auto func = aura::CLLibrary::Get().clGetDeviceIauraInfoQCOM;
    if (func)
    {
        return func(device, iaura_width, iaura_height, iaura_format, param_name, param_value_size, param_value, param_value_size_ret);
    }
    return CL_INVALID_PLATFORM;
}