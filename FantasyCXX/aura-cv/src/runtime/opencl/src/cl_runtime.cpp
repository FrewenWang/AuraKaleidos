#include "cl_runtime_impl.hpp"
#include "aura/runtime/logger.h"

#include <memory>
#include <regex>

namespace aura
{

Status FindFirstGPU(std::shared_ptr<cl::Platform> &cl_platform, std::shared_ptr<cl::Device> &cl_device)
{
    std::vector<cl::Platform> cl_platforms;
    cl::Platform::get(&cl_platforms);

    for (auto plat : cl_platforms)
    {
        std::vector<cl::Device> cl_devices;

        std::string platform_name;

        plat.getInfo(CL_PLATFORM_NAME, &platform_name);
        plat.getDevices(CL_DEVICE_TYPE_GPU, &cl_devices); //CL_DEVICE_TYPE_ALL
        for (auto dev : cl_devices)
        {
            cl_platform = std::make_shared<cl::Platform>(plat);
            cl_device   = std::make_shared<cl::Device>(dev);
            return Status::OK;
        }
    }

    return Status::ERROR;
}

AURA_EXPORTS std::string GetCLErrorInfo(cl_int error)
{
    std::string cl_err_str = "INVALID";

    switch (error)
    {
        case CL_SUCCESS:
        {
            cl_err_str = "CL_SUCCESS";
            break;
        }
        case CL_DEVICE_NOT_FOUND:
        {
            cl_err_str = "CL_DEVICE_NOT_FOUND";
            break;
        }
        case CL_DEVICE_NOT_AVAILABLE:
        {
            cl_err_str = "CL_DEVICE_NOT_AVAILABLE";
            break;
        }
        case CL_COMPILER_NOT_AVAILABLE:
        {
            cl_err_str = "CL_COMPILER_NOT_AVAILABLE";
            break;
        }
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:
        {
            cl_err_str = "CL_MEM_OBJECT_ALLOCATION_FAILURE";
            break;
        }
        case CL_OUT_OF_RESOURCES:
        {
            cl_err_str = "CL_OUT_OF_RESOURCES";
            break;
        }
        case CL_OUT_OF_HOST_MEMORY:
        {
            cl_err_str = "CL_OUT_OF_HOST_MEMORY";
            break;
        }
        case CL_PROFILING_INFO_NOT_AVAILABLE:
        {
            cl_err_str = "CL_PROFILING_INFO_NOT_AVAILABLE";
            break;
        }
        case CL_MEM_COPY_OVERLAP:
        {
            cl_err_str = "CL_MEM_COPY_OVERLAP";
            break;
        }
        case CL_IAURA_FORMAT_MISMATCH:
        {
            cl_err_str = "CL_IAURA_FORMAT_MISMATCH";
            break;
        }
        case CL_IAURA_FORMAT_NOT_SUPPORTED:
        {
            cl_err_str = "CL_IAURA_FORMAT_NOT_SUPPORTED";
            break;
        }
        case CL_BUILD_PROGRAM_FAILURE:
        {
            cl_err_str = "CL_BUILD_PROGRAM_FAILURE";
            break;
        }
        case CL_MAP_FAILURE:
        {
            cl_err_str = "CL_MAP_FAILURE";
            break;
        }
#if defined(CL_VERSION_1_1)
        case CL_MISALIGNED_SUB_BUFFER_OFFSET:
        {
            cl_err_str = "CL_MISALIGNED_SUB_BUFFER_OFFSET";
            break;
        }
        case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
        {
            cl_err_str = "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
            break;
        }
#endif // CL_VERSION_1_1
#if defined(CL_VERSION_1_2)
        case CL_COMPILE_PROGRAM_FAILURE:
        {
            cl_err_str = "CL_COMPILE_PROGRAM_FAILURE";
            break;
        }
        case CL_LINKER_NOT_AVAILABLE:
        {
            cl_err_str = "CL_LINKER_NOT_AVAILABLE";
            break;
        }
        case CL_LINK_PROGRAM_FAILURE:
        {
            cl_err_str = "CL_LINK_PROGRAM_FAILURE";
            break;
        }
        case CL_DEVICE_PARTITION_FAILED:
        {
            cl_err_str = "CL_DEVICE_PARTITION_FAILED";
            break;
        }
        case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:
        {
            cl_err_str = "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
            break;
        }
#endif // CL_VERSION_1_2
        case CL_INVALID_VALUE:
        {
            cl_err_str = "CL_INVALID_VALUE";
            break;
        }
        case CL_INVALID_DEVICE_TYPE:
        {
            cl_err_str = "CL_INVALID_DEVICE_TYPE";
            break;
        }
        case CL_INVALID_PLATFORM:
        {
            cl_err_str = "CL_INVALID_PLATFORM";
            break;
        }
        case CL_INVALID_DEVICE:
        {
            cl_err_str = "CL_INVALID_DEVICE";
            break;
        }
        case CL_INVALID_CONTEXT:
        {
            cl_err_str = "CL_INVALID_CONTEXT";
            break;
        }
        case CL_INVALID_QUEUE_PROPERTIES:
        {
            cl_err_str = "CL_INVALID_QUEUE_PROPERTIES";
            break;
        }
        case CL_INVALID_COMMAND_QUEUE:
        {
            cl_err_str = "CL_INVALID_COMMAND_QUEUE";
            break;
        }
        case CL_INVALID_HOST_PTR:
        {
            cl_err_str = "CL_INVALID_HOST_PTR";
            break;
        }
        case CL_INVALID_MEM_OBJECT:
        {
            cl_err_str = "CL_INVALID_MEM_OBJECT";
            break;
        }
        case CL_INVALID_IAURA_FORMAT_DESCRIPTOR:
        {
            cl_err_str = "CL_INVALID_IAURA_FORMAT_DESCRIPTOR";
            break;
        }
        case CL_INVALID_IAURA_SIZE:
        {
            cl_err_str = "CL_INVALID_IAURA_SIZE";
            break;
        }
        case CL_INVALID_SAMPLER:
        {
            cl_err_str = "CL_INVALID_SAMPLER";
            break;
        }
        case CL_INVALID_BINARY:
        {
            cl_err_str = "CL_INVALID_BINARY";
            break;
        }
        case CL_INVALID_BUILD_OPTIONS:
        {
            cl_err_str = "CL_INVALID_BUILD_OPTIONS";
            break;
        }
        case CL_INVALID_PROGRAM:
        {
            cl_err_str = "CL_INVALID_PROGRAM";
            break;
        }
        case CL_INVALID_PROGRAM_EXECUTABLE:
        {
            cl_err_str = "CL_INVALID_PROGRAM_EXECUTABLE";
            break;
        }
        case CL_INVALID_KERNEL_NAME:
        {
            cl_err_str = "CL_INVALID_KERNEL_NAME";
            break;
        }
        case CL_INVALID_KERNEL_DEFINITION:
        {
            cl_err_str = "CL_INVALID_KERNEL_DEFINITION";
            break;
        }
        case CL_INVALID_KERNEL:
        {
            cl_err_str = "CL_INVALID_KERNEL";
            break;
        }
        case CL_INVALID_ARG_INDEX:
        {
            cl_err_str = "CL_INVALID_ARG_INDEX";
            break;
        }
        case CL_INVALID_ARG_VALUE:
        {
            cl_err_str = "CL_INVALID_ARG_VALUE";
            break;
        }
        case CL_INVALID_ARG_SIZE:
        {
            cl_err_str = "CL_INVALID_ARG_SIZE";
            break;
        }
        case CL_INVALID_KERNEL_ARGS:
        {
            cl_err_str = "CL_INVALID_KERNEL_ARGS";
            break;
        }
        case CL_INVALID_WORK_DIMENSION:
        {
            cl_err_str = "CL_INVALID_WORK_DIMENSION";
            break;
        }
        case CL_INVALID_WORK_GROUP_SIZE:
        {
            cl_err_str = "CL_INVALID_WORK_GROUP_SIZE";
            break;
        }
        case CL_INVALID_WORK_ITEM_SIZE:
        {
            cl_err_str = "CL_INVALID_WORK_ITEM_SIZE";
            break;
        }
        case CL_INVALID_GLOBAL_OFFSET:
        {
            cl_err_str = "CL_INVALID_GLOBAL_OFFSET";
            break;
        }
        case CL_INVALID_EVENT_WAIT_LIST:
        {
            cl_err_str = "CL_INVALID_EVENT_WAIT_LIST";
            break;
        }
        case CL_INVALID_EVENT:
        {
            cl_err_str = "CL_INVALID_EVENT";
            break;
        }
        case CL_INVALID_OPERATION:
        {
            cl_err_str = "CL_INVALID_OPERATION";
            break;
        }
        case CL_INVALID_GL_OBJECT:
        {
            cl_err_str = "CL_INVALID_GL_OBJECT";
            break;
        }
        case CL_INVALID_BUFFER_SIZE:
        {
            cl_err_str = "CL_INVALID_BUFFER_SIZE";
            break;
        }
        case CL_INVALID_MIP_LEVEL:
        {
            cl_err_str = "CL_INVALID_MIP_LEVEL";
            break;
        }
        case CL_INVALID_GLOBAL_WORK_SIZE:
        {
            cl_err_str = "CL_INVALID_GLOBAL_WORK_SIZE";
            break;
        }
#if defined(CL_VERSION_1_1)
        case CL_INVALID_PROPERTY:
        {
            cl_err_str = "CL_INVALID_PROPERTY";
            break;
        }
#endif // CL_VERSION_1_1
#if defined(CL_VERSION_1_2)
        case CL_INVALID_IAURA_DESCRIPTOR:
        {
            cl_err_str = "CL_INVALID_IAURA_DESCRIPTOR";
            break;
        }
        case CL_INVALID_COMPILER_OPTIONS:
        {
            cl_err_str = "CL_INVALID_COMPILER_OPTIONS";
            break;
        }
        case CL_INVALID_LINKER_OPTIONS:
        {
            cl_err_str = "CL_INVALID_LINKER_OPTIONS";
            break;
        }
        case CL_INVALID_DEVICE_PARTITION_COUNT:
        {
            cl_err_str = "CL_INVALID_DEVICE_PARTITION_COUNT";
            break;
        }
#endif // CL_VERSION_1_2
#if defined(CL_VERSION_2_0)
        case CL_INVALID_PIPE_SIZE:
        {
            cl_err_str = "CL_INVALID_PIPE_SIZE";
            break;
        }
        case CL_INVALID_DEVICE_QUEUE:
        {
            cl_err_str = "CL_INVALID_DEVICE_QUEUE";
            break;
        }
#endif // CL_VERSION_2_0
#if defined(CL_VERSION_2_2)
        case CL_INVALID_SPEC_ID:
        {
            cl_err_str = "CL_INVALID_SPEC_ID";
            break;
        }
        case CL_MAX_SIZE_RESTRICTION_EXCEEDED:
        {
            cl_err_str = "CL_MAX_SIZE_RESTRICTION_EXCEEDED";
            break;
        }
#endif // CL_VERSION_2_2
        default:
        {
            break;
        }
    }

    return "Unknown cl error num (" + cl_err_str + ")";
}

AURA_EXPORTS std::string GetCLProfilingInfo(const std::string &kernel_name, cl::Event &cl_event)
{
    std::string str;

    DT_F64 t0 = cl_event.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>();
    DT_F64 t1 = cl_event.getProfilingInfo<CL_PROFILING_COMMAND_SUBMIT>();
    DT_F64 t2 = cl_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    DT_F64 t3 = cl_event.getProfilingInfo<CL_PROFILING_COMMAND_END>();

    str = kernel_name + " profiling\n    queued->submit : " + std::to_string((t1 - t0) * 1e-6) +
          "ms\n    submit->start : " + std::to_string((t2 - t1) * 1e-6) +
          "ms\n    start->end : " + std::to_string((t3 - t2) * 1e-6) + "ms\n";
    return str;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

CLRuntime::~CLRuntime()
{}

////////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(CL_VERSION_2_0)

AllocatorSVM::AllocatorSVM(CLRuntime *cl_rt)
                           : Allocator(AURA_MEM_SVM, "svm"),
                             m_valid(DT_FALSE), m_fine_grain(DT_FALSE)
{
    if (cl_rt && cl_rt->IsValid())
    {
        std::shared_ptr<cl::Device> cl_device = cl_rt->GetDevice();
        cl_device_svm_capabilities cl_cap = cl_device->getInfo<CL_DEVICE_SVM_CAPABILITIES>();

        if (cl_cap & CL_DEVICE_SVM_FINE_GRAIN_BUFFER)
        {
            m_valid      = DT_TRUE;
            m_fine_grain = DT_TRUE;
        }
        else if (cl_cap & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER)
        {
            m_valid      = DT_TRUE;
            m_fine_grain = DT_FALSE;
        }

        if (m_valid)
        {
            m_cl_context       = cl_rt->GetContext();
            m_cl_command_queue = cl_rt->GetCommandQueue();
        }
    }
}

AllocatorSVM::~AllocatorSVM(DT_VOID)
{
    m_valid      = DT_FALSE;
    m_fine_grain = DT_FALSE;
}

Buffer AllocatorSVM::Allocate(DT_S64 size, DT_S32 align)
{
    if (size > 0 && IsValid())
    {
        cl_svm_mem_flags cl_flags = (m_fine_grain) ? (CL_MEM_SVM_FINE_GRAIN_BUFFER) : (0);
        DT_VOID *svm_ptr = ::clSVMAlloc((*m_cl_context)(), cl_flags | CL_MEM_READ_WRITE, size, align);

        if (svm_ptr)
        {
            return Buffer(AURA_MEM_SVM, size, size, svm_ptr, svm_ptr, 0);
        }
    }

    return Buffer();
}

DT_VOID AllocatorSVM::Free(Buffer &buffer)
{
    if (AURA_MEM_SVM == buffer.m_type && buffer.m_origin != DT_NULL && IsValid())
    {
        ::clSVMFree((*m_cl_context)(), buffer.m_origin);
        buffer.Clear();
    }
}

Status AllocatorSVM::Map(const Buffer &buffer)
{
    if (AURA_MEM_SVM == buffer.m_type && buffer.m_origin != DT_NULL && IsValid())
    {
        if (m_fine_grain)
        {
            return Status::OK;
        }

        cl_map_flags map_flags = CL_MAP_READ | CL_MAP_WRITE;
        if (::clEnqueueSVMMap((*m_cl_command_queue)(), CL_TRUE, map_flags, buffer.m_origin,
                            buffer.m_capacity, 0, NULL, NULL) == CL_SUCCESS)
        {
            return Status::OK;
        }
    }

    return Status::ERROR;
}

Status AllocatorSVM::Unmap(const Buffer &buffer)
{
    if (AURA_MEM_SVM == buffer.m_type && buffer.m_origin != DT_NULL && IsValid())
    {
        if (m_fine_grain)
        {
            return Status::OK;
        }
        cl::Event cl_event;
        if (::clEnqueueSVMUnmap((*m_cl_command_queue)(), buffer.m_origin, 0, NULL, &(cl_event())) == CL_SUCCESS)
        {
            cl_event.wait();
            return Status::OK;
        }
    }

    return Status::ERROR;
}

DT_BOOL AllocatorSVM::IsValid() const
{
    return m_valid;
}

#endif
////////////////////////////////////////////////////////////////////////////////////////////////////
MobileCLRuntime::MobileCLRuntime(Context *m_ctx,
                                 std::shared_ptr<cl::Platform> &cl_platform,
                                 std::shared_ptr<cl::Device> &cl_device,
                                 const CLEngineConfig &cl_conf)
                                 : m_valid(DT_FALSE), m_ctx(m_ctx),
                                   m_cl_conf(std::make_shared<CLEngineConfig>(cl_conf)),
                                   m_cl_platform(cl_platform), m_cl_device(cl_device), m_cl_membk()
{
    m_is_fine_grain = DT_FALSE;

    //get cl version
    std::string driver_version = m_cl_device->getInfo<CL_DRIVER_VERSION>();
    std::regex regex(R"(\d+\.\d+)");
    std::smatch match;

     if (std::regex_search(driver_version, match, regex))
     {
        m_cl_version = std::stof(match[0].str());
     }

#if defined(CL_VERSION_2_0)
    if (m_cl_version >= 2.0f)
    {
        cl_device_svm_capabilities cl_cap = m_cl_device->getInfo<CL_DEVICE_SVM_CAPABILITIES>();
        m_is_fine_grain = (cl_cap & CL_DEVICE_SVM_FINE_GRAIN_BUFFER) ? DT_TRUE : DT_FALSE;
    }
#endif

    if (m_ctx)
    {
        //get cl extensions
        m_cl_extensions_str = m_cl_device->getInfo<CL_DEVICE_EXTENSIONS>();
        cl_int ret = CL_SUCCESS;

#if defined(CL_VERSION_2_0)
        ret = m_cl_device->getInfo(CL_DEVICE_IAURA_PITCH_ALIGNMENT, &m_iaura_pitch_align);
        if (CL_SUCCESS != ret)
        {
            std::string info = "get CL_DEVICE_IAURA_PITCH_ALIGNMENT info failed Error: " + GetCLErrorInfo(ret) + "\n";
            AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
            m_iaura_pitch_align = 64;
        }
#else
        m_iaura_pitch_align = 64;
#endif

#if defined(CL_VERSION_3_0)
        ret = m_cl_device->getInfo(CL_DEVICE_NON_UNIFORM_WORK_GROUP_SUPPORT, &m_is_support_non_uniform_workgroups);
        if (CL_SUCCESS != ret)
        {
            std::string info = "get CL_DEVICE_NON_UNIFORM_WORK_GROUP_SUPPORT info failed Error: " + GetCLErrorInfo(ret) + "\n";
            AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
            m_is_support_non_uniform_workgroups = CL_FALSE;
        }
#else
        m_is_support_non_uniform_workgroups = CL_FALSE;
#endif

        ret = m_cl_device->getInfo(CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, &m_cache_line_size);
        if (CL_SUCCESS != ret)
        {
            std::string info = "get CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE info failed Error: " + GetCLErrorInfo(ret) + "\n";
            AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
            m_cache_line_size = 64;
        }

        m_3d_iaura_write_support = (m_cl_extensions_str.find("cl_khr_3d_iaura_writes") != std::string::npos);
    }
}

MobileCLRuntime::~MobileCLRuntime()
{
#if defined(CL_VERSION_2_0)
    if (m_ctx)
    {
        if (m_ctx->GetMemPool())
        {
            m_ctx->GetMemPool()->UnregisterAllocator(AURA_MEM_SVM);
        }
    }
#endif

    std::lock_guard<std::mutex> guard(m_cl_membk_mutex);

    AURA_LOGD(m_ctx, AURA_TAG, "***********************************************\n");

    if (!m_cl_membk.empty())
    {
        AURA_LOGD(m_ctx, AURA_TAG, "****************** GPU Mem Leak *******************\n");
        AURA_LOGD(m_ctx, AURA_TAG, "****************** GPU Blk info *******************\n");

        DT_S32 counter = 0;
        DT_S32 leak_mem_size = 0;

        for (auto iter = m_cl_membk.begin(); iter != m_cl_membk.end(); ++iter)
        {
            AURA_LOGD(m_ctx, AURA_TAG, "* blk [%zu] - %p\n", counter, reinterpret_cast<DT_VOID*>(iter->first));
            AURA_LOGD(m_ctx, AURA_TAG, "*   size: %zu byte\n", iter->second);
            AURA_LOGD(m_ctx, AURA_TAG, "*\n");

            counter++;
            leak_mem_size += iter->second;
        }

        AURA_LOGD(m_ctx, AURA_TAG, "***********************************************\n");

        AURA_LOGD(m_ctx, AURA_TAG, "* total leak mem size: %.2f KB (%.4f MB)\n",
                  leak_mem_size / 1024.f, leak_mem_size / 1048576.f);

        m_cl_membk.clear();
    }
}

Status MobileCLRuntime::Initialize()
{
    Status ret = Status::ERROR;

    if (m_cl_device && m_ctx)
    {
        ret = Status::OK;
        cl_int cl_err = CL_SUCCESS;
        //create cl context
        m_cl_context = std::make_shared<cl::Context>(*m_cl_device, DT_NULL, DT_NULL, DT_NULL, &cl_err);

        if (CL_SUCCESS != cl_err)
        {
            m_cl_context.reset();
            std::string info = "create cl context failed Error: " + GetCLErrorInfo(cl_err) + "\n";
            AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
            ret = Status::ERROR;
        }
    }

    if (Status::OK == ret)
    {
        ret = CreateCLCommandQueue();
    }

    if (Status::OK == ret)
    {
        ret = CreateCLProgram(m_cl_conf->m_external_version);
    }

    m_valid = DT_TRUE;

    //svm register
    RegisterSvmAllocator();

    return ret;
}

std::shared_ptr<cl::Program> MobileCLRuntime::GetCLProgram(const std::string &program_name,
                                                           const std::string &source,
                                                           const std::string &build_options) const
{
    if (IsValid())
    {
        return m_cl_program_container->GetCLProgram(program_name, source, build_options);
    }
    return DT_NULL;
}

DT_VOID MobileCLRuntime::DeleteCLMem(DT_VOID **ptr)
{
    if ((DT_NULL == ptr) || (DT_NULL == *ptr))
    {
        return;
    }

    std::lock_guard<std::mutex> guard(m_cl_membk_mutex);

    if (!m_cl_membk.empty())
    {
        DT_UPTR_T addr = reinterpret_cast<DT_UPTR_T>(*ptr);

        if (m_cl_membk.count(addr))
        {
            delete reinterpret_cast<cl::Memory*>(*ptr);
            *ptr = DT_NULL;
            m_cl_membk.erase(addr);
        }
    }
}

Status MobileCLRuntime::CreatePrecompiledCLProgram(const std::string &file_path, const std::string &prefix)
{
    if (file_path.empty())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "file_path is empty");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    if (m_cl_program_container)
    {
        ret = m_cl_program_container->CreatePrecompiledCLProgram(file_path, prefix);
    }

    return ret;
}

std::shared_ptr<cl::Platform> MobileCLRuntime::GetPlatform()
{
    return m_cl_platform;
}

std::shared_ptr<cl::Device> MobileCLRuntime::GetDevice()
{
    return m_cl_device;
}

std::shared_ptr<cl::Context> MobileCLRuntime::GetContext()
{
    return m_cl_context;
}

std::shared_ptr<cl::CommandQueue> MobileCLRuntime::GetCommandQueue()
{
    return m_cl_command_queue;
}

DT_BOOL MobileCLRuntime::IsValid() const
{
    return m_valid;
}

DT_BOOL MobileCLRuntime::IsNonUniformWorkgroupsSupported() const
{
    return m_is_support_non_uniform_workgroups;
}

DT_S32 MobileCLRuntime::GetCLAddrAlignSize() const
{
    DT_S32 addr_align_size = 1024;

    if (m_ctx && m_cl_device)
    {
        DT_S32 ret = m_cl_device->getInfo(CL_DEVICE_MEM_BASE_ADDR_ALIGN, &addr_align_size);
        if (CL_SUCCESS != ret)
        {
            std::string info = "get CL_DEVICE_MEM_BASE_ADDR_ALIGN info failed Error: " + GetCLErrorInfo(ret) + "\n";
            AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
        }
    }

    return addr_align_size;
}

DT_BOOL MobileCLRuntime::IsMemShareSupported() const
{
    return DT_FALSE;
}

DT_S32 MobileCLRuntime::GetCLLengthAlignSize() const
{
    return m_iaura_pitch_align;
}

DT_S32 MobileCLRuntime::GetCLSliceAlignSize(const cl_iaura_format &cl_fmt, size_t width, size_t height) const
{
    AURA_UNUSED(cl_fmt);
    AURA_UNUSED(width);
    AURA_UNUSED(height);

    return 1;
}

std::string MobileCLRuntime::GetCLMaxConstantSizeString(DT_S32 n)
{
    AURA_UNUSED(n);

    return std::string();
}

cl::NDRange MobileCLRuntime::GetCLDefaultLocalSize(DT_U32 max_group_size, cl::NDRange &cl_global_size)
{
    DT_U32 item_sizes = 1;
    for (DT_S32 idx = 0; idx < (DT_S32)(cl_global_size.dimensions()); idx++)
    {
        item_sizes *= cl_global_size.get()[idx];
    }

    if (item_sizes < max_group_size)
    {
        max_group_size = item_sizes;
    }

    cl::NDRange cl_local_size = cl::NDRange(32, (max_group_size / 32));

    if (cl_global_size.get()[0] < cl_local_size.get()[0])
    {
        cl_local_size.get()[0] = cl_global_size.get()[0];
        cl_local_size.get()[1] = max_group_size / cl_global_size.get()[0];
    }
    else if (cl_global_size.get()[1] < cl_local_size.get()[1])
    {
        cl_local_size.get()[1] = cl_global_size.get()[1];
        cl_local_size.get()[0] = max_group_size / cl_global_size.get()[1];
    }

    return cl_local_size;
}

GpuInfo MobileCLRuntime::GetGpuInfo() const
{
    return GpuInfo(GpuType::MOBILE);
}

Status MobileCLRuntime::CreateCLCommandQueue()
{
    Status ret = Status::ERROR;
    if (m_cl_context && m_cl_device && m_ctx)
    {
        ret = Status::OK;

        cl_int cl_err = CL_SUCCESS;
        cl_command_queue_properties cl_properties = CL_QUEUE_PROFILING_ENABLE;

        m_cl_command_queue = std::make_shared<cl::CommandQueue>(*m_cl_context, *m_cl_device, cl_properties, &cl_err);

        if (CL_SUCCESS != cl_err)
        {
            m_cl_command_queue.reset();
            std::string info = "create cl command queue failed, Error: " + GetCLErrorInfo(cl_err);
            AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
            ret = Status::ERROR;
        }
    }

    return ret;
}

Status MobileCLRuntime::CreateCLProgram(const std::string &extenal_version)
{
    if (m_ctx && m_cl_device && m_cl_context)
    {
        std::string cl_driver_version = m_cl_device->getInfo<CL_DRIVER_VERSION>();

        m_cl_program_container = std::make_shared<CLProgramContainer>(m_ctx, m_cl_device,
                                                                   m_cl_context, cl_driver_version,
                                                                   m_ctx->GetVersion(),
                                                                   extenal_version,
                                                                   m_cl_conf);

        return Status::OK;
    }

    return Status::ERROR;
}

cl::Buffer* MobileCLRuntime::CreateCLBuffer(cl_mem_flags cl_flags, size_t size)
{
    if (0 == size)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "size must greater 0");
        return DT_NULL;
    }

    cl_int cl_err = CL_SUCCESS;
    cl::Buffer *cl_buffer = new cl::Buffer(*m_cl_context, cl_flags, size, DT_NULL, &cl_err);
    if (cl_err != CL_SUCCESS)
    {
        std::string info = "create buffer failed, Error: " + GetCLErrorInfo(cl_err);
        AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
        if (cl_buffer)
        {
            delete cl_buffer;
            cl_buffer = DT_NULL;
        }
        return DT_NULL;
    }

    std::lock_guard<std::mutex> guard(m_cl_membk_mutex);
    m_cl_membk.emplace(reinterpret_cast<DT_UPTR_T>(cl_buffer), size);

    return cl_buffer;
}

cl::Iaura2D* MobileCLRuntime::CreateCLIaura2D(cl_mem_flags cl_flags, const cl_iaura_format &cl_fmt, size_t width, size_t height)
{
    if (width < 1 || height < 1)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "iaura2d iaura width height must greater 0");
        return DT_NULL;
    }

    cl_int cl_err = CL_SUCCESS;
    cl::Iaura2D *cl_iaura2d = new cl::Iaura2D(*m_cl_context, cl_flags,
                                              cl::IauraFormat(cl_fmt.iaura_channel_order,
                                                              cl_fmt.iaura_channel_data_type),
                                              width, height, 0, DT_NULL, &cl_err);
    if (cl_err != CL_SUCCESS)
    {
        std::string info = "create iaura2d failed, Error: " + GetCLErrorInfo(cl_err);
        AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
        if (cl_iaura2d)
        {
            delete cl_iaura2d;
            cl_iaura2d = DT_NULL;
        }

        return DT_NULL;
    }

    std::lock_guard<std::mutex> guard(m_cl_membk_mutex);
    size_t row_pitch = cl_iaura2d->getIauraInfo<CL_IAURA_ROW_PITCH>();
    m_cl_membk.emplace(reinterpret_cast<DT_UPTR_T>(cl_iaura2d), row_pitch * height);

    return cl_iaura2d;
}

cl::Iaura3D* MobileCLRuntime::CreateCLIaura3D(cl_mem_flags cl_flags, const cl_iaura_format &cl_fmt, size_t width, size_t height, size_t depth)
{
    if (depth < 2 || width < 1 || height < 1)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "iaura3d iaura width height must greater 0 and depth must greater 1");
        return DT_NULL;
    }

    cl_int cl_err = CL_SUCCESS;
    cl::Iaura3D *cl_iaura3d = new cl::Iaura3D(*m_cl_context, cl_flags,
                                              cl::IauraFormat(cl_fmt.iaura_channel_order,
                                                              cl_fmt.iaura_channel_data_type),
                                              width, height, depth, 0, 0, DT_NULL, &cl_err);
    if (cl_err != CL_SUCCESS)
    {
        std::string info = "create iaura3d failed, Error: " + GetCLErrorInfo(cl_err);
        AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
        if (cl_iaura3d)
        {
            delete cl_iaura3d;
            cl_iaura3d = DT_NULL;
        }

        return DT_NULL;
    }

    std::lock_guard<std::mutex> guard(m_cl_membk_mutex);
    size_t slice_pitch = cl_iaura3d->getIauraInfo<CL_IAURA_SLICE_PITCH>();
    m_cl_membk.emplace(reinterpret_cast<DT_UPTR_T>(cl_iaura3d), (depth * slice_pitch));

    return cl_iaura3d;
}

#if defined(CL_VERSION_2_0)
cl::Buffer* MobileCLRuntime::InitCLBufferWithSvm(const Buffer &buffer, cl_mem_flags cl_flags, CLMemSyncMethod &cl_sync_method)
{
    if ((!buffer.IsValid()) || (buffer.m_type != AURA_MEM_SVM))
    {
        return DT_NULL;
    }

    cl::Buffer *cl_buffer = DT_NULL;
    cl_int cl_err = CL_SUCCESS;

    cl_mem cl_mem_buffer = clCreateBuffer(m_cl_context->get(), cl_flags | CL_MEM_USE_HOST_PTR, buffer.m_capacity, buffer.m_origin, &cl_err);
    if (cl_err != CL_SUCCESS)
    {
        if (cl_mem_buffer)
        {
            clReleaseMemObject(cl_mem_buffer);
        }
        return cl_buffer;
    }

    DT_S32 offset = buffer.GetOffset();
    DT_S32 addr_align_size = GetCLAddrAlignSize();

    if (0 == offset)
    {
        cl_buffer = new cl::Buffer(cl_mem_buffer);
    }
    else if ((offset > 0) && (offset % addr_align_size == 0))
    {
        cl_buffer_region region;
        region.origin = offset;
        region.size   = buffer.m_size;

        cl_mem cl_mem_sub_buffer = clCreateSubBuffer(cl_mem_buffer, cl_flags, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err);
        if (CL_SUCCESS ==  cl_err)
        {
            cl_buffer = new cl::Buffer(cl_mem_sub_buffer);
        }
        else
        {
            if (cl_mem_sub_buffer)
            {
                clReleaseMemObject(cl_mem_sub_buffer);
            }
        }

        clReleaseMemObject(cl_mem_buffer);
    }
    else
    {
        clReleaseMemObject(cl_mem_buffer);
        return cl_buffer;
    }

    cl_sync_method = m_is_fine_grain ? CLMemSyncMethod::AUTO : CLMemSyncMethod::FLUSH;

    std::lock_guard<std::mutex> guard(m_cl_membk_mutex);
    m_cl_membk.emplace(reinterpret_cast<DT_UPTR_T>(cl_buffer), sizeof(cl::Buffer));

    return cl_buffer;
}

cl::Iaura2D* MobileCLRuntime::InitCLIaura2DWithSvm(const Buffer &buffer, cl_mem_flags cl_flags, const cl_iaura_format &cl_fmt, size_t width,
                                                   size_t height, size_t pitch, CLMemSyncMethod &cl_sync_method)
{
    if ((!buffer.IsValid()) || (buffer.m_type != AURA_MEM_SVM) || (buffer.GetOffset() != 0) || (pitch % m_iaura_pitch_align != 0))
    {
        return DT_NULL;
    }

    cl_int cl_err = CL_SUCCESS;

    cl_mem cl_buffer = clCreateBuffer(m_cl_context->get(), cl_flags | CL_MEM_USE_HOST_PTR, buffer.m_capacity, buffer.m_origin, &cl_err);
    if (cl_err != CL_SUCCESS)
    {
        clReleaseMemObject(cl_buffer);
        return DT_NULL;
    }

    cl_iaura_format img_fmt;
    cl_iaura_desc img_desc;

    memset(&img_fmt, 0, sizeof(img_fmt));
    memset(&img_desc, 0, sizeof(img_desc));

    // init cl_fmt
    img_fmt.iaura_channel_order     = cl_fmt.iaura_channel_order;
    img_fmt.iaura_channel_data_type = cl_fmt.iaura_channel_data_type;

    // init desc
    img_desc.iaura_type   = CL_MEM_OBJECT_IAURA2D;
    img_desc.iaura_width  = width;
    img_desc.iaura_height = height;
    img_desc.mem_object   = cl_buffer;

    cl_mem iaura = clCreateIaura(m_cl_context->get(), cl_flags, &img_fmt, &img_desc, DT_NULL, &cl_err);

    cl::Iaura2D *cl_iaura2d = new cl::Iaura2D(iaura);

    cl_sync_method = m_is_fine_grain ? CLMemSyncMethod::AUTO : CLMemSyncMethod::FLUSH;

    clReleaseMemObject(cl_buffer);

    if (CL_SUCCESS == cl_err)
    {
        std::lock_guard<std::mutex> guard(m_cl_membk_mutex);
        m_cl_membk.emplace(reinterpret_cast<DT_UPTR_T>(cl_iaura2d), sizeof(cl::Iaura2D));
    }

    return cl_iaura2d;
}
#endif

DT_VOID MobileCLRuntime::RegisterSvmAllocator()
{
#if defined(CL_VERSION_2_0)
    //register svm memory
    if (m_ctx->GetMemPool())
    {
        Allocator *svm_allocator = new AllocatorSVM(this);
        Status ret = m_ctx->GetMemPool()->RegisterAllocator(AURA_MEM_SVM, svm_allocator);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "AURA_MEM_SVM RegisterAllocator failed");
        }
    }
#endif
}

static std::unordered_map<std::string, std::pair<const DT_CHAR*, std::vector<std::string>>>& GetProgramStringMap()
{
    static std::unordered_map<std::string, std::pair<const DT_CHAR*, std::vector<std::string>>> program_string_map;
    return program_string_map;
}

CLProgramString::CLProgramString(const std::string &name, const DT_CHAR *source, const std::vector<std::string> &incs)
{
    auto &program_string_map = GetProgramStringMap();
    program_string_map[name] = {source, incs};
}

Status CLProgramString::Register()
{
    return Status::OK;
}

std::string GetClProgramString(const std::string &name)
{
    auto &program_string_map = GetProgramStringMap();

    if (program_string_map.find(name) == program_string_map.end())
    {
        return std::string();
    }

    std::string source;
    for (const std::string &inc : program_string_map[name].second)
    {
        if (program_string_map.find(inc) == program_string_map.end())
        {
            return std::string();
        }

        source += program_string_map[inc].first;
    }

    source += program_string_map[name].first;

    return source;
}

} // namespace aura