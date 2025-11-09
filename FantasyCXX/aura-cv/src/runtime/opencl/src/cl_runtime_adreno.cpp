#include "cl_runtime_impl.hpp"
#include "aura/runtime/logger.h"

#include <sstream>
#include <string>
#include <memory>

// Adreno extensions
// Adreno performance hints
typedef cl_uint cl_perf_hint;

#define CL_CONTEXT_PERF_HINT_QCOM       (0x40C2)
#define CL_PERF_HINT_HIGH_QCOM          (0x40C3)
#define CL_PERF_HINT_NORMAL_QCOM        (0x40C4)
#define CL_PERF_HINT_LOW_QCOM           (0x40C5)

// Adreno priority hints
typedef cl_uint cl_priority_hint;

#define CL_PRIORITY_HINT_NONE_QCOM      (0)
#define CL_CONTEXT_PRIORITY_HINT_QCOM   (0x40C9)
#define CL_PRIORITY_HINT_HIGH_QCOM      (0x40CA)
#define CL_PRIORITY_HINT_NORMAL_QCOM    (0x40CB)
#define CL_PRIORITY_HINT_LOW_QCOM       (0x40CC)

/* Accepted by clGetKernelWorkGroupInfo */
#define CL_KERNEL_WAVE_SIZE_QCOM        (0xAA02)

#if !defined(CL_MEM_HOST_IOCOHERENT_QCOM)
// Cache policy specifying io-coherence
#  define CL_MEM_HOST_IOCOHERENT_QCOM     (0x40A9)
#endif // CL_MEM_HOST_IOCOHERENT_QCOM

namespace aura
{

AdrenoCLRuntime::AdrenoCLRuntime(Context *ctx,
                                 std::shared_ptr<cl::Platform> &cl_platform,
                                 std::shared_ptr<cl::Device> &cl_device,
                                 const CLEngineConfig &cl_conf)
                                 : MobileCLRuntime(ctx, cl_platform, cl_device, cl_conf),
                                   m_qcom_ext_mem_padding(0), m_qcom_page_size(0),
                                   m_qcom_host_cache_policy(CL_MEM_HOST_WRITEBACK_QCOM),
                                   m_cl_ion_type(AdrenoCLIonType::CL_ION_INVALID)

{}

Status AdrenoCLRuntime::Initialize()
{
    Status ret = Status::ERROR;

    if (m_cl_device && m_ctx)
    {
        ret = Status::OK;

        std::vector<cl_context_properties> cl_properties;
        if (m_cl_version >= 2.0f)
        {
            cl_properties = ParseContextProps(m_cl_conf->m_cl_perf_level, m_cl_conf->m_cl_priority_level);
        }

        cl_int cl_err = CL_SUCCESS;
        const cl_context_properties* props = cl_properties.empty() ? DT_NULL : cl_properties.data();
        m_cl_context = std::make_shared<cl::Context>(*m_cl_device, props, DT_NULL, DT_NULL, &cl_err);

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

    m_cl_ion_type = AdrenoCLIonType::CL_ION_INVALID;

    if ((std::string::npos == m_cl_extensions_str.find("cl_qcom_dmabuf_host_ptr")) &&
        (std::string::npos == m_cl_extensions_str.find("cl_qcom_ion_host_ptr")))
    {
        m_cl_ion_type = AdrenoCLIonType::CL_ION_NOT_SUPPOSE;
        return ret;
    }

    if (std::string::npos == m_cl_extensions_str.find("cl_qcom_ext_host_ptr_iocoherent"))
    {
        m_cl_ion_type = AdrenoCLIonType::CL_ION_UNCACHED;
        m_qcom_host_cache_policy = CL_MEM_HOST_UNCACHED_QCOM;
    }
    else
    {
        m_cl_ion_type = AdrenoCLIonType::CL_ION_CACHED;
        m_qcom_host_cache_policy = CL_MEM_HOST_IOCOHERENT_QCOM;
    }

    m_cl_device->getInfo(CL_DEVICE_EXT_MEM_PADDING_IN_BYTES_QCOM, &m_qcom_ext_mem_padding);
    m_cl_device->getInfo(CL_DEVICE_PAGE_SIZE_QCOM, &m_qcom_page_size);

    m_valid = DT_TRUE;

    //svm register
    RegisterSvmAllocator();

    return ret;
}

Status AdrenoCLRuntime::CreateCLProgram(const std::string &extenal_version)
{
    if (m_ctx && m_cl_device && m_cl_context)
    {
        std::string cl_driver_version = m_cl_device->getInfo<CL_DRIVER_VERSION>();

        DT_S32 index = cl_driver_version.find("Compiler");
        if (index != -1)
        {
            std::string match_str = cl_driver_version.substr(index + 8);

            match_str.erase(std::remove(match_str.begin(), match_str.end(), '.'), match_str.end());
            match_str.erase(std::remove(match_str.begin(), match_str.end(), ' '), match_str.end());
            cl_driver_version = match_str;
        }

        m_cl_program_container = std::make_shared<CLProgramContainer>(m_ctx, m_cl_device,
                                                                      m_cl_context, cl_driver_version,
                                                                      m_ctx->GetVersion(),
                                                                      extenal_version,
                                                                      m_cl_conf);

        return Status::OK;
    }

    return Status::ERROR;
}

DT_S32 AdrenoCLRuntime::GetIauraRowPitch(DT_S32 width, DT_S32 height, cl_iaura_format cl_fmt) const
{
    size_t row_pitch = 0;

    if (DT_NULL == m_ctx || DT_NULL == m_cl_device)
    {
        return row_pitch;
    }

    cl_int cl_err = clGetDeviceIauraInfoQCOM((*m_cl_device)(), width, height, &cl_fmt,
                                            CL_IAURA_ROW_PITCH, sizeof(row_pitch), &row_pitch, NULL);

    if (cl_err != CL_SUCCESS)
    {
        std::string info = "clGetDeviceIauraInfoQCOM row_pitch failed Error: " + GetCLErrorInfo(cl_err);
        AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
    }

    return static_cast<DT_S32>(row_pitch);
}

DT_S32 AdrenoCLRuntime::GetIauraSlicePitch(DT_S32 width, DT_S32 height, cl_iaura_format cl_fmt) const
{
    size_t slice_pitch = 0;

    if (DT_NULL == m_ctx || DT_NULL == m_cl_device)
    {
        return slice_pitch;
    }

    cl_int cl_err = clGetDeviceIauraInfoQCOM(m_cl_device->get(), width, height, &cl_fmt,
                                             CL_IAURA_SLICE_PITCH, sizeof(slice_pitch), &slice_pitch, NULL);

    if (cl_err != CL_SUCCESS)
    {
        std::string info = "clGetDeviceIauraInfoQCOM slice_pitch failed Error: " + GetCLErrorInfo(cl_err);
        AURA_ADD_ERROR_STRING(m_ctx, "get cl iaura3d row pitch failed");
    }

    return static_cast<DT_S32>(slice_pitch);
}

DT_S32 AdrenoCLRuntime::GetCLAddrAlignSize() const
{
    DT_S32 addr_align_size = 128;

    return addr_align_size;
}

DT_BOOL AdrenoCLRuntime::IsMemShareSupported() const
{
    return ((AdrenoCLIonType::CL_ION_CACHED == m_cl_ion_type) || (AdrenoCLIonType::CL_ION_UNCACHED == m_cl_ion_type));
}

DT_S32 AdrenoCLRuntime::GetCLLengthAlignSize() const
{
    return m_iaura_pitch_align;
}

DT_S32 AdrenoCLRuntime::GetCLSliceAlignSize(const cl_iaura_format &cl_fmt, size_t width, size_t height) const
{
    size_t slice_align = 4096;

    if (DT_NULL == m_ctx || DT_NULL == m_cl_device)
    {
        return slice_align;
    }

    if (m_ctx && m_cl_device)
    {
        cl_int cl_err = clGetDeviceIauraInfoQCOM(m_cl_device->get(), width, height, &cl_fmt,
                                                 CL_IAURA_SLICE_ALIGNMENT_QCOM, sizeof(slice_align), &slice_align, NULL);
        if (cl_err != CL_SUCCESS)
        {
            slice_align = 4096;
        }
    }

    return static_cast<DT_S32>(slice_align);
}

std::vector<cl_context_properties> AdrenoCLRuntime::ParseContextProps(CLPerfLevel cl_perf_level, CLPriorityLevel cl_priority_level)
{
    std::vector<cl_context_properties> cl_properties;

    if ((CLPerfLevel::PERF_DEFAULT == cl_perf_level) && (CLPriorityLevel::PRIORITY_DEFAULT == cl_priority_level))
    {
        return  cl_properties;
    }

    switch (cl_perf_level)
    {
        case CLPerfLevel::PERF_DEFAULT:
        {
            break;
        }
        case CLPerfLevel::PERF_LOW:
        {
            cl_properties.push_back(CL_CONTEXT_PERF_HINT_QCOM);
            cl_properties.push_back(CL_PERF_HINT_LOW_QCOM);
            break;
        }
        case CLPerfLevel::PERF_NORMAL:
        {
            cl_properties.push_back(CL_CONTEXT_PERF_HINT_QCOM);
            cl_properties.push_back(CL_PERF_HINT_NORMAL_QCOM);
            break;
        }
        case CLPerfLevel::PERF_HIGH:
        {
            cl_properties.push_back(CL_CONTEXT_PERF_HINT_QCOM);
            cl_properties.push_back(CL_PERF_HINT_HIGH_QCOM);
            break;
        }
        default:
        {
            break;
        }
    }

    switch (cl_priority_level)
    {
        case CLPriorityLevel::PRIORITY_DEFAULT:
        {
            break;
        }
        case CLPriorityLevel::PRIORITY_LOW:
        {
            cl_properties.push_back(CL_CONTEXT_PRIORITY_HINT_QCOM);
            cl_properties.push_back(CL_PRIORITY_HINT_LOW_QCOM);
            break;
        }
        case CLPriorityLevel::PRIORITY_NORMAL:
        {
            cl_properties.push_back(CL_CONTEXT_PRIORITY_HINT_QCOM);
            cl_properties.push_back(CL_PRIORITY_HINT_NORMAL_QCOM);
            break;
        }
        case CLPriorityLevel::PRIORITY_HIGH:
        {
            cl_properties.push_back(CL_CONTEXT_PRIORITY_HINT_QCOM);
            cl_properties.push_back(CL_PRIORITY_HINT_HIGH_QCOM);
            break;
        }
        default:
        {
            break;
        }
    }

    cl_properties.push_back(0);

    return cl_properties;
}

cl::Buffer* AdrenoCLRuntime::InitCLBuffer(const Buffer &buffer, cl_mem_flags cl_flags, CLMemSyncMethod &cl_sync_method)
{
    cl::Buffer *cl_buffer = DT_NULL;

    if (AURA_MEM_SVM == buffer.m_type)
    {
#if defined(CL_VERSION_2_0)
        cl_buffer = InitCLBufferWithSvm(buffer, cl_flags, cl_sync_method);
#endif
    }
    else if ((AURA_MEM_DMA_BUF_HEAP == buffer.m_type) && IsMemShareSupported())
    {
        // the first mem block or uncontinues mem, can be used by zero copy
        cl_mem_ion_host_ptr ion_mem;
        // init cl_mem_ion_host_ptr
        ion_mem.ext_host_ptr.allocation_type   = CL_MEM_ION_HOST_PTR_QCOM;
        ion_mem.ext_host_ptr.host_cache_policy = m_qcom_host_cache_policy;
        ion_mem.ion_filedesc                   = buffer.m_property;
        ion_mem.ion_hostptr                    = buffer.m_origin;

        cl_int cl_err     = CL_SUCCESS;
        cl_mem cl_ion_mem = clCreateBuffer(m_cl_context->get(), cl_flags | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM, buffer.m_capacity, &ion_mem, &cl_err);

        if (CL_SUCCESS == cl_err)
        {
            DT_S32 roi_offset = buffer.GetOffset();
            DT_S32 addr_align_size = GetCLAddrAlignSize();

            if (roi_offset > 0)
            {
                if (roi_offset % addr_align_size == 0)
                {
                    cl_buffer_region region;
                    region.origin            = roi_offset;
                    region.size              = buffer.m_size;
                    cl_mem cl_mem_sub_buffer = clCreateSubBuffer(cl_ion_mem, cl_flags, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err);

                    if (CL_SUCCESS == cl_err)
                    {
                        cl_buffer = new cl::Buffer(cl_mem_sub_buffer);
                        cl_sync_method = CLMemSyncMethod::AUTO;
                    }
                    else
                    {
                        if (cl_mem_sub_buffer)
                        {
                            clReleaseMemObject(cl_mem_sub_buffer);
                        }
                    }

                }
                clReleaseMemObject(cl_ion_mem);
            }
            else
            {
                cl_buffer = new cl::Buffer(cl_ion_mem);
                cl_sync_method = CLMemSyncMethod::AUTO;
            }

            if (CL_SUCCESS == cl_err && cl_buffer)
            {
                std::lock_guard<std::mutex> guard(m_cl_membk_mutex);
                m_cl_membk.emplace(reinterpret_cast<DT_UPTR_T>(cl_buffer), sizeof(cl::Buffer));
            }
        }
    }

    return cl_buffer;
}

cl::Iaura2D* AdrenoCLRuntime::InitCLIaura2D(const Buffer &buffer, cl_mem_flags cl_flags, const cl_iaura_format &cl_fmt, size_t width,
                                            size_t height, size_t pitch, CLMemSyncMethod &cl_sync_method)
{
    cl_int cl_err           = CL_SUCCESS;
    cl::Iaura2D *cl_iaura2d = DT_NULL;

    if (AURA_MEM_SVM == buffer.m_type)
    {
#if defined(CL_VERSION_2_0)
        cl_iaura2d = InitCLIaura2DWithSvm(buffer, cl_flags, cl_fmt, width, height, pitch, cl_sync_method);
#endif
    }
    else if ((AURA_MEM_DMA_BUF_HEAP == buffer.m_type) && IsMemShareSupported() &&
             (buffer.GetOffset() == 0) && (pitch % GetCLLengthAlignSize() == 0))
    {
        // the first mem block or uncontinues mem, can be used by zero copy
        cl_mem_ion_host_ptr ion_mem;
        // init cl_mem_ion_host_ptr
        ion_mem.ext_host_ptr.allocation_type    = CL_MEM_ION_HOST_PTR_QCOM;
        ion_mem.ext_host_ptr.host_cache_policy  = m_qcom_host_cache_policy;
        ion_mem.ion_filedesc                    = buffer.m_property;
        ion_mem.ion_hostptr                     = buffer.m_origin;

        cl_iaura2d = new cl::Iaura2D(*m_cl_context,
                                     cl_flags | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
                                     cl::IauraFormat(cl_fmt.iaura_channel_order,
                                                     cl_fmt.iaura_channel_data_type),
                                     width, height, pitch, &ion_mem, &cl_err);

        cl_sync_method = CLMemSyncMethod::AUTO;

        if (CL_SUCCESS == cl_err)
        {
            std::lock_guard<std::mutex> guard(m_cl_membk_mutex);
            m_cl_membk.emplace(reinterpret_cast<DT_UPTR_T>(cl_iaura2d), sizeof(cl::Iaura2D));
        }
        else
        {
            if (cl_iaura2d)
            {
                delete cl_iaura2d;
                cl_iaura2d = DT_NULL;
            }
        }
    }

    return cl_iaura2d;
}

cl::Iaura3D* AdrenoCLRuntime::InitCLIaura3D(const Buffer &buffer, cl_mem_flags cl_flags, const cl_iaura_format &cl_fmt, size_t width,
                                            size_t height, size_t depth, size_t row_pitch, size_t slice_pitch, CLMemSyncMethod &cl_sync_method)
{
    cl_int cl_err           = CL_SUCCESS;
    cl::Iaura3D *cl_iaura3d = DT_NULL;

    DT_S32 slice_align = GetCLSliceAlignSize(cl_fmt, width, height);

    if ((AURA_MEM_DMA_BUF_HEAP == buffer.m_type) && IsMemShareSupported() && (buffer.GetOffset() == 0) &&
        (row_pitch % GetCLLengthAlignSize() == 0) && slice_pitch % slice_align == 0)
    {
        cl_mem_ion_host_ptr ion_mem;
        // init cl_mem_ion_host_ptr
        ion_mem.ext_host_ptr.allocation_type   = CL_MEM_ION_HOST_PTR_QCOM;
        ion_mem.ext_host_ptr.host_cache_policy = m_qcom_host_cache_policy;
        ion_mem.ion_filedesc                   = buffer.m_property;
        ion_mem.ion_hostptr                    = buffer.m_origin;

        cl_iaura3d = new cl::Iaura3D(*m_cl_context, cl_flags | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
                                     cl::IauraFormat(cl_fmt.iaura_channel_order,
                                                     cl_fmt.iaura_channel_data_type),
                                     width, height, depth, row_pitch, slice_pitch, &ion_mem, &cl_err);

        cl_sync_method = CLMemSyncMethod::AUTO;

        if (CL_SUCCESS == cl_err)
        {
            std::lock_guard<std::mutex> guard(m_cl_membk_mutex);
            m_cl_membk.emplace(reinterpret_cast<DT_UPTR_T>(cl_iaura3d), sizeof(cl::Iaura3D));
        }
        else
        {
            if (cl_iaura3d)
            {
                delete cl_iaura3d;
                cl_iaura3d = DT_NULL;
            }
        }
    }

    return cl_iaura3d;
}

std::string AdrenoCLRuntime::GetCLMaxConstantSizeString(DT_S32 n)
{
    DT_CHAR str[128];
    snprintf(str, sizeof(str), "__attribute__((max_constant_size(%d)))", n);

    std::string max_constant_size_str(str);

    return max_constant_size_str;
}

GpuInfo AdrenoCLRuntime::GetGpuInfo() const
{
    return GpuInfo(GpuType::ADRENO);
}

} // namespace aura