#include "cl_runtime_impl.hpp"
#include "aura/runtime/logger.h"

#include <sstream>
#include <string>
#include <memory>

namespace aura
{

void PfnNotify(const char *errinfo, const void *private_info, size_t cb, void *user_data)
{
    AURA_UNUSED(errinfo);
    AURA_UNUSED(private_info);
    AURA_UNUSED(cb);
    AURA_UNUSED(user_data);
}

// (void)(*callback)(const char *buffer, size_t len, size_t complete, void *user_data)
// - buffer is a pointer to a character array of size len created by printf.
// - len is the number of new characters in buffer.
// - complete is set to a non zero value if there is no more data in the device’s printf buffer.
// - user_data is the user_data parameter specified to clCreateContext.

void CLArmPrintCB(const char *buffer, size_t len, size_t complete, void *user_data)
{
    (DT_VOID)(complete);
    MaliCLRuntime *cl_rt = reinterpret_cast<MaliCLRuntime*>(user_data);
    if (cl_rt)
    {
        AURA_LOGE(cl_rt->m_ctx, "aura CL kernel", "%.*s", len, buffer);
    }
}

MaliCLRuntime::MaliCLRuntime(Context *ctx,
                             std::shared_ptr<cl::Platform> &cl_platform,
                             std::shared_ptr<cl::Device> &cl_device,
                             const CLEngineConfig &cl_conf)
                             : MobileCLRuntime(ctx, cl_platform, cl_device, cl_conf)
{}

MaliCLRuntime::~MaliCLRuntime()
{}

Status MaliCLRuntime::Initialize()
{
    Status ret = Status::ERROR;

    if (m_cl_device && m_ctx && m_cl_platform && m_cl_conf)
    {
        ret = Status::OK;

        const cl_context_properties cl_properties[] =
        {
            CL_CONTEXT_PLATFORM,      reinterpret_cast<cl_context_properties>((*m_cl_platform)()),
            CL_PRINTF_CALLBACK_ARM,   reinterpret_cast<cl_context_properties>(CLArmPrintCB),
            CL_PRINTF_BUFFERSIZE_ARM, static_cast<cl_context_properties>(0x1000),
            0
        };

        cl_int cl_err = CL_SUCCESS;
        const cl_context_properties *cl_props = cl_properties;
        m_cl_context = std::make_shared<cl::Context>(*m_cl_device, cl_props, PfnNotify, this, &cl_err);

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
        cl_command_queue_properties properties = 0;
        properties |= CL_QUEUE_PROFILING_ENABLE;

        cl_queue_properties queue_properties[12] = {CL_QUEUE_PROPERTIES, properties, 0};

        if (m_cl_version >= 2.0f)
        {
            DT_S32 ind = 2;
            ParseContextProps(m_cl_conf->m_cl_perf_level, m_cl_conf->m_cl_priority_level, queue_properties, ind);
        }

        cl_int cl_err = CL_SUCCESS;

#if defined(CL_VERSION_2_0)
        cl_command_queue queue = clCreateCommandQueueWithProperties(m_cl_context->get(), m_cl_device->get(), queue_properties, &cl_err);
        if (cl_err != CL_SUCCESS)
        {
            cl_queue_properties queue_comm_properties[] = {CL_QUEUE_PROPERTIES, properties, 0};
            queue = clCreateCommandQueueWithProperties(m_cl_context->get(), m_cl_device->get(), queue_comm_properties, &cl_err);
            if (cl_err != CL_SUCCESS)
            {
                m_cl_context.reset();
                std::string info = "create cl clCreateCommandQueueWithProperties failed Error: " + GetCLErrorInfo(cl_err) + "\n";
                AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
                ret = Status::ERROR;
            }
        }
#else
        cl_command_queue queue = clCreateCommandQueue(m_cl_context->get(), m_cl_device->get(), properties, &cl_err);
        if (cl_err != CL_SUCCESS)
        {
            m_cl_context.reset();
            std::string info = "create cl clCreateCommandQueue failed Error: " + GetCLErrorInfo(cl_err) + "\n";
            AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
            ret = Status::ERROR;
        }
#endif
        m_cl_command_queue = std::make_shared<cl::CommandQueue>(queue);
    }

    if (Status::OK == ret)
    {
        ret = CreateCLProgram(m_cl_conf->m_external_version);
    }

    m_cl_import_func = (ClImportMemoryARMFunc)clGetExtensionFunctionAddressForPlatform((*m_cl_platform)(), "clImportMemoryARM");

    m_valid = DT_TRUE;

    //svm register
    RegisterSvmAllocator();

    m_dma_buf_heap_name = m_ctx->GetMemPool()->GetAllocator(AURA_MEM_DMA_BUF_HEAP)->GetName();

    return ret;
}

Status MaliCLRuntime::ParseContextProps(CLPerfLevel cl_perf_level, CLPriorityLevel cl_priority_level, cl_queue_properties *queue_properties, DT_S32 ind)
{
    Status ret = Status::OK;

    if ((CLPerfLevel::PERF_DEFAULT == cl_perf_level) && (CLPriorityLevel::PRIORITY_DEFAULT == cl_priority_level))
    {
        return ret;
    }

    switch (cl_perf_level)
    {
        case CLPerfLevel::PERF_DEFAULT:
        {
            break;
        }
        case CLPerfLevel::PERF_LOW:
        {
            queue_properties[ind++] = CL_QUEUE_THROTTLE_KHR;
            queue_properties[ind++] = CL_QUEUE_THROTTLE_LOW_KHR;
            break;
        }
        case CLPerfLevel::PERF_NORMAL:
        {
            queue_properties[ind++] = CL_QUEUE_THROTTLE_KHR;
            queue_properties[ind++] = CL_QUEUE_THROTTLE_MED_KHR;
            break;
        }
        case CLPerfLevel::PERF_HIGH:
        {
            queue_properties[ind++] = CL_QUEUE_THROTTLE_KHR;
            queue_properties[ind++] = CL_QUEUE_THROTTLE_HIGH_KHR;
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
            queue_properties[ind++] = CL_QUEUE_PRIORITY_KHR;
            queue_properties[ind++] = CL_QUEUE_PRIORITY_LOW_KHR;
            break;
        }
        case CLPriorityLevel::PRIORITY_NORMAL:
        {
            queue_properties[ind++] = CL_QUEUE_PRIORITY_KHR;
            queue_properties[ind++] = CL_QUEUE_PRIORITY_MED_KHR;
            break;
        }
        case CLPriorityLevel::PRIORITY_HIGH:
        {
            queue_properties[ind++] = CL_QUEUE_PRIORITY_KHR;
            queue_properties[ind++] = CL_QUEUE_PRIORITY_HIGH_KHR;
            break;
        }
        default:
        {
            break;
        }
    }

    queue_properties[ind] = 0;

    return ret;
}

cl::Buffer* MaliCLRuntime::InitCLBuffer(const Buffer &buffer, cl_mem_flags cl_flags, CLMemSyncMethod &cl_sync_method)
{
    cl::Buffer *cl_buffer = DT_NULL;

    if (AURA_MEM_SVM == buffer.m_type)
    {
#if defined(CL_VERSION_2_0)
        cl_buffer = InitCLBufferWithSvm(buffer, cl_flags, cl_sync_method);
#endif
    }
    else if (((AURA_MEM_DMA_BUF_HEAP == buffer.m_type) || (AURA_MEM_HEAP == buffer.m_type)) && ((buffer.m_capacity % GetCLLengthAlignSize()) == 0) &&
            (((DT_UPTR_T)buffer.m_origin % GetCLLengthAlignSize()) == 0) && IsMemShareSupported())
    {
        cl_int cl_err = CL_SUCCESS;
        cl_mem ion_mem = DT_NULL;

        if (AURA_MEM_HEAP == buffer.m_type)
        {
            ion_mem        = m_cl_import_func(m_cl_context->get(), cl_flags, DT_NULL, buffer.m_origin, buffer.m_capacity, &cl_err);
            cl_sync_method = CLMemSyncMethod::AUTO;
        }
        else
        {
            cl_import_properties_arm prop[5] = {CL_IMPORT_TYPE_ARM, CL_IMPORT_TYPE_DMA_BUF_ARM, 0};
            ion_mem        = m_cl_import_func(m_cl_context->get(), cl_flags, prop, (DT_VOID*)(&buffer.m_property), buffer.m_capacity, &cl_err);
            cl_sync_method = CLMemSyncMethod::FLUSH;
        }

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
                    cl_mem cl_mem_sub_buffer = clCreateSubBuffer(ion_mem, cl_flags, CL_BUFFER_CREATE_TYPE_REGION, &region, &cl_err);

                    if (CL_SUCCESS == cl_err)
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
                }
                clReleaseMemObject(ion_mem);
            }
            else
            {
                cl_buffer = new cl::Buffer(ion_mem);
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

cl::Iaura2D* MaliCLRuntime::InitCLIaura2D(const Buffer &buffer, cl_mem_flags cl_flags, const cl_iaura_format &cl_fmt, size_t width,
                                          size_t height, size_t pitch, CLMemSyncMethod &cl_sync_method)
{
    cl::Iaura2D *cl_iaura2d = DT_NULL;

    AURA_UNUSED(pitch);

#if defined(CL_VERSION_2_0)
    if (AURA_MEM_SVM == buffer.m_type)
    {
        cl_iaura2d = InitCLIaura2DWithSvm(buffer, cl_flags, cl_fmt, width, height, pitch, cl_sync_method);
    }
    else if (AURA_MEM_DMA_BUF_HEAP == buffer.m_type && ((pitch % GetCLLengthAlignSize()) == 0) &&
            (((DT_UPTR_T)buffer.m_origin % GetCLLengthAlignSize()) == 0) && IsMemShareSupported())
    {
        cl_int cl_err = CL_SUCCESS;
        cl_import_properties_arm prop[3] = {CL_IMPORT_TYPE_ARM, CL_IMPORT_TYPE_DMA_BUF_ARM, 0};
        cl_mem ion_mem = m_cl_import_func((*m_cl_context)(), cl_flags, prop, (DT_VOID*)(&buffer.m_property), buffer.m_capacity, &cl_err);

        if (CL_SUCCESS == cl_err)
        {
            cl_iaura2d = new cl::Iaura2D(*m_cl_context,
                                         cl::IauraFormat(cl_fmt.iaura_channel_order,
                                                        cl_fmt.iaura_channel_data_type),
                                         cl::Buffer(ion_mem), width, height, pitch, &cl_err);
            cl_sync_method = CLMemSyncMethod::FLUSH;

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
    }
#endif

    return cl_iaura2d;
}

cl::Iaura3D* MaliCLRuntime::InitCLIaura3D(const Buffer &buffer, cl_mem_flags cl_flags, const cl_iaura_format &cl_fmt, size_t width,
                                          size_t height, size_t depth, size_t row_pitch, size_t slice_pitch, CLMemSyncMethod &cl_sync_method)
{
    AURA_UNUSED(buffer);
    AURA_UNUSED(row_pitch);
    AURA_UNUSED(slice_pitch);

    cl::Iaura3D *cl_iaura3d = CreateCLIaura3D(cl_flags, cl_fmt, width, height, depth);
    cl_sync_method = CLMemSyncMethod::ENQUEUE;

    return cl_iaura3d;
}

DT_S32 MaliCLRuntime::GetCLAddrAlignSize() const
{
    DT_S32 addr_align_size = 128;

    return addr_align_size;
}

DT_BOOL MaliCLRuntime::IsMemShareSupported() const
{
    return (m_cl_import_func != DT_NULL);
}

DT_S32 MaliCLRuntime::GetCLLengthAlignSize() const
{
    return m_cache_line_size;
}

std::string MaliCLRuntime::GetCLMaxConstantSizeString(DT_S32 n)
{
    AURA_UNUSED(n);

    return "__attribute__(())";
}

GpuInfo MaliCLRuntime::GetGpuInfo() const
{
    return GpuInfo(GpuType::MALI);
}

} // namespace aura