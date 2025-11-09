#include "median_impl.hpp"
#include "aura/runtime/opencl.h"
#include "aura/runtime/logger.h"

namespace aura
{

static DT_VOID GetCLGlobalSize(GpuType gpu_type, DT_S32 ksize, DT_S32 channel, DT_S32 height,
                               DT_S32 width, cl::NDRange &range)
{
    DT_S32 row_elem_count = width * channel;

    // ksize = 3
    if (ksize == 3 && channel == 1)
    {
        range = cl::NDRange((row_elem_count + 3) / 4, (height + 1) >> 1);
    }
    else if (ksize == 3 && (channel == 2 || channel == 3))
    {
        range = cl::NDRange((row_elem_count + 7) / 8, (height + 1) >> 1);
    }
    // ksize = 5
    else if (ksize == 5 && (channel == 1 || channel == 2))
    {
        range = cl::NDRange((row_elem_count + 7) / 8, (height + 1) >> 1);
    }
    else if (ksize == 5 && channel == 3)
    {
        range = cl::NDRange((row_elem_count + 5) / 6, (height + 1) >> 1);
    }
    // ksize = 7; gpu = adreno
    else if ((GpuType::ADRENO == gpu_type) && ksize == 7 && channel == 1)
    {
        range = cl::NDRange((row_elem_count + 1) / 2, (height + 1) >> 1);
    }
    else if ((GpuType::ADRENO == gpu_type) && ksize == 7 && channel == 2)
    {
        range = cl::NDRange((row_elem_count + 3) / 4, (height + 1) >> 1);
    }
    else if ((GpuType::ADRENO == gpu_type) && ksize == 7 && channel == 3)
    {
        range = cl::NDRange((row_elem_count + 5) / 6, (height + 1) >> 1);
    }
    // ksize = 7; gpu = mali
    else if ((GpuType::MALI == gpu_type) && ksize == 7 && (channel == 1 || channel == 2))
    {
        range = cl::NDRange((row_elem_count + 7) / 8, (height + 1) >> 1);
    }
    else if ((GpuType::MALI == gpu_type) && ksize == 7 && channel == 3)
    {
        range = cl::NDRange((row_elem_count + 5) / 6, (height + 1) >> 1);
    }
}

static std::string GetCLBuildOptions(Context *ctx, ElemType elem_type)
{
    CLBuildOptions cl_build_opt(ctx);
    cl_build_opt.AddOption("Tp", CLTypeString(elem_type));

    return cl_build_opt.ToString();
}

static Status GetCLName(Context *ctx, DT_S32 channel, DT_S32 ksize, std::string &program_name, std::string &kernel_name)
{
    if (DT_NULL == ctx)
    {
        return Status::ERROR;
    }

    if (ctx->GetCLEngine() == DT_NULL || ctx->GetCLEngine()->GetCLRuntime() == DT_NULL)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetCLEngine failed");
        return Status::ERROR;
    }

    GpuType gpu_type = ctx->GetCLEngine()->GetCLRuntime()->GetGpuInfo().m_type;

    std::string program_postfix = std::to_string(ksize) + "x" + std::to_string(ksize) + "_c" + std::to_string(channel);
    std::string kernel_postfix  = std::to_string(ksize) + "x" + std::to_string(ksize) + "C"  + std::to_string(channel);

    program_name = "aura_median_" + program_postfix;
    kernel_name  = "MedianFilter" + kernel_postfix;

    if (7 == ksize && (1 == channel || 2 == channel))
    {
        program_name += "_" + GpuTypeToString(gpu_type);
    }

    return Status::OK;
}

MedianCL::MedianCL(Context *ctx, const OpTarget &target) : MedianImpl(ctx, target), m_profiling_string()
{}

Status MedianCL::SetArgs(const Array *src, Array *dst, DT_S32 ksize)
{
    if (MedianImpl::SetArgs(src, dst, ksize) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "MedianImpl::Initialize failed");
        return Status::ERROR;
    }

    if (ksize != 3 && ksize != 5 && ksize != 7)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "only ksize 3/5/7 supported");
        return Status::ERROR;
    }

    if (src->GetSizes().m_channel != 1 && src->GetSizes().m_channel != 2 && src->GetSizes().m_channel != 3)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "only channel 1/2/3 supported");
        return Status::ERROR;
    }

    if (src->GetSizes().m_height < 2 || src->GetSizes().m_width < 64)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src size is too small");
        return Status::ERROR;
    }

    return Status::OK;
}

Status MedianCL::Initialize()
{
    if (MedianImpl::Initialize() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "MedianImpl::Initialize() failed");
        return Status::ERROR;
    }

    // 1. init cl_mem
    m_cl_src = CLMem::FromArray(m_ctx, *m_src, CLMemParam(CL_MEM_READ_ONLY));
    if (!m_cl_src.IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "m_cl_src is invalid");
        return Status::ERROR;
    }

    m_cl_dst = CLMem::FromArray(m_ctx, *m_dst, CLMemParam(CL_MEM_WRITE_ONLY));
    if (!m_cl_dst.IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "m_cl_dst is invalid");
        return Status::ERROR;
    }

    // 2. init kernel
    m_cl_kernels = MedianCL::GetCLKernels(m_ctx, m_dst->GetElemType(), m_dst->GetSizes().m_channel, m_ksize);

    if (CheckCLKernels(m_ctx, m_cl_kernels) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Median CheckCLKernels failed");
        return Status::ERROR;
    }

    // 3. sync start
    if (m_cl_src.Sync(CLMemSyncType::WRITE) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "m_cl_src Sync start failed");
        return Status::ERROR;
    }

    return Status::OK;
}

Status MedianCL::DeInitialize()
{
    m_cl_src.Release();
    m_cl_dst.Release();

    for (auto &cl_kernel : m_cl_kernels)
    {
        cl_kernel.DeInitialize();
    }

    if (MedianImpl::DeInitialize() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "MedianImpl::DeInitialize() failed");
        return Status::ERROR;
    }

    return Status::OK;
}

Status MedianCL::Run()
{
    DT_S32 istep  = m_src->GetRowPitch() / ElemTypeSize(m_src->GetElemType());
    DT_S32 ostep  = m_dst->GetRowPitch() / ElemTypeSize(m_dst->GetElemType());
    DT_S32 height = m_dst->GetSizes().m_height;
    DT_S32 width  = m_dst->GetSizes().m_width;

    std::shared_ptr<CLRuntime> cl_rt = m_ctx->GetCLEngine()->GetCLRuntime();
    // 1. get center_area and cl_global_size
    cl::NDRange cl_global_size;

    GetCLGlobalSize(cl_rt->GetGpuInfo().m_type, m_ksize, m_dst->GetSizes().m_channel, height, width, cl_global_size);

    // 2. opencl run
    cl::Event cl_event;
    cl_int cl_ret = CL_SUCCESS;
    Status ret    = Status::ERROR;

    cl_ret = m_cl_kernels[0].Run<cl::Buffer, DT_S32, cl::Buffer, DT_S32, DT_S32, DT_S32, DT_S32, DT_S32>(
                                 m_cl_src.GetCLMemRef<cl::Buffer>(), istep,
                                 m_cl_dst.GetCLMemRef<cl::Buffer>(), ostep,
                                 height, width,
                                 (DT_S32)(cl_global_size.get()[1]),
                                 (DT_S32)(cl_global_size.get()[0]),
                                 cl_global_size,
                                 cl_rt->GetCLDefaultLocalSize(m_cl_kernels[0].GetMaxGroupSize(),
                                 cl_global_size), &cl_event);
    if (cl_ret != CL_SUCCESS)
    {
        AURA_ADD_ERROR_STRING(m_ctx, ("cl_kernel run failed : " + GetCLErrorInfo(cl_ret)).c_str());
        goto EXIT;
    }

    // 3. cl wait
    if ((DT_TRUE == m_target.m_data.opencl.profiling) || (m_dst->GetArrayType() != ArrayType::CL_MEMORY))
    {
        cl_ret = cl_event.wait();
        if (cl_ret != CL_SUCCESS)
        {
            AURA_ADD_ERROR_STRING(m_ctx, ("Wait cl_kernel failed : " + GetCLErrorInfo(cl_ret)).c_str());
            goto EXIT;
        }

        if (DT_TRUE == m_target.m_data.opencl.profiling)
        {
            m_profiling_string = " " + GetCLProfilingInfo(m_cl_kernels[0].GetKernelName(), cl_event);
        }
    }

    ret = m_cl_dst.Sync(aura::CLMemSyncType::READ);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Sync end fail");
    }

EXIT:
    AURA_RETURN(m_ctx, ret);
}

std::string MedianCL::ToString() const
{
    return MedianImpl::ToString() + m_profiling_string;
}

std::vector<CLKernel> MedianCL::GetCLKernels(Context *ctx, ElemType elem_type, DT_S32 channel, DT_S32 ksize)
{
    std::vector<CLKernel> cl_kernels;
    std::string build_opt = GetCLBuildOptions(ctx, elem_type);

    std::string program_name, kernel_name;
    if (GetCLName(ctx, channel, ksize, program_name, kernel_name) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetCLName failed");
        return cl_kernels;
    }

    cl_kernels.push_back({ctx, program_name, kernel_name, "", build_opt});

    return cl_kernels;
}

} // namespace aura
