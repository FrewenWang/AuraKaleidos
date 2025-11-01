#include "aura/runtime/opencl.h"
#include "aura/runtime/logger.h"
#include "filter2d_impl.hpp"

namespace aura
{

static Status GetCLName(Context *ctx, MI_S32 channel, MI_S32 ksize, std::string program_name[2], std::string kernel_name[2])
{
    if (MI_NULL == ctx)
    {
        return Status::ERROR;
    }

    if (ctx->GetCLEngine() == MI_NULL || ctx->GetCLEngine()->GetCLRuntime() == MI_NULL)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetCLEngine failed");
        return Status::ERROR;
    }

    GpuType gpu_type = ctx->GetCLEngine()->GetCLRuntime()->GetGpuInfo().m_type;

    std::string program_postfix = std::to_string(ksize) + "x" + std::to_string(ksize) + "_c" + std::to_string(channel);
    std::string kernel_postfix = std::to_string(ksize) + "x" + std::to_string(ksize) + "C" + std::to_string(channel);

    program_name[0] = "aura_filter2d_main_" + program_postfix;
    program_name[1] = "aura_filter2d_remain_" + program_postfix;

    kernel_name[0] = "Filter2dMain" + kernel_postfix;
    kernel_name[1] = "Filter2dRemain" + kernel_postfix;

    if ((5 == ksize) && (2 == channel))
    {
        program_name[0] += "_" + GpuTypeToString(gpu_type);
    }

    if ((3 == ksize) && (3 == channel))
    {
        program_name[0] += "_" + GpuTypeToString(gpu_type);
    }

    if (1 == channel)
    {
        program_name[0] += "_" + GpuTypeToString(gpu_type);
    }

    return Status::OK;
}

static std::string GetCLBuildOptions(Context *ctx, ElemType elem_type, BorderType border_type)
{
    CLBuildOptions cl_build_opt(ctx);

    switch (border_type)
    {
        case BorderType::CONSTANT:
        {
            cl_build_opt.AddOption("BORDER_CONSTANT", "1");
            break;
        }
        case BorderType::REPLICATE:
        {
            cl_build_opt.AddOption("BORDER_REPLICATE", "1");
            break;
        }
        case BorderType::REFLECT_101:
        {
            cl_build_opt.AddOption("BORDER_REFLECT_101", "1");
            break;
        }
        default:
        {
            break;
        }
    }

    cl_build_opt.AddOption("MAX_CONSTANT_SIZE", ctx->GetCLEngine()->GetCLRuntime()->GetCLMaxConstantSizeString(196));
    cl_build_opt.AddOption("Tp", CLTypeString(elem_type));

    return cl_build_opt.ToString(elem_type);
}

static AURA_VOID GetCLGlobalSize(GpuType gpu_type, MI_S32 ksize, MI_S32 channel, MI_S32 height, MI_S32 width,
                               cl::NDRange range[2], MI_S32 &main_width)
{
    MI_S32 elem_counts = 0;
    MI_S32 main_size   = 0;
    MI_S32 main_rest   = 0;
    MI_S32 border      = (ksize >> 1) << 1;

    if ((GpuType::ADRENO == gpu_type) && (5 == ksize) && (1 == channel))
    {
        elem_counts = 4;
        main_size   = (width - border) / elem_counts;
        main_rest   = (width - border) % elem_counts;
        range[0]    = cl::NDRange(main_size, (height + 3) >> 2);
    }
    else if ((GpuType::ADRENO == gpu_type) && (5 == ksize) && (2 == channel))
    {
        elem_counts = 4;
        main_size   = (width - border) / elem_counts;
        main_rest   = (width - border) % elem_counts;
        range[0]    = cl::NDRange(main_size, (height + 1) >> 1);
    }
    else if ((GpuType::MALI == gpu_type) && (5 == ksize) && (1 == channel))
    {
        elem_counts = 4;
        main_size   = (width - border) / elem_counts;
        main_rest   = (width - border) % elem_counts;
        range[0]    = cl::NDRange(main_size, (height + 1) >> 1);
    }
    else if ((GpuType::MALI == gpu_type) && (3 == ksize) && (3 == channel))
    {
        elem_counts = 3;
        main_size   = (width - border) / elem_counts - 1;
        main_rest   = (width - border) % elem_counts + 3;
        range[0]    = cl::NDRange(main_size, height);
    }
    else if ((5 == ksize) && (3 == channel))
    {
        elem_counts = 1;
        main_rest   = elem_counts;
        main_size   = width - border - elem_counts;
        range[0]    = cl::NDRange(main_size, (height + 1) >> 1);
    }
    else
    {
        elem_counts = (3 == ksize) ? 6 : ((5 == ksize) ? 4 : 2);
        main_size   = (width - border) / elem_counts;
        main_rest   = (width - border) % elem_counts;
        range[0]    = cl::NDRange(main_size, height);
    }

    // remain global size
    range[1] = cl::NDRange(main_rest + border, height);
    main_width = main_size * elem_counts;
}

Filter2dCL::Filter2dCL(Context *ctx, const OpTarget &target) : Filter2dImpl(ctx, target)
{}

Status Filter2dCL::SetArgs(const Array *src, Array *dst, const Array *kmat,
                           BorderType border_type, const Scalar &border_value)
{
    if (Filter2dImpl::SetArgs(src, dst, kmat, border_type, border_value) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Filter2dImpl::SetArgs failed");
        return Status::ERROR;
    }

    MI_S32 ch = src->GetSizes().m_channel;

    if (m_ksize != 3 && m_ksize != 5 && m_ksize != 7)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "ksize is only suppose 3/5/7");
        return Status::ERROR;
    }

    if (ch != 1 && ch != 2 && ch != 3)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "channel is only suppose 1/2/3");
        return Status::ERROR;
    }

    if (7 == m_ksize && ch != 1)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "CL impl cannot support channel > 1 when ksize == 7");
        return Status::ERROR;
    }

    return Status::OK;
}

Status Filter2dCL::Initialize()
{
    if (Filter2dImpl::Initialize() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Filter2dImpl::Initialize() failed");
        return Status::ERROR;
    }

    // 1. init cl_mem
    m_cl_src = CLMem::FromArray(m_ctx, *m_src, CLMemParam(CL_MEM_READ_ONLY));
    if (!m_cl_src.IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "m_cl_src is invalid");
        return Status::ERROR;
    }

    m_cl_kmat = CLMem::FromArray(m_ctx, *m_kmat, CLMemParam(CL_MEM_READ_ONLY));
    if (!m_cl_kmat.IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "m_cl_kmat is invalid");
        return Status::ERROR;
    }

    m_cl_dst = CLMem::FromArray(m_ctx, *m_dst, CLMemParam(CL_MEM_WRITE_ONLY));
    if (!m_cl_dst.IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "m_cl_dst is invalid");
        return Status::ERROR;
    }

    // 2. init kernel
    m_cl_kernels = Filter2dCL::GetCLKernels(m_ctx, m_src->GetElemType(), m_dst->GetSizes().m_channel, m_ksize, m_border_type);

    if (CheckCLKernels(m_ctx, m_cl_kernels) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "CheckCLKernels failed");
        return Status::ERROR;
    }

    // 3. sync start
    if (m_cl_src.Sync(CLMemSyncType::WRITE) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "m_cl_src Sync start failed");
        return Status::ERROR;
    }

    if (m_cl_kmat.Sync(CLMemSyncType::WRITE) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "m_cl_kmat Sync start failed");
        return Status::ERROR;
    }

    return Status::OK;
}

Status Filter2dCL::DeInitialize()
{
    m_cl_src.Release();
    m_cl_kmat.Release();
    m_cl_dst.Release();

    for (auto &cl_kernel : m_cl_kernels)
    {
        cl_kernel.DeInitialize();
    }

    if (Filter2dImpl::DeInitialize() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Filter2dImpl::DeInitialize() failed");
        return Status::ERROR;
    }

    return Status::OK;
}

Status Filter2dCL::Run()
{
    MI_S32 istep   = m_src->GetRowPitch() / ElemTypeSize(m_src->GetElemType());
    MI_S32 ostep   = m_dst->GetRowPitch() / ElemTypeSize(m_dst->GetElemType());
    MI_S32 height  = m_dst->GetSizes().m_height;
    MI_S32 width   = m_dst->GetSizes().m_width;
    MI_S32 channel = m_dst->GetSizes().m_channel;

    std::shared_ptr<CLRuntime> rt = m_ctx->GetCLEngine()->GetCLRuntime();
    CLScalar cl_border_value = clScalar(m_border_value);

    // 1. get global_size and local_size
    cl::NDRange global_size[2];
    MI_S32 main_width = 0;

    GetCLGlobalSize(rt->GetGpuInfo().m_type, m_ksize, channel, height, width, global_size, main_width);

    // 2. opencl run
    cl::Event cl_event[2];
    cl_int cl_ret = CL_SUCCESS;
    Status ret    = Status::ERROR;

    cl_ret = m_cl_kernels[0].Run<cl::Buffer, MI_S32, cl::Buffer, MI_S32, MI_S32, MI_S32, MI_S32, cl::Buffer, CLScalar>(
                                 m_cl_src.GetCLMemRef<cl::Buffer>(), istep,
                                 m_cl_dst.GetCLMemRef<cl::Buffer>(), ostep,
                                 height,
                                 (MI_S32)(global_size[0].get()[1]),
                                 (MI_S32)(global_size[0].get()[0]),
                                 m_cl_kmat.GetCLMemRef<cl::Buffer>(),
                                 cl_border_value,
                                 global_size[0], rt->GetCLDefaultLocalSize(m_cl_kernels[0].GetMaxGroupSize(), global_size[0]),
                                 &(cl_event[0]));
    if (cl_ret != CL_SUCCESS)
    {
        AURA_ADD_ERROR_STRING(m_ctx, ("Run cl_kernel[0] main failed : " + GetCLErrorInfo(cl_ret)).c_str());
        goto EXIT;
    }

    cl_ret = m_cl_kernels[1].Run<cl::Buffer, MI_S32, cl::Buffer, MI_S32, MI_S32, MI_S32, MI_S32, MI_S32, cl::Buffer, MI_S32, CLScalar>(
                                 m_cl_src.GetCLMemRef<cl::Buffer>(), istep,
                                 m_cl_dst.GetCLMemRef<cl::Buffer>(), ostep,
                                 height, width,
                                 (MI_S32)(global_size[1].get()[1]),
                                 (MI_S32)(global_size[1].get()[0]),
                                 m_cl_kmat.GetCLMemRef<cl::Buffer>(),
                                 main_width, cl_border_value,
                                 global_size[1], rt->GetCLDefaultLocalSize(m_cl_kernels[1].GetMaxGroupSize(), global_size[1]),
                                 &(cl_event[1]), {cl_event[0]});
    if (cl_ret != CL_SUCCESS)
    {
        AURA_ADD_ERROR_STRING(m_ctx, ("Run cl_kernel[1] remain failed : " + GetCLErrorInfo(cl_ret)).c_str());
        goto EXIT;
    }

    // 3. cl wait
    if ((MI_TRUE == m_target.m_data.opencl.profiling) || (m_dst->GetArrayType() != ArrayType::CL_MEMORY))
    {
        cl_ret = cl_event[1].wait();
        if (cl_ret != CL_SUCCESS)
        {
            AURA_ADD_ERROR_STRING(m_ctx, ("Wait cl_kernel failed : " + GetCLErrorInfo(cl_ret)).c_str());
            goto EXIT;
        }

        if (MI_TRUE == m_target.m_data.opencl.profiling)
        {
            m_profiling_string = " " + GetCLProfilingInfo(m_cl_kernels[0].GetKernelName(), cl_event[0]) +
                                       GetCLProfilingInfo(m_cl_kernels[1].GetKernelName(), cl_event[1]);
        }
    }

    ret = m_cl_dst.Sync(CLMemSyncType::READ);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Sync end fail");
    }

EXIT:
    AURA_RETURN(m_ctx, ret);
}

std::string Filter2dCL::ToString() const
{
    return Filter2dImpl::ToString() + m_profiling_string;
}

std::vector<CLKernel> Filter2dCL::GetCLKernels(Context *ctx, ElemType elem_type, MI_S32 channel, MI_S32 ksize, BorderType border_type)
{
    std::vector<CLKernel> cl_kernels;
    std::string build_opt = GetCLBuildOptions(ctx, elem_type, border_type);

    std::string program_name[2], kernel_name[2];
    if (GetCLName(ctx, channel, ksize, program_name, kernel_name) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetCLName failed");
        return cl_kernels;
    }

    cl_kernels.push_back({ctx, program_name[0], kernel_name[0], "", build_opt});
    cl_kernels.push_back({ctx, program_name[1], kernel_name[1], "", build_opt});

    return cl_kernels;
}

} // namespace aura
