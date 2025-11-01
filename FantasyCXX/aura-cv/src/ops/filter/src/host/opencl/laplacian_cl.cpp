#include "laplacian_impl.hpp"
#include "aura/runtime/logger.h"

namespace aura
{

static Status GetCLName(Context *ctx,  MI_S32 channel, MI_S32 ksize, std::string program_name[2], std::string kernel_name[2])
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

    program_name[0] = "aura_laplacian_main_" + program_postfix;
    program_name[1] = "aura_laplacian_remain_" + program_postfix;

    kernel_name[0] = "LaplacianMain" + kernel_postfix;
    kernel_name[1] = "LaplacianRemain" + kernel_postfix;

    if ((1 == ksize) && (3 == channel))
    {
        program_name[0] += "_" + GpuTypeToString(gpu_type);
    }

    if ((3 == ksize) && (3 == channel))
    {
        program_name[0] += "_" + GpuTypeToString(gpu_type);
    }

    if ((5 == ksize) && (3 == channel))
    {
        program_name[0] += "_" + GpuTypeToString(gpu_type);
    }

    return Status::OK;
}

static std::string GetCLBuildOptions(Context *ctx, ElemType elem_type, BorderType border_type)
{
    const std::vector<std::string> tbl =
    {
                     "U8,    U16,    S16,   F32,   F16",  // elem type
        "St",        "uchar, ushort, short, float, half", // source
        "Dt",        "short, ushort, short, float, half", // destination
        "InterType", "short, float,  float, float, float" // internal type
    };

    CLBuildOptions cl_build_opt(ctx, tbl);

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

    return cl_build_opt.ToString(elem_type);
}

static AURA_VOID GetCLGlobalSize(GpuType gpu_type, MI_S32 ksize, MI_S32 channel, MI_S32 height, MI_S32 width,
                               cl::NDRange range[2], MI_S32 &main_width)
{
    ksize = ((1 == ksize) ? 3 : ksize);

    MI_S32 elem_counts = 0;
    MI_S32 main_size   = 0;
    MI_S32 main_rest   = 0;
    MI_S32 border      = (ksize >> 1) << 1;

    if ((GpuType::MALI == gpu_type) && (5 == ksize) && (3 == channel))
    {
        elem_counts = 1;
        main_size   = width - border - elem_counts;
        main_rest   = elem_counts;
        range[0]    = cl::NDRange(main_size, height);
    }
    else if ((GpuType::MALI == gpu_type) && ((1 == ksize) || (3 == ksize)) && (3 == channel))
    {
        elem_counts = 3;
        main_size   = (width - border) / elem_counts - 1;
        main_rest   = (width - border) % elem_counts + 3;
        range[0]    = cl::NDRange(main_size, height);
    }
    else
    {
        ksize       = (1 == ksize) ? 3 : ksize;
        elem_counts = (3 == ksize) ? 6 : ((5 == ksize) ? 4 : 2);
        main_size   = (width - border) / elem_counts;
        main_rest   = (width - border) % elem_counts;
        range[0]    = cl::NDRange(main_size, height);
    }

    // remain global size
    range[1] = cl::NDRange(main_rest + border, height);
    main_width = main_size * elem_counts;
}

LaplacianCL::LaplacianCL(Context *ctx, const OpTarget &target) : LaplacianImpl(ctx, target)
{}

Status LaplacianCL::SetArgs(const Array *src, Array *dst, MI_S32 ksize,
                            BorderType border_type, const Scalar &border_value)
{
    if (LaplacianImpl::SetArgs(src, dst, ksize, border_type, border_value) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "LaplacianImpl::SetArgs failed");
        return Status::ERROR;
    }

    MI_S32 pattern = AURA_MAKE_PATTERN(src->GetElemType(), dst->GetElemType());
    if (pattern != AURA_MAKE_PATTERN(ElemType::U8,  ElemType::S16) &&
        pattern != AURA_MAKE_PATTERN(ElemType::U16, ElemType::U16) &&
        pattern != AURA_MAKE_PATTERN(ElemType::S16, ElemType::S16) &&
        pattern != AURA_MAKE_PATTERN(ElemType::F16, ElemType::F16) &&
        pattern != AURA_MAKE_PATTERN(ElemType::F32, ElemType::F32))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src/dst elemtype is not supported");
        return Status::ERROR;
    }

    if (ksize != 1 && ksize != 3 && ksize != 5 && ksize != 7)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "ksize only support 1/3/5/7");
        return Status::ERROR;
    }

    MI_S32 ch = src->GetSizes().m_channel;

    if (ch != 1 && ch != 2 && ch != 3)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "channel only support 1/2/3");
        return Status::ERROR;
    }

    if (7 == ksize && ch != 1)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "opencl target cannot support channel > 1 when ksize = 7");
        return Status::ERROR;
    }

    return Status::OK;
}

Status LaplacianCL::Initialize()
{
    if (LaplacianImpl::Initialize() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "LaplacianImpl::Initialize() failed");
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
    m_cl_kernels = LaplacianCL::GetCLKernels(m_ctx, m_src->GetElemType(), m_dst->GetSizes().m_channel, m_ksize, m_border_type);

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

    return Status::OK;
}

Status LaplacianCL::DeInitialize()
{
    m_cl_src.Release();
    m_cl_dst.Release();

    for (auto &cl_kernel : m_cl_kernels)
    {
        cl_kernel.DeInitialize();
    }

    if (LaplacianImpl::DeInitialize() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "LaplacianImpl::DeInitialize() failed");
        return Status::ERROR;
    }

    return Status::OK;
}

Status LaplacianCL::Run()
{
    MI_S32 istep   = m_src->GetRowPitch() / ElemTypeSize(m_src->GetElemType());
    MI_S32 ostep   = m_dst->GetRowPitch() / ElemTypeSize(m_dst->GetElemType());
    MI_S32 height  = m_dst->GetSizes().m_height;
    MI_S32 width   = m_dst->GetSizes().m_width;
    MI_S32 channel = m_dst->GetSizes().m_channel;

    std::shared_ptr<CLRuntime> cl_rt = m_ctx->GetCLEngine()->GetCLRuntime();
    CLScalar cl_border_value         = clScalar(m_border_value);

    // 1. get center_area and cl_global_size
    cl::NDRange cl_global_size[2];
    MI_S32 main_width = 0;

    GetCLGlobalSize(cl_rt->GetGpuInfo().m_type, m_ksize, channel, height, width, cl_global_size, main_width);

    // 2. opencl run
    cl::Event cl_event[2];
    cl_int cl_ret = CL_SUCCESS;
    Status ret    = Status::ERROR;

    cl_ret = m_cl_kernels[0].Run<cl::Buffer, MI_S32, cl::Buffer, MI_S32, MI_S32, MI_S32, MI_S32, CLScalar>(
                                 m_cl_src.GetCLMemRef<cl::Buffer>(), istep,
                                 m_cl_dst.GetCLMemRef<cl::Buffer>(), ostep,
                                 height,
                                 (MI_S32)(cl_global_size[0].get()[1]),
                                 (MI_S32)(cl_global_size[0].get()[0]),
                                 cl_border_value,
                                 cl_global_size[0], cl_rt->GetCLDefaultLocalSize(m_cl_kernels[0].GetMaxGroupSize(), cl_global_size[0]),
                                 &(cl_event[0]));
    if (cl_ret != CL_SUCCESS)
    {
        AURA_ADD_ERROR_STRING(m_ctx, ("Run cl_kernel main failed : " + GetCLErrorInfo(cl_ret)).c_str());
        goto EXIT;
    }

    cl_ret |= m_cl_kernels[1].Run<cl::Buffer, MI_S32, cl::Buffer, MI_S32, MI_S32, MI_S32, MI_S32, MI_S32, MI_S32, CLScalar>(
                                  m_cl_src.GetCLMemRef<cl::Buffer>(), istep,
                                  m_cl_dst.GetCLMemRef<cl::Buffer>(), ostep,
                                  height, width,
                                  (MI_S32)(cl_global_size[1].get()[1]),
                                  (MI_S32)(cl_global_size[1].get()[0]),
                                  main_width, cl_border_value,
                                  cl_global_size[1], cl_rt->GetCLDefaultLocalSize(m_cl_kernels[1].GetMaxGroupSize(), cl_global_size[1]),
                                  &(cl_event[1]), {cl_event[0]});
    if (cl_ret != CL_SUCCESS)
    {
        AURA_ADD_ERROR_STRING(m_ctx, ("Run cl_kernel remain failed : " + GetCLErrorInfo(cl_ret)).c_str());
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

std::string LaplacianCL::ToString() const
{
    return LaplacianImpl::ToString() + m_profiling_string;
}

std::vector<CLKernel> LaplacianCL::GetCLKernels(Context *ctx, ElemType elem_type, MI_S32 channel, MI_S32 ksize, BorderType border_type)
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
