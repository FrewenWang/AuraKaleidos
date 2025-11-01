#include "bilateral_impl.hpp"
#include "aura/runtime/logger.h"

namespace aura
{

static Status GetCLName(MI_S32 channel, MI_S32 ksize, std::string program_name[2], std::string kernel_name[2])
{
    std::string program_postfix = std::to_string(ksize) + "x" + std::to_string(ksize) + "_c" + std::to_string(channel);
    std::string kernel_postfix  = std::to_string(ksize) + "x" + std::to_string(ksize) + "C"  + std::to_string(channel);

    program_name[0] = "aura_bilateral_main_"   + program_postfix;
    program_name[1] = "aura_bilateral_remain_" + program_postfix;

    kernel_name[0] = "BilateralMain"   + kernel_postfix;
    kernel_name[1] = "BilateralRemain" + kernel_postfix;

    return Status::OK;
}

static std::string GetCLBuildOptions(Context *ctx, ElemType elem_type, BorderType border_type, MI_S32 valid_num)
{
    const std::vector<std::string> tbl =
    {
               "U8,     F16,   F32",    // elem type
        "St",  "uchar,  half,  float",  // source
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

    cl_build_opt.AddOption("MAX_CONSTANT_SIZE", ctx->GetCLEngine()->GetCLRuntime()->GetCLMaxConstantSizeString(valid_num * 4));
    return cl_build_opt.ToString(elem_type);
}

static AURA_VOID GetCLGlobalSize(MI_S32 height, MI_S32 width, MI_S32 ksize, cl::NDRange cl_range[2], MI_S32 &main_width)
{
    MI_S32 elem_counts = 4;
    MI_S32 border      = (ksize >> 1) << 1;

    MI_S32 main_size = (width - border) / elem_counts;
    MI_S32 main_rest = (width - border) % elem_counts;

    // main global size
    cl_range[0] = cl::NDRange(main_size, height);

    // remain global size
    cl_range[1] = cl::NDRange(main_rest + border, height);
    main_width = main_size * elem_counts;
}

BilateralCL::BilateralCL(Context *ctx, const OpTarget &target) : BilateralImpl(ctx, target), m_profiling_string()
{}

Status BilateralCL::SetArgs(const Array *src, Array *dst, MI_F32 sigma_color, MI_F32 sigma_space, MI_S32 ksize,
                            BorderType border_type, const Scalar &border_value)
{
    if (BilateralImpl::SetArgs(src, dst, sigma_color, sigma_space, ksize, border_type, border_value) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "BilateralImpl::Initialize failed");
        return Status::ERROR;
    }

    if ((src->GetArrayType() != ArrayType::MAT) || (dst->GetArrayType() != ArrayType::MAT))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src dst must be mat type");
        return Status::ERROR;
    }

    if (ksize != 3)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "ksize only support 3");
        return Status::ERROR;
    }

    MI_S32 ch = src->GetSizes().m_channel;
    if (ch != 1 && ch != 3)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "channel only support 1/3");
        return Status::ERROR;
    }

    return Status::OK;
}

Status BilateralCL::Initialize()
{
    if (BilateralImpl::Initialize() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "BilateralImpl::Initialize() failed");
        return Status::ERROR;
    }

    // 1. init cl_mem
    m_cl_src = CLMem::FromArray(m_ctx, *m_src, CLMemParam(CL_MEM_READ_ONLY));
    if (!m_cl_src.IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "m_cl_src is invalid");
        return Status::ERROR;
    }

    m_cl_space = CLMem::FromArray(m_ctx, m_space_weight, CLMemParam(CL_MEM_READ_ONLY));
    if (!m_cl_space.IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "m_cl_space is invalid");
        return Status::ERROR;
    }

    m_cl_color = CLMem::FromArray(m_ctx, m_color_weight, CLMemParam(CL_MEM_READ_ONLY, CL_R));
    if (!m_cl_color.IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "m_cl_color is invalid");
        return Status::ERROR;
    }

    m_cl_dst = CLMem::FromArray(m_ctx, *m_dst, CLMemParam(CL_MEM_WRITE_ONLY));
    if (!m_cl_dst.IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "m_cl_dst is invalid");
        return Status::ERROR;
    }

    // 2. init kernel
    m_cl_kernels = BilateralCL::GetCLKernels(m_ctx, m_dst->GetElemType(), m_dst->GetSizes().m_channel, m_ksize, m_border_type, m_valid_num);

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

    if (m_cl_space.Sync(CLMemSyncType::WRITE) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "m_cl_space Sync start failed");
        return Status::ERROR;
    }

    if (m_cl_color.Sync(CLMemSyncType::WRITE) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "m_cl_color Sync start failed");
        return Status::ERROR;
    }

    return Status::OK;
}

Status BilateralCL::DeInitialize()
{
    m_cl_src.Release();
    m_cl_space.Release();
    m_cl_color.Release();
    m_cl_dst.Release();

    for (auto &cl_kernel : m_cl_kernels)
    {
        cl_kernel.DeInitialize();
    }

    if (BilateralImpl::DeInitialize() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "BilateralImpl::DeInitialize() failed");
        return Status::ERROR;
    }

    return Status::OK;
}

Status BilateralCL::Run()
{
    MI_S32 istep  = m_src->GetRowPitch() / ElemTypeSize(m_src->GetElemType());
    MI_S32 ostep  = m_dst->GetRowPitch() / ElemTypeSize(m_dst->GetElemType());
    MI_S32 height = m_dst->GetSizes().m_height;
    MI_S32 width  = m_dst->GetSizes().m_width;

    std::shared_ptr<CLRuntime> cl_rt = m_ctx->GetCLEngine()->GetCLRuntime();
    CLScalar cl_border_value = clScalar(m_border_value);

    // 1. get center_area and cl_global_size
    cl::NDRange cl_global_size[2];
    MI_S32 main_width = 0;

    GetCLGlobalSize(height, width, m_ksize, cl_global_size, main_width);

    // 2. opencl run
    cl::Event cl_event[2];
    cl_int cl_ret = CL_SUCCESS;
    Status ret    = Status::ERROR;

    cl_ret = m_cl_kernels[0].Run<cl::Buffer, MI_S32, cl::Buffer, MI_S32, MI_S32, MI_S32, MI_S32, MI_S32, cl::Buffer,
                                 cl::Iaura2D, MI_S32, MI_F32, CLScalar>(
                                 m_cl_src.GetCLMemRef<cl::Buffer>(), istep,
                                 m_cl_dst.GetCLMemRef<cl::Buffer>(), ostep,
                                 width, height,
                                 (MI_S32)(cl_global_size[0].get()[0]),
                                 (MI_S32)(cl_global_size[0].get()[1]),
                                 m_cl_space.GetCLMemRef<cl::Buffer>(),
                                 m_cl_color.GetCLMemRef<cl::Iaura2D>(), m_color_weight.GetSizes().m_width,
                                 m_scale_index, cl_border_value,
                                 cl_global_size[0], cl_rt->GetCLDefaultLocalSize(m_cl_kernels[0].GetMaxGroupSize(), cl_global_size[0]),
                                 &(cl_event[0]));
    if (cl_ret != CL_SUCCESS)
    {
        AURA_ADD_ERROR_STRING(m_ctx, ("Run cl_kernel main failed : " + GetCLErrorInfo(cl_ret)).c_str());
        goto EXIT;
    }

    cl_ret |= m_cl_kernels[1].Run<cl::Buffer, MI_S32, cl::Buffer, MI_S32, MI_S32, MI_S32, MI_S32, MI_S32, cl::Buffer,
                                  cl::Iaura2D, MI_S32, MI_F32, MI_S32, CLScalar>(
                                  m_cl_src.GetCLMemRef<cl::Buffer>(), istep,
                                  m_cl_dst.GetCLMemRef<cl::Buffer>(), ostep,
                                  width, height,
                                  (MI_S32)(cl_global_size[1].get()[0]),
                                  (MI_S32)(cl_global_size[1].get()[1]),
                                  m_cl_space.GetCLMemRef<cl::Buffer>(),
                                  m_cl_color.GetCLMemRef<cl::Iaura2D>(), m_color_weight.GetSizes().m_width,
                                  m_scale_index, main_width, cl_border_value,
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

std::string BilateralCL::ToString() const
{
    return BilateralImpl::ToString() + m_profiling_string;
}

std::vector<CLKernel> BilateralCL::GetCLKernels(Context *ctx, ElemType elem_type, MI_S32 channel, MI_S32 ksize, BorderType border_type, MI_S32 valid_num)
{
    std::vector<CLKernel> cl_kernels;
    std::string build_opt = GetCLBuildOptions(ctx, elem_type, border_type, valid_num);

    std::string program_name[2], kernel_name[2];
    if (GetCLName(channel, ksize, program_name, kernel_name) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetCLName failed");
        return cl_kernels;
    }

    cl_kernels.push_back({ctx, program_name[0], kernel_name[0], "", build_opt});
    cl_kernels.push_back({ctx, program_name[1], kernel_name[1], "", build_opt});

    return cl_kernels;
}

Sizes BilateralCL::GetColorMatStride(Sizes3 color_size)
{
    Sizes color_stride;

    std::shared_ptr<CLRuntime> cl_rt = m_ctx->GetCLEngine()->GetCLRuntime();

    color_stride.m_width = AURA_ALIGN(color_size.m_width * sizeof(MI_F32), cl_rt->GetCLLengthAlignSize());

    return color_stride;
}

} // namespace aura
