#include "sobel_impl.hpp"
#include "aura/runtime/logger.h"

namespace aura
{

static Status GetCLName(MI_S32 channel, MI_S32 ksize, std::string program_name[2], std::string kernel_name[2])
{
    std::string program_postfix = std::to_string(ksize) + "x" + std::to_string(ksize) + "_c" + std::to_string(channel);
    std::string kernel_postfix  = std::to_string(ksize) + "x" + std::to_string(ksize) + "C"  + std::to_string(channel);

    program_name[0] = "aura_sobel_main_"   + program_postfix;
    program_name[1] = "aura_sobel_remain_" + program_postfix;

    kernel_name[0] = "SobelMain"   + kernel_postfix;
    kernel_name[1] = "SobelRemain" + kernel_postfix;

    return Status::OK;
}

static std::string GetCLBuildOptions(Context *ctx, MI_S32 dx, MI_S32 dy, MI_F32 scale, BorderType border_type,
                                     ElemType src_elem_type, ElemType dst_elem_type)
{
    CLBuildOptions cl_build_opt(ctx);

    cl_build_opt.AddOption("St",  CLTypeString(src_elem_type));
    cl_build_opt.AddOption("Dt",  CLTypeString(dst_elem_type));
    if (ElemType::U8 == src_elem_type)
    {
        cl_build_opt.AddOption("InterType",  "short");
    }
    else
    {
        cl_build_opt.AddOption("InterType",  "float");
    }

    cl_build_opt.AddOption("DX", std::to_string(dx));
    cl_build_opt.AddOption("DY", std::to_string(dy));

    if (!NearlyEqual(scale, 1.f))
    {
        cl_build_opt.AddOption("WITH_SCALE", "1");
    }

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

    return cl_build_opt.ToString();
}

static AURA_VOID GetCLGlobalSize(MI_S32 ksize, MI_S32 height, MI_S32 width, cl::NDRange range[2], MI_S32 &main_width)
{
    ksize              = (1 == ksize) ? 3 : ksize;

    MI_S32 elem_counts = (3 == ksize) ? 6 : ((5 == ksize) ? 4 : 2);
    MI_S32 border      = (ksize >> 1) << 1;
    MI_S32 main_size   = (width - border) / elem_counts;
    MI_S32 main_rest   = (width - border) % elem_counts;

    // main global size
    range[0] = cl::NDRange(main_size, height);

    // remain global size
    range[1] = cl::NDRange(main_rest + border, height);
    main_width = main_size * elem_counts;
}

SobelCL::SobelCL(Context *ctx, const OpTarget &target) : SobelImpl(ctx, target)
{}

Status SobelCL::SetArgs(const Array *src, Array *dst, MI_S32 dx, MI_S32 dy, MI_S32 ksize,
                        MI_F32 scale, BorderType border_type, const Scalar &border_value)
{
    if (SobelImpl::SetArgs(src, dst, dx, dy, ksize, scale, border_type, border_value) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "SobelImpl::SetArgs failed");
        return Status::ERROR;
    }

    MI_S32 pattern = AURA_MAKE_PATTERN(src->GetElemType(), dst->GetElemType());
    if (pattern != AURA_MAKE_PATTERN(ElemType::U8,  ElemType::S16) &&
        pattern != AURA_MAKE_PATTERN(ElemType::U8,  ElemType::F32) &&
        pattern != AURA_MAKE_PATTERN(ElemType::U16, ElemType::U16) &&
        pattern != AURA_MAKE_PATTERN(ElemType::S16, ElemType::S16) &&
        pattern != AURA_MAKE_PATTERN(ElemType::F16, ElemType::F16) &&
        pattern != AURA_MAKE_PATTERN(ElemType::F32, ElemType::F32))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src/dst elemtype is not supported");
        return Status::ERROR;
    }

    if (m_ksize <= 0)
    {
        m_dx = m_dx > 0 ? m_dx : 3;
        m_dy = m_dy > 0 ? m_dy : 3;
        m_ksize = 3;
    }

    if ((m_dx > 0) && (m_dy > 0) && (1 == m_ksize))
    {
        m_ksize = 3;
    }

    if (m_ksize > 5)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "ksize only support 1/3/5");
        return Status::ERROR;
    }

    MI_S32 ch = src->GetSizes().m_channel;

    if (ch != 1 && ch != 2 && ch != 3)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "channel only support 1/2/3");
        return Status::ERROR;
    }

    return Status::OK;
}

Status SobelCL::Initialize()
{
    if (SobelImpl::Initialize() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "SobelImpl::Initialize failed");
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
    m_cl_kernels = SobelCL::GetCLKernels(m_ctx, m_dx, m_dy, m_ksize, m_scale, m_border_type,
                                         m_dst->GetSizes().m_channel, m_src->GetElemType(), m_dst->GetElemType());

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

Status SobelCL::DeInitialize()
{
    m_cl_src.Release();
    m_cl_dst.Release();

    for (auto &cl_kernel : m_cl_kernels)
    {
        cl_kernel.DeInitialize();
    }

    if (SobelImpl::DeInitialize() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "SobelImpl::DeInitialize failed");
        return Status::ERROR;
    }

    return Status::OK;
}

Status SobelCL::Run()
{
    MI_S32 istep  = m_src->GetRowPitch() / ElemTypeSize(m_src->GetElemType());
    MI_S32 ostep  = m_dst->GetRowPitch() / ElemTypeSize(m_dst->GetElemType());
    MI_S32 height = m_dst->GetSizes().m_height;
    MI_S32 width  = m_dst->GetSizes().m_width;

    std::shared_ptr<CLRuntime> cl_rt = m_ctx->GetCLEngine()->GetCLRuntime();
    CLScalar cl_border_value         = clScalar(m_border_value);

    // 1. get center_area and cl_global_size
    cl::NDRange cl_global_size[2];
    MI_S32 main_width = 0;

    GetCLGlobalSize(m_ksize, height, width, cl_global_size, main_width);

    // 2. opencl run
    cl::Event cl_event[2];
    cl_int cl_ret = CL_SUCCESS;
    Status ret    = Status::ERROR;

    cl_ret = m_cl_kernels[0].Run<cl::Buffer, MI_S32, cl::Buffer, MI_S32, MI_S32, MI_F32, MI_S32, MI_S32, CLScalar>(
                                 m_cl_src.GetCLMemRef<cl::Buffer>(), istep,
                                 m_cl_dst.GetCLMemRef<cl::Buffer>(), ostep,
                                 height, m_scale,
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

    cl_ret |= m_cl_kernels[1].Run<cl::Buffer, MI_S32, cl::Buffer, MI_S32, MI_S32, MI_S32, MI_F32, MI_S32, MI_S32, MI_S32, CLScalar>(
                                  m_cl_src.GetCLMemRef<cl::Buffer>(), istep,
                                  m_cl_dst.GetCLMemRef<cl::Buffer>(), ostep,
                                  height, width, m_scale,
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

std::string SobelCL::ToString() const
{
    return SobelImpl::ToString() + m_profiling_string;
}

std::vector<CLKernel> SobelCL::GetCLKernels(Context *ctx, MI_S32 dx, MI_S32 dy, MI_S32 ksize, MI_F32 scale, BorderType border_type,
                                            MI_S32 channel, ElemType src_elem_type, ElemType dst_elem_type)
{
    std::vector<CLKernel> cl_kernels;
    std::string build_opt = GetCLBuildOptions(ctx, dx, dy, scale, border_type, src_elem_type, dst_elem_type);

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

} // namespace aura
