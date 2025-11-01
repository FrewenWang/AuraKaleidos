#include "pyrdown_impl.hpp"
#include "aura/runtime/logger.h"

namespace aura
{

static Status GetCLBuildOptions(Context *ctx, BorderType border_type, ElemType elem_type, std::string &build_opt)
{
    const std::vector<std::string> tbl =
    {
                      "U8,     U16,    S16",   // elem type
        "Tp",         "uchar,  ushort, short", // source
        "Kt",         "ushort, uint,   int",   // kernel
        "InterType",  "uint,   float,  float", // internal type
        "Q",          "16,     28,     28",    // shift bit
        "ELEM_COUNTS","4,      2,      2",     // element counts
    };

    CLBuildOptions cl_build_opt(ctx, tbl);

    switch (border_type)
    {
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
            AURA_ADD_ERROR_STRING(ctx, "pyrdown border type error");
            break;
        }
    }

    cl_build_opt.AddOption("MAX_CONSTANT_SIZE", ctx->GetCLEngine()->GetCLRuntime()->GetCLMaxConstantSizeString(20));
    build_opt = cl_build_opt.ToString(elem_type);

    return Status::OK;
}

static Status GetCLName(MI_S32 channel, MI_S32 ksize, std::vector<std::string> &program_names, std::vector<std::string> &kernel_names)
{
    std::string program_postfix = std::to_string(ksize) + "x" + std::to_string(ksize) + "_c" + std::to_string(channel);
    std::string kernel_postfix = std::to_string(ksize) + "x" + std::to_string(ksize) + "C" + std::to_string(channel);

    program_names.emplace_back("aura_pyrdown_main_" + program_postfix);
    program_names.emplace_back("aura_pyrdown_remain_" + program_postfix);

    kernel_names.emplace_back("PyrDownMain" + kernel_postfix);
    kernel_names.emplace_back("PyrDownRemain" + kernel_postfix);

    return Status::OK;
}

static AURA_VOID GetCLGlobalSize(MI_S32 height, MI_S32 width, MI_S32 ksize, cl::NDRange cl_range[2], MI_S32 &main_width, ElemType elem_type)
{
    MI_S32 elem_counts = 0;
    MI_S32 res_counts  = 0;
    MI_S32 border      = (ksize >> 1) << 1;

    if (ElemType::U8 == elem_type)
    {
        elem_counts = 4;
        res_counts  = 5;
    }
    else
    {
        elem_counts = 2;
        res_counts  = 1;
    }

    MI_S32 main_size = (width - border - res_counts) / elem_counts;
    MI_S32 main_rest = (width - border - res_counts) % elem_counts;

    // main global size
    cl_range[0] = cl::NDRange(main_size, height);

    // remain global size
    cl_range[1] = cl::NDRange(main_rest + border + res_counts, height);
    main_width  = main_size * elem_counts;
}

PyrDownCL::PyrDownCL(Context *ctx, const OpTarget &target) : PyrDownImpl(ctx, target)
{}

Status PyrDownCL::SetArgs(const Array *src, Array *dst, MI_S32 ksize, MI_F32 sigma,
                          BorderType border_type)
{
    if (PyrDownImpl::SetArgs(src, dst, ksize, sigma, border_type) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "PyrDownImpl::SetArgs failed");
        return Status::ERROR;
    }

    if ((dst->GetSizes().m_width < 16 && ElemType::U8 == dst->GetElemType()) ||
        (dst->GetSizes().m_width < 8 && ((ElemType::U16 == dst->GetElemType()) || (ElemType::S16 == dst->GetElemType()))))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "width is too small");
        return Status::ERROR;
    }

    return Status::OK;
}

Status PyrDownCL::Initialize()
{
    if (PyrDownImpl::Initialize() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "PyrDownImpl::Initialize() failed");
        return Status::ERROR;
    }

    // 1. init cl mem
    m_cl_src = CLMem::FromArray(m_ctx, *m_src, CLMemParam(CL_MEM_READ_ONLY));
    if (!m_cl_src.IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "m_cl_src is invalid");
        return Status::ERROR;
    }

    m_cl_kmat = CLMem::FromArray(m_ctx, m_kmat, CLMemParam(CL_MEM_READ_ONLY));
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
    m_cl_kernels = GetCLKernels(m_ctx, m_src->GetElemType(), m_src->GetSizes().m_channel, m_ksize, m_border_type);

    if (CheckCLKernels(m_ctx, m_cl_kernels) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "CheckCLKernels failed");
        return Status::ERROR;
    }

    // 3. sync inputs
    if (m_cl_src.Sync(CLMemSyncType::WRITE) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Sync src failed");
        return Status::ERROR;
    }

    if (m_cl_kmat.Sync(CLMemSyncType::WRITE) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Sync kmat failed");
        return Status::ERROR;
    }


    return Status::OK;
}

Status PyrDownCL::DeInitialize()
{
    m_cl_src.Release();
    m_cl_kmat.Release();
    m_cl_dst.Release();
    m_cl_kernels.clear();

    if (PyrDownImpl::DeInitialize() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "PyrDownImpl::DeInitialize() failed");
        return Status::ERROR;
    }

    return Status::OK;
}

Status PyrDownCL::Run()
{
    Status ret = Status::ERROR;
    MI_S32 istep   = m_src->GetRowPitch() / ElemTypeSize(m_src->GetElemType());
    MI_S32 ostep   = m_dst->GetRowPitch() / ElemTypeSize(m_dst->GetElemType());
    MI_S32 iheight = m_src->GetSizes().m_height;
    MI_S32 iwidth  = m_src->GetSizes().m_width;
    MI_S32 oheight = m_dst->GetSizes().m_height;
    MI_S32 owidth  = m_dst->GetSizes().m_width;
    std::shared_ptr<CLRuntime> cl_rt = m_ctx->GetCLEngine()->GetCLRuntime();

    // 1. get center_area and global_size
    cl::NDRange cl_global_size[2];
    MI_S32 main_width = 0;

    GetCLGlobalSize(oheight, owidth, m_ksize, cl_global_size, main_width, m_src->GetElemType());

    cl::Event cl_event[2];
    cl_int cl_ret   = CL_SUCCESS;
    Status ret_sync = Status::ERROR;

    // 2. opencl run
    cl_ret = m_cl_kernels[0].Run<cl::Buffer, MI_S32,  MI_S32, cl::Buffer, MI_S32, MI_S32, MI_S32, cl::Buffer>(
                                m_cl_src.GetCLMemRef<cl::Buffer>(), istep, iheight,
                                m_cl_dst.GetCLMemRef<cl::Buffer>(), ostep,
                                (MI_S32)(cl_global_size[0].get()[1]),
                                (MI_S32)(cl_global_size[0].get()[0]),
                                m_cl_kmat.GetCLMemRef<cl::Buffer>(),
                                cl_global_size[0],
                                cl_rt->GetCLDefaultLocalSize(m_cl_kernels[0].GetMaxGroupSize(), cl_global_size[0]),
                                &(cl_event[0]));
    if (cl_ret != CL_SUCCESS)
    {
        AURA_ADD_ERROR_STRING(m_ctx, ("Run cl_kernel main failed : " + GetCLErrorInfo(cl_ret)).c_str());
        goto EXIT;
    }

    cl_ret |= m_cl_kernels[1].Run<cl::Buffer, MI_S32, MI_S32, MI_S32, cl::Buffer, MI_S32, MI_S32, MI_S32, MI_S32, cl::Buffer>(
                                 m_cl_src.GetCLMemRef<cl::Buffer>(), istep, iheight, iwidth,
                                 m_cl_dst.GetCLMemRef<cl::Buffer>(), ostep,
                                 (MI_S32)(cl_global_size[1].get()[1]),
                                 (MI_S32)(cl_global_size[1].get()[0]),
                                 main_width,
                                 m_cl_kmat.GetCLMemRef<cl::Buffer>(),
                                 cl_global_size[1],
                                 cl_rt->GetCLDefaultLocalSize(m_cl_kernels[1].GetMaxGroupSize(), cl_global_size[1]),
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

    ret = Status::OK;
EXIT:
    ret_sync = m_cl_dst.Sync(CLMemSyncType::READ);
    if (ret_sync != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Sync READ fail");
    }
    AURA_RETURN(m_ctx, (ret | ret_sync));
}

std::string PyrDownCL::ToString() const
{
    return PyrDownImpl::ToString() + m_profiling_string;
}

std::vector<CLKernel> PyrDownCL::GetCLKernels(Context *ctx, ElemType elem_type, MI_S32 channel, MI_S32 ksize, BorderType border_type)
{
    std::vector<CLKernel> cl_kernels;
    std::vector<std::string> program_names, kernel_names;
    std::string build_opt;

    if (GetCLBuildOptions(ctx, border_type, elem_type, build_opt) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetCLBuildOptions failed");
        return cl_kernels;
    }

    if (GetCLName(channel, ksize, program_names, kernel_names) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetCLName failed");
        return cl_kernels;
    }

    for (MI_S32 i = 0; i < static_cast<MI_S32>(program_names.size()); ++i)
    {
        cl_kernels.emplace_back(ctx, program_names.at(i), kernel_names.at(i), "", build_opt);
    }

    return cl_kernels;
}

} // namespace aura
