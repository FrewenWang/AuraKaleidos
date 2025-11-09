#include "pyrup_impl.hpp"
#include "aura/runtime/logger.h"

namespace aura
{

static Status GetCLBuildOptions(Context *ctx, BorderType border_type, ElemType elem_type, std::string &build_opt)
{
    const std::vector<std::string> tbl =
    {
                     "U8,     U16,    S16",    // elem type
        "Tp",        "uchar,  ushort, short",  // source
        "Kt",        "ushort, uint,   int",    // kernel
        "InterType", "uint,   float,  float",  // internal type
        "Q",         "18,     26,     26",     // shift
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
            AURA_ADD_ERROR_STRING(ctx, "unsupported border type");
            return Status::ERROR;
        }
    }

    cl_build_opt.AddOption("MAX_CONSTANT_SIZE", ctx->GetCLEngine()->GetCLRuntime()->GetCLMaxConstantSizeString(32));
    build_opt = cl_build_opt.ToString(elem_type);
    return Status::OK;
}

static Status GetCLName(DT_S32 channel, DT_S32 ksize, std::vector<std::string> &program_names, std::vector<std::string> &kernel_names)
{
    std::string program_postfix = std::to_string(ksize) + "x" + std::to_string(ksize) + "_c" + std::to_string(channel);
    std::string kernel_postfix = std::to_string(ksize) + "x" + std::to_string(ksize) + "C" + std::to_string(channel);

    program_names.emplace_back("aura_pyrup_main_" + program_postfix);
    program_names.emplace_back("aura_pyrup_remain_" + program_postfix);

    kernel_names.emplace_back("PyrUpMain" + kernel_postfix);
    kernel_names.emplace_back("PyrUpRemain" + kernel_postfix);

    return Status::OK;
}

static DT_VOID GetCLGlobalSize(DT_S32 height, DT_S32 width, DT_S32 ksize, cl::NDRange cl_range[2], DT_S32 &main_width)
{
    DT_S32 elem_counts = 6;
    DT_S32 border      = (ksize >> 1) << 1;

    DT_S32 main_size = (width - border) / elem_counts;
    DT_S32 main_rest = (width - border) % elem_counts;

    // main global size
    cl_range[0] = cl::NDRange(main_size, height);

    // remain global size
    cl_range[1] = cl::NDRange(main_rest + border, height);
    main_width = main_size * elem_counts;
}

PyrUpCL::PyrUpCL(Context *ctx, const OpTarget &target) : PyrUpImpl(ctx, target)
{}

Status PyrUpCL::SetArgs(const Array *src, Array *dst, DT_S32 ksize, DT_F32 sigma,
                        BorderType border_type)
{
    if (PyrUpImpl::SetArgs(src, dst, ksize, sigma, border_type) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "PyrUpImpl::SetArgs failed");
        return Status::ERROR;
    }

    if (src->GetSizes().m_width < 8)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "width is too small");
        return Status::ERROR;
    }

    return Status::OK;
}

Status PyrUpCL::Initialize()
{
    if (PyrUpImpl::Initialize() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "PyrUpImpl::Initialize() failed");
        return Status::ERROR;
    }

    // 1. init cl_mem
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

Status PyrUpCL::DeInitialize()
{
    m_cl_src.Release();
    m_cl_kmat.Release();
    m_cl_dst.Release();
    m_cl_kernels.clear();

    if (PyrUpImpl::DeInitialize() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "PyrUpImpl::DeInitialize() failed");
        return Status::ERROR;
    }

    return Status::OK;
}

Status PyrUpCL::Run()
{
    Status ret = Status::OK;
    const DT_S32 ksh = 3;

    DT_S32 istep = m_src->GetRowPitch() / ElemTypeSize(m_src->GetElemType());
    DT_S32 ostep = m_dst->GetRowPitch() / ElemTypeSize(m_dst->GetElemType());
    DT_S32 iheight = m_src->GetSizes().m_height;
    DT_S32 iwidth  = m_src->GetSizes().m_width;

    std::shared_ptr<CLRuntime> cl_rt = m_ctx->GetCLEngine()->GetCLRuntime();
    // 1. get center_area and global_size
    cl::NDRange cl_global_size[2];
    DT_S32 main_width = 0;

    GetCLGlobalSize(iheight, iwidth, ksh, cl_global_size, main_width);

    cl::Event cl_event[2];
    cl_int cl_ret   = CL_SUCCESS;
    Status ret_sync = Status::ERROR;

    // 2. opencl run
    cl_ret = m_cl_kernels[0].Run<cl::Buffer, DT_S32, DT_S32, cl::Buffer, DT_S32, DT_S32, DT_S32, cl::Buffer>(
                                m_cl_src.GetCLMemRef<cl::Buffer>(), istep, iheight,
                                m_cl_dst.GetCLMemRef<cl::Buffer>(), ostep,
                                (DT_S32)(cl_global_size[0].get()[1]),
                                (DT_S32)(cl_global_size[0].get()[0]),
                                m_cl_kmat.GetCLMemRef<cl::Buffer>(),
                                cl_global_size[0],
                                cl_rt->GetCLDefaultLocalSize(m_cl_kernels[0].GetMaxGroupSize(), cl_global_size[0]),
                                &(cl_event[0]));
    if (cl_ret != CL_SUCCESS)
    {
        AURA_ADD_ERROR_STRING(m_ctx, ("Run cl_kernel main failed : " + GetCLErrorInfo(cl_ret)).c_str());
        goto EXIT;
    }

    cl_ret |= m_cl_kernels[1].Run<cl::Buffer, DT_S32, DT_S32, DT_S32, cl::Buffer, DT_S32, DT_S32, DT_S32, DT_S32, cl::Buffer>(
                                 m_cl_src.GetCLMemRef<cl::Buffer>(), istep,
                                 iheight, iwidth,
                                 m_cl_dst.GetCLMemRef<cl::Buffer>(), ostep,
                                 (DT_S32)(cl_global_size[1].get()[1]),
                                 (DT_S32)(cl_global_size[1].get()[0]),
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
    if ((DT_TRUE == m_target.m_data.opencl.profiling) || (m_dst->GetArrayType() != ArrayType::CL_MEMORY))
    {
        cl_ret = cl_event[1].wait();
        if (cl_ret != CL_SUCCESS)
        {
            AURA_ADD_ERROR_STRING(m_ctx, ("Wait cl_kernel failed : " + GetCLErrorInfo(cl_ret)).c_str());
            goto EXIT;
        }

        if (DT_TRUE == m_target.m_data.opencl.profiling)
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

std::string PyrUpCL::ToString() const
{
    return PyrUpImpl::ToString() + m_profiling_string;
}

std::vector<CLKernel> PyrUpCL::GetCLKernels(Context *ctx, ElemType elem_type, DT_S32 channel, DT_S32 ksize, BorderType border_type)
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

    for (DT_S32 i = 0; i < static_cast<DT_S32>(program_names.size()); ++i)
    {
        cl_kernels.emplace_back(ctx, program_names.at(i), kernel_names.at(i), "", build_opt);
    }

    return cl_kernels;
}

} //namespace aura