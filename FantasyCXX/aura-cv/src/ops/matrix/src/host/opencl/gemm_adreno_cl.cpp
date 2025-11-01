#include "gemm_impl.hpp"
#include "aura/runtime/opencl.h"
#include "aura/runtime/logger.h"

namespace aura
{

static std::string GetCLBuildOptions(Context *ctx, MI_S32 elem_counts, MI_S32 load_size)
{
    CLBuildOptions cl_build_opt(ctx);
    cl_build_opt.AddOption("ELEM_COUNTS",      std::to_string(elem_counts));
    cl_build_opt.AddOption("HALF_ELEM_COUNTS", std::to_string(elem_counts / 2));
    cl_build_opt.AddOption("LOAD_SIZE",        std::to_string(load_size));

    return cl_build_opt.ToString();
}

static AURA_VOID GetCLSize(MI_S32 m, MI_S32 n, MI_S32 bm, MI_S32 bn,
                         MI_S32 elem_counts, cl::NDRange &cl_global_size, cl::NDRange &cl_local_size)
{
    const MI_S32 local_size_x  = bn / elem_counts;
    const MI_S32 local_size_y  = bm / elem_counts;
    const MI_S32 group_size_x  = (n + bn - 1) / bn;
    const MI_S32 group_size_y  = (m + bm - 1) / bm;
    const MI_S32 global_size_x = group_size_x * local_size_x;
    const MI_S32 global_size_y = group_size_y * local_size_y;

    cl_global_size = cl::NDRange(global_size_x, global_size_y);
    cl_local_size  = cl::NDRange(local_size_x,  local_size_y);
}

GemmAdrenoCL::GemmAdrenoCL(Context *ctx, const OpTarget &target) : GemmImpl(ctx, target), m_profiling_string()
{}

Status GemmAdrenoCL::Initialize()
{
    if (GemmImpl::Initialize() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "GemmImpl::Initialize() failed");
        return Status::ERROR;
    }

    std::shared_ptr<CLRuntime> cl_rt = m_ctx->GetCLEngine()->GetCLRuntime();

    // 1. init cl_mem
    m_cl_src0 = CLMem::FromArray(m_ctx, *m_src0, CLMemParam(CL_MEM_READ_ONLY));
    if (!m_cl_src0.IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "m_cl_src is invalid");
        return Status::ERROR;
    }

    m_cl_src1 = CLMem::FromArray(m_ctx, *m_src1, CLMemParam(CL_MEM_READ_ONLY));
    if (!m_cl_src1.IsValid())
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
    m_elem_counts    = 8;
    m_local_size_x   = m_bn / m_elem_counts;
    m_local_size_y   = m_bm / m_elem_counts;
    MI_S32 load_size = m_bm * m_bk / (2 * m_local_size_x * m_local_size_y);

    m_cl_kernels = GemmAdrenoCL::GetCLKernels(m_ctx, m_elem_counts, load_size);
    if (CheckCLKernels(m_ctx, m_cl_kernels) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "CheckCLKernels failed");
        return Status::ERROR;
    }

    // 3. sync start
    if (m_cl_src0.Sync(CLMemSyncType::WRITE) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "m_cl_src0 sync start failed");
        return Status::ERROR;
    }

    if (m_cl_src1.Sync(CLMemSyncType::WRITE) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "m_cl_src1 sync start failed");
        return Status::ERROR;
    }

    return Status::OK;
}

Status GemmAdrenoCL::DeInitialize()
{
    m_cl_src0.Release();
    m_cl_src1.Release();
    m_cl_dst.Release();

    m_cl_kernels[0].DeInitialize();

    if (GemmImpl::DeInitialize() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "GemmImpl::DeInitialize() failed");
        return Status::ERROR;
    }

    return Status::OK;
}

Status GemmAdrenoCL::Run()
{
    MI_S32 m = m_src0->GetSizes().m_height;
    MI_S32 k = m_src0->GetSizes().m_width;
    MI_S32 n = m_src1->GetSizes().m_width;

    const MI_S32 istep0 = m_src0->GetRowPitch() / ElemTypeSize(m_src0->GetElemType());
    const MI_S32 istep1 = m_src1->GetRowPitch() / ElemTypeSize(m_src1->GetElemType());
    const MI_S32 ostep  =  m_dst->GetRowPitch() / ElemTypeSize( m_dst->GetElemType());

    cl::Event cl_event;
    cl_int cl_ret = CL_SUCCESS;
    Status ret    = Status::ERROR;

    // 1. get cl_global_size and cl_local_size
    cl::NDRange cl_global_size;
    cl::NDRange cl_local_size;

    GetCLSize(m, n, m_bm, m_bn, m_elem_counts, cl_global_size, cl_local_size);

    // 2. opencl run
    cl_ret = m_cl_kernels[0].Run<cl::Buffer, MI_S32, cl::Buffer, MI_S32, cl::Buffer, MI_S32, cl::LocalSpaceArg,
                                 cl::LocalSpaceArg, MI_S32, MI_S32, MI_S32, MI_S32, MI_S32, MI_S32>(
                                 m_cl_src0.GetCLMemRef<cl::Buffer>(), istep0,
                                 m_cl_src1.GetCLMemRef<cl::Buffer>(), istep1,
                                 m_cl_dst.GetCLMemRef<cl::Buffer>(),  ostep,
                                 cl::Local(m_bm * m_bk * sizeof(m_dst->GetElemType())),
                                 cl::Local(m_bk * m_bn * sizeof(m_dst->GetElemType())),
                                 m, n, k, m_bm, m_bn, m_bk,
                                 cl_global_size, cl_local_size,
                                 &cl_event);

    if (cl_ret != CL_SUCCESS)
    {
        AURA_ADD_ERROR_STRING(m_ctx, ("Run cl_kernel main failed : " + GetCLErrorInfo(cl_ret)).c_str());
        goto EXIT;
    }

    // 3. cl wait
    if ((MI_TRUE == m_target.m_data.opencl.profiling) || (m_dst->GetArrayType() != ArrayType::CL_MEMORY))
    {
        cl_ret = cl_event.wait();
        if (cl_ret != CL_SUCCESS)
        {
            AURA_ADD_ERROR_STRING(m_ctx, ("Wait cl_kernel failed : " + GetCLErrorInfo(cl_ret)).c_str());
            goto EXIT;
        }

        if (MI_TRUE == m_target.m_data.opencl.profiling)
        {
            m_profiling_string = " " + GetCLProfilingInfo(m_cl_kernels[0].GetKernelName(), cl_event);
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

std::string GemmAdrenoCL::ToString() const
{
    return GemmImpl::ToString() + m_profiling_string;
}

std::vector<CLKernel> GemmAdrenoCL::GetCLKernels(Context *ctx, MI_S32 elem_counts, MI_S32 load_size)
{
    std::vector<CLKernel> cl_kernels;
    std::string build_opt = GetCLBuildOptions(ctx, elem_counts, load_size);

    std::string program_name = "aura_gemm_adreno";
    std::string kernel_name  = "Gemm";

    cl_kernels.push_back({ctx, program_name, kernel_name, "", build_opt});

    return cl_kernels;
}

} // namespace aura