#include "gemm_impl.hpp"
#include "aura/runtime/opencl.h"
#include "aura/runtime/logger.h"

namespace aura
{

static std::string GetCLBuildOptions(Context *ctx, DT_S32 elem_counts)
{
    CLBuildOptions cl_build_opt(ctx);
    cl_build_opt.AddOption("ELEM_COUNTS", std::to_string(elem_counts));

    return cl_build_opt.ToString();
}

static DT_VOID GetCLSize(Context *ctx, std::vector<CLKernel> &cl_kernels, DT_S32 m, DT_S32 n, DT_S32 elem_counts,
                         cl::NDRange &cl_global_size, cl::NDRange &cl_local_size)
{
    std::shared_ptr<CLRuntime> cl_rt      = ctx->GetCLEngine()->GetCLRuntime();
    std::shared_ptr<cl::Device> cl_device = cl_rt->GetDevice();
    size_t preferred_group_size;
    cl_kernels[0].GetClKernel()->getWorkGroupInfo(*cl_device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, &preferred_group_size);
    DT_U32 max_group_size       = cl_kernels[0].GetMaxGroupSize();
    DT_S32 recommend_group_size = Min((DT_S32)max_group_size, 2 * (DT_S32)preferred_group_size);
    const DT_S32 local_size_x   = 16;
    const DT_S32 local_size_y   = recommend_group_size / local_size_x;

    DT_S32 global_size_x = (n + elem_counts - 1) / elem_counts;
    DT_S32 global_size_y = (m + elem_counts - 1) / elem_counts;

    cl_global_size = cl::NDRange(global_size_x, global_size_y);
    cl_local_size  = cl::NDRange(local_size_x,  local_size_y);
}

GemmMaliCL::GemmMaliCL(Context *ctx, const OpTarget &target) : GemmImpl(ctx, target), m_profiling_string()
{}

Status GemmMaliCL::Initialize()
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

    m_elem_counts = 4;

    // 2. init kernel
    m_cl_kernels = GemmMaliCL::GetCLKernels(m_ctx);
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

Status GemmMaliCL::DeInitialize()
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

Status GemmMaliCL::Run()
{
    DT_S32 m = m_src0->GetSizes().m_height;
    DT_S32 k = m_src0->GetSizes().m_width;
    DT_S32 n = m_src1->GetSizes().m_width;

    const DT_S32 istep0 = m_src0->GetRowPitch() / ElemTypeSize(m_src0->GetElemType());
    const DT_S32 istep1 = m_src1->GetRowPitch() / ElemTypeSize(m_src1->GetElemType());
    const DT_S32 ostep  =  m_dst->GetRowPitch() / ElemTypeSize( m_dst->GetElemType());

    cl::Event cl_event;
    cl_int cl_ret = CL_SUCCESS;
    Status ret    = Status::ERROR;

    // 1. get cl_global_size and cl_local_size
    cl::NDRange cl_global_size;
    cl::NDRange cl_local_size;
    GetCLSize(m_ctx, m_cl_kernels, m, n, m_elem_counts, cl_global_size, cl_local_size);

    // 2. opencl run
    cl_ret = m_cl_kernels[0].Run<cl::Buffer, DT_S32, cl::Buffer, DT_S32,
                                 cl::Buffer, DT_S32, DT_S32, DT_S32, DT_S32>(
                                 m_cl_src0.GetCLMemRef<cl::Buffer>(), istep0,
                                 m_cl_src1.GetCLMemRef<cl::Buffer>(), istep1,
                                 m_cl_dst.GetCLMemRef<cl::Buffer>(),  ostep,
                                 m, n, k,
                                 cl_global_size, cl_local_size,
                                 &cl_event);

    if (cl_ret != CL_SUCCESS)
    {
        AURA_ADD_ERROR_STRING(m_ctx, ("Run cl_kernel main failed : " + GetCLErrorInfo(cl_ret)).c_str());
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

    ret = m_cl_dst.Sync(CLMemSyncType::READ);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Sync end fail");
        goto EXIT;
    }

EXIT:
    AURA_RETURN(m_ctx, ret);
}

std::string GemmMaliCL::ToString() const
{
    return GemmImpl::ToString() + m_profiling_string;
}

std::vector<CLKernel> GemmMaliCL::GetCLKernels(Context *ctx)
{
    std::vector<CLKernel> cl_kernels;
    std::string build_opt = GetCLBuildOptions(ctx, 4);

    std::string program_name = "aura_gemm_mali";
    std::string kernel_name  = "Gemm";

    cl_kernels.push_back({ctx, program_name, kernel_name, "", build_opt});

    return cl_kernels;
}

} // namespace aura