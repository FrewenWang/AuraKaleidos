#include "mul_spectrums_impl.hpp"
#include "aura/runtime/opencl.h"
#include "aura/runtime/logger.h"

namespace aura
{

const static MI_S32 g_adreno_elem_counts = 8;
const static MI_S32 g_mail_elem_counts   = 2;

static std::string GetCLBuildOptions(Context *ctx, ElemType elem_type, MI_BOOL conj_src1)
{
    CLBuildOptions cl_build_opt(ctx);

    GpuType gpu_type = ctx->GetCLEngine()->GetCLRuntime()->GetGpuInfo().m_type;
    MI_S32 elem_counts = (GpuType::ADRENO == gpu_type) ? g_adreno_elem_counts : g_mail_elem_counts;

    cl_build_opt.AddOption("Tp",          CLTypeString(elem_type));
    cl_build_opt.AddOption("CONJ",        conj_src1 ? "1" : "0");
    cl_build_opt.AddOption("ELEM_COUNTS", std::to_string(elem_counts * 2));

    return cl_build_opt.ToString(elem_type);
}

MulSpectrumsCL::MulSpectrumsCL(Context *ctx, const OpTarget &target) : MulSpectrumsImpl(ctx, target), m_profiling_string()
{}

Status MulSpectrumsCL::SetArgs(const Array *src0, const Array *src1, Array *dst, MI_BOOL conj_src1)
{
    if (MulSpectrumsImpl::SetArgs(src0, src1, dst, conj_src1) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "MulSpectrumsImpl::SetArgs failed");
        return Status::ERROR;
    }

    return Status::OK;
}

Status MulSpectrumsCL::Initialize()
{
    if (MulSpectrumsImpl::Initialize() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "MulSpectrumsImpl::Initialize() failed");
        return Status::ERROR;
    }

    // 1. init cl_mem
    m_cl_src0 = CLMem::FromArray(m_ctx, *m_src0, CLMemParam(CL_MEM_READ_ONLY));
    if (!m_cl_src0.IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "m_cl_src0 is invalid");
        return Status::ERROR;
    }

    m_cl_src1 = CLMem::FromArray(m_ctx, *m_src1, CLMemParam(CL_MEM_READ_ONLY));
    if (!m_cl_src1.IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "m_cl_src1 is invalid");
        return Status::ERROR;
    }

    m_cl_dst = CLMem::FromArray(m_ctx, *m_dst, CLMemParam(CL_MEM_WRITE_ONLY));
    if (!m_cl_dst.IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "m_cl_dst is invalid");
        return Status::ERROR;
    }

    // 2. init kernel
    m_cl_kernels = MulSpectrumsCL::GetCLKernels(m_ctx, m_dst->GetElemType(), m_conj_src1);

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

Status MulSpectrumsCL::DeInitialize()
{
    m_cl_src0.Release();
    m_cl_src1.Release();
    m_cl_dst.Release();

    m_cl_kernels[0].DeInitialize();

    if (MulSpectrumsImpl::DeInitialize() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "MulSpectrumsImpl::DeInitialize() failed");
        return Status::ERROR;
    }

    return Status::OK;
}

Status MulSpectrumsCL::Run()
{
    MI_S32 istep0 = m_src0->GetRowPitch() / ElemTypeSize(m_src0->GetElemType());
    MI_S32 istep1 = m_src1->GetRowPitch() / ElemTypeSize(m_src1->GetElemType());
    MI_S32 ostep  = m_dst->GetRowPitch() / ElemTypeSize(m_dst->GetElemType());
    MI_S32 height = m_dst->GetSizes().m_height;
    MI_S32 width  = m_dst->GetSizes().m_width;

    std::shared_ptr<CLRuntime> cl_rt = m_ctx->GetCLEngine()->GetCLRuntime();

    // 1. get center_area and cl_global_size
    GpuType gpu_type = m_ctx->GetCLEngine()->GetCLRuntime()->GetGpuInfo().m_type;
    MI_S32 elem_counts = (GpuType::ADRENO == gpu_type) ? g_adreno_elem_counts : g_mail_elem_counts;

    cl::NDRange cl_global_size = cl::NDRange((width + elem_counts - 1) / elem_counts, height);
    cl::Event cl_event;
    cl_int cl_ret = CL_SUCCESS;
    Status ret    = Status::ERROR;

    // 2. opencl run
    cl_ret = m_cl_kernels[0].Run<cl::Buffer, MI_S32, cl::Buffer, MI_S32, cl::Buffer, MI_S32, MI_S32, MI_S32, MI_S32>(
                                 m_cl_src0.GetCLMemRef<cl::Buffer>(), istep0,
                                 m_cl_src1.GetCLMemRef<cl::Buffer>(), istep1,
                                 m_cl_dst.GetCLMemRef<cl::Buffer>(), ostep,
                                 width,
                                 (MI_S32)(cl_global_size.get()[1]),
                                 (MI_S32)(cl_global_size.get()[0]),
                                 cl_global_size, cl_rt->GetCLDefaultLocalSize(m_cl_kernels[0].GetMaxGroupSize(), cl_global_size),
                                 &(cl_event));

    if (cl_ret != CL_SUCCESS)
    {
        AURA_ADD_ERROR_STRING(m_ctx, ("Run cl_kernel failed : " + GetCLErrorInfo(cl_ret)).c_str());
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

std::string MulSpectrumsCL::ToString() const
{
    return MulSpectrumsImpl::ToString() + m_profiling_string;
}

std::vector<CLKernel> MulSpectrumsCL::GetCLKernels(Context *ctx, ElemType elem_type, MI_BOOL conj_src1)
{
    std::vector<CLKernel> cl_kernels;
    std::vector<std::string> program_names, kernel_names;

    std::string build_opt = GetCLBuildOptions(ctx, elem_type, conj_src1);

    std::string program_name = "aura_mulspectrums";
    std::string kernel_name  = "MulAndScaleSpectrums";

    cl_kernels.push_back({ctx, program_name, kernel_name, "", build_opt});

    return cl_kernels;
}

} // namespace aura
