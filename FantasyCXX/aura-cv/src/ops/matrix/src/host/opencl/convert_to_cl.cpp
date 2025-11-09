#include "convert_to_impl.hpp"
#include "aura/runtime/mat.h"
#include "aura/runtime/opencl.h"
#include "aura/runtime/logger.h"

namespace aura
{

static std::string GetCLBuildOptions(Context *ctx, ElemType src_type, ElemType dst_type)
{
    CLBuildOptions cl_build_opt(ctx);
    cl_build_opt.AddOption("St", CLTypeString(src_type));
    cl_build_opt.AddOption("Dt", CLTypeString(dst_type));

    return cl_build_opt.ToString();
}

static DT_VOID GetCLGlobalSize(DT_S32 height, DT_S32 width, DT_S32 channel, DT_S32 elem_counts, cl::NDRange &cl_range)
{
    cl_range = cl::NDRange((width * channel + elem_counts - 1) / elem_counts, height);
}

ConvertToCL::ConvertToCL(Context *ctx, const OpTarget &target) : ConvertToImpl(ctx, target), m_is_same_mat(DT_FALSE), m_profiling_string()
{}

Status ConvertToCL::Initialize()
{
    if (ConvertToImpl::Initialize() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "ConvertToImpl::Initialize() failed");
        return Status::ERROR;
    }

    // 0. special case
    DT_BOOL no_scale = (Abs(m_alpha - 1.0) < DBL_EPSILON) && (Abs(m_beta) < DBL_EPSILON);
    m_is_same_mat    = 0;
    if (no_scale && (m_src->GetElemType() == m_dst->GetElemType()) &&
        (ArrayType::MAT == m_src->GetArrayType()) && (ArrayType::MAT == m_dst->GetArrayType()))
    {
        m_is_same_mat = 1;
        return Status::OK;
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
    m_cl_kernels = ConvertToCL::GetCLKernels(m_ctx, m_src->GetElemType(), m_dst->GetElemType(), (!no_scale));
    if (CheckCLKernels(m_ctx, m_cl_kernels) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "CheckCLKernels failed");
        return Status::ERROR;
    }

    // 3. sync start
    if (m_cl_src.Sync(CLMemSyncType::WRITE) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "m_cl_src sync start failed");
        return Status::ERROR;
    }

    return Status::OK;
}

Status ConvertToCL::DeInitialize()
{
    if (!m_is_same_mat)
    {
        m_cl_src.Release();
        m_cl_dst.Release();

        m_cl_kernels[0].DeInitialize();
    }

    if (ConvertToImpl::DeInitialize() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "ConvertToImpl::DeInitialize() failed");
        return Status::ERROR;
    }

    return Status::OK;
}

Status ConvertToCL::Run()
{
    Status ret = Status::ERROR;

    // 0. the same size will copy mat
    if (m_is_same_mat)
    {
        const Mat *src = dynamic_cast<const Mat*>(m_src);
        Mat *dst       = dynamic_cast<Mat*>(m_dst);

        ret = src->CopyTo(*dst);
        AURA_RETURN(m_ctx, ret);
    }

    DT_S32 istep   = m_src->GetRowPitch() / ElemTypeSize(m_src->GetElemType());
    DT_S32 ostep   = m_dst->GetRowPitch() / ElemTypeSize(m_dst->GetElemType());
    DT_S32 height  = m_src->GetSizes().m_height;
    DT_S32 width   = m_src->GetSizes().m_width;
    DT_S32 channel = m_src->GetSizes().m_channel;

    std::shared_ptr<CLRuntime> cl_rt = m_ctx->GetCLEngine()->GetCLRuntime();

    // 1. get cl_global_size
    cl::NDRange cl_global_size;
    constexpr DT_S32 ELEM_COUNTS = 4;
    GetCLGlobalSize(height, width, channel, ELEM_COUNTS, cl_global_size);
    cl_int cl_ret = CL_SUCCESS;
    cl::Event cl_event;

    // 2. opencl run
    cl_ret = m_cl_kernels[0].Run<cl::Buffer, DT_S32, cl::Buffer, DT_S32, DT_S32, DT_S32, DT_F32, DT_F32>(
                                  m_cl_src.GetCLMemRef<cl::Buffer>(), istep,
                                  m_cl_dst.GetCLMemRef<cl::Buffer>(), ostep,
                                  width * channel, height,
                                  m_alpha, m_beta,
                                  cl_global_size, cl_rt->GetCLDefaultLocalSize(m_cl_kernels[0].GetMaxGroupSize(), cl_global_size),
                                  &cl_event);
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

    ret = m_cl_dst.Sync(CLMemSyncType::READ);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Sync end fail");
    }

EXIT:
    AURA_RETURN(m_ctx, ret);
}

std::string ConvertToCL::ToString() const
{
    return ConvertToImpl::ToString() + m_profiling_string;
}

std::vector<CLKernel> ConvertToCL::GetCLKernels(Context *ctx, ElemType src_elem_type, ElemType dst_elem_type, DT_BOOL scale)
{
    std::vector<CLKernel> cl_kernels;
    std::string build_opt = GetCLBuildOptions(ctx, src_elem_type, dst_elem_type);

    std::string program_name = scale ? "aura_convert_to_scale" : "aura_convert_to_no_scale";
    std::string kernel_name  = scale ? "ConvertToScale" : "ConvertToNoScale";

    cl_kernels.push_back({ctx, program_name, kernel_name, "", build_opt});

    return cl_kernels;
}
} // namespace aura
