#include "rotate_impl.hpp"
#include "aura/runtime/opencl.h"
#include "aura/runtime/logger.h"

namespace aura
{

static std::string GetCLBuildOptions(Context *ctx, ElemType elem_type, MI_S32 elem_counts, RotateType rot_type)
{
    CLBuildOptions cl_build_opt(ctx);

    std::string type_str;
    switch (ElemTypeSize(elem_type))
    {
        case 1:
        {
            type_str = "uchar";
            break;
        }
        case 2:
        {
            type_str = "ushort";
            break;
        }
        case 4:
        {
            type_str = "uint";
            break;
        }
        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "unsupported element type");
        }
    }

    cl_build_opt.AddOption("Tp", type_str);
    cl_build_opt.AddOption("ELEM_COUNTS", std::to_string(elem_counts));
    cl_build_opt.AddOption("ROTATE_TYPE", RotateTypeToString(rot_type));

    return cl_build_opt.ToString(elem_type);
}

static AURA_VOID GetCLGlobalSize(MI_S32 height, MI_S32 width, MI_S32 elem_counts, cl::NDRange &cl_range)
{
    cl_range = cl::NDRange((width + elem_counts - 1) / elem_counts, (height + elem_counts - 1) / elem_counts);
}

RotateCL::RotateCL(Context *ctx, const OpTarget &target) : RotateImpl(ctx, target), m_elem_counts(0), m_profiling_string()
{}

Status RotateCL::SetArgs(const Array *src, Array *dst, RotateType type)
{
    if (RotateImpl::SetArgs(src, dst, type) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "RotateImpl::SetArgs failed");
        return Status::ERROR;
    }

    if (src->GetSizes().m_channel > 4)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "opencl target cannot support channel > 4");
        return Status::ERROR;
    }

    return Status::OK;
}

Status RotateCL::Initialize()
{
    if (RotateImpl::Initialize() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "RotateImpl::Initialize() failed");
        return Status::ERROR;
    }

    MI_S32 ochannel = m_dst->GetSizes().m_channel;

    if (RotateImpl::Initialize() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "RotateImpl::Initialize() failed");
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
    m_elem_counts = (1 == ochannel) ? 8 : 4;
    m_cl_kernels  = GetCLKernels(m_ctx, m_dst->GetElemType(), ochannel, m_type);
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

Status RotateCL::DeInitialize()
{
    m_cl_src.Release();
    m_cl_dst.Release();

    m_cl_kernels[0].DeInitialize();

    if (RotateImpl::DeInitialize() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "RotateImpl::DeInitialize() failed");
        return Status::ERROR;
    }

    return Status::OK;
}

Status RotateCL::Run()
{
    MI_S32 istep  = m_src->GetRowPitch() / ElemTypeSize(m_src->GetElemType());
    MI_S32 ostep  = m_dst->GetRowPitch() / ElemTypeSize(m_dst->GetElemType());
    MI_S32 height = m_dst->GetSizes().m_height;
    MI_S32 width  = m_dst->GetSizes().m_width;

    std::shared_ptr<CLRuntime> cl_rt = m_ctx->GetCLEngine()->GetCLRuntime();

    // 1. get center_area and cl_global_size
    cl::NDRange cl_global_size;
    GetCLGlobalSize(height, width, m_elem_counts, cl_global_size);

    cl::Event cl_event;
    cl_int cl_ret = CL_SUCCESS;
    Status ret    = Status::ERROR;

    // 2. opencl run
    cl_ret = m_cl_kernels[0].Run<cl::Buffer, MI_S32, cl::Buffer, MI_S32, MI_S32, MI_S32>(
                                 m_cl_src.GetCLMemRef<cl::Buffer>(), istep,
                                 m_cl_dst.GetCLMemRef<cl::Buffer>(), ostep,
                                 width, height,
                                 cl_global_size, cl_rt->GetCLDefaultLocalSize(m_cl_kernels[0].GetMaxGroupSize(), cl_global_size),
                                 &(cl_event));
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

std::string RotateCL::ToString() const
{
    return RotateImpl::ToString() + m_profiling_string;
}

std::vector<CLKernel> RotateCL::GetCLKernels(Context *ctx, ElemType elem_type, MI_S32 ochannel, RotateType type)
{
    std::vector<CLKernel> cl_kernels;
    MI_S32 elem_counts    = (1 == ochannel) ? 8 : 4;
    std::string build_opt = GetCLBuildOptions(ctx, elem_type, elem_counts, type);

    std::string program_name = "aura_rotate_c" + std::to_string(ochannel);
    std::string kernel_name  = "RotateC" + std::to_string(ochannel);

    cl_kernels.push_back({ctx, program_name, kernel_name, "", build_opt});

    return cl_kernels;
}

} // namespace aura
