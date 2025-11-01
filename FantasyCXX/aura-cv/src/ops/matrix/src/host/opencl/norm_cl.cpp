#include "norm_impl.hpp"
#include "aura/runtime/opencl.h"
#include "aura/runtime/logger.h"

namespace aura
{

static ElemType GetAbsType(ElemType elem_type)
{
    switch (elem_type)
    {
        case ElemType::S8:
        {
            return ElemType::U8;
        }
        case ElemType::S16:
        {
            return ElemType::U16;
        }
        case ElemType::S32:
        {
            return ElemType::U32;
        }
        default:
        {
            return elem_type;
        }
    }
}

static ElemType GetSumType(ElemType elem_type)
{
    switch (elem_type)
    {
        case ElemType::U8:
        {
            return ElemType::U32;
        }
        case ElemType::S8:
        {
            return ElemType::U32;
        }
        default:
        {
            return ElemType::F32;
        }
    }
}

static ElemType GetSqSumType(ElemType elem_type)
{
    AURA_UNUSED(elem_type);
    return ElemType::F32;
}

static ElemType GetDstType(ElemType elem_type, NormType norm_type)
{
    switch (norm_type)
    {
        case NormType::NORM_INF:
        {
            return GetAbsType(elem_type);
        }
        case NormType::NORM_L1:
        {
            return GetSumType(elem_type);
        }
        case NormType::NORM_L2:
        case NormType::NORM_L2SQR:
        {
            return GetSqSumType(elem_type);
        }

        default:
        {
            return ElemType::INVALID;
        }
    }
}

static MI_F64 CastIntoF64(AURA_VOID *data, ElemType type)
{
    switch (type)
    {
        case ElemType::U8:
        {
            return static_cast<MI_F64>(*((MI_U8*)data));
        }
        case ElemType::S8:
        {
            return static_cast<MI_F64>(*((MI_S8*)data));
        }
        case ElemType::U16:
        {
            return static_cast<MI_F64>(*((MI_U16*)data));
        }
        case ElemType::S16:
        {
            return static_cast<MI_F64>(*((MI_S16*)data));
        }
        case ElemType::U32:
        {
            return static_cast<MI_F64>(*((MI_U32*)data));
        }
        case ElemType::S32:
        {
            return static_cast<MI_F64>(*((MI_S32*)data));
        }
        case ElemType::F16:
        {
            return static_cast<MI_F64>(*((MI_F16*)data));
        }
        case ElemType::F32:
        {
            return static_cast<MI_F64>(*((MI_F32*)data));
        }
        default:
        {
            return 0.0;
        }
    }
}

static std::string GetCLMainBuildOptions(Context *ctx, ElemType elem_type, MI_S32 elem_counts)
{
    CLBuildOptions cl_build_opt(ctx);

    cl_build_opt.AddOption("St",          CLTypeString(elem_type));
    cl_build_opt.AddOption("AbsType",     CLTypeString(GetAbsType(elem_type)));
    cl_build_opt.AddOption("SumType",     CLTypeString(GetSumType(elem_type)));
    cl_build_opt.AddOption("SqSumType",   CLTypeString(GetSqSumType(elem_type)));
    cl_build_opt.AddOption("ELEM_COUNTS", std::to_string(elem_counts));

    if (ElemType::F32 == elem_type || ElemType::F16 == elem_type)
    {
        cl_build_opt.AddOption("ABS", "fabs");
    }
    else
    {
        cl_build_opt.AddOption("ABS", "abs");
    }

    return cl_build_opt.ToString();
}

static std::string GetCLRemainBuildOptions(Context *ctx, ElemType src_type, ElemType sum_type)
{
    CLBuildOptions cl_build_opt(ctx);
    cl_build_opt.AddOption("St",      CLTypeString(src_type));
    cl_build_opt.AddOption("SumType", CLTypeString(sum_type));

    return cl_build_opt.ToString();
}

static AURA_VOID GetCLName(NormType type, std::string program_name[2], std::string kernel_name[2])
{
    switch (type)
    {
        case NormType::NORM_INF:
        {
            program_name[0] = "aura_norminf_main";
            kernel_name[0]  = "NormInfMain";
            program_name[1] = "aura_norminf_remain";
            kernel_name[1]  = "NormInfRemain";
            break;
        }
        case NormType::NORM_L1:
        {
            program_name[0] = "aura_abssum_main";
            kernel_name[0]  = "AbsSumMain";
            program_name[1] = "aura_sum_remain";
            kernel_name[1]  = "SumRemain";
            break;
        }
        case NormType::NORM_L2:
        case NormType::NORM_L2SQR:
        {
            program_name[0] = "aura_sqsum_main";
            kernel_name[0]  = "SqSumMain";
            program_name[1] = "aura_sum_remain";
            kernel_name[1]  = "SumRemain";
            break;
        }

        default:
        {
            break;
        }
    }
}

static AURA_VOID GetCLSize(Context *ctx, std::vector<CLKernel> cl_kernels, MI_S32 &m_group_size_x_main, MI_S32 &m_group_size_y_main,
                         cl::NDRange cl_global_size[2], cl::NDRange cl_local_size[2], MI_S32 &load_length)
{
    std::shared_ptr<cl::Device> cl_device = ctx->GetCLEngine()->GetCLRuntime()->GetDevice();
    size_t preferred_group_size;
    cl_kernels[0].GetClKernel()->getWorkGroupInfo(*cl_device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, &preferred_group_size);
    MI_S32 max_group_size     = cl_kernels[0].GetMaxGroupSize();
    MI_S32 max_local_mem_size = cl_device->getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();

    MI_S32 local_size_x_main  = Min(max_local_mem_size, Min(max_group_size, 2 * (MI_S32)preferred_group_size));
    MI_S32 local_size_y_main  = 1;
    MI_S32 group_size_main    = m_group_size_x_main * m_group_size_y_main;
    MI_S32 global_size_x_main = local_size_x_main * m_group_size_x_main;
    MI_S32 global_size_y_main = local_size_y_main * m_group_size_y_main;
    MI_S32 global_size_remain = Min(max_local_mem_size, Min(group_size_main, (MI_S32)cl_kernels[1].GetMaxGroupSize()));
    load_length               = (group_size_main + global_size_remain - 1) / global_size_remain;

    cl_global_size[0] = cl::NDRange(global_size_x_main, global_size_y_main);
    cl_local_size[0]  = cl::NDRange(local_size_x_main, local_size_y_main);
    cl_global_size[1] = cl::NDRange(global_size_remain);
    cl_local_size[1]  = cl::NDRange(global_size_remain);
}

NormCL::NormCL(Context *ctx, const OpTarget &target) : NormImpl(ctx, target), m_group_size_x_main(0),
                                                       m_group_size_y_main(0), m_profiling_string()
{}

Status NormCL::SetArgs(const Array *src, MI_F64 *result, NormType type)
{
    if (NormImpl::SetArgs(src, result, type) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "NormImpl::SetArgs failed");
        return Status::ERROR;
    }

    if (NormType::NORM_MINMAX == m_type)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "NormType::NORM_MINMAX is used for normalize function.");
        return Status::ERROR;
    }

    return Status::OK;
}

Status NormCL::Initialize()
{
    if (NormImpl::Initialize() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "NormImpl::Initialize() failed");
        return Status::ERROR;
    }

    ElemType src_type        = m_src->GetElemType();
    const MI_S32 elem_counts = 16 / ElemTypeSize(src_type);
    MI_S32 height            = m_src->GetSizes().m_height;
    MI_S32 width             = m_src->GetSizes().m_width * m_src->GetSizes().m_channel;

    ElemType dst_type      = GetDstType(src_type, m_type);
    m_group_size_x_main    = (width + m_blk_w * elem_counts - 1) / (m_blk_w * elem_counts);
    m_group_size_y_main    = (height + m_blk_h - 1) / m_blk_h;
    MI_S32 group_size_main = m_group_size_x_main * m_group_size_y_main;

    // 1. init cl_mem
    m_cl_src = CLMem::FromArray(m_ctx, *m_src, CLMemParam(CL_MEM_READ_ONLY));
    if (!m_cl_src.IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "m_cl_src is invalid");
        return Status::ERROR;
    }

    m_cl_partial = CLMem(m_ctx, CLMemParam(CL_MEM_READ_WRITE), dst_type, Sizes3(1, group_size_main * ElemTypeSize(dst_type)));
    if (!m_cl_partial.IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "cl_partial is invalid");
        return Status::ERROR;
    }

    dst_mat = Mat(m_ctx, ElemType::U8, {1, static_cast<MI_S32>(ElemTypeSize(dst_type))});

    m_cl_dst = CLMem::FromArray(m_ctx, dst_mat, CLMemParam(CL_MEM_WRITE_ONLY));
    if (!m_cl_dst.IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "m_cl_dst is invalid");
        return Status::ERROR;
    }

    // 2. init kernel
    m_cl_kernels = GetCLKernels(m_ctx, src_type, dst_type, m_type);
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

Status NormCL::DeInitialize()
{
    m_cl_src.Release();
    m_cl_partial.Release();
    m_cl_dst.Release();

    m_cl_kernels[0].DeInitialize();
    m_cl_kernels[1].DeInitialize();

    if (NormImpl::DeInitialize() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "NormImpl::DeInitialize() failed");
        return Status::ERROR;
    }

    return Status::OK;
}

Status NormCL::Run()
{
    MI_S32 istep      = m_src->GetRowPitch() / ElemTypeSize(m_src->GetElemType());
    MI_S32 height     = m_src->GetSizes().m_height;
    MI_S32 width      = m_src->GetSizes().m_width * m_src->GetSizes().m_channel;
    ElemType src_type = m_src->GetElemType();
    ElemType dst_type = GetDstType(src_type, m_type);

    // 1. get center_area and cl_global_size
    cl::NDRange cl_global_size[2], cl_local_size[2];
    MI_S32 load_length = 0;
    GetCLSize(m_ctx, m_cl_kernels, m_group_size_x_main, m_group_size_y_main,
              cl_global_size, cl_local_size, load_length);

    cl::Event cl_event[2];
    cl_int cl_ret   = CL_SUCCESS;
    Status ret      = Status::ERROR;

    // 2. opencl run
    cl_ret = m_cl_kernels[0].Run<cl::Buffer, MI_S32,
                                 cl::Buffer, cl::LocalSpaceArg,
                                 MI_S32, MI_S32, MI_S32, MI_S32>(
                                 m_cl_src.GetCLMemRef<cl::Buffer>(), istep,
                                 m_cl_partial.GetCLMemRef<cl::Buffer>(),
                                 cl::Local((MI_S32)(cl_local_size[0].get()[0]) *
                                 (MI_S32)(cl_local_size[0].get()[1]) * ElemTypeSize(dst_type)),
                                 m_blk_w, m_blk_h, width, height,
                                 cl_global_size[0],
                                 cl_local_size[0],
                                 &(cl_event[0]));
    if (cl_ret != CL_SUCCESS)
    {
        AURA_ADD_ERROR_STRING(m_ctx, ("Run cl_kernel main failed : " + GetCLErrorInfo(cl_ret)).c_str());
        goto EXIT;
    }

    cl_ret = m_cl_kernels[1].Run<cl::Buffer, cl::Buffer,
                                 cl::LocalSpaceArg,
                                 MI_S32, MI_S32>(
                                 m_cl_partial.GetCLMemRef<cl::Buffer>(),
                                 m_cl_dst.GetCLMemRef<cl::Buffer>(),
                                 cl::Local((MI_S32)(cl_global_size[1].get()[0]) * ElemTypeSize(dst_type)),
                                 load_length, m_group_size_x_main * m_group_size_y_main,
                                 cl_global_size[1],
                                 cl_local_size[1],
                                 &(cl_event[1]), {cl_event[0]});
    if (cl_ret != CL_SUCCESS)
    {
        AURA_ADD_ERROR_STRING(m_ctx, ("Run cl_kernel main failed : " + GetCLErrorInfo(cl_ret)).c_str());
        goto EXIT;
    }

    // 3. cl wait
    if ((MI_TRUE == m_target.m_data.opencl.profiling) || (dst_mat.GetArrayType() != ArrayType::CL_MEMORY))
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
        goto EXIT;
    }

    *m_result = CastIntoF64(dst_mat.GetData(), dst_type);
    if (NormType::NORM_L2 == m_type)
    {
        *m_result = Sqrt(*m_result);
    }

EXIT:
    AURA_RETURN(m_ctx, ret);
}

std::string NormCL::ToString() const
{
    return NormImpl::ToString() + m_profiling_string;
}

std::vector<CLKernel> NormCL::GetCLKernels(Context *ctx, ElemType src_elem_type, ElemType dst_elem_type, NormType type)
{
    MI_S32 elem_counts = 16 / ElemTypeSize(src_elem_type);

    std::vector<CLKernel> cl_kernels;
    std::string build_opt[2];
    build_opt[0] = GetCLMainBuildOptions(ctx, src_elem_type, elem_counts);
    build_opt[1] = GetCLRemainBuildOptions(ctx, dst_elem_type, dst_elem_type);

    std::string program_name[2], kernel_name[2];
    GetCLName(type, program_name, kernel_name);

    cl_kernels.push_back({ctx, program_name[0], kernel_name[0], "", build_opt[0]});
    cl_kernels.push_back({ctx, program_name[1], kernel_name[1], "", build_opt[1]});

    return cl_kernels;
}

} // namespace aura