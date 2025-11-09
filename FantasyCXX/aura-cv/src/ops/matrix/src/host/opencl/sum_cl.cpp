#include "sum_impl.hpp"
#include "aura/runtime/opencl.h"
#include "aura/runtime/logger.h"

namespace aura
{

static ElemType GetDstType(ElemType src_type)
{
    ElemType dst_type = ElemType::F32;
    if (ElemType::U8 == src_type)
    {
        dst_type = ElemType::U32;
    }
    if (ElemType::S8 == src_type)
    {
        dst_type = ElemType::S32;
    }

    return dst_type;
}

static std::string GetCLMainBuildOptions(Context *ctx, const DT_S32 elem_counts, ElemType src_type, ElemType sum_type)
{
    CLBuildOptions cl_build_opt(ctx);
    cl_build_opt.AddOption("St",          CLTypeString(src_type));
    cl_build_opt.AddOption("SumType",     CLTypeString(sum_type));
    cl_build_opt.AddOption("ELEM_COUNTS", std::to_string(elem_counts));

    return cl_build_opt.ToString();
}

static std::string GetCLRemainBuildOptions(Context *ctx, ElemType src_type, ElemType dst_type)
{
    CLBuildOptions cl_build_opt(ctx);
    cl_build_opt.AddOption("St",      CLTypeString(src_type));
    cl_build_opt.AddOption("SumType", CLTypeString(dst_type));

    return cl_build_opt.ToString();
}

SumCL::SumCL(Context *ctx, const OpTarget &target) : SumImpl(ctx, target), m_group_size_x_main(0), m_group_size_y_main(0)
{}

Status SumCL::SetArgs(const Array *src, Scalar *result)
{
    if (SumImpl::SetArgs(src, result) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "SumImpl::SetArgs failed");
        return Status::ERROR;
    }

    if (src->GetSizes().m_channel != 1)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "opencl target only support channel 1");
        return Status::ERROR;
    }

    return Status::OK;
}

Status SumCL::Initialize()
{
    if (SumImpl::Initialize() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "SumImpl::Initialize() failed");
        return Status::ERROR;
    }

    ElemType src_type             = m_src->GetElemType();
    const DT_S32 elem_counts_main = 16 / ElemTypeSize(src_type);
    DT_S32 height                 = m_src->GetSizes().m_height;
    DT_S32 width                  = m_src->GetSizes().m_width * m_src->GetSizes().m_channel;

    ElemType dst_type      = GetDstType(src_type);
    m_group_size_x_main    = (width + m_blk_w * elem_counts_main - 1) / (m_blk_w * elem_counts_main);
    m_group_size_y_main    = (height + m_blk_h - 1) / m_blk_h;
    DT_S32 group_size_main = m_group_size_x_main * m_group_size_y_main;

    // 1. init cl_mem
    m_cl_src = CLMem::FromArray(m_ctx, *m_src, CLMemParam(CL_MEM_READ_ONLY));
    if (!m_cl_src.IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "m_cl_src is invalid");
        return Status::ERROR;
    }

    m_cl_partial = CLMem(m_ctx, CLMemParam(CL_MEM_READ_WRITE), dst_type, {1, group_size_main}, {1, group_size_main * ElemTypeSize(dst_type)});
    if (!m_cl_partial.IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "cl_partial is invalid");
        return Status::ERROR;
    }

    dst_mat = Mat(m_ctx, ElemType::U8, {1, static_cast<DT_S32>(ElemTypeSize(dst_type))});

    m_cl_dst = CLMem::FromArray(m_ctx, dst_mat, CLMemParam(CL_MEM_WRITE_ONLY));
    if (!m_cl_dst.IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "m_cl_dst is invalid");
        return Status::ERROR;
    }

    // 2. init kernel
    m_cl_kernels = GetCLKernels(m_ctx, src_type, dst_type);
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

Status SumCL::DeInitialize()
{
    m_cl_src.Release();
    m_cl_partial.Release();
    m_cl_dst.Release();

    m_cl_kernels[0].DeInitialize();
    m_cl_kernels[1].DeInitialize();

    if (SumImpl::DeInitialize() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "SumImpl::DeInitialize() failed");
        return Status::ERROR;
    }

    return Status::OK;
}

Status SumCL::Run()
{
    std::shared_ptr<CLRuntime> cl_rt      = m_ctx->GetCLEngine()->GetCLRuntime();
    std::shared_ptr<cl::Device> cl_device = cl_rt->GetDevice();
    DT_S32 istep                          = m_src->GetRowPitch() / ElemTypeSize(m_src->GetElemType());
    DT_S32 height                         = m_src->GetSizes().m_height;
    DT_S32 width                          = m_src->GetSizes().m_width * m_src->GetSizes().m_channel;
    ElemType src_type                     = m_src->GetElemType();
    ElemType dst_type                     = GetDstType(src_type);

    // 1. get center_area and cl_global_size
    size_t preferred_group_size;
    m_cl_kernels[0].GetClKernel()->getWorkGroupInfo(*cl_device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, &preferred_group_size);
    DT_S32 max_group_size     = m_cl_kernels[0].GetMaxGroupSize();
    DT_S32 max_local_mem_size = cl_device->getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();

    DT_S32 local_size_x_main  = Min(max_local_mem_size, Min(max_group_size, 2 * (DT_S32)preferred_group_size));
    DT_S32 local_size_y_main  = 1;
    DT_S32 group_size_main    = m_group_size_x_main * m_group_size_y_main;
    DT_S32 global_size_x_main = local_size_x_main * m_group_size_x_main;
    DT_S32 global_size_y_main = local_size_y_main * m_group_size_y_main;
    DT_S32 global_size_remain = Min(max_local_mem_size, Min(group_size_main, (DT_S32)m_cl_kernels[1].GetMaxGroupSize()));
    DT_S32 elem_counts_remain = (group_size_main + global_size_remain - 1) / global_size_remain;

    cl::Event cl_event[2];
    cl_int cl_ret = CL_SUCCESS;
    Status ret    = Status::ERROR;

    // 2. opencl run
    cl_ret = m_cl_kernels[0].Run<cl::Buffer, DT_S32,
                                 cl::Buffer, cl::LocalSpaceArg,
                                 DT_S32, DT_S32, DT_S32, DT_S32>(
                                 m_cl_src.GetCLMemRef<cl::Buffer>(), istep,
                                 m_cl_partial.GetCLMemRef<cl::Buffer>(),
                                 cl::Local(local_size_x_main * local_size_y_main * ElemTypeSize(dst_type)),
                                 m_blk_w, m_blk_h, width, height,
                                 cl::NDRange(global_size_x_main, global_size_y_main),
                                 cl::NDRange(local_size_x_main, local_size_y_main),
                                 &(cl_event[0]));
    if (cl_ret != CL_SUCCESS)
    {
        AURA_ADD_ERROR_STRING(m_ctx, ("Run cl_kernel main failed : " + GetCLErrorInfo(cl_ret)).c_str());
        goto EXIT;
    }

    cl_ret = m_cl_kernels[1].Run<cl::Buffer, cl::Buffer,
                                 cl::LocalSpaceArg,
                                 DT_S32, DT_S32>(
                                 m_cl_partial.GetCLMemRef<cl::Buffer>(),
                                 m_cl_dst.GetCLMemRef<cl::Buffer>(),
                                 cl::Local(global_size_remain * ElemTypeSize(dst_type)),
                                 elem_counts_remain, m_group_size_x_main * m_group_size_y_main,
                                 cl::NDRange(global_size_remain),
                                 cl::NDRange(global_size_remain),
                                 &(cl_event[1]), {cl_event[0]});
    if (cl_ret != CL_SUCCESS)
    {
        AURA_ADD_ERROR_STRING(m_ctx, ("Run cl_kernel main failed : " + GetCLErrorInfo(cl_ret)).c_str());
        goto EXIT;
    }

    // 3. cl wait
    if ((DT_TRUE == m_target.m_data.opencl.profiling) || (dst_mat.GetArrayType() != ArrayType::CL_MEMORY))
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

    ret = m_cl_dst.Sync(CLMemSyncType::READ);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "m_cl_dst sync end fail");
    }

    switch (dst_type)
    {
        case ElemType::U32:
        {
            DT_U32 *dst_ptr = dst_mat.Ptr<DT_U32>(0);
            m_result->m_val[0] = static_cast<DT_F64>(dst_ptr[0]);
            break;
        }
        case ElemType::S32:
        {
            DT_S32 *dst_ptr = dst_mat.Ptr<DT_S32>(0);
            m_result->m_val[0] = static_cast<DT_F64>(dst_ptr[0]);
            break;
        }
        default:
        {
            DT_F32 *dst_ptr = dst_mat.Ptr<DT_F32>(0);
            m_result->m_val[0] = static_cast<DT_F64>(dst_ptr[0]);
            break;
        }
    }

EXIT:
    AURA_RETURN(m_ctx, ret);
}

std::string SumCL::ToString() const
{
    return SumImpl::ToString() + m_profiling_string;
}

std::vector<CLKernel> SumCL::GetCLKernels(Context *ctx, ElemType src_elem_type, ElemType dst_elem_type)
{
    std::vector<CLKernel> cl_kernels;
    std::string build_opt[2];

    DT_S32 elem_counts_main = 16 / ElemTypeSize(src_elem_type);

    build_opt[0] = GetCLMainBuildOptions(ctx, elem_counts_main, src_elem_type, dst_elem_type);
    build_opt[1] = GetCLRemainBuildOptions(ctx, dst_elem_type, dst_elem_type);

    std::string program_name[2], kernel_name[2];
    program_name[0] = "aura_sum_main";
    kernel_name[0]  = "SumMain";
    program_name[1] = "aura_sum_remain";
    kernel_name[1]  = "SumRemain";

    cl_kernels.push_back({ctx, program_name[0], kernel_name[0], "", build_opt[0]});
    cl_kernels.push_back({ctx, program_name[1], kernel_name[1], "", build_opt[1]});

    return cl_kernels;
}

MeanCL::MeanCL(Context *ctx, const OpTarget &target) : SumCL(ctx, target)
{}

Status MeanCL::Run()
{
    Status ret = SumCL::Run();
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "SumCL run failed.");
        return Status::ERROR;
    }

    const DT_S32 height = m_src->GetSizes().m_height;
    const DT_S32 width  = m_src->GetSizes().m_width;
    *m_result           = (*m_result) / static_cast<DT_F64>(height * width);

    return Status::OK;
}

} // namespace aura