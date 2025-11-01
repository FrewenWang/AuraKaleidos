#include "connect_component_label_impl.hpp"

namespace aura
{

static Status CheckCLExtensions(Context *ctx, MI_BOOL &ballot_flag, MI_BOOL &shuffle_flag)
{
    std::shared_ptr<CLRuntime> cl_rt = ctx->GetCLEngine()->GetCLRuntime();

    if (cl_rt)
    {
        std::string extensions_info_str = cl_rt->GetDevice()->getInfo<CL_DEVICE_EXTENSIONS>();
        size_t pos   = extensions_info_str.find("cl_khr_subgroup_ballot");
        ballot_flag  = (pos != std::string::npos);
        pos          = extensions_info_str.find("cl_khr_subgroup_shuffle");
        shuffle_flag = (pos != std::string::npos);
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "CLRuntime is nullptr");
        return Status::ERROR;
    }

#if CL_TARGET_OPENCL_VERSION >= 200
    return Status::OK;
#  else
    AURA_ADD_ERROR_STRING(ctx, "opencl 1.2 is too low to suport subgroup functions in adreno/mali");
    return Status::ERROR;
#  endif // CL_TARGET_OPENCL_VERSION >= 200
}

static Status GetCLBuildOptions(Context *ctx, ConnectivityType connectivity_type,
                                const ElemType label_type, std::string &build_opt)
{
    const std::vector<std::string> tbl =
    {
                "U32,    S32",   // label elem type
        "Tp",   "uint,   int",   // source
    };

    CLBuildOptions cl_build_opt(ctx, tbl);

    if (ConnectivityType::SQUARE == connectivity_type)
    {
        cl_build_opt.AddOption("CONNECTIVITY_CUBE", "1");
    }

    std::shared_ptr<CLRuntime> cl_rt = ctx->GetCLEngine()->GetCLRuntime();

    MI_BOOL ballot_flag = MI_FALSE, shuffle_flag = MI_FALSE;
    if (CheckCLExtensions(ctx, ballot_flag, shuffle_flag) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "OpenCL CheckCLExtensions fail");
        return Status::ERROR;
    }

    if (ballot_flag)
    {
        cl_build_opt.AddOption("BALLOT", "1");
    }
    if (shuffle_flag)
    {
        cl_build_opt.AddOption("SHUFFLE", "1");
    }

    if (GpuType::ADRENO == cl_rt->GetGpuInfo().m_type)
    {
        cl_build_opt.AddOption("ADRENO", "1");
    }
    else if (GpuType::MALI == cl_rt->GetGpuInfo().m_type)
    {
        cl_build_opt.AddOption("MALI", "1");
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "unsupported GPU platform type");
        return Status::ERROR;
    }

    build_opt = cl_build_opt.ToString(label_type);

    return Status::OK;
}

static Status GetCLName(Context *ctx, CCLAlgo algo_type, std::vector<std::string> &program_names,
                        std::vector<std::string> &kernel_names)
{
    if (MI_NULL == ctx)
    {
        return Status::ERROR;
    }

    program_names.emplace_back("aura_connect_component_ha_strip_labeling");
    program_names.emplace_back("aura_connect_component_ha_strip_merge");
    program_names.emplace_back("aura_connect_component_ha_relabeling");

    kernel_names.emplace_back("CCL" + CCLAlgoTypeToString(algo_type) + "StripLabeling");
    kernel_names.emplace_back("CCL" + CCLAlgoTypeToString(algo_type) + "StripMerge");
    kernel_names.emplace_back("CCL" + CCLAlgoTypeToString(algo_type) + "Relabeling");

    return Status::OK;
}

static AURA_VOID GetCLSize(Context *ctx, MI_S32 height, MI_S32 width, MI_S32 max_group_size,
                         ConnectivityType connectivity_type, cl::NDRange cl_local_size[3], cl::NDRange cl_global_size[3])
{
    std::shared_ptr<CLRuntime> cl_rt = ctx->GetCLEngine()->GetCLRuntime();

    const MI_S32 group_h = 4;
    MI_S32 subgroup_size = 128;

    if (GpuType::ADRENO == cl_rt->GetGpuInfo().m_type)
    {
        subgroup_size = 64;
    }
    else if (GpuType::MALI == cl_rt->GetGpuInfo().m_type)
    {
        subgroup_size = 16;
    }

    if (ConnectivityType::CROSS == connectivity_type)
    {
        MI_S32 strip_merge_global_h = ((height + group_h - 1) / group_h + group_h - 1) / group_h;

        cl_local_size[0] = cl::NDRange(subgroup_size,  group_h);
        cl_local_size[1] = cl::NDRange(subgroup_size,  group_h);
        cl_local_size[2] = cl::NDRange(subgroup_size,  group_h);

        cl_global_size[0] = cl::NDRange(subgroup_size,                    AURA_ALIGN(height, group_h));
        cl_global_size[1] = cl::NDRange(AURA_ALIGN(width, subgroup_size), strip_merge_global_h * group_h);
        cl_global_size[2] = cl::NDRange(AURA_ALIGN(width, subgroup_size), AURA_ALIGN(height, group_h));
    }
    else
    {
        MI_S32 warp_nums_merge      = std::min((width + subgroup_size - 1) / subgroup_size, max_group_size / subgroup_size);
        MI_S32 strip_merge_global_w = std::max((width + subgroup_size * (subgroup_size - 2) - 1) / (subgroup_size * (subgroup_size - 1)), 1);

        cl_local_size[0] = cl::NDRange(subgroup_size,  group_h);
        cl_local_size[1] = cl::NDRange(subgroup_size,  1,       warp_nums_merge);
        cl_local_size[2] = cl::NDRange(subgroup_size,  group_h);

        cl_global_size[0] = cl::NDRange(subgroup_size,                        AURA_ALIGN(height, group_h), 1);
        cl_global_size[1] = cl::NDRange(strip_merge_global_w * subgroup_size, (height - 1) / group_h,      warp_nums_merge);
        cl_global_size[2] = cl::NDRange(AURA_ALIGN(width, subgroup_size),     AURA_ALIGN(height, group_h), 1);
    }
}

ConnectComponentLabelCL::ConnectComponentLabelCL(Context *ctx, const OpTarget &target) : ConnectComponentLabelImpl(ctx, target), m_profiling_string()
{}

Status ConnectComponentLabelCL::SetArgs(const Array *src, Array *dst, CCLAlgo algo_type,
                                        ConnectivityType connectivity_type, EquivalenceSolver solver_type)
{
    if (ConnectComponentLabelImpl::SetArgs(src, dst, algo_type, connectivity_type, solver_type) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "ConnectComponentLabelImpl::SetArgs failed");
        return Status::ERROR;
    }

    if (CCLAlgo::HA_GPU != algo_type)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "opencl impl only support HA algorithms");
        return Status::ERROR;
    }

    if (dst->GetElemType() != aura::ElemType::U32 && dst->GetElemType() != aura::ElemType::S32)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "dst elem type must be U32/S32");
        return Status::ERROR;
    }

    MI_S32 width  = src->GetSizes().m_width;
    MI_S32 height = src->GetSizes().m_height;
    if (width < 16 || height < 4)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "size is too small for opencl");
        return Status::ERROR;
    }

    return Status::OK;
}

Status ConnectComponentLabelCL::Initialize()
{
    if (ConnectComponentLabelImpl::Initialize() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "ConnectComponentLabelImpl::Initialize() failed");
        return Status::ERROR;
    }

    if (m_ctx->GetCLEngine()->GetCLRuntime()->GetGpuInfo().m_type != GpuType::ADRENO)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "ConnectComponentLabelCL only support QCOM Adreno GPU for now");
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
    m_cl_kernels = ConnectComponentLabelCL::GetCLKernels(m_ctx, m_dst->GetElemType(), m_algo_type, m_connectivity_type);

    if (CheckCLKernels(m_ctx, m_cl_kernels) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "CheckCLKernels failed");
        return Status::ERROR;
    }

    // 3. sync src
    if (m_cl_src.Sync(CLMemSyncType::WRITE) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Sync src failed");
        return Status::ERROR;
    }

    return Status::OK;
}

Status ConnectComponentLabelCL::DeInitialize()
{
    m_cl_src.Release();
    m_cl_dst.Release();
    m_cl_kernels.clear();

    if (ConnectComponentLabelImpl::DeInitialize() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "ConnectComponentLabelImpl::DeInitialize() failed");
        return Status::ERROR;
    }

    return Status::OK;
}

Status ConnectComponentLabelCL::Run()
{
    MI_S32 istep  = m_src->GetRowPitch() / ElemTypeSize(m_src->GetElemType());
    MI_S32 ostep  = m_dst->GetRowPitch() / ElemTypeSize(m_dst->GetElemType());
    MI_S32 height = m_dst->GetSizes().m_height;
    MI_S32 width  = m_dst->GetSizes().m_width;

    std::shared_ptr<CLRuntime> cl_rt = m_ctx->GetCLEngine()->GetCLRuntime();

    // 1. get center_area and cl_global_size
    cl::NDRange cl_local_size[3];
    cl::NDRange cl_global_size[3];

    MI_S32 max_group_size = m_cl_kernels[1].GetMaxGroupSize();
    GetCLSize(m_ctx, height, width, max_group_size, m_connectivity_type, cl_local_size, cl_global_size);

    cl::Event cl_event[3];
    cl_int cl_ret   = CL_SUCCESS;
    Status ret      = Status::ERROR;
    Status ret_sync = Status::ERROR;

    // 2. opencl run
    // 1st kernel
    {
        cl_ret = m_cl_kernels[0].Run<cl::Buffer, MI_S32, cl::Buffer, MI_S32, MI_S32, MI_S32>(
                                    m_cl_src.GetCLMemRef<cl::Buffer>(), istep,
                                    m_cl_dst.GetCLMemRef<cl::Buffer>(), ostep,
                                    height, width,
                                    cl_global_size[0], cl_local_size[0],
                                    &(cl_event[0]));
        if (cl_ret != CL_SUCCESS)
        {
            AURA_ADD_ERROR_STRING(m_ctx, ("Run cl_kernel main failed : " + GetCLErrorInfo(cl_ret)).c_str());
            goto EXIT;
        }
    }

    // 2nd kernel and 3rd kernel
    for (MI_S32 i = 1; i < 3; ++i)
    {
        cl_ret = m_cl_kernels[i].Run<cl::Buffer, MI_S32, cl::Buffer, MI_S32, MI_S32, MI_S32>(
                                    m_cl_src.GetCLMemRef<cl::Buffer>(), istep,
                                    m_cl_dst.GetCLMemRef<cl::Buffer>(), ostep,
                                    height, width,
                                    cl_global_size[i], cl_local_size[i],
                                    &(cl_event[i]), {cl_event[i - 1]});
        if (cl_ret != CL_SUCCESS)
        {
            AURA_ADD_ERROR_STRING(m_ctx, ("Run cl_kernel main failed : " + GetCLErrorInfo(cl_ret)).c_str());
            goto EXIT;
        }
    }

    // 3. opencl wait
    if ((MI_TRUE == m_target.m_data.opencl.profiling) || (m_dst->GetArrayType() != ArrayType::CL_MEMORY))
    {
        cl_ret = cl_event[2].wait();
        if (cl_ret != CL_SUCCESS)
        {
            AURA_ADD_ERROR_STRING(m_ctx, ("Wait cl_kernel failed : " + GetCLErrorInfo(cl_ret)).c_str());
            goto EXIT;
        }
        m_profiling_string += " "  + GetCLProfilingInfo(m_cl_kernels[0].GetKernelName(), cl_event[0]) + 
                              GetCLProfilingInfo(m_cl_kernels[1].GetKernelName(), cl_event[1]) +
                              GetCLProfilingInfo(m_cl_kernels[2].GetKernelName(), cl_event[2]);
    }

    ret_sync = m_cl_dst.Sync(aura::CLMemSyncType::READ);
    if (ret_sync != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Sync end fail");
    }

    ret = Status::OK;

EXIT:
    AURA_RETURN(m_ctx, (ret | ret_sync));
}

std::string ConnectComponentLabelCL::ToString() const
{
    return ConnectComponentLabelImpl::ToString() + m_profiling_string;
}

std::vector<CLKernel> ConnectComponentLabelCL::GetCLKernels(Context *ctx, ElemType dst_elem_type, CCLAlgo algo_type, ConnectivityType connectivity_type)
{
    std::vector<CLKernel> cl_kernels;
    std::vector<std::string> program_names, kernel_names;
    std::string build_opt;

    if (GetCLBuildOptions(ctx, connectivity_type, dst_elem_type, build_opt) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetCLBuildOptions failed");
        return cl_kernels;
    }

    if (GetCLName(ctx, algo_type, program_names, kernel_names) != Status::OK)
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