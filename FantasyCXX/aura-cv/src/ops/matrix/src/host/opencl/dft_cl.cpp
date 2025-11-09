#include "dft_impl.hpp"

#include "aura/runtime/mat.h"
#include "aura/runtime/opencl.h"
#include "aura/runtime/logger.h"

namespace aura
{

static std::string GetCLBuildOptions(Context *ctx, ElemType src_elem_type, ElemType dst_elem_type, DT_BOOL with_scale,
                                     DT_BOOL is_dst_c1)
{
    CLBuildOptions cl_build_opt(ctx);

    cl_build_opt.AddOption("St", CLTypeString(src_elem_type));
    cl_build_opt.AddOption("Dt", CLTypeString(dst_elem_type));
    cl_build_opt.AddOption("WITH_SCALE", with_scale ? "1" : "0");
    cl_build_opt.AddOption("IS_DST_C1", is_dst_c1 ? "1" : "0");

    return cl_build_opt.ToString();
}

static DT_VOID GetCLSize(Context *ctx, std::vector<CLKernel> &cl_kernels, DT_S32 width, DT_S32 height, cl::NDRange cl_global_size[2],
                         cl::NDRange cl_local_size[2])
{
    auto device = ctx->GetCLEngine()->GetCLRuntime()->GetDevice();
    DT_S32 max_group_sz_row = cl_kernels[0].GetClKernel()->getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(*device);
    DT_S32 max_group_sz_col = cl_kernels[1].GetClKernel()->getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(*device);

    // Row Param
    {
        DT_S32 local_size_x = Min(width / 2, max_group_sz_row);
        DT_S32 local_size_y = 1;

        const DT_S32 group_size_x = 1; // because we use local memory for compute, x direction only has one group
        const DT_S32 group_size_y = (height + local_size_y - 1) / local_size_y;

        const DT_S32 global_size_x = local_size_x * group_size_x;
        const DT_S32 global_size_y = local_size_y * group_size_y;

        cl_global_size[0] = cl::NDRange(global_size_x, global_size_y);
        cl_local_size[0]  = cl::NDRange(local_size_x,  local_size_y);
    }
    // Col Param
    {
        size_t local_size_x = 1;
        size_t local_size_y = Min(height / 2, max_group_sz_col);

        const size_t group_size_x = (width + local_size_x - 1) / local_size_x; // width;
        const size_t group_size_y = 1;

        const size_t global_size_x = local_size_x * group_size_x;
        const size_t global_size_y = local_size_y * group_size_y;

        cl_global_size[1] = cl::NDRange(global_size_x, global_size_y);
        cl_local_size[1]  = cl::NDRange(local_size_x, local_size_y);
    }
}

DftCL::DftCL(Context *ctx, const OpTarget &target) : DftImpl(ctx, target), m_buffer_pitch(0),
                                                     m_local_buffer_bytes(0), m_exp_total_bytes(0),
                                                     m_profiling_string()
{}

Status DftCL::SetArgs(const Array *src, Array *dst)
{
    if (DftImpl::SetArgs(src, dst) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "DftImpl::SetArgs failed");
        return Status::ERROR;
    }

    if (src->GetElemType() == ElemType::S32 || src->GetElemType() == ElemType::U32 || src->GetElemType() == ElemType::F64)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "currently src does not support DT_S32/DT_U32/DT_F64 type.");
        return Status::ERROR;
    }

    Sizes3 sz = src->GetSizes();
    if (sz.m_height < 16 || sz.m_width < 16)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "size is too small");
        return Status::ERROR;
    }

    if (!IsPowOf2(sz.m_width) || !IsPowOf2(sz.m_height))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "CL impl only support 2^n width/height.");
        return Status::ERROR;
    }

    return Status::OK;
}

Status DftCL::Initialize()
{
    if (DftImpl::Initialize() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "DftImpl::Initialize() failed");
        return Status::ERROR;
    }

    Sizes3 sz     = m_src->GetSizes();
    DT_S32 width  = sz.m_width;
    DT_S32 height = sz.m_height;

    // Cpu get exp tables and index table
    DT_S32 idx_total_bytes = Max(width, height) * sizeof(DT_U16);
    m_exp_total_bytes      = Max(width, height) * sizeof(DT_F32);
    m_local_buffer_bytes   = Max(width, height) * 2 * sizeof(DT_F32);

    DT_S32 buffer_sz = AURA_ALIGN(idx_total_bytes + m_exp_total_bytes, 64);
    m_buffer_pitch   = buffer_sz;

    // query device info
    auto   device             = m_ctx->GetCLEngine()->GetCLRuntime()->GetDevice();
    DT_S32 max_local_mem_size = device->getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();

    if (m_exp_total_bytes + m_local_buffer_bytes > max_local_mem_size)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "CL Local memory is not large enough for do fft.");
        return Status::ERROR;
    }

    m_param = Mat(m_ctx, ElemType::U8, {2, buffer_sz});
    if (!m_param.IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Create m_param mat failed.");
        return Status::ERROR;
    }

    // 0. compute m_param in cpu
    {
        DT_U16 *idx_table = m_param.Ptr<DT_U16>(0);
        auto exp_table = reinterpret_cast<std::complex<DT_F32>*>(idx_table + width);
        GetReverseIndex(idx_table, width);
        GetDftExpTable<0>(exp_table, width);

        idx_table = m_param.Ptr<DT_U16>(1);
        exp_table = reinterpret_cast<std::complex<DT_F32>*>(idx_table + height);
        GetReverseIndex(idx_table, height);
        GetDftExpTable<0>(exp_table, height);
    }

    // 1. init cl_mem
    m_cl_src = CLMem::FromArray(m_ctx, *m_src, CLMemParam(CL_MEM_READ_ONLY));
    if (!m_cl_src.IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "cl_src is invalid");
        return Status::ERROR;
    }

    m_cl_dst = CLMem::FromArray(m_ctx, *m_dst, CLMemParam(CL_MEM_READ_WRITE));
    if (!m_cl_dst.IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "cl_dst is invalid");
        return Status::ERROR;
    }

    m_cl_param = CLMem::FromArray(m_ctx, m_param, CLMemParam(CL_MEM_READ_ONLY));
    if (!m_cl_param.IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "cl_param is invalid");
        return Status::ERROR;
    }

    // 2. build option
    m_cl_kernels = DftCL::GetCLKernels(m_ctx, m_src->GetElemType(), m_dst->GetElemType());
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

    if (m_cl_param.Sync(CLMemSyncType::WRITE) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "m_cl_param sync start failed");
        return Status::ERROR;
    }

    return Status::OK;
}

Status DftCL::DeInitialize()
{
    m_cl_src.Release();
    m_cl_param.Release();
    m_cl_dst.Release();
    m_param.Release();

    m_cl_kernels[0].DeInitialize();
    m_cl_kernels[1].DeInitialize();

    if (DftImpl::DeInitialize() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "DftImpl::DeInitialize() failed");
        return Status::ERROR;
    }

    return Status::OK;
}

Status DftCL::Run()
{
    DT_S32 istep   = m_src->GetRowPitch() / ElemTypeSize(m_src->GetElemType());
    DT_S32 ostep   = m_dst->GetRowPitch() / ElemTypeSize(m_dst->GetElemType());
    DT_S32 height  = m_src->GetSizes().m_height;
    DT_S32 width   = m_src->GetSizes().m_width;

    // 1. get cl_global_size and cl_local_size
    cl::NDRange cl_global_size[2];
    cl::NDRange cl_local_size[2];
    GetCLSize(m_ctx, m_cl_kernels, width, height, cl_global_size, cl_local_size);

    cl::Event cl_event[2];
    cl_int cl_ret = CL_SUCCESS;
    Status ret    = Status::ERROR;

    // 2. CL run
    cl_ret = m_cl_kernels[0].Run<cl::Buffer, DT_S32, cl::Buffer, DT_S32, cl::Buffer, DT_S32, cl::LocalSpaceArg, cl::LocalSpaceArg, DT_S32, DT_S32>(
                                 m_cl_src.GetCLMemRef<cl::Buffer>(), istep,
                                 m_cl_dst.GetCLMemRef<cl::Buffer>(), ostep,
                                 m_cl_param.GetCLMemRef<cl::Buffer>(), m_buffer_pitch,
                                 cl::Local(m_local_buffer_bytes * 1),
                                 cl::Local(m_exp_total_bytes),
                                 width, height,
                                 cl_global_size[0],
                                 cl_local_size[0],
                                 &cl_event[0]);
    if (cl_ret != CL_SUCCESS)
    {
        AURA_ADD_ERROR_STRING(m_ctx, ("Run m_cl_kernel main failed : " + GetCLErrorInfo(cl_ret)).c_str());
        goto EXIT;
    }

    cl_ret = m_cl_kernels[1].Run<cl::Buffer, DT_S32, cl::Buffer, DT_S32, cl::LocalSpaceArg, cl::LocalSpaceArg, DT_S32, DT_S32>(
                                 m_cl_dst.GetCLMemRef<cl::Buffer>(), ostep,
                                 m_cl_param.GetCLMemRef<cl::Buffer>(), m_buffer_pitch,
                                 cl::Local(m_local_buffer_bytes * 1),
                                 cl::Local(m_exp_total_bytes),
                                 width, height,
                                 cl_global_size[1],
                                 cl_local_size[1],
                                 &cl_event[1], {cl_event[0]});

    if (cl_ret != CL_SUCCESS)
    {
        AURA_ADD_ERROR_STRING(m_ctx, ("cl_kernel run failed : " + GetCLErrorInfo(cl_ret)).c_str());
        goto EXIT;
    }

    // 3. cl wait
    if ((DT_TRUE == m_target.m_data.opencl.profiling) || (m_dst->GetArrayType() != ArrayType::CL_MEMORY))
    {
        cl_ret = cl_event[1].wait();
        if (cl_ret != CL_SUCCESS)
        {
            AURA_ADD_ERROR_STRING(m_ctx, ("Wait m_cl_kernel failed : " + GetCLErrorInfo(cl_ret)).c_str());
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
        AURA_ADD_ERROR_STRING(m_ctx, "Sync end fail");
    }

EXIT:
    AURA_RETURN(m_ctx, ret);
}

std::string DftCL::ToString() const
{
    return DftImpl::ToString() + m_profiling_string;
}

std::vector<CLKernel> DftCL::GetCLKernels(Context *ctx, ElemType src_elem_type, ElemType dst_elem_type)
{
    std::vector<CLKernel> cl_kernels;
    std::string build_opt = GetCLBuildOptions(ctx, src_elem_type, dst_elem_type, DT_FALSE, DT_FALSE);

    std::string program_name[2] = {"aura_dft_row_process", "aura_dft_col_process"};
    std::string kernel_name[2]  = {"DftRowProcess", "DftColProcess"};
    cl_kernels.push_back({ctx, program_name[0], kernel_name[0], "", build_opt});
    cl_kernels.push_back({ctx, program_name[1], kernel_name[1], "", build_opt});

    return cl_kernels;
}

InverseDftCL::InverseDftCL(Context *ctx, const OpTarget &target) : InverseDftImpl(ctx, target), m_buffer_pitch(0),
                                                                   m_local_buffer_bytes(0), m_exp_total_bytes(0),
                                                                   m_profiling_string()
{}

Status InverseDftCL::SetArgs(const Array *src, Array *dst, DT_BOOL with_scale)
{
    if (InverseDftImpl::SetArgs(src, dst, with_scale) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "InverseDftImpl::SetArgs failed");
        return Status::ERROR;
    }

    Sizes3 sz = src->GetSizes();
    if (sz.m_height < 16 || sz.m_width < 16)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "size is too small");
        return Status::ERROR;
    }

    if (!IsPowOf2(sz.m_width) || !IsPowOf2(sz.m_height))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "CL impl only support 2^n width/height.");
        return Status::ERROR;
    }

    return Status::OK;
}

Status InverseDftCL::Initialize()
{
    if (InverseDftImpl::Initialize() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "InverseDftImpl::Initialize() failed");
        return Status::ERROR;
    }

    Sizes3 sz       = m_src->GetSizes();
    Sizes3 dst_size = m_dst->GetSizes();
    DT_S32 width    = sz.m_width;
    DT_S32 height   = sz.m_height;

    // Cpu get exp tables and index table
    DT_S32 idx_total_bytes = Max(width, height) * sizeof(DT_U16);
    m_exp_total_bytes      = Max(width, height) * sizeof(DT_F32);
    m_local_buffer_bytes   = Max(width, height) * 2 * sizeof(DT_F32);

    DT_S32 buffer_sz = AURA_ALIGN(idx_total_bytes + m_exp_total_bytes, 64);
    m_buffer_pitch   = buffer_sz;

    // query device info
    auto   device             = m_ctx->GetCLEngine()->GetCLRuntime()->GetDevice();
    DT_S32 max_local_mem_size = device->getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();

    if (m_exp_total_bytes + m_local_buffer_bytes > max_local_mem_size)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "CL Local memory is not large enough for do fft.");
        return Status::ERROR;
    }

    m_param = Mat(m_ctx, ElemType::U8, {2, buffer_sz});
    if (!m_param.IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Create m_param mat failed.");
        return Status::ERROR;
    }

    // 0. compute m_param in cpu
    {
        DT_U16 *idx_table = m_param.Ptr<DT_U16>(0);
        auto exp_table = reinterpret_cast<std::complex<DT_F32>*>(idx_table + width);
        GetReverseIndex(idx_table, width);
        GetDftExpTable<1>(exp_table, width);

        idx_table = m_param.Ptr<DT_U16>(1);
        exp_table = reinterpret_cast<std::complex<DT_F32>*>(idx_table + height);
        GetReverseIndex(idx_table, height);
        GetDftExpTable<1>(exp_table, height);
    }

    // 1. init cl_mem
    m_cl_src = CLMem::FromArray(m_ctx, *m_src, CLMemParam(CL_MEM_READ_ONLY));
    if (!m_cl_src.IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "cl_src is invalid");
        return Status::ERROR;
    }

    m_cl_dst = CLMem::FromArray(m_ctx, *m_dst, CLMemParam(CL_MEM_READ_WRITE));
    if (!m_cl_dst.IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "cl_dst is invalid");
        return Status::ERROR;
    }

    m_cl_param = CLMem::FromArray(m_ctx, m_param, CLMemParam(CL_MEM_READ_ONLY));
    if (!m_cl_param.IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "cl_param is invalid");
        return Status::ERROR;
    }

    DT_BOOL is_dst_c1 = (1 == dst_size.m_channel) ? DT_TRUE : DT_FALSE;

    if (is_dst_c1 || (ElemType::F32 != m_dst->GetElemType()))
    {
        m_cl_mid = CLMem::FromArray(m_ctx, m_mid, CLMemParam(CL_MEM_READ_WRITE));
        if (!m_cl_mid.IsValid())
        {
            AURA_ADD_ERROR_STRING(m_ctx, "m_cl_mid is invalid");
            return Status::ERROR;
        }
    }
    else
    {
        m_cl_mid = m_cl_dst;
    }

    // 2. init kernel
    m_cl_kernels = GetCLKernels(m_ctx, m_src->GetElemType(), m_dst->GetElemType(), m_with_scale, is_dst_c1);
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

    if (m_cl_param.Sync(CLMemSyncType::WRITE) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "m_cl_param sync start failed");
        return Status::ERROR;
    }

    return Status::OK;
}

Status InverseDftCL::DeInitialize()
{
    if (InverseDftImpl::DeInitialize() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "DftImpl::DeInitialize() failed");
        return Status::ERROR;
    }

    m_cl_src.Release();
    m_cl_param.Release();
    m_cl_dst.Release();
    m_param.Release();
    m_cl_mid.Release();

    m_cl_kernels[0].DeInitialize();
    m_cl_kernels[1].DeInitialize();

    return Status::OK;
}

Status InverseDftCL::Run()
{
    DT_S32 istep    = m_src->GetRowPitch() / ElemTypeSize(m_src->GetElemType());
    DT_S32 ostep    = m_dst->GetRowPitch() / ElemTypeSize(m_dst->GetElemType());
    DT_S32 step_mid = m_cl_mid.GetRowPitch() / ElemTypeSize(m_cl_mid.GetElemType());
    DT_S32 height   = m_src->GetSizes().m_height;
    DT_S32 width    = m_src->GetSizes().m_width;

    // 1. get cl_global_size and cl_local_size
    cl::NDRange cl_global_size[2];
    cl::NDRange cl_local_size[2];
    GetCLSize(m_ctx, m_cl_kernels, width, height, cl_global_size, cl_local_size);

    cl::Event cl_event[2];
    cl_int cl_ret = CL_SUCCESS;
    Status ret    = Status::ERROR;

    // 2. CL run
    cl_ret = m_cl_kernels[0].Run<cl::Buffer, DT_S32, cl::Buffer, DT_S32, cl::Buffer, DT_S32, cl::LocalSpaceArg, cl::LocalSpaceArg, DT_S32, DT_S32>(
                                 m_cl_src.GetCLMemRef<cl::Buffer>(), istep,
                                 m_cl_mid.GetCLMemRef<cl::Buffer>(), step_mid,
                                 m_cl_param.GetCLMemRef<cl::Buffer>(), m_buffer_pitch,
                                 cl::Local(m_local_buffer_bytes * 1),
                                 cl::Local(m_exp_total_bytes),
                                 width, height,
                                 cl_global_size[0],
                                 cl_local_size[0],
                                 &cl_event[0]);
    if (cl_ret != CL_SUCCESS)
    {
        AURA_ADD_ERROR_STRING(m_ctx, ("Run m_cl_kernel main failed : " + GetCLErrorInfo(cl_ret)).c_str());
        goto EXIT;
    }

    cl_ret = m_cl_kernels[1].Run<cl::Buffer, DT_S32, cl::Buffer, DT_S32, cl::Buffer, DT_S32 ,cl::LocalSpaceArg, cl::LocalSpaceArg, DT_S32, DT_S32>(
                                 m_cl_mid.GetCLMemRef<cl::Buffer>(), step_mid,
                                 m_cl_param.GetCLMemRef<cl::Buffer>(), m_buffer_pitch,
                                 m_cl_dst.GetCLMemRef<cl::Buffer>(), ostep,
                                 cl::Local(m_local_buffer_bytes * 1),
                                 cl::Local(m_exp_total_bytes),
                                 width, height,
                                 cl_global_size[1],
                                 cl_local_size[1],
                                 &cl_event[1], {cl_event[0]});

    if (cl_ret != CL_SUCCESS)
    {
        AURA_ADD_ERROR_STRING(m_ctx, ("cl_kernel run failed : " + GetCLErrorInfo(cl_ret)).c_str());
        goto EXIT;
    }

    // 3. cl wait
    if ((DT_TRUE == m_target.m_data.opencl.profiling) || (m_dst->GetArrayType() != ArrayType::CL_MEMORY))
    {
        cl_ret = cl_event[1].wait();
        if (cl_ret != CL_SUCCESS)
        {
            AURA_ADD_ERROR_STRING(m_ctx, ("Wait m_cl_kernel failed : " + GetCLErrorInfo(cl_ret)).c_str());
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
        AURA_ADD_ERROR_STRING(m_ctx, "Sync end fail");
    }

EXIT:
    AURA_RETURN(m_ctx, ret);
}

std::string InverseDftCL::ToString() const
{
    return InverseDftImpl::ToString() + m_profiling_string;
}

std::vector<CLKernel> InverseDftCL::GetCLKernels(Context *ctx, ElemType src_elem_type, ElemType dst_elem_type, DT_S32 with_scale, DT_BOOL is_dst_c1)
{
    std::vector<CLKernel> cl_kernels;
    std::string build_opt = GetCLBuildOptions(ctx, src_elem_type, dst_elem_type, with_scale, is_dst_c1);

    std::string program_name[2] = {"aura_idft_row_process", "aura_idft_col_process"};
    std::string kernel_name[2]  = {"IdftRowProcess", "IdftColProcess"};
    cl_kernels.push_back({ctx, program_name[0], kernel_name[0], "", build_opt});
    cl_kernels.push_back({ctx, program_name[1], kernel_name[1], "", build_opt});

    return cl_kernels;
}
} // namespace aura