#include "grid_dft_impl.hpp"
#include "dft_impl.hpp"
#include "aura/runtime/mat.h"
#include "aura/runtime/opencl.h"
#include "aura/runtime/logger.h"

namespace aura
{

static AURA_VOID GetCLName(MI_S32 grid_len, std::string &program_name, std::string &kernel_name, MI_BOOL is_inverse)
{
    if (is_inverse)
    {
        program_name = "aura_grid_idft_" + std::to_string(grid_len) + "x" + std::to_string(grid_len);
        kernel_name = "GridIDft" + std::to_string(grid_len) + "x" + std::to_string(grid_len);
    }
    else
    {
        program_name = "aura_grid_dft_" + std::to_string(grid_len) + "x" + std::to_string(grid_len);
        kernel_name = "GridDft" + std::to_string(grid_len) + "x" + std::to_string(grid_len);
    }
}

static std::string GetCLBuildOptions(Context *ctx, ElemType elem_type, MI_BOOL with_scale, MI_BOOL save_real_only)
{
    MI_S32 int_with_scale = static_cast<MI_S32>(with_scale);
    MI_S32 int_real_only  = static_cast<MI_S32>(save_real_only);

    CLBuildOptions cl_build_opt(ctx);
    cl_build_opt.AddOption("Tp",                CLTypeString(elem_type));
    cl_build_opt.AddOption("WITH_SCALE",        std::to_string(int_with_scale));
    cl_build_opt.AddOption("SAVE_REAL_ONLY",    std::to_string(int_real_only));
    cl_build_opt.AddOption("MAX_CONSTANT_SIZE", ctx->GetCLEngine()->GetCLRuntime()->GetCLMaxConstantSizeString(256));

    return cl_build_opt.ToString();
}

static AURA_VOID GetCLSize(MI_S32 width, MI_S32 height, MI_S32 grid_len, cl::NDRange &cl_global_size, cl::NDRange &cl_local_size)
{
    MI_S32 local_size_x = 1;
    MI_S32 local_size_y = (32 == grid_len) ? grid_len : (grid_len >> 1);

    MI_S32 group_size_x = width  / grid_len;
    MI_S32 group_size_y = height / grid_len;

    MI_S32 global_size_x = local_size_x * group_size_x;
    MI_S32 global_size_y = local_size_y * group_size_y;

    cl_global_size = cl::NDRange(global_size_x, global_size_y);
    cl_local_size  = cl::NDRange(local_size_x,  local_size_y);
}

GridDftCL::GridDftCL(Context *ctx, const OpTarget &target) : GridDftImpl(ctx, target), m_local_buffer_size(0), m_profiling_string()
{}

Status GridDftCL::SetArgs(const Array *src, Array *dst, MI_S32 grid_len)
{
    if (GridDftImpl::SetArgs(src, dst, grid_len) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "GridDftImpl::SetArgs failed");
        return Status::ERROR;
    }

    if (ElemType::S32 == src->GetElemType() || ElemType::U32 == src->GetElemType() || ElemType::F64 == src->GetElemType())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "currently src does not support MI_S32/MI_U32/MI_F64 type.");
        return Status::ERROR;
    }

    Sizes3 sz = src->GetSizes();

    if (sz.m_height < 32 || sz.m_width < 32)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "size is too small");
        return Status::ERROR;
    }

    if (grid_len < 4 || grid_len > 32)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "grid_len must be in [4, 32]");
        return Status::ERROR;
    }

    return Status::OK;
}

Status GridDftCL::Initialize()
{
    if (GridDftImpl::Initialize() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "GridDftImpl::Initialize() failed");
        return Status::ERROR;
    }

    //0. cal exp table
    m_local_buffer_size = m_grid_len * m_grid_len * sizeof(MI_F32) * 2;
    MI_S32 param_size = m_grid_len * sizeof(MI_F32) * 2;

    m_param = Mat(m_ctx, ElemType::U8, {1, param_size});
    if (!m_param.IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "m_param is invalid");
        return Status::ERROR;
    }

    std::complex<MI_F32> *exp_table = (std::complex<MI_F32> *)m_param.GetData();
    GetDftExpTable<0>(exp_table, m_grid_len);

    //1. init cl_mem
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

    // 2. init kernel
    ElemType elem_type = (m_src->GetElemType());

    m_cl_kernels = GetCLKernels(m_ctx, elem_type, m_grid_len);
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

Status GridDftCL::DeInitialize()
{
    m_cl_src.Release();
    m_cl_dst.Release();
    m_cl_param.Release();
    m_param.Release();

    m_cl_kernels[0].DeInitialize();

    if (GridDftImpl::DeInitialize() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "GridDftImpl::DeInitialize() failed");
        return Status::ERROR;
    }

    return Status::OK;
}

Status GridDftCL::Run()
{
    Sizes3 sz     = m_dst->GetSizes();
    MI_S32 width  = sz.m_width;
    MI_S32 height = sz.m_height;
    MI_S32 istep  = m_src->GetRowPitch() / ElemTypeSize(m_src->GetElemType());
    MI_S32 ostep  = m_dst->GetRowPitch() / ElemTypeSize(m_dst->GetElemType());

    cl::Event event;
    cl_int cl_ret = CL_SUCCESS;
    Status ret = Status::ERROR;

    // 1. get cl_global_size and opencl run
    if (m_grid_len <= 8)  // grid_len is 4 or 8
    {
        std::shared_ptr<CLRuntime> cl_rt = m_ctx->GetCLEngine()->GetCLRuntime();
        cl::NDRange cl_global_size = cl::NDRange(width / m_grid_len, height / m_grid_len);
        // cl kernel run
        cl_ret = m_cl_kernels[0].Run<cl::Buffer, MI_S32, cl::Buffer, MI_S32, MI_S32, MI_S32>(
                                     m_cl_src.GetCLMemRef<cl::Buffer>(), istep,
                                     m_cl_dst.GetCLMemRef<cl::Buffer>(), ostep,
                                     width, height,
                                     cl_global_size,
                                     cl_rt->GetCLDefaultLocalSize(m_cl_kernels[0].GetMaxGroupSize(), cl_global_size),
                                     &event);
        if (cl_ret != CL_SUCCESS)
        {
            AURA_ADD_ERROR_STRING(m_ctx, ("Run cl_kernel failed : " + GetCLErrorInfo(cl_ret)).c_str());
            goto EXIT;
        }
    }
    else // grid_len is 16 or 32
    {
        // get global_size and local_size
        cl::NDRange cl_global_size, cl_local_size;
        GetCLSize(width, height, m_grid_len, cl_global_size, cl_local_size);
        // cl kernel run
        cl_ret = m_cl_kernels[0].Run<cl::Buffer, MI_S32, cl::Buffer, MI_S32, cl::Buffer, cl::LocalSpaceArg, MI_S32, MI_S32>(
                                     m_cl_src.GetCLMemRef<cl::Buffer>(), istep,
                                     m_cl_dst.GetCLMemRef<cl::Buffer>(), ostep,
                                     m_cl_param.GetCLMemRef<cl::Buffer>(),
                                     cl::Local(m_local_buffer_size),
                                     width, height,
                                     cl_global_size,
                                     cl_local_size,
                                     &event);
        if (cl_ret != CL_SUCCESS)
        {
            AURA_ADD_ERROR_STRING(m_ctx, ("Run cl_kernel failed : " + GetCLErrorInfo(cl_ret)).c_str());
            goto EXIT;
        }
    }

    // 2. cl event wait
    if ((MI_TRUE == m_target.m_data.opencl.profiling) || (m_dst->GetArrayType() != ArrayType::CL_MEMORY))
    {
        cl_ret = event.wait();
        if (cl_ret != CL_SUCCESS)
        {
            AURA_ADD_ERROR_STRING(m_ctx, ("Wait m_cl_kernel failed : " + GetCLErrorInfo(cl_ret)).c_str());
            goto EXIT;
        }

        if (MI_TRUE == m_target.m_data.opencl.profiling)
        {
            m_profiling_string = " " + GetCLProfilingInfo(m_cl_kernels[0].GetKernelName(), event);
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

std::string GridDftCL::ToString() const
{
    return GridDftImpl::ToString() + m_profiling_string;
}

std::vector<CLKernel> GridDftCL::GetCLKernels(Context *ctx, ElemType elem_type, MI_S32 grid_len)
{
    std::vector<CLKernel> cl_kernels;
    std::string build_opt = GetCLBuildOptions(ctx, elem_type, MI_FALSE, MI_FALSE);

    std::string program_name, kernel_name;
    GetCLName(grid_len, program_name, kernel_name, MI_FALSE);

    cl_kernels.push_back({ctx, program_name, kernel_name, "", build_opt});

    return cl_kernels;
}

GridIDftCL::GridIDftCL(Context *ctx, const OpTarget &target) : GridIDftImpl(ctx, target), m_local_buffer_size(0), m_profiling_string()
{}

Status GridIDftCL::SetArgs(const Array *src, Array *dst, MI_S32 grid_len, MI_BOOL with_scale)
{
    if (GridIDftImpl::SetArgs(src, dst, grid_len, with_scale) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "GridDftImpl::SetArgs failed");
        return Status::ERROR;
    }

    if (ElemType::F64 == dst->GetElemType())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "currently dst does not support MI_F64 type.");
        return Status::ERROR;
    }

    Sizes3 sz = src->GetSizes();
    if (sz.m_height < 32 || sz.m_width < 32)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "size is too small");
        return Status::ERROR;
    }

    if (grid_len < 4 || grid_len > 32)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "grid_len must be in [4, 32]");
        return Status::ERROR;
    }

    return Status::OK;
}

Status GridIDftCL::Initialize()
{
    if (GridIDftImpl::Initialize() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "GridIDftImpl::Initialize() failed");
        return Status::ERROR;
    }

    MI_BOOL save_real_only = (1 == m_dst->GetSizes().m_channel);

    // cal exp table
    m_local_buffer_size = m_grid_len * m_grid_len * sizeof(MI_F32) * 2;
    MI_S32 param_size = m_grid_len * sizeof(MI_F32) * 2;

    m_param = Mat(m_ctx, ElemType::U8, {1, param_size});
    if (!m_param.IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "m_param is invalid");
        return Status::ERROR;
    }

    std::complex<MI_F32> *exp_table = (std::complex<MI_F32> *)m_param.GetData();
    GetDftExpTable<1>(exp_table, m_grid_len);

    // init cl_mem
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

    // init kernel
    ElemType elem_type = (m_dst->GetElemType());
    m_cl_kernels = GetCLKernels(m_ctx, elem_type, m_grid_len, m_with_scale, save_real_only);
    if (CheckCLKernels(m_ctx, m_cl_kernels) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "CheckCLKernels failed");
        return Status::ERROR;
    }

    // sync start
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

Status GridIDftCL::DeInitialize()
{
    if (GridIDftImpl::DeInitialize() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "GridIDftImpl::DeInitialize() failed");
        return Status::ERROR;
    }

    m_cl_src.Release();
    m_cl_dst.Release();
    m_cl_param.Release();
    m_param.Release();

    m_cl_kernels[0].DeInitialize();

    return Status::OK;
}

Status GridIDftCL::Run()
{
    Sizes3 sz     = m_dst->GetSizes();
    MI_S32 width  = sz.m_width;
    MI_S32 height = sz.m_height;
    MI_S32 istep  = m_src->GetRowPitch() / ElemTypeSize(m_src->GetElemType());
    MI_S32 ostep  = m_dst->GetRowPitch() / ElemTypeSize(m_dst->GetElemType());

    cl::Event event;
    cl_int cl_ret = CL_SUCCESS;
    Status ret    = Status::ERROR;

    if (4 == m_grid_len || 8 == m_grid_len)
    {
        std::shared_ptr<CLRuntime> cl_rt = m_ctx->GetCLEngine()->GetCLRuntime();
        cl::NDRange global_size = cl::NDRange(width / m_grid_len, height / m_grid_len);
        // cl kernel run
        cl_ret = m_cl_kernels[0].Run<cl::Buffer, MI_S32, cl::Buffer, MI_S32, MI_S32, MI_S32>(
                                     m_cl_src.GetCLMemRef<cl::Buffer>(), istep,
                                     m_cl_dst.GetCLMemRef<cl::Buffer>(), ostep,
                                     width, height,
                                     global_size,
                                     cl_rt->GetCLDefaultLocalSize(m_cl_kernels[0].GetMaxGroupSize(), global_size),
                                     &event);
        if (cl_ret != CL_SUCCESS)
        {
            AURA_ADD_ERROR_STRING(m_ctx, ("Run cl_kernel failed : " + GetCLErrorInfo(cl_ret)).c_str());
            goto EXIT;
        }
    }
    else
    {
        // get global_size and local_size
        cl::NDRange global_size, local_size;
        GetCLSize(width, height, m_grid_len, global_size, local_size);
        // cl kernel run
        cl_ret = m_cl_kernels[0].Run<cl::Buffer, MI_S32, cl::Buffer, MI_S32, cl::Buffer, cl::LocalSpaceArg, MI_S32, MI_S32>(
                                     m_cl_src.GetCLMemRef<cl::Buffer>(), istep,
                                     m_cl_dst.GetCLMemRef<cl::Buffer>(), ostep,
                                     m_cl_param.GetCLMemRef<cl::Buffer>(),
                                     cl::Local(m_local_buffer_size),
                                     width, height,
                                     global_size,
                                     local_size,
                                     &event);
        if (cl_ret != CL_SUCCESS)
        {
            AURA_ADD_ERROR_STRING(m_ctx, ("Run cl_kernel failed : " + GetCLErrorInfo(cl_ret)).c_str());
            goto EXIT;
        }
    }

    // cl event wait
    if ((MI_TRUE == m_target.m_data.opencl.profiling) || (m_dst->GetArrayType() != ArrayType::CL_MEMORY))
    {
        cl_ret = event.wait();
        if (cl_ret != CL_SUCCESS)
        {
            AURA_ADD_ERROR_STRING(m_ctx, ("Wait m_cl_kernel failed : " + GetCLErrorInfo(cl_ret)).c_str());
            goto EXIT;
        }

        if (MI_TRUE == m_target.m_data.opencl.profiling)
        {
            m_profiling_string = " " + GetCLProfilingInfo(m_cl_kernels[0].GetKernelName(), event);
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

std::string GridIDftCL::ToString() const
{
    return GridIDftImpl::ToString() + m_profiling_string;
}

std::vector<CLKernel> GridIDftCL::GetCLKernels(Context *ctx, ElemType elem_type, MI_S32 grid_len, MI_S32 with_scale, MI_BOOL save_real_only)
{
    std::vector<CLKernel> cl_kernels;
    std::string build_opt = GetCLBuildOptions(ctx, elem_type, with_scale, save_real_only);

    std::string program_name, kernel_name;
    GetCLName(grid_len, program_name, kernel_name, MI_TRUE);

    cl_kernels.push_back({ctx, program_name, kernel_name, "", build_opt});

    return cl_kernels;
}

} // namespace aura