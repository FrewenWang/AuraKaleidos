#include "resize_impl.hpp"
#include "aura/runtime/opencl.h"
#include "aura/runtime/logger.h"

namespace aura
{

static Status GetCLBuildOptions(Context *ctx, ElemType elem_type, DT_S32 channel, std::string &build_opt)
{
    CLBuildOptions cl_build_opt(ctx);
    cl_build_opt.AddOption("Tp", CLTypeString(elem_type));
    cl_build_opt.AddOption("CHANNEL", std::to_string(channel));

    build_opt = cl_build_opt.ToString();

    return Status::OK;
}

static Status GetCLName(std::string &program_name, std::string &kernel_name)
{
    program_name = "aura_resize_nn";
    kernel_name  = "ResizeNn";

    return Status::OK;
}

static DT_VOID GetCLGlobalSize(DT_S32 width, DT_S32 height, cl::NDRange &cl_range)
{
    cl_range = cl::NDRange((width + 3) / 4, height);
}

ResizeNnCL::ResizeNnCL(Context *ctx, const OpTarget &target) : ResizeCL(ctx, target)
{}

Status ResizeNnCL::Initialize()
{
    if (ResizeCL::Initialize() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "ResizeCL::Initialize() failed");
        return Status::ERROR;
    }

    // 2. init kernel
    m_cl_kernels = GetCLKernels(m_ctx, m_cl_src.GetElemType(), m_cl_src.GetSizes().m_channel);
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

Status ResizeNnCL::Run()
{
    Status ret = Status::ERROR;
    Status ret_sync = Status::ERROR;

    Sizes3 src_sz = m_cl_src.GetSizes();
    Sizes3 dst_sz = m_cl_dst.GetSizes();

    DT_S32 iwidth  = src_sz.m_width;
    DT_S32 iheight = src_sz.m_height;
    DT_S32 owidth  = dst_sz.m_width;
    DT_S32 oheight = dst_sz.m_height;
    DT_S32 istep   = m_src->GetRowPitch() / ElemTypeSize(m_src->GetElemType());
    DT_S32 ostep   = m_dst->GetRowPitch() / ElemTypeSize(m_dst->GetElemType());

    DT_F32 scale_x = DT_F32(owidth) / DT_F32(iwidth);
    DT_F32 scale_y = DT_F32(oheight) / DT_F32(iheight);

    DT_F32 inv_scale_x = 1.0f / scale_x;
    DT_F32 inv_scale_y = 1.0f / scale_y;

    DT_S32 x_max = Min(Ceil((iwidth - 1) * scale_x), owidth);
    DT_S32 y_max = Min(Ceil((iheight - 1) * scale_y), oheight);

    std::shared_ptr<CLRuntime> cl_rt = m_ctx->GetCLEngine()->GetCLRuntime();

    // 1. get center_area and cl_global_size
    cl::NDRange cl_global_size;
    GetCLGlobalSize(owidth, oheight, cl_global_size);
    cl::NDRange cl_local_size = cl_rt->GetCLDefaultLocalSize(m_cl_kernels[0].GetMaxGroupSize(), cl_global_size);

    cl::Event cl_event;
    cl_int cl_ret = CL_SUCCESS;

    // 2. opencl run
     cl_ret = m_cl_kernels[0].Run<cl::Buffer, DT_S32, cl::Buffer, DT_S32, DT_S32, DT_S32, DT_S32, DT_S32, DT_F32, DT_F32, DT_S32, DT_S32>(
                                 m_cl_src.GetCLMemRef<cl::Buffer>(), istep,
                                 m_cl_dst.GetCLMemRef<cl::Buffer>(), ostep,
                                 iwidth, iheight, owidth, oheight,
                                 inv_scale_x, inv_scale_y, x_max, y_max,
                                 cl_global_size, cl_local_size, &(cl_event));

    if (cl_ret != CL_SUCCESS)
    {
        AURA_ADD_ERROR_STRING(m_ctx, ("Run cl_kernel main failed : " + GetCLErrorInfo(cl_ret)).c_str());
        goto EXIT;
    }

    // 3. cl wait
    if (m_target.m_data.opencl.profiling || (m_dst->GetArrayType() != ArrayType::CL_MEMORY))
    {
        cl_ret = cl_event.wait();
        if (cl_ret != CL_SUCCESS)
        {
            AURA_ADD_ERROR_STRING(m_ctx, ("Wait cl_kernel failed : " + GetCLErrorInfo(cl_ret)).c_str());
            goto EXIT;
        }
        if (m_target.m_data.opencl.profiling)
        {
            m_profiling_string = " " + GetCLProfilingInfo(m_cl_kernels[0].GetKernelName(), cl_event);
        }
    }

    ret_sync = m_cl_dst.Sync(CLMemSyncType::READ);
    if (ret_sync != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Sync end fail");
    }

    ret = Status::OK;

EXIT:
    AURA_RETURN(m_ctx, (ret | ret_sync));
}

Status ResizeNnCL::DeInitialize()
{
    m_cl_kernels.clear();

    if (ResizeCL::DeInitialize() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "ResizeCL::DeInitialize() failed");
        return Status::ERROR;
    }

    return Status::OK;
}

std::vector<CLKernel> ResizeNnCL::GetCLKernels(Context *ctx, ElemType elem_type, DT_S32 channel)
{
    std::vector<CLKernel> cl_kernels;

    std::string build_opt;
    if (GetCLBuildOptions(ctx, elem_type, channel, build_opt) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetCLBuildOptions failed");
        return cl_kernels;
    }

    std::string program_name;
    std::string kernel_name;

    if (GetCLName(program_name, kernel_name) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetCLName failed");
        return cl_kernels;
    }

    cl_kernels.emplace_back(ctx, program_name, kernel_name, "", build_opt);

    return cl_kernels;
}

} // namespace aura