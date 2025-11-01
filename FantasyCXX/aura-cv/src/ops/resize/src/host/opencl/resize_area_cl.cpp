#include "resize_impl.hpp"
#include "aura/runtime/opencl.h"
#include "aura/runtime/logger.h"
namespace aura
{

static Status GetKernelParamStr(Context *ctx, MI_S32 iwidth, MI_S32 iheight, MI_S32 owidth, MI_S32 oheight,
                                MI_S32 &elem_x, MI_S32 &elem_y)
{
    if ((iwidth == 2 * owidth) && (iheight == 2 * oheight))
    {
        elem_y = 1;
        elem_x = 4;
    }
    else if ((owidth == 2 * iwidth) && (oheight == 2 * iheight))
    {
        elem_y = 2;
        elem_x = 4;
    }
    else if ((iwidth == 4 * owidth) && (iheight == 4 * oheight))
    {
        elem_y = 1;
        elem_x = 4;
    }
    else if ((owidth == 4 * iwidth) && (oheight == 4 * iheight))
    {
        elem_y = 4;
        elem_x = 4;
    }
    else if ((iwidth < owidth) || (iheight < oheight))
    {
        elem_y = 1;
        elem_x = 4;
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "unsupported scale");
        return Status::ERROR;
    }

    return Status::OK;
}

static Status GetCLBuildOptions(Context *ctx, ElemType elem_type, std::string &build_opt)
{
    const std::vector<std::string> tbl =
    {
                     "U8,     S8,    U16,    S16,   F32,   F16",   // element type
        "Tp",        "uchar,  char,  ushort, short, float, half",  // source type
        "InterType", "ushort, short, uint,   int,   float, float", // internal type
    };

    CLBuildOptions cl_build_opt(ctx, tbl);

    build_opt = cl_build_opt.ToString(elem_type);

    return Status::OK;
}

static Status GetCLName(Context *ctx, MI_S32 iwidth, MI_S32 iheight, MI_S32 owidth, MI_S32 oheight, MI_S32 channel,
                        std::string &program_name, std::string &kernel_name)
{
    if (MI_NULL == ctx)
    {
        return Status::ERROR;
    }

    std::string prog_str, kernel_str;

    if ((iwidth == 2 * owidth) && (iheight == 2 * oheight))
    {
        kernel_str = "DownX2";
        prog_str   = "fast_down_x2";
    }
    else if ((owidth == 2 * iwidth) && (oheight == 2 * iheight))
    {
        kernel_str = "UpX2";
        prog_str   = "fast_up_x2";
    }
    else if ((iwidth == 4 * owidth) && (iheight == 4 * oheight))
    {
        kernel_str = "DownX4";
        prog_str   = "fast_down_x4";
    }
    else if ((owidth == 4 * iwidth) && (oheight == 4 * iheight))
    {
        kernel_str = "UpX4";
        prog_str   = "fast_up_x4";
    }
    else if ((iwidth < owidth) || (iheight < oheight))
    {
        kernel_str = "UpComm";
        prog_str   = "comm_up";
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "unsupported scale");
        return Status::ERROR;
    }

    kernel_name = "ResizeArea" + kernel_str + "C" + std::to_string(channel);

    program_name = "aura_resize_area_" + prog_str + "_c" + std::to_string(channel);

    return Status::OK;
}

static AURA_VOID GetGlobalSize(MI_S32 width, MI_S32 height, MI_S32 elem_x, MI_S32 elem_y, cl::NDRange &range)
{
    range = cl::NDRange((width + elem_x - 1) / elem_x, height / elem_y);
}

ResizeAreaCL::ResizeAreaCL(Context *ctx, const OpTarget &target) : ResizeCL(ctx, target)
{}

Status ResizeAreaCL::Initialize()
{
    if (ResizeCL::Initialize() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "ResizeCL::Initialize() failed");
        return Status::ERROR;
    }

    MI_S32 iwidth  = m_src->GetSizes().m_width;
    MI_S32 iheight = m_src->GetSizes().m_height;
    MI_S32 channel = m_src->GetSizes().m_channel;
    MI_S32 owidth  = m_dst->GetSizes().m_width;
    MI_S32 oheight = m_dst->GetSizes().m_height;

    if (GetKernelParamStr(m_ctx, iwidth, iheight, owidth, oheight, m_elem_x, m_elem_y) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "GetKernelParamStr failed");
        return Status::ERROR;
    }

    // 2.init kernel
    m_cl_kernels = GetCLKernels(m_ctx, m_src->GetElemType(), iwidth, iheight, owidth, oheight, channel);

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

Status ResizeAreaCL::Run()
{
    MI_S32 iwidth  = m_src->GetSizes().m_width;
    MI_S32 iheight = m_src->GetSizes().m_height;
    MI_S32 istep   = m_src->GetRowPitch() / ElemTypeSize(m_src->GetElemType());
    MI_S32 owidth  = m_dst->GetSizes().m_width;
    MI_S32 oheight = m_dst->GetSizes().m_height;
    MI_S32 ostep   = m_dst->GetRowPitch() / ElemTypeSize(m_dst->GetElemType());
    MI_F32 scale_x = static_cast<MI_F32>(iwidth) / owidth;
    MI_F32 scale_y = static_cast<MI_F32>(iheight) / oheight;

    std::shared_ptr<CLRuntime> cl_rt = m_ctx->GetCLEngine()->GetCLRuntime();

    // 1. get center_area and global_size
    cl::NDRange global_size;
    GetGlobalSize(owidth, oheight, m_elem_x, m_elem_y, global_size);

    cl::Event event;
    cl_int cl_ret = CL_SUCCESS;
    Status ret = Status::ERROR;
    Status ret_sync = Status::ERROR;

    // 2. opencl run
    cl_ret = m_cl_kernels[0].Run<cl::Buffer, MI_S32, cl::Buffer, MI_S32, MI_F32, MI_F32, MI_S32, MI_S32, MI_S32, MI_S32>(
                         m_cl_src.GetCLMemRef<cl::Buffer>(), istep,
                         m_cl_dst.GetCLMemRef<cl::Buffer>(), ostep,
                         scale_x, scale_y, iwidth, iheight, owidth, oheight,
                         global_size, cl_rt->GetCLDefaultLocalSize(m_cl_kernels[0].GetMaxGroupSize(), global_size),
                         &event);
    if (cl_ret != CL_SUCCESS)
    {
        AURA_ADD_ERROR_STRING(m_ctx, ("Run cl_kernel main failed : " + GetCLErrorInfo(cl_ret)).c_str());
        goto EXIT;
    }

    // 3. cl wait
    if (m_target.m_data.opencl.profiling || (m_dst->GetArrayType() != ArrayType::CL_MEMORY))
    {
        cl_ret = event.wait();
        if (cl_ret != CL_SUCCESS)
        {
            AURA_ADD_ERROR_STRING(m_ctx, ("Wait cl_kernel failed : " + GetCLErrorInfo(cl_ret)).c_str());
            goto EXIT;
        }

        if (m_target.m_data.opencl.profiling)
        {
            m_profiling_string = " " + GetCLProfilingInfo(m_cl_kernels[0].GetKernelName(), event);
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

Status ResizeAreaCL::DeInitialize()
{
    m_cl_kernels.clear();

    if (ResizeCL::DeInitialize() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "ResizeCL::DeInitialize() failed");
        return Status::ERROR;
    }

    return Status::OK;
}

std::vector<CLKernel> ResizeAreaCL::GetCLKernels(Context *ctx, ElemType elem_type, MI_S32 iwidth, MI_S32 iheight,
                                                 MI_S32 owidth, MI_S32 oheight, MI_S32 channel)
{
    std::vector<CLKernel> cl_kernels;
    std::string build_opt;

    if (GetCLBuildOptions(ctx, elem_type, build_opt) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetCLBuildOptions failed");
        return cl_kernels;
    }

    std::string kernel_name, program_name;

    if (GetCLName(ctx, iwidth, iheight, owidth, oheight, channel, program_name, kernel_name) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetCLName failed");
        return cl_kernels;
    }

    cl_kernels.emplace_back(ctx, program_name, kernel_name, "", build_opt);

    return cl_kernels;
}

} // namespace aura