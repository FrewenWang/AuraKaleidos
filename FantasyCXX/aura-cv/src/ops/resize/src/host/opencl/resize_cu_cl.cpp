#include "resize_impl.hpp"
#include "aura/runtime/opencl.h"
#include "aura/runtime/logger.h"

namespace aura
{

static Status GetKernelParams(MI_S32 iwidth, MI_S32 iheight, MI_S32 owidth, MI_S32 oheight, MI_S32 &border)
{
    MI_S32 left = 0, right = 0;
    MI_F32 scale_x = static_cast<MI_F32>(iwidth) / owidth;

    if (!(((iwidth == 2 * owidth) && (iheight == 2 * oheight))) &&
        !((iwidth == 4 * owidth) && (iheight == 4 * oheight)))
    {
        left = right = 0;

        for (MI_S32 i = 0; i < owidth; i++)
        {
            MI_F32 fx = (i + 0.5f) * scale_x - 0.5f;
            MI_S32 sx = Floor(fx) - 1;

            if (sx < 0)
            {
                left++; // count left side border
            }
            else if ((sx + 3) >= iwidth)
            {
                right++; // count right side border
            }
        }

        border = Max(left, right);
    }

    return Status::OK;
}

static Status GetCLBuildOptions(Context *ctx, ElemType elem_type, std::string &build_opt)
{
    CLBuildOptions cl_build_opt(ctx);
    cl_build_opt.AddOption("Tp", CLTypeString(elem_type));
    build_opt = cl_build_opt.ToString();

    return Status::OK;
}

static Status GetCLName(MI_S32 iwidth, MI_S32 iheight, MI_S32 owidth, MI_S32 oheight, MI_S32 channel,
                        std::vector<std::string> &program_names, std::vector<std::string> &kernel_names)
{
    std::string kernel_str, program_str;

    if ((iwidth == 2 * owidth) && (iheight == 2 * oheight)) //down_x2
    {
        kernel_str = "DownX2";
        program_str = "fast_down_x2";
    }
    else if ((iwidth == 4 * owidth) && (iheight == 4 * oheight)) //down_x4
    {
        kernel_str = "DownX4";
        program_str = "fast_down_x4";
    }
    else
    {
        program_str = "main_comm";
    }

    // e.g. ResizeCuMainC2
    kernel_names.emplace_back("ResizeCuMain" + kernel_str + "C" + std::to_string(channel));
    kernel_names.emplace_back("ResizeCuRemainC" + std::to_string(channel));

    program_names.emplace_back("aura_resize_cu_" + program_str + "_c" + std::to_string(channel));
    program_names.emplace_back("aura_resize_cu_remain_comm_c" + std::to_string(channel));

    return Status::OK;
}

static AURA_VOID GetGlobalSize(MI_S32 width, MI_S32 height, cl::NDRange *range, MI_S32 border)
{
    range[0] = cl::NDRange(width - 2 * border, height);
    range[1] = cl::NDRange(border, height); //calc two output once
}

ResizeCuCL::ResizeCuCL(Context *ctx, const OpTarget &target) : ResizeCL(ctx, target)
{}

Status ResizeCuCL::Initialize()
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

    m_border = 0;
    if (GetKernelParams(iwidth, iheight, owidth, oheight, m_border) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "GetKernelParams failed");
        return Status::ERROR;
    }

    // 2. init kernel
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

Status ResizeCuCL::Run()
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

    // 1. global_size
    cl::NDRange global_size[2];
    GetGlobalSize(owidth, oheight, global_size, m_border);

    cl::Event event[2];
    cl_int cl_ret = CL_SUCCESS;
    Status ret = Status::ERROR;
    Status ret_sync = Status::ERROR;

    // 2. opencl run
    cl_ret = m_cl_kernels[0].Run<cl::Buffer, MI_S32, cl::Buffer, MI_S32, MI_F32, MI_F32, MI_S32, MI_S32, MI_S32, MI_S32, MI_S32, MI_S32, MI_S32>(
                                m_cl_src.GetCLMemRef<cl::Buffer>(), istep,
                                m_cl_dst.GetCLMemRef<cl::Buffer>(), ostep,
                                scale_x, scale_y, m_border,
                                iwidth, iheight, owidth, oheight,
                                (MI_S32)(global_size[0].get()[0]),
                                (MI_S32)(global_size[0].get()[1]),
                                global_size[0], cl_rt->GetCLDefaultLocalSize(m_cl_kernels[0].GetMaxGroupSize(), global_size[0]),
                                &event[0]);

    if (cl_ret != CL_SUCCESS)
    {
        AURA_ADD_ERROR_STRING(m_ctx, ("Run cl_kernel main failed : " + GetCLErrorInfo(cl_ret)).c_str());
        goto EXIT;
    }

    if (m_border)
    {
        cl_ret |= m_cl_kernels[1].Run<cl::Buffer, MI_S32, cl::Buffer, MI_S32, MI_F32, MI_F32, MI_S32, MI_S32, MI_S32, MI_S32, MI_S32>(
                                     m_cl_src.GetCLMemRef<cl::Buffer>(), istep,
                                     m_cl_dst.GetCLMemRef<cl::Buffer>(), ostep,
                                     scale_x, scale_y, iwidth, iheight, owidth,
                                     (MI_S32)(global_size[1].get()[0]),
                                     (MI_S32)(global_size[1].get()[1]),
                                     global_size[1], cl_rt->GetCLDefaultLocalSize(m_cl_kernels[1].GetMaxGroupSize(), global_size[1]),
                                     &event[1], {event[0]});
        if (cl_ret != CL_SUCCESS)
        {
            AURA_ADD_ERROR_STRING(m_ctx, ("Run cl_kernel main failed : " + GetCLErrorInfo(cl_ret)).c_str());
            goto EXIT;
        }
    }

    // 3. cl wait
    if (m_target.m_data.opencl.profiling || (m_dst->GetArrayType() != ArrayType::CL_MEMORY))
    {
        if (m_border)
        {
            cl_ret = event[1].wait();
            if (cl_ret != CL_SUCCESS)
            {
                AURA_ADD_ERROR_STRING(m_ctx, ("Wait cl_kernel failed : " + GetCLErrorInfo(cl_ret)).c_str());
                goto EXIT;
            }

            if (m_target.m_data.opencl.profiling)
            {
                m_profiling_string = " " + GetCLProfilingInfo(m_cl_kernels[0].GetKernelName(), event[0]) +
                                           GetCLProfilingInfo(m_cl_kernels[1].GetKernelName(), event[1]);
            }
        }
        else
        {
            cl_ret = event[0].wait();
            if (cl_ret != CL_SUCCESS)
            {
                AURA_ADD_ERROR_STRING(m_ctx, ("Wait cl_kernel failed : " + GetCLErrorInfo(cl_ret)).c_str());
                goto EXIT;
            }

            if (m_target.m_data.opencl.profiling)
            {
                m_profiling_string = " " + GetCLProfilingInfo(m_cl_kernels[0].GetKernelName(), event[0]);
            }
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

Status ResizeCuCL::DeInitialize()
{
    m_cl_kernels.clear();

    if (ResizeCL::DeInitialize() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "ResizeCL::DeInitialize() failed");
        return Status::ERROR;
    }

    return Status::OK;
}

std::vector<CLKernel> ResizeCuCL::GetCLKernels(Context *ctx, ElemType elem_type, MI_S32 iwidth, MI_S32 iheight,
                                               MI_S32 owidth, MI_S32 oheight, MI_S32 channel)
{
    std::vector<CLKernel> cl_kernels;

    std::string build_opt;
    if (GetCLBuildOptions(ctx, elem_type, build_opt) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetCLBuildOptions failed");
        return cl_kernels;
    }

    std::vector<std::string> program_names;
    std::vector<std::string> kernel_names;

    if (GetCLName(iwidth, iheight, owidth, oheight, channel, program_names, kernel_names) != Status::OK)
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