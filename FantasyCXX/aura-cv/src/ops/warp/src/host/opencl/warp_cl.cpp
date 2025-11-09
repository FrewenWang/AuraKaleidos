#include "warp_impl.hpp"
#include "aura/runtime/opencl.h"
#include "aura/runtime/logger.h"

namespace aura
{

static Status GetCLKernelParam(WarpType warp_type, InterpType interp_type, DT_S32 channel, DT_S32 &elem_counts, DT_S32 &elem_height)
{
    if (WarpType::PERSPECTIVE == warp_type && interp_type != InterpType::LINEAR)
    {
        elem_counts = 1;
        elem_height = 2;
    }
    else if (WarpType::PERSPECTIVE == warp_type && InterpType::LINEAR == interp_type)
    {
        elem_height = 1;
        elem_counts = (1 == channel) ? 2 : 1;
    }
    else if (WarpType::AFFINE == warp_type && InterpType::NEAREST == interp_type)
    {
        elem_height = 1;
        elem_counts = (1 == channel) ? 1 : 2;
    }
    else
    {
        elem_height = 1;
        elem_counts = 1;
    }

    return Status::OK;
}

static Status GetCLBuildOptions(Context *ctx, ElemType elem_type, std::string &build_opt, WarpType warp_type,
                                 BorderType border_type, DT_S32 elem_counts, DT_S32 elem_height, DT_S32 channel)
{
    const std::vector<std::string> tbl =
    {
                      "U8,    S8,   U16,    S16,   U32,  S32, F32,   F16",    // elem type
        "Tp",         "uchar, char, ushort, short, uint, int, float, half",   // data type
        "SelectType", "uchar, char, ushort, short, uint, int, int  , ushort", // select type
    };

    CLBuildOptions cl_build_opt(ctx, tbl);
    cl_build_opt.AddOption("CHANNEL", std::to_string(channel));
    cl_build_opt.AddOption("ELEM_HEIGHT", std::to_string(elem_height));
    cl_build_opt.AddOption("ELEM_COUNTS", std::to_string(elem_counts));

    switch (warp_type)
    {
        case WarpType::AFFINE:
        {
            cl_build_opt.AddOption("WARP_AFFINE", "1");
            break;
        }

        case WarpType::PERSPECTIVE:
        {
            cl_build_opt.AddOption("WARP_PERSPECTIVE", "1");
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "Unknown warp type");
            return Status::ERROR;
            break;
        }
    }

    switch (border_type)
    {
        case BorderType::CONSTANT:
        {
            cl_build_opt.AddOption("BORDER_CONSTANT", "1");
            break;
        }
        case BorderType::REPLICATE:
        {
            cl_build_opt.AddOption("BORDER_REPLICATE", "1");
            break;
        }
        case BorderType::REFLECT_101:
        {
            cl_build_opt.AddOption("BORDER_REFLECT_101", "1");
            break;
        }
        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "Unknown border type");
            return Status::ERROR;
            break;
        }
    }

    build_opt = cl_build_opt.ToString(elem_type);

    return Status::OK;
}

static std::string InterpTypeToCLString(InterpType interp_type)
{
    switch (interp_type)
    {
        case InterpType::NEAREST:
            return std::string("Nn");

        case InterpType::LINEAR:
            return std::string("Bn");

        case InterpType::CUBIC:
            return std::string("Cu");

        default:
            break;
    }

    return std::string();
}

static Status GetCLName(Context *ctx, DT_S32 channel, InterpType interp_type, std::string &kernel_name, std::string &program_name)
{
    std::string interp_type_str = InterpTypeToCLString(interp_type);

    if (interp_type_str.empty())
    {
        AURA_ADD_ERROR_STRING(ctx, "Unsupported interp type");
        return Status::ERROR;
    }

    kernel_name                 = "Warp" + interp_type_str + "C" + std::to_string(channel);
    std::transform(interp_type_str.begin(), interp_type_str.begin() + 1, interp_type_str.begin(), ::tolower);
    program_name = "aura_warp_" + interp_type_str + "_c" + std::to_string(channel);

    return Status::OK;
}

WarpCL::WarpCL(Context *ctx, WarpType warp_type, const OpTarget &target) : WarpImpl(ctx, warp_type, target),
                                                                           m_elem_counts(0), m_elem_height(0)
{}

Status WarpCL::SetArgs(const Array *src, const Array *matrix, Array *dst, InterpType interp_type,
                       BorderType border_type, const Scalar &border_value)
{
    if (WarpImpl::SetArgs(src, matrix, dst, interp_type, border_type, border_value) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "WarpImpl::SetArgs failed");
        return Status::ERROR;
    }

    if (ElemType::F64 == src->GetElemType())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Unsupported src type for opencl");
        return Status::ERROR;
    }

    DT_S32 channel = src->GetSizes().m_channel;
    if (channel != 1 && channel != 2)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "channel only support 1/2");
        return Status::ERROR;
    }

    return Status::OK;
}

Status WarpCL::Initialize()
{
    if (WarpImpl::Initialize() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "WarpImpl::Initialize failed");
        return Status::ERROR;
    }

    const Mat *matrix = dynamic_cast<const Mat*>(m_matrix);
    if (DT_NULL == matrix)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "matrix is not mat");
        return Status::ERROR;
    }

    DT_S32 map_height  = m_dst->GetSizes().m_height;
    DT_S32 map_width   = m_dst->GetSizes().m_width;
    DT_S32 map_channel = (WarpType::AFFINE == m_warp_type) ? 2 : 3;

    m_map_x = Mat(m_ctx, ElemType::F32, aura::Sizes3(1, map_width, map_channel));
    m_map_y = Mat(m_ctx, ElemType::F32, aura::Sizes3(1, map_height, map_channel));

    if (InitMapOffset(m_ctx, *matrix, m_map_x, m_map_y, m_warp_type) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "InitMapOffset failed");
        return Status::ERROR;
    }

    DT_S32 channel = m_dst->GetSizes().m_channel;

    // 1. init cl_mem
    cl_channel_order cl_ch_order      = (1 == channel) ? CL_R : CL_RG;
    CLMemParam       cl_mem_param_src = CLMemParam(CL_MEM_READ_ONLY, cl_ch_order); // cl iaura

    m_cl_src = CLMem::FromArray(m_ctx, *m_src, cl_mem_param_src);
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

    m_cl_map_x = CLMem::FromArray(m_ctx, m_map_x, CLMemParam(CL_MEM_READ_ONLY));
    if (!m_cl_dst.IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "m_cl_map_x is invalid");
        return Status::ERROR;
    }

    m_cl_map_y = CLMem::FromArray(m_ctx, m_map_y, CLMemParam(CL_MEM_READ_ONLY));
    if (!m_cl_dst.IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "m_cl_map_y is invalid");
        return Status::ERROR;
    }

    if (GetCLKernelParam(m_warp_type, m_interp_type, channel, m_elem_counts, m_elem_height) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "GetCLKernelParam failed");
        return Status::ERROR;
    }

    // 2. init kernel
    m_cl_kernels = GetCLKernels(m_ctx, m_dst->GetElemType(), channel, m_border_type, m_warp_type, m_interp_type);

    if (CheckCLKernels(m_ctx, m_cl_kernels) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "CheckCLKernels failed");
        return Status::ERROR;
    }

    // 3. sync inputs
    if (m_cl_src.Sync(CLMemSyncType::WRITE) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Sync src failed");
        return Status::ERROR;
    }

    if (m_cl_map_x.Sync(CLMemSyncType::WRITE) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Sync map_x failed");
        return Status::ERROR;
    }

    if (m_cl_map_y.Sync(CLMemSyncType::WRITE) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Sync map_y failed");
        return Status::ERROR;
    }

    return Status::OK;
}

Status WarpCL::DeInitialize()
{
    m_map_x.Release();
    m_map_y.Release();
    m_cl_src.Release();
    m_cl_dst.Release();
    m_cl_map_x.Release();
    m_cl_map_y.Release();
    m_cl_kernels.clear();

    if (WarpImpl::DeInitialize() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "WarpImpl::DeInitialize() failed");
        return Status::ERROR;
    }

    return Status::OK;
}

Status WarpCL::Run()
{
    DT_S32 ostep   = m_dst->GetRowPitch() / ElemTypeSize(m_dst->GetElemType());
    DT_S32 oheight = m_dst->GetSizes().m_height;
    DT_S32 owidth  = m_dst->GetSizes().m_width;
    DT_S32 channel = m_dst->GetSizes().m_channel;
    DT_S32 iheight = m_src->GetSizes().m_height;
    DT_S32 iwidth  = m_src->GetSizes().m_width;

    CLScalar                   cl_border_value = clScalar(m_border_value);
    std::shared_ptr<CLRuntime> cl_rt           = m_ctx->GetCLEngine()->GetCLRuntime();

    // 1. get cl_global_size
    cl::NDRange cl_global_size = cl::NDRange((owidth + m_elem_counts - 1) / m_elem_counts,
                                             (oheight + m_elem_height - 1) / m_elem_height);
    cl::Event cl_event;
    cl_int    cl_ret   = CL_SUCCESS;
    Status    ret      = Status::ERROR;
    Status    ret_sync = Status::ERROR;

    // 3. opencl run
    cl_ret = m_cl_kernels[0].Run<cl::Iaura2D, DT_S32, DT_S32, cl::Buffer, DT_S32, DT_S32, DT_S32, cl::Buffer, cl::Buffer, CLScalar>(
                             m_cl_src.GetCLMemRef<cl::Iaura2D>(), iheight, iwidth,
                             m_cl_dst.GetCLMemRef<cl::Buffer>(), oheight, owidth * channel, ostep,
                             m_cl_map_x.GetCLMemRef<cl::Buffer>(),
                             m_cl_map_y.GetCLMemRef<cl::Buffer>(),
                             cl_border_value,
                             cl_global_size, cl_rt->GetCLDefaultLocalSize(m_cl_kernels[0].GetMaxGroupSize(), cl_global_size), &(cl_event));
    if (cl_ret != CL_SUCCESS)
    {
        AURA_ADD_ERROR_STRING(m_ctx, ("Run cl_kernel main failed : " + GetCLErrorInfo(cl_ret)).c_str());
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

    ret_sync = m_cl_dst.Sync(aura::CLMemSyncType::READ);
    if (ret_sync != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Sync end fail");
        goto EXIT;
    }

    ret = Status::OK;

EXIT:
    AURA_RETURN(m_ctx, (ret | ret_sync));
}

std::string WarpCL::ToString() const
{
    return WarpImpl::ToString() + m_profiling_string;
}

std::vector<CLKernel> WarpCL::GetCLKernels(Context *ctx, ElemType elem_type, DT_S32 channel,
                                           BorderType border_type, WarpType warp_type, InterpType interp_type)
{
    std::vector<CLKernel> cl_kernels;

    // 2. build option
    DT_S32 elem_counts = 0, elem_height = 0;
    if (GetCLKernelParam(warp_type, interp_type, channel, elem_counts, elem_height) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetCLKernelParam failed");
        return cl_kernels;
    }

    std::string build_opt;
    if (GetCLBuildOptions(ctx, elem_type, build_opt, warp_type, border_type, elem_counts, elem_height, channel) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetCLBuildOptions failed");
        return cl_kernels;
    }

    // 3. init kernel
    std::string program_name, kernel_name;
    if (GetCLName(ctx, channel, interp_type, kernel_name, program_name) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetCLName failed");
        return cl_kernels;
    }

    cl_kernels.emplace_back(ctx, program_name, kernel_name, "", build_opt);

    return cl_kernels;
}

} // namespace aura
