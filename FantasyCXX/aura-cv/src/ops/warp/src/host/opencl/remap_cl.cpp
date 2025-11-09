#include "remap_impl.hpp"
#include "aura/runtime/opencl.h"
#include "aura/runtime/logger.h"

namespace aura
{

static Status GetCLBuildOptions(Context *ctx, ElemType elem_type, ElemType map_type, BorderType border_type,
                                 DT_S32 elem_counts, DT_S32 elem_height, DT_S32 channel, std::string &build_opt)
{
    const std::vector<std::string> tbl =
    {
                      "U8,    S8,   U16,    S16,   U32,  S32, F32,   F16"   , // elem type
        "Tp",         "uchar, char, ushort, short, uint, int, float, half"  , // data type
        "SelectType", "uchar, char, ushort, short, uint, int, int  , ushort", // select type
    };

    CLBuildOptions cl_build_opt(ctx, tbl);
    cl_build_opt.AddOption("MapType"    , CLTypeString(map_type));
    cl_build_opt.AddOption("CHANNEL"    , std::to_string(channel));
    cl_build_opt.AddOption("ELEM_HEIGHT", std::to_string(elem_height));
    cl_build_opt.AddOption("ELEM_COUNTS", std::to_string(elem_counts));

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
            break;
        }
    }

    build_opt = cl_build_opt.ToString(elem_type);

    return Status::OK;
}

static Status GetCLKernelParam(Context *ctx, InterpType interp_type, BorderType border_type, ElemType map_elem_type,
                               DT_S32 channel, DT_S32 &elem_counts, DT_S32 &elem_height)
{
    std::shared_ptr<CLRuntime> cl_rt = ctx->GetCLEngine()->GetCLRuntime();
    GpuType gpu_type = cl_rt->GetGpuInfo().m_type;

    if (GpuType::ADRENO == gpu_type && InterpType::NEAREST == interp_type)
    {
        elem_height = 1 + ((ElemType::S16 == map_elem_type) || (BorderType::REFLECT_101 == border_type));
        elem_counts = 2; // param after fine tuning
    }
    else if (GpuType::ADRENO == gpu_type && InterpType::NEAREST != interp_type)
    {
        elem_height = 1;
        elem_counts = 2 - (channel == 2);
    }
    else if (GpuType::MALI == gpu_type && InterpType::NEAREST == interp_type)
    {
        elem_height = 1;
        elem_counts = 4 - 2 * (BorderType::REFLECT_101 == border_type);
    }
    else if (GpuType::MALI == gpu_type && InterpType::NEAREST != interp_type)
    {
        elem_height = 1;
        elem_counts = (InterpType::LINEAR == interp_type || BorderType::CONSTANT == border_type) ? 1 : 2;
    }
    else
    {
        // set default value
        elem_counts = 1;
        elem_height = 1;
    }

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

    kernel_name = "Remap" + interp_type_str + "C" + std::to_string(channel);
    std::transform(interp_type_str.begin(), interp_type_str.begin() + 1, interp_type_str.begin(), ::tolower);
    program_name = "aura_remap_" + interp_type_str + "_c" + std::to_string(channel);

    return Status::OK;
}

static DT_VOID GetCLGlobalSize(Array *dst, DT_S32 elem_counts, DT_S32 elem_height, cl::NDRange &range)
{
    range = cl::NDRange((dst->GetSizes().m_width  + elem_counts - 1) / elem_counts,
                        (dst->GetSizes().m_height + elem_height - 1) / elem_height);
}

RemapCL::RemapCL(Context *ctx, const OpTarget &target) : RemapImpl(ctx, target)
{}

Status RemapCL::SetArgs(const Array *src, const Array *map, Array *dst, InterpType interp_type,
                        BorderType border_type, const Scalar &border_value)
{
    if (RemapImpl::SetArgs(src, map, dst, interp_type, border_type, border_value) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "RemapImpl::SetArgs failed");
        return Status::ERROR;
    }

    if (ElemType::F64 == src->GetElemType())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Unsupported src type for opencl");
        return Status::ERROR;
    }

    if (((map->GetElemType() != ElemType::F32) && (interp_type != InterpType::NEAREST)) || (src->GetSizes().m_channel > 2))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "map mat only supports f32 for remap bn/cu, and remap only supports channel 1,2.");
        return Status::ERROR;
    }

    return Status::OK;
}

Status RemapCL::Initialize()
{
    if (RemapImpl::Initialize() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "RemapImpl::Initialize() failed");
        return Status::ERROR;
    }

    DT_S32 channel = m_dst->GetSizes().m_channel;

    // 1. init cl_mem
    cl_channel_order cl_ch_order = (1 == channel) ? CL_R : CL_RG;
    CLMemParam cl_mem_param_src  = CLMemParam(CL_MEM_READ_ONLY, cl_ch_order);// cl iaura
    m_cl_src = CLMem::FromArray(m_ctx, *m_src, cl_mem_param_src);
    if (!m_cl_src.IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "m_cl_src is invalid");
        return Status::ERROR;
    }

    m_cl_map = CLMem::FromArray(m_ctx, *m_map, CLMemParam(CL_MEM_READ_ONLY));
    if (!m_cl_map.IsValid())
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

    GetCLKernelParam(m_ctx, m_interp_type, m_border_type, m_map->GetElemType(), channel, m_elem_counts, m_elem_height);

    // 2.init kernel
    m_cl_kernels = GetCLKernels(m_ctx, m_map->GetElemType(), m_dst->GetElemType(), channel, m_border_type, m_interp_type);

    if (CheckCLKernels(m_ctx, m_cl_kernels) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Remap CheckCLKernels failed");
        return Status::ERROR;
    }

    // 3. sync inputs
    if (m_cl_src.Sync(CLMemSyncType::WRITE) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Sync src failed");
        return Status::ERROR;
    }

    if (m_cl_map.Sync(CLMemSyncType::WRITE) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Sync map failed");
        return Status::ERROR;
    }

    return Status::OK;
}

Status RemapCL::DeInitialize()
{
    m_cl_src.Release();
    m_cl_map.Release();
    m_cl_dst.Release();
    m_cl_kernels.clear();

    if (RemapImpl::DeInitialize() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "RemapImpl::DeInitialize() failed");
        return Status::ERROR;
    }

    return Status::OK;
}

Status RemapCL::Run()
{
    DT_S32 ostep   = m_dst->GetRowPitch() / ElemTypeSize(m_dst->GetElemType());
    DT_S32 mstep   = m_map->GetRowPitch() / ElemTypeSize(m_map->GetElemType());
    DT_S32 oheight = m_dst->GetSizes().m_height;
    DT_S32 owidth  = m_dst->GetSizes().m_width;
    DT_S32 channel = m_dst->GetSizes().m_channel;
    DT_S32 iheight = m_src->GetSizes().m_height;
    DT_S32 iwidth  = m_src->GetSizes().m_width;
    CLScalar cl_border_value = clScalar(m_border_value);
    std::shared_ptr<CLRuntime> cl_rt = m_ctx->GetCLEngine()->GetCLRuntime();

    // 1. get cl_global_size
    cl::NDRange cl_global_size;
    GetCLGlobalSize(m_dst, m_elem_counts, m_elem_height, cl_global_size);

    cl::Event cl_event;
    cl_int cl_ret   = CL_SUCCESS;
    Status ret      = Status::ERROR;
    Status ret_sync = Status::ERROR;

    // 3. opencl run
    cl_ret = m_cl_kernels[0].Run<cl::Iaura2D, DT_S32, DT_S32, cl::Buffer, DT_S32, DT_S32, DT_S32, cl::Buffer, DT_S32, CLScalar>(
                             m_cl_src.GetCLMemRef<cl::Iaura2D>(), iheight, iwidth,
                             m_cl_dst.GetCLMemRef<cl::Buffer>(),  oheight, owidth * channel, ostep,
                             m_cl_map.GetCLMemRef<cl::Buffer>(),  mstep,
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

    ret_sync = m_cl_dst.Sync(CLMemSyncType::READ);
    if (ret_sync != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Sync end fail");
        goto EXIT;
    }

    ret = Status::OK;

EXIT:
    AURA_RETURN(m_ctx, (ret | ret_sync));
}

std::string RemapCL::ToString() const
{
    return RemapImpl::ToString() + m_profiling_string;
}

std::vector<CLKernel> RemapCL::GetCLKernels(Context *ctx, ElemType map_elem_type, ElemType dst_elem_type, DT_S32 channel,
                                            BorderType border_type, InterpType interp_type)
{
    std::vector<CLKernel> cl_kernels;
    DT_S32 elem_counts, elem_height;

    if (GetCLKernelParam(ctx, interp_type, border_type, map_elem_type, channel, elem_counts, elem_height) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetCLKernelParam() failed");
        return cl_kernels;
    }

    std::string build_opt;
    if (GetCLBuildOptions(ctx, dst_elem_type, map_elem_type, border_type, elem_counts, elem_height, channel, build_opt) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetCLBuildOptions() failed");
        return cl_kernels;
    }

    std::string program_name, kernel_name;
    if (GetCLName(ctx, channel, interp_type, kernel_name, program_name) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetCLName() failed");
        return cl_kernels;
    }

    cl_kernels.emplace_back(ctx, program_name, kernel_name, "", build_opt);

    return cl_kernels;
}

} // namespace aura
