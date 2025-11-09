#include "morph_impl.hpp"

namespace aura
{

static std::string GetCLBuildOptions(Context *ctx, ElemType elem_type, DT_S32 ksize, DT_S32 elem_counts,
                                     DT_S32 elem_height, MorphType type)
{
    CLBuildOptions cl_build_opt(ctx);

    cl_build_opt.AddOption("Tp",          CLTypeString(elem_type));
    cl_build_opt.AddOption("MORPH_TYPE",  MorphTypeToString(type));
    cl_build_opt.AddOption("ELEM_COUNTS", std::to_string(elem_counts));
    cl_build_opt.AddOption("KERNEL_SIZE", std::to_string(ksize));
    cl_build_opt.AddOption("ELEM_HEIGHT", std::to_string(elem_height));

    return cl_build_opt.ToString(elem_type);
}

static Status GetCLName(Context *ctx, DT_S32 channel, MorphShape shape, std::string &program_name, std::string &kernel_name)
{
    if (DT_NULL == ctx)
    {
        return Status::ERROR;
    }

    switch (shape)
    {
        case MorphShape::RECT:
        {
            program_name = "aura_morph_rect";
            kernel_name  = "MorphRectC" + std::to_string(channel);
            break;
        }

        case MorphShape::CROSS:
        {
            program_name = "aura_morph_cross";
            kernel_name  = "MorphCrossC" + std::to_string(channel);
            break;
        }

        case MorphShape::ELLIPSE:
        {
            program_name = "aura_morph_ellipse";
            kernel_name  = "MorphEllipseC" + std::to_string(channel);
            break;
        }

        default:
        {
            break;
        }
    }

    return Status::OK;
}

static DT_VOID GetCLGlobalSize(DT_S32 height, DT_S32 width, DT_S32 elem_counts,
                               DT_S32 elem_height, cl::NDRange &cl_global_size)
{
    cl_global_size = cl::NDRange((width + elem_counts - 1) / elem_counts, (height + elem_height - 1) / elem_height);
}

static DT_VOID GetCLKernelParam(MorphShape shape, DT_S32 &elem_counts, DT_S32 &elem_height)
{
    elem_counts = 4;
    elem_height = (MorphShape::RECT == shape) ? 6 : ((MorphShape::CROSS == shape) ? 5 : 4);
}

Status MorphCL::MorphCLImpl(CLMem &cl_src, CLMem &cl_dst)
{
    DT_S32 istep  = cl_src.GetRowPitch() / ElemTypeSize(cl_src.GetElemType());
    DT_S32 ostep  = cl_dst.GetRowPitch() / ElemTypeSize(cl_dst.GetElemType());
    DT_S32 height = cl_dst.GetSizes().m_height;
    DT_S32 width  = cl_dst.GetSizes().m_width;

    // 1. get cl_global_size
    cl::NDRange cl_global_size;
    DT_S32 elem_counts, elem_height;

    GetCLKernelParam(m_shape, elem_counts, elem_height);
    GetCLGlobalSize(height, width, elem_counts, elem_height, cl_global_size);

    // 2. opencl run
    cl::Event cl_event;
    cl_int cl_ret = CL_SUCCESS;
    Status ret    = Status::ERROR;

    cl_ret = m_cl_kernels[0].Run<cl::Buffer, DT_S32, cl::Buffer, DT_S32, DT_S32, DT_S32>(
                                 cl_src.GetCLMemRef<cl::Buffer>(), istep,
                                 cl_dst.GetCLMemRef<cl::Buffer>(), ostep,
                                 height, width, cl_global_size,
                                 cl::NDRange(32, (m_cl_kernels[0].GetMaxGroupSize() / 32)), &(cl_event));
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

    // 4. sync end
    ret = cl_dst.Sync(CLMemSyncType::READ);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "cl_dst Sync end fail");
    }

EXIT:
    AURA_RETURN(m_ctx, ret);
}

MorphCL::MorphCL(Context *ctx, MorphType type, const OpTarget &target) : MorphImpl(ctx, type, target)
{}

Status MorphCL::SetArgs(const Array *src, Array *dst, DT_S32 ksize, MorphShape shape, DT_S32 iterations)
{
    if (MorphImpl::SetArgs(src, dst, ksize, shape, iterations) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "MorphImpl::SetArgs failed");
        return Status::ERROR;
    }

    if (ksize > 7)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "opencl impl cannot support ksize > 7");
        return Status::ERROR;
    }

    if (src->GetSizes().m_channel != 1)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "opencl impl cannot support multi-channel");
        return Status::ERROR;
    }

    return Status::OK;
}

Status MorphCL::Initialize()
{
    if (MorphImpl::Initialize() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "MorphImpl::Initialize() failed");
        return Status::ERROR;
    }

    // 1. init cl_mem
    if (m_iterations > 1)
    {
        m_tmp = Mat(m_ctx, m_dst->GetElemType(), m_dst->GetSizes(), AURA_MEM_DMA_BUF_HEAP, m_dst->GetStrides());

        m_cl_tmp = CLMem::FromArray(m_ctx, m_tmp, CLMemParam(CL_MEM_READ_WRITE));
        if (!m_cl_tmp.IsValid())
        {
            AURA_ADD_ERROR_STRING(m_ctx, "m_cl_dst is invalid");
            return Status::ERROR;
        }
    }

    m_cl_src = CLMem::FromArray(m_ctx, *m_src, CLMemParam(CL_MEM_READ_ONLY));
    if (!m_cl_src.IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "m_cl_src is invalid");
        return Status::ERROR;
    }

    m_cl_dst = CLMem::FromArray(m_ctx, *m_dst, CLMemParam(CL_MEM_READ_WRITE));
    if (!m_cl_dst.IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "m_cl_dst is invalid");
        return Status::ERROR;
    }

    // 2. init kernel
    m_cl_kernels = MorphCL::GetCLKernels(m_ctx, m_dst->GetElemType(), m_dst->GetSizes().m_channel, m_ksize, m_shape, m_type);

    if (CheckCLKernels(m_ctx, m_cl_kernels) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Morph CheckCLKernels failed");
        return Status::ERROR;
    }

    // 3. sync src
    if (m_cl_src.Sync(CLMemSyncType::WRITE) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "m_cl_src Sync src failed");
        return Status::ERROR;
    }

    return Status::OK;
}

Status MorphCL::Run()
{
    Status ret = Status::ERROR;

    CLMem *temp_cl_src = &m_cl_src;
    CLMem *temp_cl_dst = ((m_iterations & 1) == 1) ? &m_cl_dst : &m_cl_tmp;

    ret = MorphCLImpl(*temp_cl_src, *temp_cl_dst);
    if (ret != Status::OK)
    {
        std::string info = "Morph_" + MorphTypeToString(m_type) + std::to_string(m_ksize) + "x" + std::to_string(m_ksize) + "Opencl failed";
        AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
        return ret;
    }

    for (DT_S32 i = 1; i < m_iterations; i++)
    {
        temp_cl_src = (((i + m_iterations) & 1) == 0) ? &m_cl_dst : &m_cl_tmp;
        temp_cl_dst = (((i + m_iterations) & 1) == 0) ? &m_cl_tmp : &m_cl_dst;

        ret = temp_cl_src->Sync(CLMemSyncType::WRITE);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "m_cl_src Sync start failed");
            return ret;
        }

        ret = MorphCLImpl(*temp_cl_src, *temp_cl_dst);
        if (ret != Status::OK)
        {
            std::string info = "Morph_" + MorphTypeToString(m_type) + std::to_string(m_ksize) + "x" + std::to_string(m_ksize) + "Opencl failed";
            AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
            return ret;
        }
    }

    AURA_RETURN(m_ctx, ret);
}

Status MorphCL::DeInitialize()
{
    m_cl_src.Release();
    m_cl_tmp.Release();
    m_cl_dst.Release();
    m_tmp.Release();
    m_cl_kernels.clear();

    if (MorphImpl::DeInitialize() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "MorphImpl::DeInitialize() failed");
        return Status::ERROR;
    }

    return Status::OK;
}

std::string MorphCL::ToString() const
{
    return MorphImpl::ToString() + m_profiling_string;
}

std::vector<CLKernel> MorphCL::GetCLKernels(Context *ctx, ElemType elem_type, DT_S32 channel, DT_S32 ksize, MorphShape shape, MorphType type)
{
    std::vector<CLKernel> cl_kernels;
    DT_S32 elem_counts, elem_height;

    GetCLKernelParam(shape, elem_counts, elem_height);
    std::string build_opt = GetCLBuildOptions(ctx, elem_type, ksize, elem_counts, elem_height, type);

    std::string program_name, kernel_name;
    if (GetCLName(ctx, channel, shape, program_name, kernel_name) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetCLName failed");
        return cl_kernels;
    }

    cl_kernels.push_back({ctx, program_name, kernel_name, "", build_opt});

    return cl_kernels;
}

} // namespace aura
