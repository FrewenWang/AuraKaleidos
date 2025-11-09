#include "aura/ops/filter/gaussian.hpp"
#include "gaussian_impl.hpp"
#include "aura/runtime/logger.h"

namespace aura
{

template <typename Tp>
struct GaussianTraits
{
    // Tp = DT_U8, DT_U16 DT_S16 MI_F16 DT_F32
    // 表达式拆分
    // 表达式1 = std::is_same<Tp, DT_U8>::value,   这是类型特性工具 std::is_same 的调用，用于判断模板参数 TP 是否与 U8 是同一类型
    //          若 TP 是 U8（例如 8 位无符号整数），则表达式返回 true；否则返回 false
    // 表达式2 = typename Promote<Tp>::Type,
    // 表达式3 = DT_F32
    // typename std::conditional<表达式1,表达式2,表达式3>::type  这是一个编译期条件选择模板，根据第一个参数（布尔值）选择返回类型
    // 综合上述逻辑，KernelType 的类型为：
    // 当 TP 是 U8 时候：KernelType = Promote<U8>::Type
    // 当 TP 不是 U8 时：KernelType = F32
    using KernelType = typename std::conditional<std::is_same<Tp, DT_U8>::value, typename Promote<Tp>::Type, DT_F32>::type;
    // Tp = DT_U8, DT_U16 DT_S16 MI_F16 DT_F32
    static constexpr DT_U32 Q = std::is_same<Tp, DT_U8>::value ? 8 : 0;
};

static std::string GetCLBuildOptions(Context *ctx, ElemType elem_type, BorderType border_type)
{
    const std::vector<std::string> tbl =
    {
                     "U8,     U16,    S16,   F32,   F16",   // elem type
        "St",        "uchar,  ushort, short, float, half",  // source
        "Kt",        "ushort, float,  float, float, float", // kernel
        "InterType", "uint,   float,  float, float, float", // internal type
    };

    CLBuildOptions cl_build_opt(ctx, tbl);

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

    cl_build_opt.AddOption("MAX_CONSTANT_SIZE", ctx->GetCLEngine()->GetCLRuntime()->GetCLMaxConstantSizeString(36));
    return cl_build_opt.ToString(elem_type);
}

static Status GetCLName(Context *ctx, DT_S32 channel, DT_S32 ksize, std::string program_name[2], std::string kernel_name[2])
{
    if (DT_NULL == ctx)
    {
        return Status::ERROR;
    }

    if (ctx->GetCLEngine() == DT_NULL || ctx->GetCLEngine()->GetCLRuntime() == DT_NULL)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetCLEngine failed");
        return Status::ERROR;
    }

    GpuType gpu_type = ctx->GetCLEngine()->GetCLRuntime()->GetGpuInfo().m_type;

    std::string program_postfix = std::to_string(ksize) + "x" + std::to_string(ksize) + "_c" + std::to_string(channel);
    std::string kernel_postfix = std::to_string(ksize) + "x" + std::to_string(ksize) + "C" + std::to_string(channel);

    program_name[0] = "aura_gaussian_main_" + program_postfix;
    program_name[1] = "aura_gaussian_remain_" + program_postfix;

    kernel_name[0] = "GaussianMain" + kernel_postfix;
    kernel_name[1] = "GaussianRemain" + kernel_postfix;

    if ((7 == ksize) && (1 == channel))
    {
        program_name[0] += "_" + GpuTypeToString(gpu_type);
    }

    if ((5 == ksize) && ((2 == channel) || (3 == channel)))
    {
        program_name[0] += "_" + GpuTypeToString(gpu_type);
    }

    return Status::OK;
}

static DT_VOID GetCLGlobalSize(GpuType gpu_type, DT_S32 ksize, DT_S32 channel, DT_S32 height,
                               DT_S32 width, ElemType elem_type, cl::NDRange range[2], DT_S32 &main_width)
{
    DT_S32 elem_counts = 0;
    DT_S32 main_size   = 0;
    DT_S32 main_rest   = 0;
    DT_S32 border      = (ksize >> 1) << 1;

    if ((GpuType::ADRENO == gpu_type) && (7 == ksize) && (1 == channel))
    {
        elem_counts = 10;
        main_size   = (width - border) / elem_counts;
        main_rest   = (width - border) % elem_counts;
        range[0]    = cl::NDRange(main_size, ElemType::U8 == elem_type ? height : (height + 1) >> 1);
    }
    else if ((GpuType::MALI == gpu_type) && (5 == ksize) && (2 == channel))
    {
        elem_counts = 1;
        main_rest   = elem_counts * 3;
        main_size   = width - border - main_rest;
        range[0]    = cl::NDRange(main_size, height);
    }
    else if ((GpuType::MALI == gpu_type) && (5 == ksize) && (3 == channel))
    {
        elem_counts = 1;
        main_rest   = elem_counts;
        main_size   = width - border - main_rest;
        range[0]    = cl::NDRange(main_size, height);
    }
    else if ((3 == ksize) && (3 == channel))
    {
        elem_counts = 3;
        main_size   = (width - border) / elem_counts - 1;
        main_rest   = (width - border) % elem_counts + 3;
        range[0]    = cl::NDRange(main_size, height);
    }
    else
    {
        elem_counts = (3 == ksize) ? 6 : ((5 == ksize) ? 4 : 2);
        main_size   = (width - border) / elem_counts;
        main_rest   = (width - border) % elem_counts;
        range[0]    = cl::NDRange(main_size, height);
    }

    // remain global size
    range[1]   = cl::NDRange(main_rest + border, height);
    main_width = main_size * elem_counts;
}

GaussianCL::GaussianCL(Context *ctx, const OpTarget &target) : GaussianImpl(ctx, target)
{}

/**
 * 步骤一：设置相关参数
 * @param src
 * @param dst
 * @param ksize  高斯核的大小
 * @param sigma  高斯核的sigma
 * @param border_type  边界处理类型
 * @param border_value
 * @return
 */
Status GaussianCL::SetArgs(const Array *src, Array *dst, DT_S32 ksize, DT_F32 sigma,
                           BorderType border_type, const Scalar &border_value)
{
    if (GaussianImpl::SetArgs(src, dst, ksize, sigma, border_type, border_value) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "GaussianImpl::SetArgs failed");
        return Status::ERROR;
    }

    if (ksize != 3 && ksize != 5 && ksize != 7)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "ksize is only suppose 3/5/7");
        return Status::ERROR;
    }

    DT_S32 ch = src->GetSizes().m_channel;

    if (ch != 1 && ch != 2 && ch != 3)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "channel is only suppose 1/2/3");
        return Status::ERROR;
    }

    if (7 == ksize && (ch != 1))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "opencl target cannot support channel > 1 when ksize == 7");
        return Status::ERROR;
    }

    return Status::OK;
}

/**
 * TODO 这个方法的代码需要好好学习
 * @return
 */
Status GaussianCL::PrepareKmat()
{
    /// 获取高斯核的权重数据
    std::vector<DT_F32> kernel = GetGaussianKernel(m_ksize, m_sigma);

    // 获取原始src的数据类型
    switch (m_src->GetElemType())
    {
        case ElemType::U8:
        {
            /// 或是高斯核的类型
            using KernelType   = GaussianTraits<DT_U8>::KernelType;
            constexpr DT_U32 Q = GaussianTraits<DT_U8>::Q;
            m_kmat = GetGaussianKmat<KernelType, Q>(m_ctx, kernel);
            break;
        }

        case ElemType::U16:
        case ElemType::S16:
        case ElemType::F32:
#if defined(AURA_BUILD_HOST)
        case ElemType::F16:
#endif // AURA_BUILD_HOST
        {
            m_kmat = GetGaussianKmat(m_ctx, kernel);
            break;
        }

        default:
        {
            m_kmat = Mat();
            AURA_ADD_ERROR_STRING(m_ctx, "Unsupported source format");
            return Status::ERROR;
        }
    }

    return Status::OK;
}

/**
 * 步骤二：开始机型初始化
 * @return
 */
Status GaussianCL::Initialize()
{
    /// TODO 这个地方进行初始化.但是好像GaussianImpl上层并没有进行初始化
    if (GaussianImpl::Initialize() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "GaussianImpl::Initialize() failed");
        return Status::ERROR;
    }

    // 0. prepare kmat。 开始进行初始化高斯内核
    if (PrepareKmat() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "PrepareKmat failed");
        return Status::ERROR;
    }

    // 1. init cl_mem
    // 创建CL内存
    m_cl_src = CLMem::FromArray(m_ctx, *m_src, CLMemParam(CL_MEM_READ_ONLY));
    if (!m_cl_src.IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "m_cl_src is invalid");
        return Status::ERROR;
    }

    // 创建CL kmat
    m_cl_kmat = CLMem::FromArray(m_ctx, m_kmat, CLMemParam(CL_MEM_READ_ONLY));
    if (!m_cl_kmat.IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "m_cl_kmat is invalid");
        return Status::ERROR;
    }
    /// 创建输出数据
    m_cl_dst = CLMem::FromArray(m_ctx, *m_dst, CLMemParam(CL_MEM_WRITE_ONLY));
    if (!m_cl_dst.IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "m_cl_dst is invalid");
        return Status::ERROR;
    }

    // 2. init kernel
    /// 初始化kernel
    m_cl_kernels = GaussianCL::GetCLKernels(m_ctx, m_dst->GetElemType(), m_dst->GetSizes().m_channel, m_ksize, m_border_type);

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

    if (m_cl_kmat.Sync(CLMemSyncType::WRITE) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "m_cl_kmat Sync start failed");
        return Status::ERROR;
    }

    return Status::OK;
}

Status GaussianCL::DeInitialize()
{
    m_cl_src.Release();
    m_cl_kmat.Release();
    m_cl_dst.Release();

    for (auto &cl_kernel : m_cl_kernels)
    {
        cl_kernel.DeInitialize();
    }

    m_kmat.Release();

    if (GaussianImpl::DeInitialize() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "GaussianImpl::DeInitialize() failed");
        return Status::ERROR;
    }

    return Status::OK;
}

Status GaussianCL::Run()
{
    DT_S32 istep   = m_src->GetRowPitch() / ElemTypeSize(m_src->GetElemType());
    DT_S32 ostep   = m_dst->GetRowPitch() / ElemTypeSize(m_dst->GetElemType());
    DT_S32 height  = m_dst->GetSizes().m_height;
    DT_S32 width   = m_dst->GetSizes().m_width;
    DT_S32 channel = m_dst->GetSizes().m_channel;

    std::shared_ptr<CLRuntime> cl_rt = m_ctx->GetCLEngine()->GetCLRuntime();
    CLScalar cl_border_value         = clScalar(m_border_value);

    // 1. get center_area and cl_global_size
    cl::NDRange cl_global_size[2];
    DT_S32 main_width = 0;

    GetCLGlobalSize(cl_rt->GetGpuInfo().m_type, m_ksize, channel, height, width,  m_src->GetElemType(), cl_global_size, main_width);

    // 2. opencl run
    cl::Event cl_event[2];
    cl_int cl_ret = CL_SUCCESS;
    Status ret    = Status::ERROR;

    cl_ret = m_cl_kernels[0].Run<cl::Buffer, DT_S32, cl::Buffer, DT_S32, DT_S32, DT_S32, DT_S32, cl::Buffer, CLScalar>(
                                 m_cl_src.GetCLMemRef<cl::Buffer>(), istep,
                                 m_cl_dst.GetCLMemRef<cl::Buffer>(), ostep,
                                 height,
                                 (DT_S32)(cl_global_size[0].get()[1]),
                                 (DT_S32)(cl_global_size[0].get()[0]),
                                 m_cl_kmat.GetCLMemRef<cl::Buffer>(),
                                 cl_border_value,
                                 cl_global_size[0], cl_rt->GetCLDefaultLocalSize(m_cl_kernels[0].GetMaxGroupSize(), cl_global_size[0]),
                                 &(cl_event[0]));
    if (cl_ret != CL_SUCCESS)
    {
        AURA_ADD_ERROR_STRING(m_ctx, ("Run cl_kernel main failed : " + GetCLErrorInfo(cl_ret)).c_str());
        goto EXIT;
    }

    cl_ret |= m_cl_kernels[1].Run<cl::Buffer, DT_S32, cl::Buffer, DT_S32, DT_S32, DT_S32, DT_S32, DT_S32, DT_S32, cl::Buffer, CLScalar>(
                                  m_cl_src.GetCLMemRef<cl::Buffer>(), istep,
                                  m_cl_dst.GetCLMemRef<cl::Buffer>(), ostep,
                                  height, width, main_width,
                                  (DT_S32)(cl_global_size[1].get()[1]),
                                  (DT_S32)(cl_global_size[1].get()[0]),
                                  m_cl_kmat.GetCLMemRef<cl::Buffer>(),
                                  cl_border_value,
                                  cl_global_size[1], cl_rt->GetCLDefaultLocalSize(m_cl_kernels[1].GetMaxGroupSize(), cl_global_size[1]),
                                  &(cl_event[1]), {cl_event[0]});

    if (cl_ret != CL_SUCCESS)
    {
        AURA_ADD_ERROR_STRING(m_ctx, ("Run cl_kernel remain failed : " + GetCLErrorInfo(cl_ret)).c_str());
        goto EXIT;
    }

    // 3. cl wait
    if ((DT_TRUE == m_target.m_data.opencl.profiling) || (m_dst->GetArrayType() != ArrayType::CL_MEMORY))
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
        AURA_ADD_ERROR_STRING(m_ctx, "m_cl_dst Sync end fail");
    }

EXIT:
    AURA_RETURN(m_ctx, ret);
}

std::string GaussianCL::ToString() const
{
    return GaussianImpl::ToString() + m_profiling_string;
}

std::vector<CLKernel> GaussianCL::GetCLKernels(Context *ctx, ElemType elem_type, DT_S32 channel, DT_S32 ksize, BorderType border_type)
{
    std::vector<CLKernel> cl_kernels;
    std::string build_opt = GetCLBuildOptions(ctx, elem_type, border_type);

    std::string program_name[2], kernel_name[2];
    if (GetCLName(ctx, channel, ksize, program_name, kernel_name) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetCLName failed");
        return cl_kernels;
    }

    cl_kernels.push_back({ctx, program_name[0], kernel_name[0], "", build_opt});
    cl_kernels.push_back({ctx, program_name[1], kernel_name[1], "", build_opt});

    return cl_kernels;
}

} // namespace aura
