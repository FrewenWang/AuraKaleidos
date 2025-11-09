#include "aura/ops/filter/gaussian.hpp"
#include "gaussian_impl.hpp"
#include "aura/ops/matrix.h"
#include "aura/runtime/logger.h"
#include "aura/runtime/thread_buffer.h"
#include "aura/runtime/worker_pool.h"

namespace aura
{

template <typename Tp>
struct GaussianTraits
{
    // 针对对应的数据类型Tp,来进行做4字节类型提升
    // Tp = DT_U32 DT_S32 DT_F32, DT_U8 DT_U16 DT_S16 MI_F16
    // 这是一个模板结构体，接收一个类型 Tp，用来根据 Tp 的类型特性确定处理高斯卷积时的内部处理类型和量化参数。
    // std::conditional<cond, A, B>::type 根据 cond 为真或假，选择类型 A 或 B。
    // 条件是 sizeof(Tp) == 4，即 Tp 占 4 字节（通常是 float 或 32-bit int）。
    // 如果 Tp 是 4 字节类型，KernelType = Tp。否则 KernelType = Promote<Tp>::Type。
    using KernelType = typename std::conditional<sizeof(Tp) == 4, Tp, typename Promote<Tp>::Type>::type;

    // Tp = MI_F16 DT_F32, DT_U8 DT_U32 DT_S32, DT_U16 DT_S16
    // 如果 Tp 是浮点类型（如 MI_F16、DT_F32），Q = 0 —— 表示不需要量化。
    // 如果 Tp 是整数类型：
    // 
    static constexpr DT_U32 Q = is_floating_point<Tp>::value ? 0 : (sizeof(Tp) > 1 ? 14 : 8);
};

/**
 * 高斯模糊的具体实现
 * @tparam Tp
 * @param ctx
 * @param src
 * @param dst
 * @param ksize
 * @param kmat
 * @param thread_buffer
 * @param start_row
 * @param end_row
 * @return
 */
template <typename Tp>
static Status GaussianNoneImpl(Context *ctx, const Mat &src, Mat &dst, DT_S32 ksize, const Mat &kmat,
                               ThreadBuffer &thread_buffer, DT_S32 start_row, DT_S32 end_row)
{
    using KernelType = typename GaussianTraits<Tp>::KernelType;
    using SumType    = typename Promote<KernelType>::Type;

    const KernelType *kernel = kmat.Ptr<KernelType>(0);

    DT_S32 iwidth  = src.GetSizes().m_width;
    DT_S32 owidth  = dst.GetSizes().m_width;
    DT_S32 channel = dst.GetSizes().m_channel;
    DT_S32 ksh = ksize >> 1;

    SumType *sum_row = thread_buffer.GetThreadData<SumType>();

    if (DT_NULL == sum_row)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get Buffer failed");
        return Status::ERROR;
    }

    std::vector<const Tp*> src_rows(ksize);
    for (DT_S32 k = 0; k < ksize; k++)
    {
        src_rows[k] = src.Ptr<Tp>(start_row + k);
    }

    for (DT_S32 y = start_row; y < end_row; y++)
    {
        Tp *dst_row = dst.Ptr<Tp>(y);
        for (DT_S32 x = 0; x < iwidth; x++)
        {
            for (DT_S32 ch = 0; ch < channel; ch++)
            {
                SumType sum = 0;
                for (DT_S32 k = 0; k < ksh; k++)
                {
                    sum += (static_cast<SumType>(src_rows[k][x * channel + ch]) + static_cast<SumType>(src_rows[ksize - k - 1][x * channel + ch])) * kernel[k];
                }
                sum += static_cast<SumType>(src_rows[ksh][x * channel + ch]) * kernel[ksh];
                sum_row[x * channel + ch] = sum;
            }
        }

        for (DT_S32 x = 0; x < owidth; x++)
        {
            for (DT_S32 ch = 0; ch < channel; ch++)
            {
                SumType sum = 0;
                for (DT_S32 k = 0; k < ksh; k++)
                {
                    sum += (sum_row[(x + k) * channel + ch] + sum_row[(x + ksize - k - 1) * channel + ch]) * kernel[k];
                }
                sum += sum_row[(x + ksh) * channel + ch] * kernel[ksh];
                dst_row[x * channel + ch] = ShiftSatCast<SumType, Tp, GaussianTraits<Tp>::Q << 1>(sum);
            }
        }

        for (DT_S32 i = 0; i < ksize - 1; i++)
        {
            src_rows[i] = src_rows[i + 1];
        }
        src_rows[ksize - 1] = const_cast<Tp*>(src.Ptr<Tp>(y + ksize));
    }

    return Status::OK;
}

GaussianNone::GaussianNone(Context *ctx, const OpTarget &target) : GaussianImpl(ctx, target)
{}

/**
 * 步骤一：开始设置参数
 * @param src
 * @param dst
 * @param ksize
 * @param sigma
 * @param border_type
 * @param border_value
 * @return
 */
Status GaussianNone::SetArgs(const Array *src, Array *dst, DT_S32 ksize, DT_F32 sigma,
                             BorderType border_type, const Scalar &border_value)
{
    // 判断GaussianImpl的SetArgs的大小
    if (GaussianImpl::SetArgs(src, dst, ksize, sigma, border_type, border_value) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "GaussianImpl::SetArgs failed");
        return Status::ERROR;
    }
    // 判断数组类型必须是MAT类型
    if ((src->GetArrayType() != ArrayType::MAT) || (dst->GetArrayType() != ArrayType::MAT))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src dst must be mat type");
        return Status::ERROR;
    }

    return Status::OK;
}

Status GaussianNone::PrepareKmat()
{
    // 获取高斯核的参数
    std::vector<DT_F32> kernel = GetGaussianKernel(m_ksize, m_sigma);

#define GET_GAUSSIAN_KMAT(type)                                     \
    using KernelType   = typename GaussianTraits<type>::KernelType; \
    constexpr DT_U32 Q = GaussianTraits<type>::Q;                   \
                                                                    \
    m_kmat = GetGaussianKmat<KernelType, Q>(m_ctx, kernel);         \

    switch (m_src->GetElemType())
    {
        case ElemType::U8:
        {
            GET_GAUSSIAN_KMAT(DT_U8)
            break;
        }

        case ElemType::U16:
        {
            GET_GAUSSIAN_KMAT(DT_U16)
            break;
        }

        case ElemType::S16:
        {
            GET_GAUSSIAN_KMAT(DT_S16)
            break;
        }

        case ElemType::U32:
        {
            GET_GAUSSIAN_KMAT(DT_U32)
            break;
        }

        case ElemType::S32:
        {
            GET_GAUSSIAN_KMAT(DT_S32)
            break;
        }

        case ElemType::F32:
        {
            m_kmat = GetGaussianKmat(m_ctx, kernel);
            break;
        }

#if defined(AURA_BUILD_HOST)
        case ElemType::F16:
        {
            m_kmat = GetGaussianKmat(m_ctx, kernel);
            break;
        }
#endif // AURA_BUILD_HOST

        default:
        {
            m_kmat = Mat();
            AURA_ADD_ERROR_STRING(m_ctx, "Unsupported source format");
            return Status::ERROR;
        }
    }

#undef GET_GAUSSIAN_KMAT

    return Status::OK;
}

/**
 * 步骤二： 进行开始初始化
 * @return
 */
Status GaussianNone::Initialize()
{
    // 初始化高斯函数，
    if (GaussianImpl::Initialize() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "GaussianImpl::Initialize() failed");
        return Status::ERROR;
    }

    // Prepare kmat 初始化高斯核的Mat数据
    if (PrepareKmat() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "PrepareKmat failed");
        return Status::ERROR;
    }

    // Get border mat sizes
    DT_S32 ksh          = m_ksize >> 1;
    Sizes3 border_sizes = m_src->GetSizes() + Sizes3(ksh << 1, ksh << 1, 0);
    m_src_border        = Mat(m_ctx, m_src->GetElemType(), border_sizes);

    if (!m_src_border.IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Create src border failed");
        return Status::ERROR;
    }

    return Status::OK;
}

Status GaussianNone::Run()
{
    const Mat *src = dynamic_cast<const Mat*>(m_src);
    Mat *dst       = dynamic_cast<Mat*>(m_dst);

    if ((DT_NULL == src) || (DT_NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src dst is null");
        return Status::ERROR;
    }

    // Get border mat sizes
    DT_S32 ksh = m_ksize >> 1;
    if (IMakeBorder(m_ctx, *src, m_src_border, ksh, ksh, ksh, ksh, m_border_type, m_border_value, m_target) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "make border fail..");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    DT_S32 oheight  = dst->GetSizes().m_height;
    DT_S32 iwidth   = m_src_border.GetSizes().m_width;
    DT_S32 ichannel = m_src_border.GetSizes().m_channel;

#define GAUSSIAN_NONE_IMPL(type)                                                                                        \
    if (m_target.m_data.none.enable_mt)                                                                                 \
    {                                                                                                                   \
        WorkerPool *wp = m_ctx->GetWorkerPool();                                                                        \
        if (DT_NULL == wp)                                                                                              \
        {                                                                                                               \
            AURA_ADD_ERROR_STRING(m_ctx, "GetWorkerpool failed");                                                       \
            return Status::ERROR;                                                                                       \
        }                                                                                                               \
                                                                                                                        \
        using KernelType = typename GaussianTraits<type>::KernelType;                                                   \
        using SumType = typename Promote<KernelType>::Type;                                                             \
                                                                                                                        \
        DT_S32 buffer_size = iwidth * ichannel * sizeof(SumType);                                                       \
        ThreadBuffer thread_buffer(m_ctx, buffer_size);                                                                 \
                                                                                                                        \
        ret = wp->ParallelFor(static_cast<DT_S32>(0), oheight, GaussianNoneImpl<type>, m_ctx, std::cref(m_src_border),  \
                              std::ref(*dst), m_ksize, std::cref(m_kmat), std::ref(thread_buffer));                     \
    }                                                                                                                   \
    else                                                                                                                \
    {                                                                                                                   \
        using KernelType = typename GaussianTraits<type>::KernelType;                                                   \
        using SumType = typename Promote<KernelType>::Type;                                                             \
                                                                                                                        \
        DT_S32 buffer_size = iwidth * ichannel * sizeof(SumType);                                                       \
        ThreadBuffer thread_buffer(m_ctx, buffer_size);                                                                 \
                                                                                                                        \
        ret = GaussianNoneImpl<type>(m_ctx, m_src_border, *dst, m_ksize, m_kmat, thread_buffer, 0, oheight);            \
    }                                                                                                                   \
    if (ret != Status::OK)                                                                                              \
    {                                                                                                                   \
        DT_CHAR error_msg[128];                                                                                         \
        std::snprintf(error_msg, sizeof(error_msg), "GaussianNoneImpl<%s> failed", #type);                              \
        AURA_ADD_ERROR_STRING(m_ctx, error_msg);                                                                        \
    }

    switch (src->GetElemType())
    {
        case ElemType::U8:
        {
            GAUSSIAN_NONE_IMPL(DT_U8)
            break;
        }

        case ElemType::U16:
        {
            GAUSSIAN_NONE_IMPL(DT_U16)
            break;
        }

        case ElemType::S16:
        {
            GAUSSIAN_NONE_IMPL(DT_S16)
            break;
        }

        case ElemType::U32:
        {
            GAUSSIAN_NONE_IMPL(DT_U32)
            break;
        }

        case ElemType::S32:
        {
            GAUSSIAN_NONE_IMPL(DT_S32)
            break;
        }

#if defined(AURA_BUILD_HOST)
        case ElemType::F16:
        {
            GAUSSIAN_NONE_IMPL(MI_F16)
            break;
        }
#endif // AURA_BUILD_HOST

        case ElemType::F32:
        {
            GAUSSIAN_NONE_IMPL(DT_F32)
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(m_ctx, "elem type error");
            ret = Status::ERROR;
        }
    }

#undef GAUSSIAN_NONE_IMPL

    AURA_RETURN(m_ctx, ret);
}

Status GaussianNone::DeInitialize()
{
    m_src_border.Release();
    m_kmat.Release();

    if (GaussianImpl::DeInitialize() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "GaussianImpl::DeInitialize() failed");
        return Status::ERROR;
    }

    return Status::OK;
}

} // namespace aura