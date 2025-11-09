#include "binary_impl.hpp"
#include "aura/runtime/worker_pool.h"
#include "aura/runtime/logger.h"

namespace aura
{

template <typename Tp = DT_VOID>
struct BinaryMinNeonFunctor
{
    using VType = typename neon::QVector<Tp>::VType;
    static constexpr VType BinarayNeonFunc(const VType& v0, const VType& v1)
    {
        return neon::vmin(v0, v1);
    }

    static constexpr Tp BinaryNoneFunc(const Tp& s0, const Tp& s1)
    {
        return (s0 < s1) ? s0 : s1;
    }
};

template <typename Tp = DT_VOID>
struct BinaryMaxNeonFunctor
{
    using VType = typename neon::QVector<Tp>::VType;
    static constexpr VType BinarayNeonFunc(const VType& v0, const VType& v1)
    {
        return neon::vmax(v0, v1);
    }

    static constexpr Tp BinaryNoneFunc(const Tp& s0, const Tp& s1)
    {
        return (s0 > s1) ? s0 : s1;
    }
};

template <typename Tp, typename Functor>
static Status BinaryNeonImpl(const Mat &src0, const Mat &src1, Mat &dst, DT_S32 y_start, DT_S32 y_end)
{
    using VType = typename neon::QVector<Tp>::VType;
    constexpr DT_S32 VEC_SIZE = 16 / sizeof(Tp);

    const DT_S32 width   = src0.GetSizes().m_width;
    const DT_S32 channel = src0.GetSizes().m_channel;
    const DT_S32 num_per_row = width * channel;
    const DT_S32 w_align = num_per_row & (-VEC_SIZE);

    for (DT_S32 y = y_start; y < y_end; y++)
    {
        const Tp *src0_row = src0.Ptr<Tp>(y);
        const Tp *src1_row = src1.Ptr<Tp>(y);
        Tp *dst_row = dst.Ptr<Tp>(y);

        DT_S32 x = 0;
        for (; x < w_align; x += VEC_SIZE)
        {
            VType vq0    = neon::vload1q(src0_row + x);
            VType vq1    = neon::vload1q(src1_row + x);
            VType vq_cmp = Functor::BinarayNeonFunc(vq0, vq1);
            neon::vstore(dst_row + x, vq_cmp);
        }

        for (; x < num_per_row; x++)
        {
            dst_row[x] = Functor::BinaryNoneFunc(src0_row[x], src1_row[x]);
        }
    }

    return Status::OK;
}

template <typename Tp, typename Functor>
static Status BinaryNeonHelper(Context *ctx, const Mat &src0, const Mat &src1, Mat &dst, const OpTarget &target)
{
    AURA_UNUSED(target);
    WorkerPool *wp = ctx->GetWorkerPool();
    if (DT_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerPool failed");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    const DT_S32 height = src0.GetSizes().m_height;

    ret = wp->ParallelFor(0, height, BinaryNeonImpl<Tp, Functor>, std::cref(src0),
                          std::cref(src1), std::ref(dst));

    AURA_RETURN(ctx, ret);
}

template <typename Tp>
static Status BinaryNeonHelper(Context *ctx, BinaryOpType type, const Mat &src0, const Mat &src1, Mat &dst, const OpTarget &target)
{
    Status ret = Status::ERROR;

    switch (type)
    {
        case BinaryOpType::MIN:
        {
            ret = BinaryNeonHelper<Tp, BinaryMinNeonFunctor<Tp>>(ctx, src0, src1, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "BinaryNeonHelper<Tp, BinaryMinNeonFunctor<Tp>> failed.");
            }
            break;
        }
        case BinaryOpType::MAX:
        {
            ret = BinaryNeonHelper<Tp, BinaryMaxNeonFunctor<Tp>>(ctx, src0, src1, dst, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "BinaryNeonHelper<Tp, BinaryMaxNeonFunctor<Tp>> failed.");
            }
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "unsupported binary operator type");
            ret = Status::ERROR;
        }
    }

    AURA_RETURN(ctx, ret);
}

BinaryNeon::BinaryNeon(Context *ctx, const OpTarget &target) : BinaryImpl(ctx, target)
{}

Status BinaryNeon::SetArgs(const Array *src0, const Array *src1, Array *dst, BinaryOpType type)
{
    if (BinaryImpl::SetArgs(src0, src1, dst, type) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "BinaryImpl::SetArgs failed");
        return Status::ERROR;
    }

    if ((src0->GetArrayType() != ArrayType::MAT) || (src1->GetArrayType() != ArrayType::MAT) || (dst->GetArrayType() != ArrayType::MAT))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src must be mat type");
        return Status::ERROR;
    }

    return Status::OK;
}

Status BinaryNeon::Run()
{
    const Mat *src0 = dynamic_cast<const Mat*>(m_src0);
    const Mat *src1 = dynamic_cast<const Mat*>(m_src1);
    Mat *dst        = dynamic_cast<Mat*>(m_dst);

    if ((NULL == src0) || (NULL == src1) || (NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src or dst is null");
        return Status::ERROR;
    }

    Status ret = Status::OK;

    switch (src0->GetElemType())
    {
        case ElemType::U8:
        {
            ret = BinaryNeonHelper<DT_U8>(m_ctx, m_type, *src0, *src1, *dst, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "BinaryNeonHelper<DT_U8> failed.");
            }
            break;
        }
        case ElemType::S8:
        {
            ret = BinaryNeonHelper<DT_S8>(m_ctx, m_type, *src0, *src1, *dst, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "BinaryNeonHelper<DT_S8> failed.");
            }
            break;
        }
        case ElemType::U16:
        {
            ret = BinaryNeonHelper<DT_U16>(m_ctx, m_type, *src0, *src1, *dst, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "BinaryNeonHelper<DT_U16> failed.");
            }
            break;
        }
        case ElemType::S16:
        {
            ret = BinaryNeonHelper<DT_S16>(m_ctx, m_type, *src0, *src1, *dst, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "BinaryNeonHelper<DT_S16> failed.");
            }
            break;
        }
        case ElemType::U32:
        {
            ret = BinaryNeonHelper<DT_U32>(m_ctx, m_type, *src0, *src1, *dst, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "BinaryNeonHelper<DT_U32> failed.");
            }
            break;
        }
        case ElemType::S32:
        {
            ret = BinaryNeonHelper<DT_S32>(m_ctx, m_type, *src0, *src1, *dst, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "BinaryNeonHelper<DT_S32> failed.");
            }
            break;
        }
#if defined(AURA_ENABLE_NEON_FP16)
        case ElemType::F16:
        {
            ret = BinaryNeonHelper<MI_F16>(m_ctx, m_type, *src0, *src1, *dst, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "BinaryNeonHelper<MI_F16> failed.");
            }
            break;
        }
#endif
        case ElemType::F32:
        {
            ret = BinaryNeonHelper<DT_F32>(m_ctx, m_type, *src0, *src1, *dst, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "BinaryNeonHelper<DT_F32> failed.");
            }
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(m_ctx, "unsupported element type");
            return Status::ERROR;
        }
    }

    AURA_RETURN(m_ctx, ret);
}

} // namespace aura