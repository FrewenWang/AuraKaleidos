#include "binary_impl.hpp"
#include "aura/runtime/logger.h"
#include "aura/runtime/worker_pool.h"

namespace aura
{

template <typename Tp = AURA_VOID>
struct BinaryMinFunctor
{
    constexpr Tp operator()(const Tp& left, const Tp& right) const
    {
        return (left < right) ? left : right;
    }
};

template <typename Tp = AURA_VOID>
struct BinaryMaxFunctor
{
    constexpr Tp operator()(const Tp& left, const Tp& right) const
    {
        return (left > right) ? left : right;
    }
};

template <typename Tp, typename Functor>
static Status BinaryNoneImpl(const Mat &src0, const Mat &src1, Mat &dst, MI_S32 start_row, MI_S32 end_row)
{
    Functor op;

    const MI_S32 width       = src0.GetSizes().m_width;
    const MI_S32 channel     = src0.GetSizes().m_channel;
    const MI_S32 num_per_row = width * channel;

    for (MI_S32 y = start_row; y < end_row; y++)
    {
        const Tp *src0_row = src0.Ptr<Tp>(y);
        const Tp *src1_row = src1.Ptr<Tp>(y);
        Tp *dst_row = dst.Ptr<Tp>(y);

        for (MI_S32 x = 0; x < num_per_row; x++)
        {
            dst_row[x] = op(src0_row[x], src1_row[x]);
        }
    }

    return Status::OK;
}

template <typename Tp>
static Status BinaryNoneHelper(Context *ctx, BinaryOpType type, const Mat &src0, const Mat &src1, Mat &dst, OpTarget &target)
{
    Status ret = Status::OK;
    MI_S32 height = dst.GetSizes().m_height;

#define BINARY_NONE_IMPL(functor)                                                                       \
    if (target.m_data.none.enable_mt)                                                                   \
    {                                                                                                   \
        WorkerPool *wp = ctx->GetWorkerPool();                                                          \
        if (MI_NULL == wp)                                                                              \
        {                                                                                               \
            AURA_ADD_ERROR_STRING(ctx, "GetWorkerpool failed");                                         \
            return Status::ERROR;                                                                       \
        }                                                                                               \
        ret = wp->ParallelFor(static_cast<MI_S32>(0), height, BinaryNoneImpl<Tp, functor>,              \
                              std::cref(src0), std::cref(src1), std::ref(dst));                         \
    }                                                                                                   \
    else                                                                                                \
    {                                                                                                   \
        ret = BinaryNoneImpl<Tp, functor>(src0, src1, dst, static_cast<MI_S32>(0), height);             \
    }                                                                                                   \
                                                                                                        \
    if (ret != Status::OK)                                                                              \
    {                                                                                                   \
        MI_CHAR error_msg[128];                                                                         \
        std::snprintf(error_msg, sizeof(error_msg), "BinaryNoneImpl<%s> failed", #functor);             \
        AURA_ADD_ERROR_STRING(ctx, error_msg);                                                          \
    }

    switch (type)
    {
        case BinaryOpType::MIN:
        {
            BINARY_NONE_IMPL(BinaryMinFunctor<Tp>);
            break;
        }
        case BinaryOpType::MAX:
        {
            BINARY_NONE_IMPL(BinaryMaxFunctor<Tp>);
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "unsupported binary operator type");
            return Status::ERROR;
        }
    }

    AURA_RETURN(ctx, ret);
}

BinaryNone::BinaryNone(Context *ctx, const OpTarget &target) : BinaryImpl(ctx, target)
{}

Status BinaryNone::SetArgs(const Array *src0, const Array *src1, Array *dst, BinaryOpType type)
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

Status BinaryNone::Run()
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
            ret = BinaryNoneHelper<MI_U8>(m_ctx, m_type, *src0, *src1, *dst, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "BinaryNoneHelper<MI_U8> failed.");
            }
            break;
        }
        case ElemType::S8:
        {
            ret = BinaryNoneHelper<MI_S8>(m_ctx, m_type, *src0, *src1, *dst, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "BinaryNoneHelper<MI_S8> failed.");
            }
            break;
        }
        case ElemType::U16:
        {
            ret = BinaryNoneHelper<MI_U16>(m_ctx, m_type, *src0, *src1, *dst, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "BinaryNoneHelper<MI_U16> failed.");
            }
            break;
        }
        case ElemType::S16:
        {
            ret = BinaryNoneHelper<MI_S16>(m_ctx, m_type, *src0, *src1, *dst, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "BinaryNoneHelper<MI_S16> failed.");
            }
            break;
        }
        case ElemType::U32:
        {
            ret = BinaryNoneHelper<MI_U32>(m_ctx, m_type, *src0, *src1, *dst, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "BinaryNoneHelper<MI_U32> failed.");
            }
            break;
        }
        case ElemType::S32:
        {
            ret = BinaryNoneHelper<MI_S32>(m_ctx, m_type, *src0, *src1, *dst, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "BinaryNoneHelper<MI_S32> failed.");
            }
            break;
        }
#if defined(AURA_BUILD_HOST)
        case ElemType::F16:
        {
            ret = BinaryNoneHelper<MI_F16>(m_ctx, m_type, *src0, *src1, *dst, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "BinaryNoneHelper<MI_F16> failed.");
            }
            break;
        }
        case ElemType::F32:
        {
            ret = BinaryNoneHelper<MI_F32>(m_ctx, m_type, *src0, *src1, *dst, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "BinaryNoneHelper<MI_F32> failed.");
            }
            break;
        }
#endif

        default:
        {
            AURA_ADD_ERROR_STRING(m_ctx, "unsupported element type");
            return Status::ERROR;
        }
    }

    AURA_RETURN(m_ctx, ret);
}

} // namespace aura