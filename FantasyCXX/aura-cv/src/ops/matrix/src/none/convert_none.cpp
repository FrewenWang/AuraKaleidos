#include "convert_to_impl.hpp"
#include "aura/ops/core.h"
#include "aura/runtime/mat.h"
#include "aura/runtime/logger.h"
#include "aura/runtime/worker_pool.h"

namespace aura
{

template <typename Tp0, typename Tp1>
AURA_INLINE Status ConvertToNoScaleNoneImpl(const Mat &src, Mat &dst, MI_S32 start_row, MI_S32 end_row)
{
    Sizes3 sz = src.GetSizes();

    for (MI_S32 y = start_row; y < end_row; ++y)
    {
        const Tp0 *src_row = src.Ptr<Tp0>(y);
        Tp1 *dst_row       = dst.Ptr<Tp1>(y);

        for (MI_S32 x = 0; x < sz.m_width * sz.m_channel; ++x)
        {
            dst_row[x] = SaturateCast<Tp1>(src_row[x]);
        }
    }

    return Status::OK;
}

template <typename Tp0, typename Tp1>
static Status ConvertToScaledNoneImpl(const Mat &src, Mat &dst, MI_F32 alpha, MI_F32 beta, MI_S32 start_row, MI_S32 end_row)
{
    Sizes3 sz = src.GetSizes();

    for (MI_S32 y = start_row; y < end_row; ++y)
    {
        const Tp0 *src_row = src.Ptr<Tp0>(y);
        Tp1 *dst_row       = dst.Ptr<Tp1>(y);

        for (MI_S32 x = 0; x < sz.m_width * sz.m_channel; ++x)
        {
            dst_row[x] = SaturateCast<Tp1>(src_row[x] * alpha + beta);
        }
    }

    return Status::OK;
}

template <typename Tp0, typename Tp1>
static Status ConvertToNoneHelper(Context *ctx, const Mat &src, Mat &dst, MI_F32 alpha, MI_F32 beta, OpTarget &target)
{
    Status ret = Status::OK;

    MI_BOOL no_scale = (Abs(alpha - 1.0) < DBL_EPSILON) && (Abs(beta) < DBL_EPSILON);
    MI_S32 height    = dst.GetSizes().m_height;

    if (no_scale)
    {
        if (src.GetElemType() == dst.GetElemType())
        {
            src.CopyTo(dst);
        }
        else
        {
            if (target.m_data.none.enable_mt)
            {
                WorkerPool *wp = ctx->GetWorkerPool();
                if (MI_NULL == wp)
                {
                    AURA_ADD_ERROR_STRING(ctx, "GetWorkerpool failed");
                    return Status::ERROR;
                }

                ret = wp->ParallelFor(static_cast<MI_S32>(0), height,  ConvertToNoScaleNoneImpl<Tp0, Tp1>,
                                      std::cref(src), std::ref(dst));
            }
            else
            {
                ret = ConvertToNoScaleNoneImpl<Tp0, Tp1>(src, dst, static_cast<MI_S32>(0), height);
            }

            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ConvertToNoScaleNoneImpl failed.");
            }
        }
    }
    else
    {   if (target.m_data.none.enable_mt)
        {
            WorkerPool *wp = ctx->GetWorkerPool();
            if (MI_NULL == wp)
            {
                AURA_ADD_ERROR_STRING(ctx, "GetWorkerpool failed");
                return Status::ERROR;
            }

            ret = wp->ParallelFor(static_cast<MI_S32>(0), height,  ConvertToScaledNoneImpl<Tp0, Tp1>,
                                  std::cref(src), std::ref(dst), alpha, beta);
        }
        else
        {
            ret = ConvertToScaledNoneImpl<Tp0, Tp1>(src, dst, alpha, beta, static_cast<MI_S32>(0), height);
        }

        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(ctx, "ConvertToScaledNoneImpl failed.");
        }
    }

    AURA_RETURN(ctx, ret);
}

template <typename Tp>
static Status ConvertToNoneHelper(Context *ctx, const Mat &src, Mat &dst, MI_F32 alpha, MI_F32 beta, OpTarget &target)
{
    Status ret = Status::OK;

    switch (dst.GetElemType())
    {
        case ElemType::U8:
        {
            ret = ConvertToNoneHelper<Tp, MI_U8>(ctx, src, dst, alpha, beta, target);
            break;
        }
        case ElemType::S8:
        {
            ret = ConvertToNoneHelper<Tp, MI_S8>(ctx, src, dst, alpha, beta, target);
            break;
        }
        case ElemType::U16:
        {
            ret = ConvertToNoneHelper<Tp, MI_U16>(ctx, src, dst, alpha, beta, target);
            break;
        }
        case ElemType::S16:
        {
            ret = ConvertToNoneHelper<Tp, MI_S16>(ctx, src, dst, alpha, beta, target);
            break;
        }
        case ElemType::U32:
        {
            ret = ConvertToNoneHelper<Tp, MI_U32>(ctx, src, dst, alpha, beta, target);
            break;
        }
        case ElemType::S32:
        {
            ret = ConvertToNoneHelper<Tp, MI_S32>(ctx, src, dst, alpha, beta, target);
            break;
        }
#if defined(AURA_BUILD_HOST)
        case ElemType::F16:
        {
            ret = ConvertToNoneHelper<Tp, MI_F16>(ctx, src, dst, alpha, beta, target);
            break;
        }
        case ElemType::F32:
        {
            ret = ConvertToNoneHelper<Tp, MI_F32>(ctx, src, dst, alpha, beta, target);
            break;
        }
#endif
        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "ConvertToNone dst type is unsupported.");
            ret = Status::ERROR;
            break;
        }
    }

    AURA_RETURN(ctx, ret);
}

ConvertToNone::ConvertToNone(Context *ctx, const OpTarget &target) : ConvertToImpl(ctx, target)
{}

Status ConvertToNone::SetArgs(const Array *src, Array *dst, MI_F32 alpha, MI_F32 beta)
{
    if (ConvertToImpl::SetArgs(src, dst, alpha, beta) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "ConvertToImpl::SetArgs failed");
        return Status::ERROR;
    }

    if ((src->GetArrayType() != ArrayType::MAT) || (dst->GetArrayType() != ArrayType::MAT))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src dst must be mat type");
        return Status::ERROR;
    }

    return Status::OK;
}

Status ConvertToNone::Run()
{
    const Mat *src = dynamic_cast<const Mat*>(m_src);
    Mat *dst       = dynamic_cast<Mat*>(m_dst);

    if ((MI_NULL == src) || (MI_NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src or dst is null");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    switch (src->GetElemType())
    {
        case ElemType::U8:
        {
            ret = ConvertToNoneHelper<MI_U8>(m_ctx, *src, *dst, m_alpha, m_beta, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "ConvertToNoneHelper<MI_U8> failed.");
            }
            break;
        }
        case ElemType::S8:
        {
            ret = ConvertToNoneHelper<MI_S8>(m_ctx, *src, *dst, m_alpha, m_beta, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "ConvertToNoneHelper<MI_S8> failed.");
            }
            break;
        }
        case ElemType::U16:
        {
            ret = ConvertToNoneHelper<MI_U16>(m_ctx, *src, *dst, m_alpha, m_beta, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "ConvertToNoneHelper<MI_U16> failed.");
            }
            break;
        }
        case ElemType::S16:
        {
            ret = ConvertToNoneHelper<MI_S16>(m_ctx, *src, *dst, m_alpha, m_beta, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "ConvertToNoneHelper<MI_S16> failed.");
            }
            break;
        }
        case ElemType::U32:
        {
            ret = ConvertToNoneHelper<MI_U32>(m_ctx, *src, *dst, m_alpha, m_beta, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "ConvertToNoneHelper<MI_U32> failed.");
            }
            break;
        }
        case ElemType::S32:
        {
            ret = ConvertToNoneHelper<MI_S32>(m_ctx, *src, *dst, m_alpha, m_beta, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "ConvertToNoneHelper<MI_S32> failed.");
            }
            break;
        }
#if defined(AURA_BUILD_HOST)
        case ElemType::F16:
        {
            ret = ConvertToNoneHelper<MI_F16>(m_ctx, *src, *dst, m_alpha, m_beta, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "ConvertToNoneHelper<MI_F16> failed.");
            }
            break;
        }
        case ElemType::F32:
        {
            ret = ConvertToNoneHelper<MI_F32>(m_ctx, *src, *dst, m_alpha, m_beta, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "ConvertToNoneHelper<MI_F32> failed.");
            }
            break;
        }
#endif
        default:
        {
            AURA_ADD_ERROR_STRING(m_ctx, "ConvertToNone src type is unsupported.");
            ret = Status::ERROR;
            break;
        }
    }

    AURA_RETURN(m_ctx, ret);
}

} // namespace aura