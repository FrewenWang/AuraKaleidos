#include "aura/ops/misc/threshold.hpp"
#include "threshold_impl.hpp"
#include "aura/runtime/logger.h"
#include "aura/runtime/worker_pool.h"

namespace aura
{

template <typename Tp0, typename Tp1>
static Status ThresholdBinaryNone(const Mat &src, Mat &dst, Tp1 thresh, Tp0 max_val, DT_S32 start_row, DT_S32 end_row)
{
    DT_S32 channel = dst.GetSizes().m_channel;
    DT_S32 width   = dst.GetSizes().m_width;

    for (DT_S32 y = start_row; y < end_row; y++)
    {
        const Tp0 *src_row = src.Ptr<Tp0>(y);
        Tp0       *dst_row = dst.Ptr<Tp0>(y);

        for (DT_S32 x = 0; x < width * channel; x++)
        {
            dst_row[x] = (src_row[x] > thresh) ? max_val : static_cast<Tp0>(0);
        }
    }

    return Status::OK;
}

template <typename Tp0, typename Tp1>
static Status ThresholdBinaryInvNone(const Mat &src, Mat &dst, Tp1 thresh, Tp0 max_val, DT_S32 start_row, DT_S32 end_row)
{
    DT_S32 channel = dst.GetSizes().m_channel;
    DT_S32 width   = dst.GetSizes().m_width;

    for (DT_S32 y = start_row; y < end_row; y++)
    {
        const Tp0 *src_row = src.Ptr<Tp0>(y);
        Tp0       *dst_row = dst.Ptr<Tp0>(y);

        for (DT_S32 x = 0; x < width * channel; x++)
        {
            dst_row[x] = (src_row[x] > thresh) ? static_cast<Tp0>(0) : max_val;
        }
    }

    return Status::OK;
}

template <typename Tp>
static Status ThresholdTruncNone(const Mat &src, Mat &dst, Tp thresh, DT_S32 start_row, DT_S32 end_row)
{
    DT_S32 channel = dst.GetSizes().m_channel;
    DT_S32 width   = dst.GetSizes().m_width;

    for (DT_S32 y = start_row; y < end_row; y++)
    {
        const Tp *src_row = src.Ptr<Tp>(y);
        Tp       *dst_row = dst.Ptr<Tp>(y);

        for (DT_S32 x = 0; x < width * channel; x++)
        {
            dst_row[x] = (src_row[x] > thresh) ? thresh : src_row[x];
        }
    }

    return Status::OK;
}

template <typename Tp0, typename Tp1>
static Status ThresholdToZeroNone(const Mat &src, Mat &dst, Tp1 thresh, DT_S32 start_row, DT_S32 end_row)
{
    DT_S32 channel = dst.GetSizes().m_channel;
    DT_S32 width   = dst.GetSizes().m_width;

    for (DT_S32 y = start_row; y < end_row; y++)
    {
        const Tp0 *src_row = src.Ptr<Tp0>(y);
        Tp0       *dst_row = dst.Ptr<Tp0>(y);

        for (DT_S32 x = 0; x < width * channel; x++)
        {
            dst_row[x] = (src_row[x] > thresh) ? src_row[x] : static_cast<Tp0>(0);
        }
    }

    return Status::OK;
}

template <typename Tp0, typename Tp1>
static Status ThresholdToZeroInvNone(const Mat &src, Mat &dst, Tp1 thresh, DT_S32 start_row, DT_S32 end_row)
{
    DT_S32 channel = dst.GetSizes().m_channel;
    DT_S32 width   = dst.GetSizes().m_width;

    for (DT_S32 y = start_row; y < end_row; y++)
    {
        const Tp0 *src_row = src.Ptr<Tp0>(y);
        Tp0       *dst_row = dst.Ptr<Tp0>(y);

        for (DT_S32 x = 0; x < width * channel; x++)
        {
            dst_row[x] = (src_row[x] > thresh) ? static_cast<Tp0>(0) : src_row[x];
        }
    }

    return Status::OK;
}

template <typename Tp0, typename Tp1>
static Status ThresholdNoneHelper(Context *ctx, const Mat &src, Mat &dst, Tp1 thresh, Tp0 max_val, DT_S32 type, const OpTarget &target)
{
    Status ret = Status::ERROR;

    DT_S32 height = dst.GetSizes().m_height;

    switch (type & AURA_THRESH_MASK_LOW)
    {
        case AURA_THRESH_BINARY:
        {
            if (target.m_data.none.enable_mt)
            {
                WorkerPool *wp = ctx->GetWorkerPool();
                if (DT_NULL == wp)
                {
                    AURA_ADD_ERROR_STRING(ctx, "GetWorkerpool failed");
                    return ret;
                }

                ret = wp->ParallelFor(static_cast<DT_S32>(0), height, ThresholdBinaryNone<Tp0, Tp1>, std::cref(src), std::ref(dst), thresh, max_val);
            }
            else
            {
                ret = ThresholdBinaryNone<Tp0, Tp1>(src, dst, thresh, max_val, 0, height);
            }
            break;
        }

        case AURA_THRESH_BINARY_INV:
        {
            if (target.m_data.none.enable_mt)
            {
                WorkerPool *wp = ctx->GetWorkerPool();
                if (DT_NULL == wp)
                {
                    AURA_ADD_ERROR_STRING(ctx, "GetWorkerpool failed");
                    return Status::ERROR;
                }

                ret = wp->ParallelFor(static_cast<DT_S32>(0), height, ThresholdBinaryInvNone<Tp0, Tp1>, std::cref(src), std::ref(dst), thresh, max_val);
            }
            else
            {
                ret = ThresholdBinaryInvNone<Tp0, Tp1>(src, dst, thresh, max_val, 0, height);
            }
            break;
        }

        case AURA_THRESH_TRUNC:
        {
            if (target.m_data.none.enable_mt)
            {
                WorkerPool *wp = ctx->GetWorkerPool();
                if (DT_NULL == wp)
                {
                    AURA_ADD_ERROR_STRING(ctx, "GetWorkerpool failed");
                    return Status::ERROR;
                }

                ret = wp->ParallelFor(static_cast<DT_S32>(0), height, ThresholdTruncNone<Tp0>, std::cref(src), std::ref(dst), SaturateCast<Tp0>(thresh));
            }
            else
            {
                ret = ThresholdTruncNone<Tp0>(src, dst, SaturateCast<Tp0>(thresh), 0, height);
            }
            break;
        }

        case AURA_THRESH_TOZERO:
        {
            if (target.m_data.none.enable_mt)
            {
                WorkerPool *wp = ctx->GetWorkerPool();
                if (DT_NULL == wp)
                {
                    AURA_ADD_ERROR_STRING(ctx, "GetWorkerpool failed");
                    return Status::ERROR;
                }

                ret = wp->ParallelFor(static_cast<DT_S32>(0), height, ThresholdToZeroNone<Tp0, Tp1>, std::cref(src), std::ref(dst), thresh);
            }
            else
            {
                ret = ThresholdToZeroNone<Tp0, Tp1>(src, dst, thresh, 0, height);
            }
            break;
        }

        case AURA_THRESH_TOZERO_INV:
        {
            if (target.m_data.none.enable_mt)
            {
                WorkerPool *wp = ctx->GetWorkerPool();
                if (DT_NULL == wp)
                {
                    AURA_ADD_ERROR_STRING(ctx, "GetWorkerpool failed");
                    return Status::ERROR;
                }

                ret = wp->ParallelFor(static_cast<DT_S32>(0), height, ThresholdToZeroInvNone<Tp0, Tp1>, std::cref(src), std::ref(dst), thresh);
            }
            else
            {
                ret = ThresholdToZeroInvNone<Tp0, Tp1>(src, dst, thresh, 0, height);
            }
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "threshold method not supported");
            ret = Status::ERROR;
        }
    }

    AURA_RETURN(ctx, ret);
}

ThresholdNone::ThresholdNone(Context *ctx, const OpTarget &target) : ThresholdImpl(ctx, target)
{}

Status ThresholdNone::SetArgs(const Array *src, Array *dst, DT_F32 thresh, DT_F32 max_val, DT_S32 type)
{
    if (ThresholdImpl::SetArgs(src, dst, thresh, max_val, type) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "ThresholdImpl::SetArgs failed");
        return Status::ERROR;
    }

    if ((src->GetArrayType() != ArrayType::MAT) || (dst->GetArrayType() != ArrayType::MAT))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src dst must be mat type");
        return Status::ERROR;
    }

    return Status::OK;
}

Status ThresholdNone::Run()
{
    Status ret = Status::ERROR;

    const Mat *src = dynamic_cast<const Mat*>(m_src);
    Mat       *dst = dynamic_cast<Mat*>(m_dst);

    if ((DT_NULL == src) || (DT_NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src or dst is null");
        return Status::ERROR;
    }

    DT_S32 ithresh = Floor(m_thresh);

    switch (src->GetElemType())
    {
        case ElemType::U8:
        {
            if (1 == src->GetSizes().m_channel)
            {
                ret = ReCalcThresh(ithresh);
                if (ret != Status::OK)
                {
                    AURA_ADD_ERROR_STRING(m_ctx, "ReCalcThresh failed");
                    return ret;
                }
            }

            DT_U8 imax_val = SaturateCast<DT_U8>(m_max_val);
            ret = ThresholdNoneHelper<DT_U8, DT_S32>(m_ctx, *src, *dst, ithresh, imax_val, m_type, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "ThresholdNoneHelper<DT_U8, DT_S32> failed");
            }
            break;
        }

        case ElemType::S8:
        {
            DT_S8 imax_val = SaturateCast<DT_S8>(m_max_val);
            ret = ThresholdNoneHelper<DT_S8, DT_S32>(m_ctx, *src, *dst, ithresh, imax_val, m_type, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "ThresholdNoneHelper<DT_S8, DT_S32> failed");
            }
            break;
        }

        case ElemType::U16:
        {
            DT_U16 imax_val = SaturateCast<DT_U16>(m_max_val);
            ret = ThresholdNoneHelper<DT_U16, DT_S32>(m_ctx, *src, *dst, ithresh, imax_val, m_type, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "ThresholdNoneHelper<DT_U16, DT_S32> failed");
            }
            break;
        }

        case ElemType::S16:
        {
            DT_S16 imax_val = SaturateCast<DT_S16>(m_max_val);
            ret = ThresholdNoneHelper<DT_S16, DT_S32>(m_ctx, *src, *dst, ithresh, imax_val, m_type, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "ThresholdNoneHelper<DT_S16, DT_S32> failed");
            }
            break;
        }

#if defined(AURA_BUILD_HOST)
        case ElemType::F16:
        {
            ret = ThresholdNoneHelper<MI_F16, MI_F16>(m_ctx, *src, *dst, m_thresh, m_max_val, m_type, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "ThresholdNoneHelper<MI_F16, MI_F16> failed");
            }
            break;
        }

        case ElemType::F32:
        {
            ret = ThresholdNoneHelper<DT_F32, DT_F32>(m_ctx, *src, *dst, m_thresh, m_max_val, m_type, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "ThresholdNoneHelper<DT_F32, DT_F32> failed");
            }
            break;
        }
#endif // AURA_BUILD_HOST

        default:
        {
            AURA_ADD_ERROR_STRING(m_ctx, "elem type error");
            ret = Status::ERROR;
        }
    }

    AURA_RETURN(m_ctx, ret);
}

} // namespace aura