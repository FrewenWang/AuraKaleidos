#include "aura/ops/misc/threshold.hpp"
#include "threshold_impl.hpp"
#include "aura/runtime/worker_pool.h"
#include "aura/runtime/logger.h"

namespace aura
{

template <typename Tp0, typename Tp1>
static Status ThresholdBinaryNeon(const Mat &src, Mat &dst, Tp1 thresh, Tp0 max_val, DT_S32 start_row, DT_S32 end_row)
{
    using VType = typename neon::QVector<Tp0>::VType;

    constexpr DT_S32 ELEM_COUNTS = 16 / sizeof(Tp0);

    VType vq_zeros, vq_thresh, vq_max_val;
    neon::vdup(vq_zeros, static_cast<Tp0>(0));
    neon::vdup(vq_thresh, SaturateCast<Tp0>(thresh));
    neon::vdup(vq_max_val, max_val);

    DT_S32 channel     = dst.GetSizes().m_channel;
    DT_S32 width_x_c   = dst.GetSizes().m_width * channel;
    DT_S32 width_align = width_x_c & (-ELEM_COUNTS);

    if (SaturateCast<Tp0>(thresh) > thresh) //special case, thresh is smaller than all src data
    {
        for (DT_S32 y = start_row; y < end_row; y++)
        {
            Tp0 *dst_row = dst.Ptr<Tp0>(y);

            DT_S32 x = 0;
            for (; x < width_align; x += ELEM_COUNTS)
            {
                neon::vstore(dst_row + x, vq_max_val);
            }

            for (; x < width_x_c; x++)
            {
                dst_row[x] = max_val;
            }
        }
    }
    else if (SaturateCast<Tp0>(thresh) < thresh) //special case, thresh is larger than all src data
    {
        for (DT_S32 y = start_row; y < end_row; y++)
        {
            Tp0 *dst_row = dst.Ptr<Tp0>(y);

            DT_S32 x = 0;
            for (; x < width_align; x += ELEM_COUNTS)
            {
                neon::vstore(dst_row + x, vq_zeros);
            }

            for (; x < width_x_c; x++)
            {
                dst_row[x] = 0;
            }
        }
    }
    else
    {
        for (DT_S32 y = start_row; y < end_row; y++)
        {
            const Tp0 *src_row = src.Ptr<Tp0>(y);
            Tp0       *dst_row = dst.Ptr<Tp0>(y);

            DT_S32 x = 0;
            for (; x < width_align; x += ELEM_COUNTS)
            {
                auto vq_src  = neon::vload1q(src_row + x);
                auto vq_sign = neon::vcgt(vq_src, vq_thresh);
                auto vq_dst  = neon::vbsl(vq_sign, vq_max_val, vq_zeros);
                neon::vstore(dst_row + x, vq_dst);
            }

            for (; x < width_x_c; x++)
            {
                dst_row[x] = (src_row[x] > thresh) ? max_val : 0;
            }
        }
    }

    return Status::OK;
}

template <typename Tp0, typename Tp1>
static Status ThresholdBinaryInvNeon(const Mat &src, Mat &dst, Tp1 thresh, Tp0 max_val, DT_S32 start_row, DT_S32 end_row)
{
    using VType = typename neon::QVector<Tp0>::VType;

    constexpr DT_S32 ELEM_COUNTS = 16 / sizeof(Tp0);

    VType vq_zeros, vq_thresh, vq_max_val;
    neon::vdup(vq_zeros, static_cast<Tp0>(0));
    neon::vdup(vq_thresh, SaturateCast<Tp0>(thresh));
    neon::vdup(vq_max_val, max_val);

    DT_S32 channel     = dst.GetSizes().m_channel;
    DT_S32 width_x_c   = dst.GetSizes().m_width * channel;
    DT_S32 width_align = width_x_c & (-ELEM_COUNTS);

    if (SaturateCast<Tp0>(thresh) > thresh) //special case, thresh is smaller than all src data
    {
        for (DT_S32 y = start_row; y < end_row; y++)
        {
            Tp0 *dst_row = dst.Ptr<Tp0>(y);

            DT_S32 x = 0;
            for (; x < width_align; x += ELEM_COUNTS)
            {
                neon::vstore(dst_row + x, vq_zeros);
            }

            for (; x < width_x_c; x++)
            {
                dst_row[x] = 0;
            }
        }
    }
    else if (SaturateCast<Tp0>(thresh) < thresh) //special case, thresh is larger than all src data
    {
        for (DT_S32 y = start_row; y < end_row; y++)
        {
            Tp0 *dst_row = dst.Ptr<Tp0>(y);

            DT_S32 x = 0;
            for (; x < width_align; x += ELEM_COUNTS)
            {
                neon::vstore(dst_row + x, vq_max_val);
            }

            for (; x < width_x_c; x++)
            {
                dst_row[x] = max_val;
            }
        }
    }
    else
    {
        for (DT_S32 y = start_row; y < end_row; y++)
        {
            const Tp0 *src_row = src.Ptr<Tp0>(y);
            Tp0       *dst_row = dst.Ptr<Tp0>(y);

            DT_S32 x = 0;
            for (; x < width_align; x += ELEM_COUNTS)
            {
                auto vq_src  = neon::vload1q(src_row + x);
                auto vq_sign = neon::vcgt(vq_src, vq_thresh);
                auto vq_dst  = neon::vbsl(vq_sign, vq_zeros, vq_max_val);
                neon::vstore(dst_row + x, vq_dst);
            }

            for (; x < width_x_c; x++)
            {
                dst_row[x] = (src_row[x] > thresh) ? 0 : max_val;
            }
        }
    }

    return Status::OK;
}

template <typename Tp>
static Status ThresholdTruncNeon(const Mat &src, Mat &dst, Tp thresh, DT_S32 start_row, DT_S32 end_row)
{
    constexpr DT_S32 ELEM_COUNTS = 16 / sizeof(Tp);

    using VType = typename neon::QVector<Tp>::VType;

    VType vq_thresh;
    neon::vdup(vq_thresh, thresh);

    DT_S32 channel     = dst.GetSizes().m_channel;
    DT_S32 width_x_c   = dst.GetSizes().m_width * channel;
    DT_S32 width_align = width_x_c & (-ELEM_COUNTS);

    for (DT_S32 y = start_row; y < end_row; y++)
    {
        const Tp *src_row = src.Ptr<Tp>(y);
        Tp       *dst_row = dst.Ptr<Tp>(y);

        DT_S32 x = 0;
        for (; x < width_align; x += ELEM_COUNTS)
        {
            auto vq_src = neon::vload1q(src_row + x);
            auto vq_dst = neon::vmin(vq_src, vq_thresh);
            neon::vstore(dst_row + x, vq_dst);
        }

        for (; x < width_x_c; x++)
        {
            dst_row[x] = (src_row[x] > thresh) ? thresh : src_row[x];
        }
    }

    return Status::OK;
}

template <typename Tp0, typename Tp1>
static Status ThresholdToZeroNeon(const Mat &src, Mat &dst, Tp1 thresh, DT_S32 start_row, DT_S32 end_row)
{
    using VType = typename neon::QVector<Tp0>::VType;

    constexpr DT_S32 ELEM_COUNTS = 16 / sizeof(Tp0);

    VType vq_thresh, vq_zeros;
    neon::vdup(vq_zeros,  static_cast<Tp0>(0));
    neon::vdup(vq_thresh, SaturateCast<Tp0>(thresh));

    DT_S32 channel     = dst.GetSizes().m_channel;
    DT_S32 width_x_c   = dst.GetSizes().m_width * channel;
    DT_S32 width_align = width_x_c & (-ELEM_COUNTS);

    for (DT_S32 y = start_row; y < end_row; y++)
    {
        const Tp0 *src_row = src.Ptr<Tp0>(y);
        Tp0       *dst_row = dst.Ptr<Tp0>(y);

        DT_S32 x = 0;
        for (; x < width_align; x += ELEM_COUNTS)
        {
            auto vq_src  = neon::vload1q(src_row + x);
            auto vq_sign = neon::vcgt(vq_src, vq_thresh);
            auto vq_dst  = neon::vbsl(vq_sign, vq_src, vq_zeros);
            neon::vstore(dst_row + x, vq_dst);
        }

        for (; x < width_x_c; x++)
        {
            dst_row[x] = (src_row[x] > thresh) ? src_row[x] : 0;
        }
    }

    return Status::OK;
}

template <typename Tp0, typename Tp1>
static Status ThresholdToZeroInvNeon(const Mat &src, Mat &dst, Tp1 thresh, DT_S32 start_row, DT_S32 end_row)
{
    using VType = typename neon::QVector<Tp0>::VType;

    constexpr DT_S32 ELEM_COUNTS = 16 / sizeof(Tp0);

    VType vq_thresh, vq_zeros;
    neon::vdup(vq_zeros, static_cast<Tp0>(0));
    neon::vdup(vq_thresh, SaturateCast<Tp0>(thresh));

    DT_S32 channel     = dst.GetSizes().m_channel;
    DT_S32 width_x_c   = dst.GetSizes().m_width * channel;
    DT_S32 width_align = width_x_c & (-ELEM_COUNTS);

    for (DT_S32 y = start_row; y < end_row; y++)
    {
        const Tp0 *src_row = src.Ptr<Tp0>(y);
        Tp0       *dst_row = dst.Ptr<Tp0>(y);

        DT_S32 x = 0;
        for (; x < width_align; x += ELEM_COUNTS)
        {
            auto vq_src  = neon::vload1q(src_row + x);
            auto vq_sign = neon::vcgt(vq_src,  vq_thresh);
            auto vq_dst  = neon::vbsl(vq_sign, vq_zeros, vq_src);
            neon::vstore(dst_row + x, vq_dst);
        }

        for (; x < width_x_c; x++)
        {
            dst_row[x] = (src_row[x] > thresh) ? 0 : src_row[x];
        }
    }

    return Status::OK;
}

template <typename Tp0, typename Tp1>
static Status ThresholdNeonHelper(Context *ctx, const Mat &src, Mat &dst, Tp1 thresh, Tp0 max_val, DT_S32 type,
                                  const OpTarget &target)
{
    Status ret = Status::ERROR;

    AURA_UNUSED(target);
    WorkerPool *wp = ctx->GetWorkerPool();
    if (DT_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerPool failed");
        return ret;
    }

    DT_S32 oheight = dst.GetSizes().m_height;

    switch(type & AURA_THRESH_MASK_LOW)
    {
        case AURA_THRESH_BINARY:
        {
            ret = wp->ParallelFor(0, oheight, ThresholdBinaryNeon<Tp0, Tp1>, std::cref(src), std::ref(dst), thresh, max_val);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ThresholdBinaryNeon<Tp0, Tp1> failed");
            }
            break;
        }

        case AURA_THRESH_BINARY_INV:
        {
            ret = wp->ParallelFor(0, oheight, ThresholdBinaryInvNeon<Tp0, Tp1>, std::cref(src), std::ref(dst), thresh, max_val);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ThresholdBinaryInvNeon<Tp0, Tp1> failed");
            }
            break;
        }

        case AURA_THRESH_TRUNC:
        {
            ret = wp->ParallelFor(0, oheight, ThresholdTruncNeon<Tp0>, std::cref(src), std::ref(dst), SaturateCast<Tp0>(thresh));
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ThresholdTruncNeon<Tp0> failed");
            }
            break;
        }

        case AURA_THRESH_TOZERO:
        {
            ret = wp->ParallelFor(0, oheight, ThresholdToZeroNeon<Tp0, Tp1>, std::cref(src), std::ref(dst), thresh);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ThresholdToZeroNeon<Tp0, Tp1> failed");
            }
            break;
        }

        case AURA_THRESH_TOZERO_INV:
        {
            ret = wp->ParallelFor(0, oheight, ThresholdToZeroInvNeon<Tp0, Tp1>, std::cref(src), std::ref(dst), thresh);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ThresholdToZeroInvNeon<Tp0, Tp1> failed");
            }
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "threshold method not supported");
            return ret;
        }
    }

    AURA_RETURN(ctx, ret);
}

ThresholdNeon::ThresholdNeon(Context *ctx, const OpTarget &target) : ThresholdImpl(ctx, target)
{}

Status ThresholdNeon::SetArgs(const Array *src, Array *dst, DT_F32 thresh, DT_F32 max_val, DT_S32 type)
{
    Status ret = Status::ERROR;

    if (ThresholdImpl::SetArgs(src, dst, thresh, max_val, type) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "ThresholdImpl::SetArgs failed");
        return ret;
    }

    if ((src->GetArrayType() != ArrayType::MAT) || (dst->GetArrayType() != ArrayType::MAT))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src dst must be mat type");
        return ret;
    }

    return Status::OK;
}

Status ThresholdNeon::Run()
{
    Status ret = Status::ERROR;

    const Mat *src = dynamic_cast<const Mat*>(m_src);
    Mat       *dst = dynamic_cast<Mat*>(m_dst);

    if ((DT_NULL == src) || (DT_NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src or dst is null");
        return ret;
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
            ret = ThresholdNeonHelper<DT_U8, DT_S32>(m_ctx, *src, *dst, ithresh, imax_val, m_type, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "ThresholdNeonHelper<DT_U8, DT_S32> failed");
            }
            break;
        }

        case ElemType::S8:
        {
            DT_S8 imax_val = SaturateCast<DT_S8>(m_max_val);
            ret = ThresholdNeonHelper<DT_S8, DT_S32>(m_ctx, *src, *dst, ithresh, imax_val, m_type, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "ThresholdNeonHelper<DT_S8, DT_S32> failed");
            }
            break;
        }

        case ElemType::U16:
        {
            DT_U16 imax_val = SaturateCast<DT_U16>(m_max_val);
            ret = ThresholdNeonHelper<DT_U16, DT_S32>(m_ctx, *src, *dst, ithresh, imax_val, m_type, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "ThresholdNeonHelper<DT_U16, DT_S32> failed");
            }
            break;
        }

        case ElemType::S16:
        {
            DT_S16 imax_val = SaturateCast<DT_S16>(m_max_val);
            ret = ThresholdNeonHelper<DT_S16, DT_S32>(m_ctx, *src, *dst, ithresh, imax_val, m_type, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "ThresholdNeonHelper<DT_S16, DT_S32> failed");
            }
            break;
        }

#if defined(AURA_ENABLE_NEON_FP16)
        case ElemType::F16:
        {
            ret = ThresholdNeonHelper<MI_F16, MI_F16>(m_ctx, *src, *dst, m_thresh, m_max_val, m_type, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "ThresholdNeonHelper<MI_F16, MI_F16> failed");
            }
            break;
        }
#endif // AURA_ENABLE_NEON_FP16

        case ElemType::F32:
        {
            ret = ThresholdNeonHelper<DT_F32, DT_F32>(m_ctx, *src, *dst, m_thresh, m_max_val, m_type, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "ThresholdNeonHelper<DT_F32, DT_F32> failed");
            }
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(m_ctx, "elem type error");
            ret = Status::ERROR;
        }
    }

    AURA_RETURN(m_ctx, ret);
}

} // namespace aura