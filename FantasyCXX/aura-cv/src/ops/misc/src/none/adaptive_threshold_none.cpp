#include "adaptive_threshold_impl.hpp"
#include "aura/ops/misc/threshold.hpp"
#include "aura/ops/filter.h"
#include "aura/ops/matrix.h"
#include "aura/runtime/logger.h"
#include "aura/runtime/worker_pool.h"

namespace aura
{

static Status AdaptiveThresholdNoneImpl(const Mat &src, Mat &dst, MI_U8 *tab, MI_S32 start_row, MI_S32 end_row)
{
    MI_S32 width  = dst.GetSizes().m_width;

    for (MI_S32 y = start_row; y < end_row; y++)
    {
        const MI_U8 *src_row = src.Ptr<MI_U8>(y);
        MI_U8       *dst_row = dst.Ptr<MI_U8>(y);

        for (MI_S32 x = 0; x < width; x++)
        {
            dst_row[x] = tab[src_row[x] - dst_row[x] + 255];
        }
    }

    return Status::OK;
}

AdaptiveThresholdNone::AdaptiveThresholdNone(Context *ctx, const OpTarget &target) : AdaptiveThresholdImpl(ctx, target)
{}

Status AdaptiveThresholdNone::SetArgs(const Array *src, Array *dst, MI_F32 max_val, AdaptiveThresholdMethod method,
                                      MI_S32 type, MI_S32 block_size, MI_F32 delta)
{
    if (AdaptiveThresholdImpl::SetArgs(src, dst, max_val, method, type, block_size, delta) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "AdaptiveThresholdImpl::SetArgs failed");
        return Status::ERROR;
    }

    if ((src->GetArrayType() != ArrayType::MAT) || (dst->GetArrayType() != ArrayType::MAT))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src dst must be mat type");
        return Status::ERROR;
    }

    return Status::OK;
}

Status AdaptiveThresholdNone::Run()
{
    const Mat *src = dynamic_cast<const Mat*>(m_src);
    Mat       *dst = dynamic_cast<Mat*>(m_dst);

    if ((MI_NULL == src) || (MI_NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src dst is null");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    MI_S32 height = dst->GetSizes().m_height;

    switch(m_method)
    {
        case AdaptiveThresholdMethod::ADAPTIVE_THRESH_MEAN_C:
        {
            // because of the division of boxfilter none impl has an error of 1, so use OpTarget::None temporarily.
            ret = IBoxfilter(m_ctx, *src, *dst, m_block_size, BorderType::REPLICATE, Scalar(0, 0, 0, 0), OpTarget(TargetType::NONE));
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "Boxfilter failed");
                return ret;
            }
            break;
        }

        case AdaptiveThresholdMethod::ADAPTIVE_THRESH_GAUSSIAN_C:
        {
            Mat src_f32(m_ctx, ElemType::F32, src->GetSizes());
            Mat dst_f32(m_ctx, ElemType::F32, dst->GetSizes());
            if (!(src_f32.IsValid() && dst_f32.IsValid()))
            {
                AURA_ADD_ERROR_STRING(m_ctx, "mat create failed");
                return Status::ERROR;
            }

            ret = IConvertTo(m_ctx, *src, src_f32, 1.0f, 0.0f, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "ConvertTo failed");
                return ret;
            }

            ret = IGaussian(m_ctx, src_f32, dst_f32, m_block_size, 0.f, BorderType::REPLICATE, aura::Scalar(0, 0, 0, 0), m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "Gaussian failed");
                return ret;
            }

            ret = IConvertTo(m_ctx, dst_f32, *dst, 1.0f, 0.0f, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "ConvertTo failed");
                return ret;
            }
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(m_ctx, "adaptive threshold method not supported");
            return Status::ERROR;
        }
    }

    MI_S32 imax_val = SaturateCast<MI_U8>(m_max_val);
    MI_S32 idelta = (AURA_THRESH_BINARY == m_type) ? Ceil(m_delta) : Floor(m_delta);

    MI_U8 tab[768];

    if (AURA_THRESH_BINARY == m_type)
    {
        for (MI_S32 i = 0; i < 768; i++)
        {
            tab[i] = static_cast<MI_U8>(i - 255 > -idelta ? imax_val : 0);
        }
    }
    else if (AURA_THRESH_BINARY_INV == m_type)
    {
        for (MI_S32 i = 0; i < 768; i++)
        {
            tab[i] = static_cast<MI_U8>(i - 255 <= -idelta ? imax_val : 0);
        }
    }
    else
    {
        AURA_ADD_ERROR_STRING(m_ctx, "AdaptiveThreshold type error");
        return Status::ERROR;
    }

    if (m_target.m_data.none.enable_mt)
    {
        WorkerPool *wp = m_ctx->GetWorkerPool();
        if (MI_NULL == wp)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "GetWorkerpool failed");
            return Status::ERROR;
        }

        ret = wp->ParallelFor(static_cast<MI_S32>(0), height, AdaptiveThresholdNoneImpl, *src, *dst, tab);
    }
    else
    {
        ret = AdaptiveThresholdNoneImpl(*src, *dst, tab, 0, height);
    }

    AURA_RETURN(m_ctx, ret);
}

} // namespace aura