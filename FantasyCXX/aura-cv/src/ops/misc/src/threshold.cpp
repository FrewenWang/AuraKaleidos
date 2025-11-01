#include "aura/ops/misc/threshold.hpp"
#include "threshold_impl.hpp"
#include "aura/ops/hist.h"
#include "aura/runtime/logger.h"
#include "aura/tools/json.h"

namespace aura
{

static std::shared_ptr<ThresholdImpl> CreateThresholdImpl(Context *ctx, const OpTarget &target)
{
    std::shared_ptr<ThresholdImpl> impl;

    switch (target.m_type)
    {
        case TargetType::NONE:
        {
            impl.reset(new ThresholdNone(ctx, target));
            break;
        }

        case TargetType::NEON:
        {
#if defined(AURA_ENABLE_NEON)
           impl.reset(new ThresholdNeon(ctx, target));
#endif
           break;
        }

        case TargetType::HVX:
        {
#if defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON)
            impl.reset(new ThresholdHvx(ctx, target));
#endif // AURA_BUILD_HEXAGON
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "target type error");
            break;
        }
    }

    return impl;
}

Threshold::Threshold(Context *ctx, const OpTarget &target) : Op(ctx, target)
{}

Status Threshold::SetArgs(const Array *src, Array *dst, MI_F32 thresh, MI_F32 max_val, MI_S32 type)
{
    if ((MI_NULL == m_ctx))
    {
        return Status::ERROR;
    }

    if ((MI_NULL == src) || (MI_NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src/dst is null ptr");
        return Status::ERROR;
    }

    OpTarget impl_target = m_target;

    // check impl type
    switch (m_target.m_type)
    {
        case TargetType::NEON:
        {
#if defined(AURA_ENABLE_NEON)
            if (CheckNeonWidth(*src) != Status::OK)
            {
                impl_target = OpTarget::None();
            }
#endif // AURA_ENABLE_NEON
            break;
        }

        case TargetType::HVX:
        {
#if defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON)
            if (CheckHvxWidth(*src) != Status::OK)
            {
                impl_target = OpTarget::None();
            }
#endif // defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON)
            break;
        }

        default:
        {
            break;
        }
    }

    // set m_impl
    if (MI_NULL == m_impl.get() || impl_target != m_impl->GetOpTarget())
    {
        m_impl = CreateThresholdImpl(m_ctx, impl_target);
    }

    // run initialize
    ThresholdImpl *threshold_impl = dynamic_cast<ThresholdImpl*>(m_impl.get());
    if (MI_NULL == threshold_impl)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "threshold_impl is null ptr");
        return Status::ERROR;
    }

    Status ret = threshold_impl->SetArgs(src, dst, thresh, max_val, type);

    AURA_RETURN(m_ctx, ret);
}

AURA_EXPORTS Status IThreshold(Context *ctx, const Mat &src, Mat &dst,
                               MI_F32 thresh, MI_F32 max_val, MI_S32 type,
                               const OpTarget &target)
{
    Threshold threshold(ctx, target);

    return OpCall(ctx, threshold, &src, &dst, thresh, max_val, type);
}

ThresholdImpl::ThresholdImpl(Context *ctx, const OpTarget &target) : OpImpl(ctx, "Threshold", target),
                                                                     m_thresh(0.f), m_max_val(0.f), m_type(0),
                                                                     m_src(MI_NULL), m_dst(MI_NULL)
{}

Status ThresholdImpl::SetArgs(const Array *src, Array *dst, MI_F32 thresh, MI_F32 max_val, MI_S32 type)
{
    if (MI_NULL == m_ctx)
    {
        return Status::ERROR;
    }

    if (!(src->IsValid() && dst->IsValid()))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "invalid src or dst");
        return Status::ERROR;
    }

    if (!src->IsSizesEqual(*dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src and dst should have the same size");
        return Status::ERROR;
    }

    m_src     = src;
    m_dst     = dst;
    m_thresh  = thresh;
    m_max_val = max_val;
    m_type    = type;

    return Status::OK;
}

static Status ThresholdOstu(Context *ctx, const Mat &src, MI_S32 &thresh, OpTarget &target)
{
    std::vector<MI_U32> histgram(256, 0);
    Scalar range = {0, 256};

    if (ICalcHist(ctx, src, 0, histgram, 256, range, Mat(), MI_FALSE, target) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "CalcHist failed");
        return Status::ERROR;
    }

    MI_S32 width  = src.GetSizes().m_width;
    MI_S32 height = src.GetSizes().m_height;

    MI_F32 value_avg = 0.0f;
    MI_F32 scale = 1.0f / (width * height);

    for (MI_S32 i = 0; i < 256; i++)
    {
        value_avg += i * static_cast<MI_F32>(histgram[i]);
    }

    value_avg *= scale;
    MI_F32 value_foreground = 0, ratio_foreground = 0;
    MI_F32 max_sigma = 0, value_max = 0;

    for (MI_S32 i = 0; i < 256; i++)
    {
        MI_F32 ratio_i, ratio_background, value_background, sigma;
        ratio_i = histgram[i] * scale;
        value_foreground *= ratio_foreground;
        ratio_foreground += ratio_i;
        ratio_background = 1. - ratio_foreground;

        if (ratio_foreground < 1e-6 || ratio_background < 1e-6)
        {
            continue;
        }

        value_foreground = (value_foreground + i * ratio_i) / ratio_foreground;
        value_background = (value_avg - ratio_foreground * value_foreground) / ratio_background;
        sigma = ratio_foreground * ratio_background * (value_foreground - value_background) * (value_foreground - value_background);

        if (sigma > max_sigma)
        {
            max_sigma = sigma;
            value_max = i;
        }
    }

    thresh = static_cast<MI_S32>(value_max);

    return Status::OK;
}

static Status ThresholdTriangle(Context *ctx, const Mat &src, MI_S32 &thresh, OpTarget &target)
{
    std::vector<MI_U32> histgram(256, 0);
    Scalar range = {0, 256};

    if (ICalcHist(ctx, src, 0, histgram, 256, range, Mat(), MI_FALSE, target) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "CalcHist failed");
        return Status::ERROR;
    }

    MI_S32 i = 0, j = 0;
    MI_S32 left_bound = 0, right_bound = 0;
    MI_S32 max_index  = 0;
    MI_U32 max_val    = 0;
    MI_S32 is_flipped = 0;

    for (i = 0; i < 256; i++)
    {
        if (histgram[i] > 0)
        {
            left_bound = i;
            break;
        }
    }

    if (left_bound > 0)
    {
        left_bound--;
    }

    for (i = 255; i > 0; i--)
    {
        if (histgram[i] > 0)
        {
            right_bound = i;
            break;
        }
    }

    if (right_bound < 255)
    {
        right_bound++;
    }

    for (i = 0; i < 256; i++)
    {
        if (histgram[i] > max_val)
        {
            max_val = histgram[i];
            max_index = i;
        }
    }

    if (max_index - left_bound < right_bound - max_index)
    {
        is_flipped = 1;
        i = 0;
        j = 255;
        MI_U32 temp_hist = 0;
        // swap data
        while (i < j)
        {
            temp_hist   = histgram[i];
            histgram[i] = histgram[j];
            histgram[j] = temp_hist;
            i++;
            j--;
        }
        left_bound = 255 - right_bound;
        max_index  = 255 - max_index;
    }

    MI_S32 threshold = left_bound;
    MI_F32 a = max_val;
    MI_F32 b = left_bound - max_index;
    MI_F32 dist = 0, tempdist = 0;

    for (i = left_bound + 1; i <= max_index; i++)
    {
        tempdist = a * i + b * histgram[i];
        if (tempdist > dist)
        {
            dist = tempdist;
            threshold = i;
        }
    }

    threshold--;

    if (is_flipped)
    {
        threshold = 255 - threshold;
    }

    thresh = threshold;

    return Status::OK;
}

Status ThresholdImpl::ReCalcThresh(MI_S32 &thresh)
{
    Status ret = Status::OK;

    const Mat *src = dynamic_cast<const Mat*>(m_src);
    if (MI_NULL == src)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src is null");
        return Status::ERROR;
    }

    OpTarget target = (TargetType::HVX == m_target.m_type) ? m_target : OpTarget(TargetType::NONE);

    switch(m_type & AURA_THRESH_MASK_HIGH)
    {
        case AURA_THRESH_OTSU:
        {
            ret = ThresholdOstu(m_ctx, *src, thresh, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "ThresholdOstu failed");
                return Status::ERROR;
            }

            break;
        }

        case AURA_THRESH_TRIANGLE:
        {
            ret = ThresholdTriangle(m_ctx, *src, thresh, target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "ThresholdTriangle failed");
                return Status::ERROR;
            }

            break;
        }

        default :
        {
            break; //normal case, no need to calc thresh
        }
    }

    AURA_RETURN(m_ctx, ret);
}

std::vector<const Array*> ThresholdImpl::GetInputArrays() const
{
    return {m_src};
}

std::vector<const Array*> ThresholdImpl::GetOutputArrays() const
{
    return {m_dst};
}

std::string ThresholdImpl::ToString() const
{
    std::string str;

    str = "op(Threshold)";
    str += " target(" + GetOpTarget().ToString() + ")";
    str += " param(" + std::to_string(m_thresh) + " | " +
           "max_val:" + std::to_string(m_max_val) + " | "
           "type:" + std::to_string(m_type) + ")\n";

    return str;
}

AURA_VOID ThresholdImpl::Dump(const std::string &prefix) const
{
    JsonWrapper json_wrapper(m_ctx, prefix, m_name);

    std::vector<std::string>  names  = {"src", "dst"};
    std::vector<const Array*> arrays = {m_src, m_dst};

    if (json_wrapper.SetArray(names, arrays) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "SetArray src failed");
        return;
    }

    AURA_JSON_SERIALIZE(m_ctx, json_wrapper, m_src, m_dst, m_thresh, m_max_val, m_type);
}

} // namespace aura