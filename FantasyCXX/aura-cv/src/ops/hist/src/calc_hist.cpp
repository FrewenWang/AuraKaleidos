#include "aura/ops/hist/calc_hist.hpp"
#include "calc_hist_impl.hpp"
#include "aura/runtime/logger.h"
#include "aura/tools/json.h"

namespace aura
{

static std::shared_ptr<CalcHistImpl> CreateCalcHistImpl(Context *ctx, const OpTarget &target)
{
    std::shared_ptr<CalcHistImpl> impl;

    switch (target.m_type)
    {
        case TargetType::NONE:
        {
            impl.reset(new CalcHistNone(ctx, target));
            break;
        }

        case TargetType::HVX:
        {
#if (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))
            impl.reset(new CalcHistHvx(ctx, target));
#endif // (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))
            break;
        }

        default :
        {
            break;
        }
    }

    return impl;
}

CalcHist::CalcHist(Context *ctx, const OpTarget &target) : Op(ctx, target)
{}

Status CalcHist::SetArgs(const Array *src, MI_S32 channel, std::vector<MI_U32> &hist, MI_S32 hist_size, 
                         const Scalar &ranges, const Array *mask, MI_BOOL accumulate)
{
    if (MI_NULL == m_ctx)
    {
        return Status::ERROR;
    }

    if ((MI_NULL == src))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src is null ptr");
        return Status::ERROR;
    }

    OpTarget impl_target = m_target;

    switch (m_target.m_type)
    {
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
        m_impl = CreateCalcHistImpl(m_ctx, impl_target);
    }

    // run SetArgs
    CalcHistImpl *calc_hist_impl = dynamic_cast<CalcHistImpl *>(m_impl.get());
    if (MI_NULL == calc_hist_impl)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "calc_hist_impl is null ptr");
        return Status::ERROR;
    }

    Status ret = calc_hist_impl->SetArgs(src, channel, hist, hist_size, ranges, mask, accumulate);

    AURA_RETURN(m_ctx, ret);
}

AURA_EXPORTS Status ICalcHist(Context *ctx, const Mat &src, MI_S32 channel, std::vector<MI_U32> &hist, MI_S32 hist_size,
                              const Scalar &ranges, const Mat &mask, MI_BOOL accumulate, const OpTarget &target)
{
    CalcHist calchist(ctx, target);

    return OpCall(ctx, calchist, &src, channel, hist, hist_size, ranges, &mask, accumulate);
}

CalcHistImpl::CalcHistImpl(Context *ctx, const OpTarget &target) : OpImpl(ctx, "CalcHist", target),
                                                                   m_channel(0), m_hist_size(0), m_accumulate(MI_FALSE),
                                                                   m_src(MI_NULL), m_mask(MI_NULL), m_hist(MI_NULL)
{}

Status CalcHistImpl::SetArgs(const Array *src, MI_S32 channel, std::vector<MI_U32> &hist, MI_S32 hist_size, 
                             const Scalar &ranges, const Array *mask, MI_BOOL accumulate)
{
    Status ret = Status::ERROR;

    if (MI_NULL == m_ctx)
    {
        return ret;
    }

    if (!(src->IsValid()))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "invalid src");
        return ret;
    }

    if (channel >= src->GetSizes().m_channel)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "the channel of input is wrong");
        return ret;
    }

    if (hist.size() < (MI_U32)hist_size)
    {
        hist.resize(hist_size);
    }

    if (mask->IsValid())
    {
        if (src->GetSizes().m_height != mask->GetSizes().m_height || src->GetSizes().m_width != mask->GetSizes().m_width)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "src and mask must have the same height and width");
            return ret;
        }

        if (mask->GetElemType() != ElemType::U8)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "mask element type must be u8");
            return ret;
        }
    }

    m_src        = src;
    m_mask       = mask;
    m_channel    = channel;
    m_hist_size  = hist_size;
    m_hist       = &hist;
    m_ranges     = ranges;
    m_accumulate = accumulate;

    return Status::OK;
}

std::vector<const Array*> CalcHistImpl::GetInputArrays() const
{
    return {m_src, m_mask};
}

std::string CalcHistImpl::ToString() const
{
    std::string str;

    str = "op(CalcHist)";
    str += " target(" + GetOpTarget().ToString() + ")";
    str += " param(" + std::to_string(m_channel) + " | " + 
           "hist_size:" + std::to_string(m_hist_size)+ ")\n";

    return str;
}

AURA_VOID CalcHistImpl::Dump(const std::string &prefix) const
{
    JsonWrapper json_wrapper(m_ctx, prefix, m_name);

    if (json_wrapper.SetArray("src", m_src) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "SetArray src failed");
        return;
    }

    if (json_wrapper.SetArray("mask", m_mask, MI_FALSE) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "SetArray src failed");
        return;
    }

    AURA_JSON_SERIALIZE(m_ctx, json_wrapper, m_src, m_mask, *m_hist, m_channel, m_hist_size, m_ranges, m_accumulate);
}

} // namespace aura