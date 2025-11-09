#include "adaptive_threshold_impl.hpp"
#include "aura/runtime/logger.h"
#include "aura/tools/json.h"

namespace aura
{

static std::shared_ptr<AdaptiveThresholdImpl> CreateAdaptiveThresholdImpl(Context *ctx, const OpTarget &target)
{
    std::shared_ptr<AdaptiveThresholdImpl> impl;

    switch (target.m_type)
    {
        case TargetType::NONE:
        {
            impl.reset(new AdaptiveThresholdNone(ctx, target));
            break;
        }

        case TargetType::NEON:
        {
#if defined(AURA_ENABLE_NEON)
            impl.reset(new AdaptiveThresholdNone(ctx, target));
#endif // AURA_ENABLE_NEON
            break;
        }

        default :
        {
            break;
        }
    }

    return impl;
}

AdaptiveThreshold::AdaptiveThreshold(Context *ctx, const OpTarget &target) : Op(ctx, target)
{}

Status AdaptiveThreshold::SetArgs(const Array *src, Array *dst, DT_F32 max_val, AdaptiveThresholdMethod method,
                                  DT_S32 type, DT_S32 block_size, DT_F32 delta)
{
    if ((DT_NULL == m_ctx))
    {
        return Status::ERROR;
    }

    if ((DT_NULL == src) || (DT_NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src/dst is null ptr");
        return Status::ERROR;
    }

    OpTarget impl_target = m_target;

    // set m_impl
    if (DT_NULL == m_impl.get() || impl_target != m_impl->GetOpTarget())
    {
        m_impl = CreateAdaptiveThresholdImpl(m_ctx, impl_target);
    }

    // run SetArgs
    AdaptiveThresholdImpl *adaptive_threshold_impl = dynamic_cast<AdaptiveThresholdImpl *>(m_impl.get());
    if (DT_NULL == adaptive_threshold_impl)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "adaptive_threshold_impl is null ptr");
        return Status::ERROR;
    }

    Status ret = adaptive_threshold_impl->SetArgs(src, dst, max_val, method, type, block_size, delta);

    AURA_RETURN(m_ctx, ret);
}

AURA_EXPORTS Status IAdaptiveThreshold(Context *ctx, const Mat &src, Mat &dst, DT_F32 max_val, AdaptiveThresholdMethod method,
                                       DT_S32 type, DT_S32 block_size, DT_F32 delta, const OpTarget &target)
{
    AdaptiveThreshold adaptive_threshold(ctx, target);

    return OpCall(ctx, adaptive_threshold, &src, &dst, max_val, method, type, block_size, delta);
}

AdaptiveThresholdImpl::AdaptiveThresholdImpl(Context *ctx, const OpTarget &target) : OpImpl(ctx, "AdaptiveThreshold", target),
                                                                                     m_method(AdaptiveThresholdMethod::ADAPTIVE_THRESH_MEAN_C),
                                                                                     m_max_val(0.f), m_type(0), m_block_size(0), m_delta(0.f),
                                                                                     m_src(DT_NULL), m_dst(DT_NULL)
{}

Status AdaptiveThresholdImpl::SetArgs(const Array *src, Array *dst, DT_F32 max_val, AdaptiveThresholdMethod method,
                                      DT_S32 type, DT_S32 block_size, DT_F32 delta)
{
    if (DT_NULL == m_ctx)
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

    if (src->GetElemType() != ElemType::U8 || dst->GetElemType() != ElemType::U8)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src/dst elem type error");
        return Status::ERROR;
    }

    if (dst->GetSizes().m_channel != 1)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "AdaptiveThreshold don't support multi channels");
        return Status::ERROR;
    }

    m_src        = src;
    m_dst        = dst;
    m_max_val    = max_val;
    m_method     = method;
    m_type       = type;
    m_block_size = block_size;
    m_delta      = delta;

    return Status::OK;
}

std::vector<const Array*> AdaptiveThresholdImpl::GetInputArrays() const
{
    return {m_src};
}

std::vector<const Array*> AdaptiveThresholdImpl::GetOutputArrays() const
{
    return {m_dst};
}

std::string AdaptiveThresholdImpl::ToString() const
{
    std::string str;

    str = "op(AdaptiveThreshold)";
    str += " target(" + GetOpTarget().ToString() + ")";
    str += " param(" + AdaptiveThresholdMethodToString(m_method) + " | " +
           "type:" + std::to_string(m_type) + " | " + "block_size:" + std::to_string(m_block_size) + " | "
           "delta_value:" + std::to_string(m_delta) + ")\n";

    return str;
}

DT_VOID AdaptiveThresholdImpl::Dump(const std::string &prefix) const
{
    JsonWrapper json_wrapper(m_ctx, prefix, m_name);

    std::vector<std::string>  names  = {"src", "dst"};
    std::vector<const Array*> arrays = {m_src, m_dst};

    if (json_wrapper.SetArray(names, arrays) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "SetArray src failed");
        return;
    }

    AURA_JSON_SERIALIZE(m_ctx, json_wrapper, m_src, m_dst, m_max_val, m_type, m_block_size, m_delta);
}

} // namespace aura