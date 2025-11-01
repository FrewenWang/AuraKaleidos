#include "fast_impl.hpp"
#include "aura/runtime/logger.h"
#include "aura/tools/json.h"

namespace aura
{

static std::shared_ptr<FastImpl> CreateFastImpl(Context *ctx, const OpTarget &target)
{
    std::shared_ptr<FastImpl> impl;

    switch (target.m_type)
    {
        case TargetType::NONE:
        {
            impl.reset(new FastNone(ctx, target));
            break;
        }

        case TargetType::NEON:
        {
#if defined(AURA_ENABLE_NEON)
            impl.reset(new FastNeon(ctx, target));
#endif // AURA_ENABLE_NEON
            break;
        }

        case TargetType::HVX:
        {
#if (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))
            impl.reset(new FastHvx(ctx, target));
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

Fast::Fast(Context *ctx, const OpTarget &target) : Op(ctx, target)
{}

Status Fast::SetArgs(const Array *src, std::vector<KeyPoint> &key_points, MI_S32 threshold,
                     MI_BOOL nonmax_suppression, FastDetectorType type, MI_U32 max_num_corners)
{

    if ((MI_NULL == m_ctx))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "m_ctx is NULL");
        return Status::ERROR;
    }

    if (MI_NULL == src)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src is null ptr");
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
        m_impl = CreateFastImpl(m_ctx, impl_target);
    }

    // run initialize
    FastImpl *fast_impl = dynamic_cast<FastImpl *>(m_impl.get());
    if (MI_NULL == fast_impl)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "fast_impl is null ptr");
        return Status::ERROR;
    }

    Status ret = fast_impl->SetArgs(src, key_points, threshold, nonmax_suppression, type, max_num_corners);

    AURA_RETURN(m_ctx, ret);
}

AURA_EXPORTS Status IFast(Context *ctx, const Mat &src, std::vector<KeyPoint> &key_points, MI_S32 threshold,
                          MI_BOOL nonmax_suppression, FastDetectorType type, MI_U32 max_num_corners, const OpTarget &target)
{
    Fast fast(ctx, target);

    return OpCall(ctx, fast, &src, key_points, threshold, nonmax_suppression, type, max_num_corners);
}

FastImpl::FastImpl(Context *ctx, const OpTarget &target) : OpImpl(ctx, "Fast", target),
                                                           m_threshold(0), m_max_num_corners(0),
                                                           m_nonmax_suppression(MI_FALSE),
                                                           m_detector_type(FastDetectorType::FAST_9_16),
                                                           m_src(MI_NULL)
{}

Status FastImpl::SetArgs(const Array *src, std::vector<KeyPoint> &key_points, MI_S32 threshold,
                         MI_BOOL nonmax_suppression, FastDetectorType type, MI_U32 max_num_corners)
{
    if (MI_NULL == m_ctx)
    {
        return Status::ERROR;
    }

    if (!(src->IsValid()))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "invalid src or dst");
        return Status::ERROR;
    }

    if(0 == max_num_corners)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "max_num_corners must be greater than 0");
        return Status::ERROR;
    }

    if (key_points.size() != 0)
    {
        key_points.clear();
    }

    if (key_points.capacity() < max_num_corners)
    {
        key_points.reserve(max_num_corners);
    }

    if (src->GetSizes().m_channel != 1)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src channel only support 1");
        return Status::ERROR;
    }

    if (src->GetElemType() != ElemType::U8)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "elem_type only support u8");
        return Status::ERROR;
    }

    m_src                       = src;
    m_key_points                = &key_points;
    m_threshold                 = threshold;
    m_nonmax_suppression        = nonmax_suppression;
    m_max_num_corners           = max_num_corners;
    m_detector_type             = type;

    return Status::OK;
}

std::vector<const Array*> FastImpl::GetInputArrays() const
{
    return {m_src};
}

std::string FastImpl::ToString() const
{
    std::string str;

    str = "op(Fast)";
    str += " target(" + GetOpTarget().ToString() + ")";
    str += " param( m_threshold:" + std::to_string(m_threshold) + " | " +
            " | " + "m_nonmax_suppression:" + std::to_string(m_nonmax_suppression) + ")\n";

    return str;
}

AURA_VOID FastImpl::Dump(const std::string &prefix) const
{
    JsonWrapper json_wrapper(m_ctx, prefix, m_name);

    if (json_wrapper.SetArray("src", m_src) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "SetArray src failed");
        return;
    }

    AURA_JSON_SERIALIZE(m_ctx, json_wrapper, m_src, *m_key_points, m_threshold, m_nonmax_suppression, m_max_num_corners, m_detector_type);
}

} // namespace aura