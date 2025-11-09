#include "aura/ops/matrix/integral.hpp"
#include "integral_impl.hpp"
#include "aura/runtime/logger.h"
#include "aura/tools/json.h"

namespace aura
{

static std::shared_ptr<IntegralImpl> CreateIntegralImpl(Context *ctx, const OpTarget &target)
{
    std::shared_ptr<IntegralImpl> impl;

    switch (target.m_type)
    {
        case TargetType::NONE:
        {
            impl.reset(new IntegralNone(ctx, target));
            break;
        }

        case TargetType::NEON:
        {
#if defined(AURA_ENABLE_NEON)
            impl.reset(new IntegralNeon(ctx, target));
#endif // AURA_ENABLE_NEON
            break;
        }

        case TargetType::HVX:
        {
#if (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))
            impl.reset(new IntegralHvx(ctx, target));
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

Integral::Integral(Context *ctx, const OpTarget &target) : Op(ctx, target)
{}

Status Integral::SetArgs(const Array *src, Array *dst, Array *dst_sq)
{
    if ((DT_NULL == m_ctx))
    {
        return Status::ERROR;
    }

    if (DT_NULL == src)
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
    if (DT_NULL == m_impl.get() || impl_target != m_impl->GetOpTarget())
    {
        m_impl = CreateIntegralImpl(m_ctx, impl_target);
    }

    // run initialize
    IntegralImpl *integral_impl = dynamic_cast<IntegralImpl *>(m_impl.get());
    if (DT_NULL == integral_impl)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "integral_impl is null ptr");
        return Status::ERROR;
    }

    Status ret = integral_impl->SetArgs(src, dst, dst_sq);
    AURA_RETURN(m_ctx, ret);
}

AURA_EXPORTS Status IIntegral(Context *ctx, const Mat &src, Mat &dst, Mat &dst_sq, const OpTarget &target)
{
    if (dst.IsValid() && dst_sq.IsValid())
    {
        Integral integral(ctx, target);
        return OpCall(ctx, integral, &src, &dst, &dst_sq);
    }
    else if (dst.IsValid())
    {
        Integral integral(ctx, target);
        return OpCall(ctx, integral, &src, &dst, DT_NULL);
    }
    else if (dst_sq.IsValid())
    {
        Integral integral(ctx, target);
        return OpCall(ctx, integral, &src, DT_NULL, &dst_sq);
    }

    return Status::ERROR;
}

IntegralImpl::IntegralImpl(Context *ctx, const OpTarget &target) : OpImpl(ctx, "Integral", target),
                                                                   m_src(DT_NULL), m_dst(DT_NULL),
                                                                   m_dst_sq(DT_NULL)
{}

Status IntegralImpl::SetArgs(const Array *src, Array *dst, Array *dst_sq)
{
    if (DT_NULL == m_ctx)
    {
        return Status::ERROR;
    }

    if (DT_NULL == src)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src or dst is null");
        return Status::ERROR;
    }

    if (!src->IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src or dst mat is invalid.");
        return Status::ERROR;
    }

    if (dst && dst->IsValid() && (dst->GetSizes() != src->GetSizes()))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src and dst size not match.");
        return Status::ERROR;
    }

    if (dst_sq && dst_sq->IsValid() && (dst_sq->GetSizes() != src->GetSizes()))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src and dst_sq size not match.");
        return Status::ERROR;
    }

    if (DT_NULL == dst && DT_NULL == dst_sq)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "both dst and dst_sq are null");
        return Status::ERROR;
    }

    if ((dst && !dst->IsValid()) && (dst_sq && !dst_sq->IsValid()))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "both dst and dst_sq are invalid.");
        return Status::ERROR;
    }

    m_src = src;
    m_dst = dst;
    m_dst_sq = dst_sq;

    return Status::OK;
}

std::vector<const Array*> IntegralImpl::GetInputArrays() const
{
    return {m_src};
}

std::vector<const Array*> IntegralImpl::GetOutputArrays() const
{
    if ((m_dst != DT_NULL) && (DT_NULL == m_dst_sq))
    {
        return {m_dst};
    }
    else if ((NULL == m_dst) && (m_dst_sq != DT_NULL))
    {
        return {m_dst_sq};
    }
    else if ((m_dst != DT_NULL) && (m_dst_sq != DT_NULL))
    {
        return {m_dst, m_dst_sq};
    }
    else
    {
        return std::vector<const Array*>();
    }
}

std::string IntegralImpl::ToString() const
{
    std::string str;

    str = "op(Integral)";
    str += " target(" + GetOpTarget().ToString() + ")\n";

    return str;
}

DT_VOID IntegralImpl::Dump(const std::string &prefix) const
{
    JsonWrapper json_wrapper(m_ctx, prefix, m_name);

    std::vector<std::string>  names  = {"src", "dst", "dst_sq"};
    std::vector<const Array*> arrays = {m_src, m_dst, m_dst_sq};

    if (json_wrapper.SetArray(names, arrays) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "SetArray failed");
        return;
    }

    AURA_JSON_SERIALIZE(m_ctx, json_wrapper, m_src, m_dst_sq, m_dst);
}

} // namespace aura