#include "aura/ops/matrix/flip.hpp"
#include "flip_impl.hpp"
#include "aura/runtime/logger.h"
#include "aura/tools/json.h"

namespace aura
{

static std::shared_ptr<FlipImpl> CreateFlipImpl(Context *ctx, const OpTarget &target)
{
    std::shared_ptr<FlipImpl> impl;

    switch (target.m_type)
    {
        case TargetType::NONE:
        {
            impl.reset(new FlipNone(ctx, target));
            break;
        }

        case TargetType::NEON:
        {
#if defined(AURA_ENABLE_NEON)
            impl.reset(new FlipNeon(ctx, target));
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

Flip::Flip(Context *ctx, const OpTarget &target) : Op(ctx, target)
{}

Status Flip::SetArgs(const Array *src, Array *dst, FlipType type)
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
        m_impl = CreateFlipImpl(m_ctx, impl_target);
    }

    // run initialize
    FlipImpl *flip_impl = dynamic_cast<FlipImpl*>(m_impl.get());
    if (DT_NULL == flip_impl)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "flip_impl is null ptr");
        return Status::ERROR;
    }

    Status ret = flip_impl->SetArgs(src, dst, type);

    AURA_RETURN(m_ctx, ret);
}

AURA_EXPORTS Status IFlip(Context *ctx, const Mat &src, Mat &dst, FlipType type, const OpTarget &target)
{
    Flip flip(ctx, target);

    return OpCall(ctx, flip, &src, &dst, type);
}

FlipImpl::FlipImpl(Context *ctx, const OpTarget &target) : OpImpl(ctx, "Flip", target), m_src(DT_NULL),
                                                           m_dst(DT_NULL), m_type(FlipType::BOTH)
{}

Status FlipImpl::SetArgs(const Array *src, Array *dst, FlipType type)
{
    if (DT_NULL == m_ctx)
    {
        return Status::ERROR;
    }

    if ((DT_NULL == src) || (DT_NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src or dst is null");
        return Status::ERROR;
    }

    if (!src->IsValid() || !dst->IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src or dst mat is invalid.");
        return Status::ERROR;
    }

    if (!src->IsEqual(*dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "failed, src and dst has different elem_type or size.");
        return Status::ERROR;
    }

    m_src = src;
    m_dst = dst;
    m_type = type;
    return Status::OK;
}

std::vector<const Array*> FlipImpl::GetInputArrays() const
{
    return {m_src};
}

std::vector<const Array*> FlipImpl::GetOutputArrays() const
{
    return {m_dst};
}

std::string FlipImpl::ToString() const
{
    std::string str;

    str = "op(Flip)";
    str += " target(" + GetOpTarget().ToString() + ")\n";

    return str;
}

DT_VOID FlipImpl::Dump(const std::string &prefix) const
{
    JsonWrapper json_wrapper(m_ctx, prefix, m_name);

    std::vector<std::string>  names  = {"src", "dst"};
    std::vector<const Array*> arrays = {m_src, m_dst};

    if (json_wrapper.SetArray(names, arrays) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "SetArray src failed");
        return;
    }

    AURA_JSON_SERIALIZE(m_ctx, json_wrapper, m_src, m_dst, m_type);
}

} // namespace aura