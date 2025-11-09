#include "aura/ops/matrix/normalize.hpp"
#include "normalize_impl.hpp"
#include "aura/runtime/logger.h"
#include "aura/tools/json.h"

namespace aura
{

static std::shared_ptr<NormalizeImpl> CreateNormalizeImpl(Context *ctx, const OpTarget &target)
{
    std::shared_ptr<NormalizeImpl> impl;

    switch (target.m_type)
    {
        case TargetType::NONE:
        {
            impl.reset(new NormalizeNone(ctx, target));
            break;
        }

        case TargetType::NEON:
        {
#if defined(AURA_ENABLE_NEON)
            impl.reset(new NormalizeNeon(ctx, target));
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

Normalize::Normalize(Context *ctx, const OpTarget &target) : Op(ctx, target)
{}

Status Normalize::SetArgs(const Array *src, Array *dst, DT_F32 alpha, DT_F32 beta, NormType type)
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

        default:
        {
            break;
        }
    }

    // set m_impl
    if (DT_NULL == m_impl.get() || impl_target != m_impl->GetOpTarget())
    {
        m_impl = CreateNormalizeImpl(m_ctx, impl_target);
    }

    // run initialize
    NormalizeImpl *normalize_impl = dynamic_cast<NormalizeImpl*>(m_impl.get());
    if (DT_NULL == normalize_impl)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "normalize_impl is null ptr");
        return Status::ERROR;
    }

    Status ret = normalize_impl->SetArgs(src, dst, alpha, beta, type);

    AURA_RETURN(m_ctx, ret);
}

AURA_EXPORTS Status INormalize(Context *ctx, const Mat &src, Mat &dst, DT_F32 alpha, DT_F32 beta, NormType type, const OpTarget &target)
{
    Normalize normalize(ctx, target);

    return OpCall(ctx, normalize, &src, &dst, alpha, beta, type);
}

NormalizeImpl::NormalizeImpl(Context *ctx, const OpTarget &target) : OpImpl(ctx, "Normalize", target),
                                                                     m_src(DT_NULL), m_dst(DT_NULL),
                                                                     m_alpha(1), m_beta(0), m_type(NormType::NORM_INF)
{}

Status NormalizeImpl::SetArgs(const Array *src, Array *dst, DT_F32 alpha, DT_F32 beta, NormType type)
{
    if (DT_NULL == m_ctx)
    {
        return Status::ERROR;
    }

    if ((DT_NULL == src) || (DT_NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src/dst is null ptr");
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

    m_src   = src;
    m_dst   = dst;
    m_alpha = alpha;
    m_beta  = beta;
    m_type  = type;
    return Status::OK;
}

std::vector<const Array*> NormalizeImpl::GetInputArrays() const
{
    return {m_src};
}

std::vector<const Array*> NormalizeImpl::GetOutputArrays() const
{
    return {m_dst};
}

std::string NormalizeImpl::ToString() const
{
    std::string str;

    str = "op(Normalize)";
    str += " target(" + GetOpTarget().ToString() + ")\n";

    return str;
}

DT_VOID NormalizeImpl::Dump(const std::string &prefix) const
{
    JsonWrapper json_wrapper(m_ctx, prefix, m_name);

    std::vector<std::string>  names  = {"src", "dst"};
    std::vector<const Array*> arrays = {m_src, m_dst};

    if (json_wrapper.SetArray(names, arrays) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "SetArray src failed");
        return;
    }


    AURA_JSON_SERIALIZE(m_ctx, json_wrapper, m_src, m_dst, m_alpha, m_beta, m_type);
}

} // namespace aura