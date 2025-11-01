#include "aura/ops/matrix/min_max_loc.hpp"
#include "min_max_loc_impl.hpp"
#include "aura/runtime/logger.h"
#include "aura/tools/json.h"

namespace aura
{

static std::shared_ptr<MinMaxLocImpl> CreateMinMaxLocImpl(Context *ctx, const OpTarget &target)
{
    std::shared_ptr<MinMaxLocImpl> impl;

    switch (target.m_type)
    {
        case TargetType::NONE:
        {
            impl.reset(new MinMaxLocNone(ctx, target));
            break;
        }

        case TargetType::NEON:
        {
#if defined(AURA_ENABLE_NEON)
            impl.reset(new MinMaxLocNeon(ctx, target));
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

MinMaxLoc::MinMaxLoc(Context *ctx, const OpTarget &target) : Op(ctx,target)
{}

Status MinMaxLoc::SetArgs(const Array *src, MI_F64 *min_val, MI_F64 *max_val, Point3i *min_pos, Point3i *max_pos)
{
    if ((MI_NULL == m_ctx))
    {
        return Status::ERROR;
    }

    if (MI_NULL == src)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src is null ptr");
        return Status::ERROR;
    }

    if ((MI_NULL == min_val) || (MI_NULL == max_val) || (MI_NULL == min_pos) || (MI_NULL == max_pos))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "output pointer is null ptr");
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
    if (MI_NULL == m_impl.get() || impl_target != m_impl->GetOpTarget())
    {
        m_impl = CreateMinMaxLocImpl(m_ctx, impl_target);
    }

    // run initialize
    MinMaxLocImpl *min_max_loc_impl = dynamic_cast<MinMaxLocImpl*>(m_impl.get());
    if (MI_NULL == min_max_loc_impl)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "min_max_loc_impl is null ptr");
        return Status::ERROR;
    }

    Status ret = min_max_loc_impl->SetArgs(src, min_val, max_val, min_pos, max_pos);

    AURA_RETURN(m_ctx, ret);
}

AURA_EXPORTS Status IMinMaxLoc(Context *ctx, const Mat &src, MI_F64 *min_val, MI_F64 *max_val, Point3i *min_pos, Point3i *max_pos, const OpTarget &target)
{
    MinMaxLoc min_max_loc(ctx, target);

    return OpCall(ctx, min_max_loc, &src, min_val, max_val, min_pos, max_pos);
}

MinMaxLocImpl::MinMaxLocImpl(Context *ctx, const OpTarget &target) : OpImpl(ctx, "MinMaxLoc", target), m_src(MI_NULL), m_min_val(MI_NULL), 
                                                                     m_max_val(MI_NULL), m_min_pos(MI_NULL), m_max_pos(MI_NULL)
{}

Status MinMaxLocImpl::SetArgs(const Array *src, MI_F64 *min_val, MI_F64 *max_val, Point3i *min_pos, Point3i *max_pos)
{
    if (MI_NULL == m_ctx)
    {
        return Status::ERROR;
    }

    if (MI_NULL == src)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src is null ptr");
        return Status::ERROR;
    }

    if (!src->IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src or dst mat is invalid.");
        return Status::ERROR;
    }

    m_src = src;
    m_min_val = min_val;
    m_max_val = max_val;
    m_min_pos = min_pos;
    m_max_pos = max_pos;
    return Status::OK;
}

std::vector<const Array*> MinMaxLocImpl::GetInputArrays() const
{
    return {m_src};
}

std::string MinMaxLocImpl::ToString() const
{
    std::string str;

    str = "op(MinMaxLoc)";
    str += " target(" + GetOpTarget().ToString() + ")\n";

    return str;
}

AURA_VOID MinMaxLocImpl::Dump(const std::string &prefix) const
{
    JsonWrapper json_wrapper(m_ctx, prefix, m_name);

    if (json_wrapper.SetArray("src", m_src) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "SetArray src failed");
        return;
    }

    AURA_JSON_SERIALIZE(m_ctx, json_wrapper, m_src, *m_min_val, *m_max_val, *m_min_pos, *m_max_pos);
}

} // namespace aura