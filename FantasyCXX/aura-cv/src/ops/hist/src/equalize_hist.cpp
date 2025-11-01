#include "aura/ops/hist/equalize_hist.hpp"
#include "equalize_hist_impl.hpp"
#include "aura/runtime/logger.h"
#include "aura/tools/json.h"

namespace aura
{

static std::shared_ptr<EqualizeHistImpl> CreateEqualizeHistImpl(Context *ctx, const OpTarget &target)
{
    std::shared_ptr<EqualizeHistImpl> impl;

    switch (target.m_type)
    {
        case TargetType::NONE:
        {
            impl.reset(new EqualizeHistNone(ctx, target));
            break;
        }

        default :
        {
            break;
        }
    }

    return impl;
}

EqualizeHist::EqualizeHist(Context *ctx, const OpTarget &target) : Op(ctx, target)
{}

Status EqualizeHist::SetArgs(const Array *src, Array *dst)
{
    Status ret = Status::ERROR;

    if ((MI_NULL == m_ctx))
    {
        return ret;
    }

    if ((MI_NULL == src) || (MI_NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src/dst is null ptr");
        return ret;
    }

    OpTarget impl_target = m_target;

    // set m_impl
    if (MI_NULL == m_impl.get() || impl_target != m_impl->GetOpTarget())
    {
        m_impl = CreateEqualizeHistImpl(m_ctx, impl_target);
    }

    // run SetArgs
    EqualizeHistImpl *equalize_hist_impl = dynamic_cast<EqualizeHistImpl *>(m_impl.get());
    if (MI_NULL == equalize_hist_impl)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "equalize_hist_impl is null ptr");
        return ret;
    }

    ret = equalize_hist_impl->SetArgs(src, dst);

    AURA_RETURN(m_ctx, ret);
}

AURA_EXPORTS Status IEqualizeHist(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target)
{
    EqualizeHist equalize_hist(ctx, target);

    return OpCall(ctx, equalize_hist, &src, &dst);
}

EqualizeHistImpl::EqualizeHistImpl(Context *ctx, const OpTarget &target) : OpImpl(ctx, "EqualizeHist", target),
                                                                           m_src(MI_NULL), m_dst(MI_NULL)
{}

Status EqualizeHistImpl::SetArgs(const Array *src, Array *dst)
{
    Status ret = Status::ERROR;

    if (MI_NULL == m_ctx)
    {
        return ret;
    }

    if (!(src->IsValid() && dst->IsValid()))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "invalid src or dst");
        return ret;
    }

    if (src->GetElemType() != ElemType::U8 || dst->GetElemType() != ElemType::U8)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src and dst elem type should be u8");
        return ret;
    }

    if (src->GetSizes().m_channel != 1 || dst->GetSizes().m_channel != 1)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src and dst channel must be 1");
        return ret;
    }

    if (src->GetSizes().m_height != dst->GetSizes().m_height || src->GetSizes().m_width != dst->GetSizes().m_width)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src and dst must have the same height and width");
        return ret;
    }

    m_src = src;
    m_dst = dst;

    return Status::OK;
}

std::vector<const Array*> EqualizeHistImpl::GetInputArrays() const
{
    return {m_src};
}

std::vector<const Array*> EqualizeHistImpl::GetOutputArrays() const
{
    return {m_dst};
}

std::string EqualizeHistImpl::ToString() const
{
    std::string str;

    str = "op(EqualizeHist)";
    str += " target(" + GetOpTarget().ToString() + ")\n";

    return str;
}

AURA_VOID EqualizeHistImpl::Dump(const std::string &prefix) const
{
    JsonWrapper json_wrapper(m_ctx, prefix, m_name);

    std::vector<std::string>  names  = {"src", "dst"};
    std::vector<const Array*> arrays = {m_src, m_dst};

    if (json_wrapper.SetArray(names, arrays) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "SetArray src failed");
        return;
    }

    AURA_JSON_SERIALIZE(m_ctx, json_wrapper, m_src, m_dst);
}

} // namespace aura