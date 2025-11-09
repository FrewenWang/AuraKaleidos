#include "aura/ops/matrix/make_border.hpp"
#include "make_border_impl.hpp"
#include "aura/runtime/logger.h"
#include "aura/tools/json.h"

namespace aura
{

static std::shared_ptr<MakeBorderImpl> CreateMakeBorderImpl(Context *ctx, const OpTarget &target)
{
    std::shared_ptr<MakeBorderImpl> impl;

    switch (target.m_type)
    {
        case TargetType::NONE:
        {
            impl.reset(new MakeBorderNone(ctx, target));
            break;
        }

        default :
        {
            break;
        }
    }

    return impl;
}

static Status CheckMakeBorderMatSize(Context *ctx, const Array *src, Array *dst, DT_S32 top, DT_S32 bottom, DT_S32 left, DT_S32 right)
{
    Sizes3 border_size(top + bottom, left + right, 0);

    const Sizes3 &src_sz = src->GetSizes();
    const Sizes3 &dst_sz = dst->GetSizes();

    if (src_sz.m_channel != dst_sz.m_channel)
    {
        AURA_ADD_ERROR_STRING(ctx, "MakeBorderNone plane channel not equal.");
        return Status::ERROR;
    }

    if ((border_size + src_sz) != dst_sz)
    {
        AURA_ADD_ERROR_STRING(ctx, "MakeBorderNone src dst plane shape doesn't match.");
        return Status::ERROR;
    }

    return Status::OK;
}

MakeBorder::MakeBorder(Context *ctx, const OpTarget &target) : Op(ctx, target)
{}

Status MakeBorder::SetArgs(const Array *src, Array *dst, DT_S32 top, DT_S32 bottom,
                           DT_S32 left, DT_S32 right, BorderType type, const Scalar &border_value)
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
        m_impl = CreateMakeBorderImpl(m_ctx, impl_target);
    }

    // run initialize
    MakeBorderImpl *make_border_impl = dynamic_cast<MakeBorderImpl*>(m_impl.get());
    if (DT_NULL == make_border_impl)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "make_border_impl is null ptr");
        return Status::ERROR;
    }

    Status ret = make_border_impl->SetArgs(src, dst, top, bottom, left, right, type, border_value);

    AURA_RETURN(m_ctx, ret);
}

AURA_EXPORTS Status IMakeBorder(Context *ctx, const Mat &src, Mat &dst, DT_S32 top, DT_S32 bottom,
                                DT_S32 left, DT_S32 right, BorderType type, const Scalar &border_value,
                                const OpTarget &target)
{
    MakeBorder make_border(ctx, target);

    return OpCall(ctx, make_border, &src, &dst, top, bottom, left, right, type, border_value);
}

MakeBorderImpl::MakeBorderImpl(Context *ctx, const OpTarget &target) : OpImpl(ctx, "MakeBorder", target),
                                                                       m_top(0), m_bottom(0), m_left(0),
                                                                       m_right(0), m_type(BorderType::REFLECT_101)

{}

Status MakeBorderImpl::SetArgs(const Array *src, Array *dst, DT_S32 top, DT_S32 bottom,
                               DT_S32 left, DT_S32 right, BorderType type, const Scalar &border_value)
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

    if (top < 0 || bottom < 0 || left < 0 || right < 0)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "MakeBorderNone invalid border size.");
        return Status::ERROR;
    }

    if (src->GetElemType() != dst->GetElemType())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "MakeBorderNone src and dst must has same elem_type.");
        return Status::ERROR;
    }

    if (CheckMakeBorderMatSize(m_ctx, src, dst, top, bottom, left, right) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "MakeBorderNone CheckMakeBorderMatSize failed.");
        return Status::ERROR;
    }

    m_src          = src;
    m_dst          = dst;
    m_top          = top;
    m_bottom       = bottom;
    m_left         = left;
    m_right        = right;
    m_type         = type;
    m_border_value = border_value;
    return Status::OK;
}

std::vector<const Array*> MakeBorderImpl::GetInputArrays() const
{
    return {m_src};
}

std::vector<const Array*> MakeBorderImpl::GetOutputArrays() const
{
    return {m_dst};
}

std::string MakeBorderImpl::ToString() const
{
    std::string str;

    str = "op(MakeBorder)";
    str += " target(" + GetOpTarget().ToString() + ")";
    str += " param(" + BorderTypeToString(m_type) + " | " +
            "top:" + std::to_string(m_top) + " | " + "bottom:" + std::to_string(m_bottom) + " | "
            "left:" + std::to_string(m_left) + " | " + "right:" + std::to_string(m_right) + " | "
            "border_value:" + m_border_value.ToString() + ")\n";

    return str;
}

DT_VOID MakeBorderImpl::Dump(const std::string &prefix) const
{
    JsonWrapper json_wrapper(m_ctx, prefix, m_name);

    std::vector<std::string>  names  = {"src", "dst"};
    std::vector<const Array*> arrays = {m_src, m_dst};

    if (json_wrapper.SetArray(names, arrays) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "SetArray src failed");
        return;
    }

    AURA_JSON_SERIALIZE(m_ctx, json_wrapper, m_src, m_dst, m_top, m_bottom, m_left, m_right, m_type, m_border_value);
}

} // namespace aura