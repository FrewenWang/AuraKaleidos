#include "aura/ops/feature2d/harris.hpp"
#include "harris_impl.hpp"
#include "aura/runtime/logger.h"
#include "aura/tools/json.h"

namespace aura
{

static std::shared_ptr<HarrisImpl> CreateHarrisImpl(Context *ctx, const OpTarget &target)
{
    std::shared_ptr<HarrisImpl> impl;

    switch (target.m_type)
    {
        case TargetType::NONE:
        {
            impl.reset(new HarrisNone(ctx, target));
            break;
        }

        case TargetType::NEON:
        {
#if defined(AURA_ENABLE_NEON)
            impl.reset(new HarrisNeon(ctx, target));
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

Harris::Harris(Context *ctx, const OpTarget &target) : Op(ctx, target)
{}

Status Harris::SetArgs(const Array *src, Array *dst, DT_S32 block_size, DT_S32 ksize, DT_F64 k,
                       BorderType border_type, const Scalar &border_value)
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
        m_impl = CreateHarrisImpl(m_ctx, impl_target);
    }

    // run SetArgs
    HarrisImpl *harris_impl = dynamic_cast<HarrisImpl *>(m_impl.get());
    if (DT_NULL == harris_impl)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "harris_impl is null ptr");
        return Status::ERROR;
    }

    Status ret = harris_impl->SetArgs(src, dst, block_size, ksize, k, border_type, border_value);

    AURA_RETURN(m_ctx, ret);
}

AURA_EXPORTS Status IHarris(Context *ctx, const Mat &src, Mat &dst, DT_S32 block_size, DT_S32 ksize, DT_F64 k,
                            BorderType border_type, const Scalar &border_value, const OpTarget &target)
{
    Harris harris(ctx, target);

    return OpCall(ctx, harris, &src, &dst, block_size, ksize, k, border_type, border_value);
}

HarrisImpl::HarrisImpl(Context *ctx, const OpTarget &target) : OpImpl(ctx, "Harris", target),
                                                               m_block_size(0), m_ksize(0), m_k(0.0),
                                                               m_border_type(BorderType::REFLECT_101),
                                                               m_src(DT_NULL), m_dst(DT_NULL)
{}

Status HarrisImpl::SetArgs(const Array *src, Array *dst, DT_S32 block_size, DT_S32 ksize, DT_F64 k,
                           BorderType border_type, const Scalar &border_value)
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

    if (!((src->GetSizes().m_height == dst->GetSizes().m_height) &&
        (src->GetSizes().m_width == dst->GetSizes().m_width)))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src and dst should have the same w/h size");
        return Status::ERROR;
    }

    if (src->GetSizes().m_channel != 1 || dst->GetSizes().m_channel != 1)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src and dst channel should be 1");
        return Status::ERROR;
    }

    if ((src->GetElemType() != ElemType::U8 && src->GetElemType() != ElemType::F32) || (dst->GetElemType() != ElemType::F32))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src only support u8/f32 and dst only support f32");
        return Status::ERROR;
    }

    m_src            = src;
    m_dst            = dst;
    m_block_size     = block_size;
    m_ksize          = ksize;
    m_k              = k;
    m_border_type    = border_type;
    m_border_value   = border_value;

    return Status::OK;
}

std::vector<const Array*> HarrisImpl::GetInputArrays() const
{
    return {m_src};
}

std::vector<const Array*> HarrisImpl::GetOutputArrays() const
{
    return {m_dst};
}

std::string HarrisImpl::ToString() const
{
    std::string str;

    DT_CHAR k_str[20];
    snprintf(k_str, sizeof(k_str), "%.2f", m_k);

    str = "op(Harris)";
    str += " target(" + GetOpTarget().ToString() + ")";
    str += " param(" + BorderTypeToString(m_border_type) + " | " +
            "block_size:" + std::to_string(m_block_size) + " | " + "ksize:" + std::to_string(m_ksize) + " | " +
            "k:" + k_str + " | " + "border_value:" + m_border_value.ToString() + ")\n";

    return str;
}

DT_VOID HarrisImpl::Dump(const std::string &prefix) const
{
    JsonWrapper json_wrapper(m_ctx, prefix, m_name);

    std::vector<std::string>  names  = {"src", "dst"};
    std::vector<const Array*> arrays = {m_src, m_dst};

    if (json_wrapper.SetArray(names, arrays) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "SetArray src failed");
        return;
    }

    AURA_JSON_SERIALIZE(m_ctx, json_wrapper, m_src, m_dst, m_block_size, m_ksize, m_k, m_border_type, m_border_value);
}

} // namespace aura