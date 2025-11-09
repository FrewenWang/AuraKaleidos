#include "aura/ops/filter/guidefilter.hpp"
#include "guidefilter_impl.hpp"
#include "aura/runtime/logger.h"
#include "aura/tools/json.h"

namespace aura
{

static std::shared_ptr<GuideFilterImpl> CreateGuideFilterImpl(Context *ctx, const OpTarget &target)
{
    std::shared_ptr<GuideFilterImpl> impl;

    switch (target.m_type)
    {
        case TargetType::NONE:
        {
            impl.reset(new GuideFilterNone(ctx, target));
            break;
        }

        case TargetType::NEON:
        {
#if defined(AURA_ENABLE_NEON)
            impl.reset(new GuideFilterNeon(ctx, target));
#endif // AURA_ENABLE_NEON
            break;
        }

        case TargetType::OPENCL:
        {
#if defined(AURA_ENABLE_OPENCL)
            // impl.reset(new GuideFilterCL(ctx, target));
#endif // AURA_ENABLE_OPENCL
            break;
        }

        case TargetType::HVX:
        {
#if (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))
            // impl.reset(new GuideFilterHvx(ctx, target));
#endif // defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON)
            break;
        }

        default :
        {
            break;
        }
    }

    return impl;
}

GuideFilter::GuideFilter(Context *ctx, const OpTarget &target) : Op(ctx, target)
{}

Status GuideFilter::SetArgs(const Array *src0, const Array *src1, Array *dst, DT_S32 ksize, DT_F32 eps,
                            GuideFilterType type, BorderType border_type, const Scalar &border_value)
{
    if ((DT_NULL == m_ctx))
    {
        return Status::ERROR;
    }

    if ((DT_NULL == src0) || (DT_NULL == src1) || (DT_NULL == dst))
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
            if (CheckNeonWidth(*src0) != Status::OK)
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
        m_impl = CreateGuideFilterImpl(m_ctx, impl_target);
    }

    // run initialize
    GuideFilterImpl *guidefilter_impl = dynamic_cast<GuideFilterImpl *>(m_impl.get());
    if (DT_NULL == guidefilter_impl)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "guidefilter_impl is null ptr");
        return Status::ERROR;
    }

    Status ret = guidefilter_impl->SetArgs(src0, src1, dst, ksize, eps, type, border_type, border_value);

    AURA_RETURN(m_ctx, ret);
}

AURA_EXPORTS Status IGuideFilter(Context *ctx, const Mat &src0, const Mat &src1, Mat &dst, DT_S32 ksize, DT_F32 eps,
                                 GuideFilterType type, BorderType border_type, const Scalar &border_value, const OpTarget &target)
{
    GuideFilter guidefilter(ctx, target);

    return OpCall(ctx, guidefilter, &src0, &src1, &dst, ksize, eps, type, border_type, border_value);
}

GuideFilterImpl::GuideFilterImpl(Context *ctx, const OpTarget &target) : OpImpl(ctx, AURA_OPS_FILTER_GUIDEFILTER_OP_NAME, target),
                                                                         m_ksize(0), m_eps(0.0f), m_type(GuideFilterType::NORMAL),
                                                                         m_border_type(BorderType::REFLECT_101), m_src0(DT_NULL),
                                                                         m_src1(DT_NULL), m_dst(DT_NULL)
{}

Status GuideFilterImpl::SetArgs(const Array *src0, const Array *src1, Array *dst, DT_S32 ksize, DT_F32 eps,
                                GuideFilterType type, BorderType border_type, const Scalar &border_value)
{
    if (DT_NULL == m_ctx)
    {
        return Status::ERROR;
    }

    if (!(src0->IsValid() && src1->IsValid() && dst->IsValid()))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "invalid src0/src1/dst");
        return Status::ERROR;
    }

    if (!(src0->IsEqual(*src1) && src0->IsEqual(*dst)))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src0/src1/dst should have the same size and the same data type");
        return Status::ERROR;
    }

    if ((ksize & 1) != 1 || ksize < 3)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "ksize must be odd and >= 3");
        return Status::ERROR;
    }

    if ((src0->GetSizes().m_height < ksize / 2) || (src0->GetSizes().m_width < ksize / 2))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "height/width must be larger than ksize/2");
        return Status::ERROR;
    }

    m_src0         = src0;
    m_src1         = src1;
    m_dst          = dst;
    m_ksize        = ksize;
    m_eps          = eps;
    m_type         = type;
    m_border_type  = border_type;
    m_border_value = border_value;

    return Status::OK;
}

std::vector<const Array*> GuideFilterImpl::GetInputArrays() const
{
    return {m_src0, m_src1};
}

std::vector<const Array*> GuideFilterImpl::GetOutputArrays() const
{
    return {m_dst};
}

std::string GuideFilterImpl::ToString() const
{
    std::string str;

    DT_CHAR eps_str[20];
    snprintf(eps_str, sizeof(eps_str), "%.4f", m_eps);

    str = "op(GuideFilter)";
    str += " target(" + GetOpTarget().ToString() + ")";
    str += " param("  + BorderTypeToString(m_border_type) + " | " +
            "ksize:"  + std::to_string(m_ksize)           + " | " +
            "eps:"    + eps_str                           + " | " +
            "type:"   + GuideFilterTypeToString(m_type)   + " | " +
            "border_value:" + m_border_value.ToString()   + ")\n";

    return str;
}

DT_VOID GuideFilterImpl::Dump(const std::string &prefix) const
{
    JsonWrapper json_wrapper(m_ctx, prefix, m_name);

    std::vector<std::string>  names  = {"src0", "src1", "dst"};
    std::vector<const Array*> arrays = {m_src0, m_src1, m_dst};

    if (json_wrapper.SetArray(names, arrays) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "SetArray src failed");
        return;
    }

    AURA_JSON_SERIALIZE(m_ctx, json_wrapper, m_src0, m_src1, m_dst, m_ksize, m_eps, m_type, m_border_type, m_border_value);
}

} // namespace aura
