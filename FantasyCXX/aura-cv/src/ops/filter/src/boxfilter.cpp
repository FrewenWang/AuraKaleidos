#include "aura/ops/filter/boxfilter.hpp"
#include "boxfilter_impl.hpp"
#include "aura/runtime/logger.h"
#include "aura/tools/json.h"

namespace aura
{

static std::shared_ptr<BoxFilterImpl> CreateBoxfilterImpl(Context *ctx, const OpTarget &target)
{
    std::shared_ptr<BoxFilterImpl> impl;

    switch (target.m_type)
    {
        case TargetType::NONE:
        {
            impl.reset(new BoxFilterNone(ctx, target));
            break;
        }

        case TargetType::NEON:
        {
#if defined(AURA_ENABLE_NEON)
            impl.reset(new BoxFilterNeon(ctx, target));
#endif // AURA_ENABLE_NEON
            break;
        }

        case TargetType::OPENCL:
        {
#if defined(AURA_ENABLE_OPENCL)
            impl.reset(new BoxFilterCL(ctx, target));
#endif // AURA_ENABLE_OPENCL
            break;
        }

        case TargetType::HVX:
        {
#if defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON)
            impl.reset(new BoxFilterHvx(ctx, target));
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

BoxFilter::BoxFilter(Context *ctx, const OpTarget &target) : Op(ctx, target)
{}

Status BoxFilter::SetArgs(const Array *src, Array *dst, MI_S32 ksize,
                          BorderType border_type, const Scalar &border_value)
{
    if (MI_NULL == m_ctx)
    {
        return Status::ERROR;
    }

    if ((MI_NULL == src) || (MI_NULL == dst))
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
        m_impl = CreateBoxfilterImpl(m_ctx, impl_target);
    }

    // run SetArgs
    BoxFilterImpl *boxfilter_impl = dynamic_cast<BoxFilterImpl *>(m_impl.get());
    if (MI_NULL == boxfilter_impl)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "boxfilter_impl is null ptr");
        return Status::ERROR;
    }

    Status ret = boxfilter_impl->SetArgs(src, dst, ksize, border_type, border_value);

    AURA_RETURN(m_ctx, ret);
}

Status BoxFilter::CLPrecompile(Context *ctx, ElemType elem_type, MI_S32 channel, MI_S32 ksize, BorderType border_type)
{
#if defined(AURA_ENABLE_OPENCL)
    if (MI_NULL == ctx)
    {
        return Status::ERROR;
    }

    std::vector<CLKernel> cl_kernels = BoxFilterCL::GetCLKernels(ctx, elem_type, channel, ksize, border_type);

    if (CheckCLKernels(ctx, cl_kernels) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "BoxFilter CheckCLKernels failed");
        return Status::ERROR;
    }
#else
    AURA_UNUSED(ctx);
    AURA_UNUSED(elem_type);
    AURA_UNUSED(channel);
    AURA_UNUSED(ksize);
    AURA_UNUSED(border_type);
#endif
    return Status::OK;
}

AURA_EXPORTS Status IBoxfilter(Context *ctx, const Mat &src, Mat &dst, MI_S32 ksize, BorderType border_type,
                               const Scalar &border_value, const OpTarget &target)
{
    BoxFilter boxfilter(ctx, target);

    return OpCall(ctx, boxfilter, &src, &dst, ksize, border_type, border_value);
}

BoxFilterImpl::BoxFilterImpl(Context *ctx, const OpTarget &target) : OpImpl(ctx, "BoxFilter", target),
                                                                     m_ksize(0), m_border_type(BorderType::REFLECT_101),
                                                                     m_src(MI_NULL), m_dst(MI_NULL)
{}

Status BoxFilterImpl::SetArgs(const Array *src, Array *dst, MI_S32 ksize,
                              BorderType border_type, const Scalar &border_value)
{
    if (MI_NULL == m_ctx)
    {
        return Status::ERROR;
    }

    if (!(src->IsValid() && dst->IsValid()))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "invalid src or dst");
        return Status::ERROR;
    }

    if (!src->IsEqual(*dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src and dst should have the same size");
        return Status::ERROR;
    }

    if ((ksize & 1) != 1)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "ksize must be odd");
        return Status::ERROR;
    }

    if ((src->GetSizes().m_height < ksize / 2) || (src->GetSizes().m_width < ksize / 2))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "height/width must bigger than ksize/2");
        return Status::ERROR;
    }

    m_src          = src;
    m_dst          = dst;
    m_ksize        = ksize;
    m_border_type  = border_type;
    m_border_value = border_value;

    return Status::OK;
}

std::vector<const Array*> BoxFilterImpl::GetInputArrays() const
{
    return {m_src};
}

std::vector<const Array*> BoxFilterImpl::GetOutputArrays() const
{
    return {m_dst};
}

std::string BoxFilterImpl::ToString() const
{
    std::string str;

    str = "op(BoxFilter)";
    str += " target(" + GetOpTarget().ToString() + ")";
    str += " param(" + BorderTypeToString(m_border_type) + " | " +
            "ksize:" + std::to_string(m_ksize) + " | "
            "border_value:" + m_border_value.ToString() + ")\n";

    return str;
}

AURA_VOID BoxFilterImpl::Dump(const std::string &prefix) const
{
    JsonWrapper json_wrapper(m_ctx, prefix, m_name);

    std::vector<std::string>  names  = {"src", "dst"};
    std::vector<const Array*> arrays = {m_src, m_dst};

    if (json_wrapper.SetArray(names, arrays) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "SetArray src failed");
        return;
    }

    AURA_JSON_SERIALIZE(m_ctx, json_wrapper, m_src, m_dst, m_ksize, m_border_type, m_border_value);
}

} // namespace aura