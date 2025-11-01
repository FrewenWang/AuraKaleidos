#include "aura/ops/filter/filter2d.hpp"
#include "filter2d_impl.hpp"
#include "aura/runtime/logger.h"
#include "aura/tools/json.h"

namespace aura
{

static std::shared_ptr<Filter2dImpl> CreateFilter2DImpl(Context *ctx, const OpTarget &target)
{
    std::shared_ptr<Filter2dImpl> impl;

    switch (target.m_type)
    {
        case TargetType::NONE:
        {
            impl.reset(new Filter2dNone(ctx, target));
            break;
        }

        case TargetType::NEON:
        {
#if defined(AURA_ENABLE_NEON)
            impl.reset(new Filter2dNeon(ctx, target));
#endif // AURA_ENABLE_NEON
            break;
        }

        case TargetType::OPENCL:
        {
#if defined(AURA_ENABLE_OPENCL)
            impl.reset(new Filter2dCL(ctx, target));
#endif // AURA_ENABLE_OPENCL
            break;
        }

        case TargetType::HVX:
        {
#if (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))
            impl.reset(new Filter2dHvx(ctx, target));
#endif // AURA_BUILD_HEXAGON
            break;
        }

        default :
        {
            break;
        }
    }

    return impl;
}

Filter2d::Filter2d(Context *ctx, const OpTarget &target) : Op(ctx, target)
{}

Status Filter2d::SetArgs(const Array *src, Array *dst, const Array *kmat,
                         BorderType border_type, const Scalar &border_value)
{
    if (MI_NULL == m_ctx)
    {
        return Status::ERROR;
    }

    if ((MI_NULL == src) || (MI_NULL == dst) || (MI_NULL == kmat))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src/dst/kmat is null ptr");
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

        case TargetType::OPENCL:
        {
#if defined(AURA_ENABLE_OPENCL)
            if (CheckCLWidth(*src) != Status::OK)
            {
                impl_target = OpTarget::None();
            }
#endif // AURA_ENABLE_OPENCL
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
        m_impl = CreateFilter2DImpl(m_ctx, impl_target);
    }

    // run initialize
    Filter2dImpl *filter2d_impl = dynamic_cast<Filter2dImpl*>(m_impl.get());
    if (MI_NULL == filter2d_impl)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "filter2d_impl is null ptr");
        return Status::ERROR;
    }

    Status ret = filter2d_impl->SetArgs(src, dst, kmat, border_type, border_value);

    AURA_RETURN(m_ctx, ret);
}

Status Filter2d::CLPrecompile(Context *ctx, ElemType elem_type, MI_S32 channel, MI_S32 ksize, BorderType border_type)
{
#if defined(AURA_ENABLE_OPENCL)
    if (MI_NULL == ctx)
    {
        return Status::ERROR;
    }

    std::vector<CLKernel> cl_kernels = Filter2dCL::GetCLKernels(ctx, elem_type, channel, ksize, border_type);

    if (CheckCLKernels(ctx, cl_kernels) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "Filter2d CheckCLKernels failed");
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

AURA_EXPORTS Status IFilter2d(Context *ctx, const Mat &src, Mat &dst, const Mat &kmat, BorderType border_type,
                              const Scalar &border_value, const OpTarget &target)
{
    Filter2d filter2d(ctx, target);

    return OpCall(ctx, filter2d, &src, &dst, &kmat, border_type, border_value);
}

Filter2dImpl::Filter2dImpl(Context *ctx, const OpTarget &target) : OpImpl(ctx, "Filter2d", target),
                                                                   m_ksize(0), m_border_type(BorderType::REFLECT_101),
                                                                   m_src(MI_NULL), m_dst(MI_NULL), m_kmat(MI_NULL)
{}

Status Filter2dImpl::SetArgs(const Array *src, Array *dst, const Array *kmat,
                             BorderType border_type, const Scalar &border_value)
{
    if (MI_NULL == m_ctx)
    {
        return Status::ERROR;
    }

    if (!(src->IsValid() && dst->IsValid() && kmat->IsValid()))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "invalid src or dst or kmat");
        return Status::ERROR;
    }

    if (!src->IsEqual(*dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src and dst should have the same size");
        return Status::ERROR;
    }

    Sizes3 ksizes = kmat->GetSizes();
    if (ksizes.m_width != ksizes.m_height || ksizes.m_channel != 1)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "kdata/ksizes shape error");
        return Status::ERROR;
    }

    if ((ksizes.m_width & 1) != 1)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "ksize must be odd");
        return Status::ERROR;
    }

    if (ksizes.m_width > src->GetSizes().m_width || ksizes.m_height > src->GetSizes().m_height)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "ksize cannot be greater than src");
        return Status::ERROR;
    }

    m_src          = src;
    m_dst          = dst;
    m_ksize        = ksizes.m_width;
    m_border_type  = border_type;
    m_border_value = border_value;
    m_kmat         = kmat;

    return Status::OK;
}

std::vector<const Array*> Filter2dImpl::GetInputArrays() const
{
    return {m_src};
}

std::vector<const Array*> Filter2dImpl::GetOutputArrays() const
{
    return {m_dst};
}

std::string Filter2dImpl::ToString() const
{
    std::string str;

    str = "op(Filter2d)";
    str += " target(" + GetOpTarget().ToString() + ")";
    str += " param(" + BorderTypeToString(m_border_type) + " | " +
            "ksize:" + std::to_string(m_ksize) + " | "
            "border_value:" + m_border_value.ToString() + ")\n";

    return str;
}

AURA_VOID Filter2dImpl::Dump(const std::string &prefix) const
{
    JsonWrapper json_wrapper(m_ctx, prefix, m_name);

    std::vector<std::string>  names  = {"src", "kmat", "dst"};
    std::vector<const Array*> arrays = {m_src, m_kmat, m_dst};

    if (json_wrapper.SetArray(names, arrays) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "SetArray failed");
        return;
    }

    AURA_JSON_SERIALIZE(m_ctx, json_wrapper, m_src, m_kmat, m_dst, m_ksize, m_border_type, m_border_value);
}

} // namespace aura