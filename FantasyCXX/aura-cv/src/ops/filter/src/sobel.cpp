#include "aura/ops/filter/sobel.hpp"
#include "sobel_impl.hpp"
#include "aura/runtime/logger.h"
#include "aura/tools/json.h"

namespace aura
{

static std::shared_ptr<SobelImpl> CreateSobelImpl(Context *ctx, const OpTarget &target)
{
    std::shared_ptr<SobelImpl> impl;

    switch (target.m_type)
    {
        case TargetType::NONE:
        {
            impl.reset(new SobelNone(ctx, target));
            break;
        }

        case TargetType::NEON:
        {
#if defined(AURA_ENABLE_NEON)
            impl.reset(new SobelNeon(ctx, target));
#endif // AURA_ENABLE_NEON
            break;
        }

        case TargetType::OPENCL:
        {
#if defined(AURA_ENABLE_OPENCL)
            impl.reset(new SobelCL(ctx, target));
#endif // AURA_ENABLE_OPENCL
            break;
        }

        case TargetType::HVX:
        {
#if (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))
            impl.reset(new SobelHvx(ctx, target));
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

Sobel::Sobel(Context *ctx, const OpTarget &target) : Op(ctx, target)
{}

Status Sobel::SetArgs(const Array *src, Array *dst, MI_S32 dx, MI_S32 dy, MI_S32 ksize, MI_F32 scale,
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
#endif // (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))
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
        m_impl = CreateSobelImpl(m_ctx, impl_target);
    }

    // run initialize
    SobelImpl *sobel_impl = dynamic_cast<SobelImpl*>(m_impl.get());
    if (MI_NULL == sobel_impl)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "SobelImpl is null ptr");
        return Status::ERROR;
    }

    Status ret = sobel_impl->SetArgs(src, dst, dx, dy, ksize, scale, border_type, border_value);

    AURA_RETURN(m_ctx, ret);
}

Status Sobel::CLPrecompile(Context *ctx, MI_S32 dx, MI_S32 dy, MI_S32 ksize, MI_F32 scale, BorderType border_type,
                           MI_S32 channel, ElemType src_elem_type, ElemType dst_elem_type)
{
#if defined(AURA_ENABLE_OPENCL)
    if (MI_NULL == ctx)
    {
        return Status::ERROR;
    }

    std::vector<CLKernel> cl_kernels = SobelCL::GetCLKernels(ctx, dx, dy, ksize, scale, border_type,
                                                             channel, src_elem_type, dst_elem_type);

    if (CheckCLKernels(ctx, cl_kernels) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "Sobel CheckCLKernels failed");
        return Status::ERROR;
    }
#else
    AURA_UNUSED(ctx);
    AURA_UNUSED(channel);
    AURA_UNUSED(ksize);
    AURA_UNUSED(dx);
    AURA_UNUSED(dy);
    AURA_UNUSED(scale);
    AURA_UNUSED(border_type);
    AURA_UNUSED(src_elem_type);
    AURA_UNUSED(dst_elem_type);
#endif
    return Status::OK;
}

AURA_EXPORTS Status ISobel(Context *ctx, const Mat &src, Mat &dst, MI_S32 dx, MI_S32 dy, MI_S32 ksize, MI_F32 scale,
                           BorderType border_type, const Scalar &border_value, const OpTarget &target)
{
    Sobel sobel(ctx, target);

    return OpCall(ctx, sobel, &src, &dst, dx, dy, ksize, scale, border_type, border_value);
}

SobelImpl::SobelImpl(Context *ctx, const OpTarget &target) : OpImpl(ctx, "Sobel", target),
                                                             m_dx(0), m_dy(0), m_ksize(0), m_scale(0.f),
                                                             m_border_type(BorderType::REFLECT_101),
                                                             m_src(MI_NULL), m_dst(MI_NULL)
{}

Status SobelImpl::SetArgs(const Array *src, Array *dst, MI_S32 dx, MI_S32 dy, MI_S32 ksize, MI_F32 scale,
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

    if (!src->IsSizesEqual(*dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src and dst should have the same size");
        return Status::ERROR;
    }

    if (dx < 0 || dy < 0 || dx + dy < 1)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "dx and dy cannot be less than 0, the sum of dx and dy cannot be less than 1");
        return Status::ERROR;
    }

    if (ksize > 0)
    {
        if ((ksize & 1) != 1)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "ksize must be odd");
            return Status::ERROR;
        }

        if (ksize > src->GetSizes().m_width || ksize > src->GetSizes().m_height)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "ksize cannot be greater than src");
            return Status::ERROR;
        }

        MI_S32 ksx = (1 == ksize && dx > 0) ? 3 : ksize;
        MI_S32 ksy = (1 == ksize && dy > 0) ? 3 : ksize;
        if (dx >= ksx || dy >= ksy)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "dx/dy cannot be greater than ksx/ksy");
            return Status::ERROR;
        }
    }
    else
    {
        if (dx + dy != 1)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "(dx + dy) should equal to 1");
            return Status::ERROR;
        }
    }

    m_src          = src;
    m_dst          = dst;
    m_dx           = dx;
    m_dy           = dy;
    m_ksize        = ksize;
    m_scale        = scale;
    m_border_type  = border_type;
    m_border_value = border_value;

    return Status::OK;
}

std::vector<const Array*> SobelImpl::GetInputArrays() const
{
    return {m_src};
}

std::vector<const Array*> SobelImpl::GetOutputArrays() const
{
    return {m_dst};
}

std::string SobelImpl::ToString() const
{
    std::string str;

    str  = "op(Sobel)";
    str += " target(" + GetOpTarget().ToString() + ")";
    str += "param(" + BorderTypeToString(m_border_type) + " | " +
           "dx:" + std::to_string(m_dx) + " | " + "dy:" + std::to_string(m_dy) + " | "
           "ksize:" + std::to_string(m_ksize) + " | " + "scale:" + std::to_string(m_scale) + " | "
           "border_value:" + m_border_value.ToString() + ")\n";

    return str;
}

AURA_VOID SobelImpl::Dump(const std::string &prefix) const
{
    JsonWrapper json_wrapper(m_ctx, prefix, m_name);

    std::vector<std::string>  names  = {"src", "dst"};
    std::vector<const Array*> arrays = {m_src, m_dst};

    if (json_wrapper.SetArray(names, arrays) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "SetArray src failed");
        return;
    }

    AURA_JSON_SERIALIZE(m_ctx, json_wrapper, m_src, m_dst, m_dx, m_dy, m_ksize, m_scale, m_border_type, m_border_value);
}

} // namespace aura