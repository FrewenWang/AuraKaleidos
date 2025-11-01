#include "aura/ops/feature2d/canny.hpp"
#include "canny_impl.hpp"
#include "aura/runtime/logger.h"
#include "aura/tools/json.h"

namespace aura
{

static std::shared_ptr<CannyImpl> CreateCannyImpl(Context *ctx, const OpTarget &target)
{
    std::shared_ptr<CannyImpl> impl;

    switch (target.m_type)
    {
        case TargetType::NONE:
        {
            impl.reset(new CannyNone(ctx, target));
            break;
        }

        case TargetType::NEON:
        {
#if defined(AURA_ENABLE_NEON)
            impl.reset(new CannyNeon(ctx, target));
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

Canny::Canny(Context *ctx, const OpTarget &target) : Op(ctx, target)
{}

Status Canny::SetArgs(const Array *src, Array *dst, MI_F64 low_thresh, MI_F64 high_thresh,
                      MI_S32 aperture_size, MI_BOOL l2_gradient)
{
    if ((MI_NULL == m_ctx))
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

        default:
        {
            break;
        }
    }

    // set m_impl
    if (MI_NULL == m_impl.get() || impl_target != m_impl->GetOpTarget())
    {
        m_impl = CreateCannyImpl(m_ctx, impl_target);
    }

    // run SetArgs
    CannyImpl *canny_impl = dynamic_cast<CannyImpl *>(m_impl.get());
    if (MI_NULL == canny_impl)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Canny_impl is null ptr");
        return Status::ERROR;
    }

    Status ret = canny_impl->SetArgs(src, dst, low_thresh, high_thresh, aperture_size, l2_gradient);

    AURA_RETURN(m_ctx, ret);
}

Status Canny::SetArgs(const Array *dx, const Array *dy, Array *dst, MI_F64 low_thresh,
                      MI_F64 high_thresh, MI_BOOL l2_gradient)
{
    if ((MI_NULL == m_ctx))
    {
        return Status::ERROR;
    }

    if ((MI_NULL == dx) || (MI_NULL == dy) || (MI_NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "dx/dy/dst is null ptr");
        return Status::ERROR;
    }

    OpTarget impl_target = m_target;

    // check impl type
    switch (m_target.m_type)
    {
        case TargetType::NEON:
        {
#if defined(AURA_ENABLE_NEON)
            if (CheckNeonWidth(*dx) != Status::OK || CheckNeonWidth(*dy) != Status::OK)
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
        m_impl = CreateCannyImpl(m_ctx, impl_target);
    }

    // run SetArgs
    CannyImpl *canny_impl = dynamic_cast<CannyImpl *>(m_impl.get());
    if (MI_NULL == canny_impl)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Canny_impl is null ptr");
        return Status::ERROR;
    }

    Status ret = canny_impl->SetArgs(dx, dy, dst, low_thresh, high_thresh, l2_gradient);

    AURA_RETURN(m_ctx, ret);
}

AURA_EXPORTS Status ICanny(Context *ctx, const Mat &src, Mat &dst, MI_F64 low_thresh, MI_F64 high_thresh,
                           MI_S32 aperture_size, MI_BOOL l2_gradient, const OpTarget &target)
{
    Canny canny(ctx, target);

    return OpCall(ctx, canny, &src, &dst, low_thresh, high_thresh, aperture_size, l2_gradient);
}

AURA_EXPORTS Status ICanny(Context *ctx, const Mat &dx, const Mat &dy, Mat &dst, MI_F64 low_thresh,
                           MI_F64 high_thresh, MI_BOOL l2_gradient, const OpTarget &target)
{
    Canny canny(ctx, target);

    return OpCall(ctx, canny, &dx, &dy, &dst, low_thresh, high_thresh, l2_gradient);
}

CannyImpl::CannyImpl(Context *ctx, const OpTarget &target) : OpImpl(ctx, "Canny", target),
                                                             m_low_thresh(0.0), m_high_thresh(0.0), m_aperture_size(0),
                                                             m_l2_gradient(MI_FALSE), m_is_aperture(MI_FALSE),
                                                             m_src(MI_NULL), m_dx(MI_NULL), m_dy(MI_NULL), m_dst(MI_NULL)

{}

Status CannyImpl::SetArgs(const Array *src, Array *dst, MI_F64 low_thresh, MI_F64 high_thresh,
                          MI_S32 aperture_size, MI_BOOL l2_gradient)
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

    if (!((src->GetSizes().m_height == dst->GetSizes().m_height) &&
         (src->GetSizes().m_width == dst->GetSizes().m_width)))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src and dst should have the same w/h size");
        return Status::ERROR;
    }

    if (dst->GetSizes().m_channel != 1)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "dst channel only support 1");
        return Status::ERROR;
    }

    if ((src->GetElemType() != ElemType::U8)|| (dst->GetElemType() != ElemType::U8))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "elem_type only support u8");
        return Status::ERROR;
    }

    m_src            = src;
    m_dx             = MI_NULL;
    m_dy             = MI_NULL;
    m_dst            = dst;
    m_low_thresh     = low_thresh;
    m_high_thresh    = high_thresh;
    m_aperture_size  = aperture_size;
    m_l2_gradient    = l2_gradient;
    m_is_aperture    = MI_TRUE;

    return Status::OK;
}

Status CannyImpl::SetArgs(const Array *dx, const Array *dy, Array *dst, MI_F64 low_thresh,
                          MI_F64 high_thresh, MI_BOOL l2_gradient)
{
    if (MI_NULL == m_ctx)
    {
        return Status::ERROR;
    }

    if (!((dx->IsValid() && dy->IsValid() && dst->IsValid())))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "invalid src or dst");
        return Status::ERROR;
    }

    if (!((dx->GetSizes().m_height == dst->GetSizes().m_height) &&
         (dx->GetSizes().m_width == dst->GetSizes().m_width)) &&
        !((dy->GetSizes().m_height == dst->GetSizes().m_height) &&
         (dy->GetSizes().m_width == dst->GetSizes().m_width)))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src and dst should have the same w/h size");
        return Status::ERROR;
    }

    if ((dx->GetElemType() != ElemType::S16) ||
        (dy->GetElemType() != ElemType::S16) ||
        (dst->GetElemType() != ElemType::U8))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "elem type error, dx/dy only support s16, dst only support u8");
        return Status::ERROR;
    }

    if (dst->GetSizes().m_channel != 1)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "dst channel only support 1");
        return Status::ERROR;
    }

    m_src            = MI_NULL;
    m_dx             = dx;
    m_dy             = dy;
    m_dst            = dst;
    m_low_thresh     = low_thresh;
    m_high_thresh    = high_thresh;
    m_l2_gradient    = l2_gradient;
    m_is_aperture    = MI_FALSE;

    return Status::OK;
}

std::vector<const Array*> CannyImpl::GetInputArrays() const
{
    if (m_is_aperture)
    {
        return {m_src};
    }
    else
    {
        return {m_dx, m_dy};
    }

}

std::vector<const Array*> CannyImpl::GetOutputArrays() const
{
    return {m_dst};
}

std::string CannyImpl::ToString() const
{
    std::string str;

    MI_CHAR low_thresh_str[20], high_thresh_str[20];
    snprintf(low_thresh_str, sizeof(m_low_thresh), "%.2f", m_low_thresh);
    snprintf(high_thresh_str, sizeof(m_high_thresh), "%.2f", m_high_thresh);

    str = "op(Canny)";
    str += " target(" + GetOpTarget().ToString() + ")";
    str += " param(l2_gradient:" + std::to_string(m_l2_gradient) + "low_thresh:" + low_thresh_str + " | " + "high_thresh:" + high_thresh_str + " | " +
            "aperture_size:" + std::to_string(m_aperture_size) + ")\n";

    return str;
}

AURA_VOID CannyImpl::Dump(const std::string &prefix) const
{
    JsonWrapper json_wrapper(m_ctx, prefix, m_name);

    if (m_is_aperture)
    {
        std::vector<std::string>  names  = {"src", "dst"};
        std::vector<const Array*> arrays = {m_src, m_dst};

        if (json_wrapper.SetArray(names, arrays) != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "SetArray src failed");
            return;
        }

        AURA_JSON_SERIALIZE(m_ctx, json_wrapper, m_src, m_dst, m_low_thresh, m_high_thresh, m_l2_gradient, m_is_aperture);
    }
    else
    {
        std::vector<std::string>  names  = {"src_dx", "src_dy", "dst"};
        std::vector<const Array*> arrays = {m_dx, m_dy, m_dst};

        if (json_wrapper.SetArray(names, arrays) != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "SetArray src failed");
            return;
        }

        AURA_JSON_SERIALIZE(m_ctx, json_wrapper, m_dx, m_dy, m_dst, m_low_thresh, m_high_thresh, m_l2_gradient, m_is_aperture);
    }
}

} // namespace aura