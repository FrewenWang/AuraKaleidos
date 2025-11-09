#include "houghcircles_impl.hpp"
#include "aura/runtime/logger.h"
#include "aura/tools/json.h"

namespace aura
{

static std::shared_ptr<HoughCirclesImpl> CreateHoughCirclesImpl(Context *ctx, const OpTarget &target)
{
    std::shared_ptr<HoughCirclesImpl> impl;

    switch (target.m_type)
    {
        case TargetType::NONE:
        {
            impl.reset(new HoughCirclesNone(ctx, target));
            break;
        }

        case TargetType::NEON:
        {
#if defined(AURA_ENABLE_NEON)
            impl.reset(new HoughCirclesNone(ctx, target));
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

HoughCircles::HoughCircles(Context *ctx, const OpTarget &target) : Op(ctx, target)
{}

Status HoughCircles::SetArgs(const Array *src, std::vector<Scalar> &circles, HoughCirclesMethod method, DT_F64 dp,
                             DT_F64 min_dist, DT_F64 canny_thresh, DT_F64 acc_thresh, DT_S32 min_radius, DT_S32 max_radius)
{
    if ((DT_NULL == m_ctx))
    {
        return Status::ERROR;
    }

    if ((DT_NULL == src))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src is null ptr");
        return Status::ERROR;
    }

    OpTarget impl_target = m_target;

    // set m_impl
    if (DT_NULL == m_impl.get() || impl_target != m_impl->GetOpTarget())
    {
        m_impl = CreateHoughCirclesImpl(m_ctx, impl_target);
    }

    // run SetArgs
    HoughCirclesImpl *hough_circles_impl = dynamic_cast<HoughCirclesImpl *>(m_impl.get());
    if (DT_NULL == hough_circles_impl)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "hough_circles_impl is null ptr");
        return Status::ERROR;
    }

    Status ret = hough_circles_impl->SetArgs(src, circles, method, dp, min_dist, canny_thresh, acc_thresh, min_radius, max_radius);

    AURA_RETURN(m_ctx, ret);
}

AURA_EXPORTS Status IHoughCircles(Context *ctx, const Mat &mat, std::vector<Scalar> &circles, HoughCirclesMethod method, DT_F64 dp,
                                  DT_F64 min_dist, DT_F64 canny_thresh, DT_F64 acc_thresh, DT_S32 min_radius, DT_S32 max_radius,
                                  const OpTarget &target)
{
    HoughCircles hough_circles(ctx, target);

    return OpCall(ctx, hough_circles, &mat, circles, method, dp, min_dist, canny_thresh, acc_thresh, min_radius, max_radius);
}

HoughCirclesImpl::HoughCirclesImpl(Context *ctx, const OpTarget &target) : OpImpl(ctx, "HoughCircles", target),
                                                                           m_method(HoughCirclesMethod::HOUGH_GRADIENT),
                                                                           m_dp(0.0), m_min_dist(0.0), m_canny_thresh(0.0),
                                                                           m_acc_thresh(0.0), m_min_radius(0), m_max_radius(0),
                                                                           m_src(DT_NULL)
{}

Status HoughCirclesImpl::SetArgs(const Array *mat, std::vector<Scalar> &circles, HoughCirclesMethod method, DT_F64 dp,
                                 DT_F64 min_dist, DT_F64 canny_thresh, DT_F64 acc_thresh, DT_S32 min_radius, DT_S32 max_radius)
{
    if (DT_NULL == m_ctx)
    {
        return Status::ERROR;
    }

    if (!(mat->IsValid()))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "invalid src");
        return Status::ERROR;
    }

    if (mat->GetElemType() != ElemType::U8)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "mat type error, should be u8");
        return Status::ERROR;
    }

    if (mat->GetSizes().m_channel != 1)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "channel number err, should be 1");
        return Status::ERROR;
    }

    if (method != HoughCirclesMethod::HOUGH_GRADIENT)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "hough circle method err, only support HOUGH_GRADIENT");
        return Status::ERROR;
    }

    if ((dp <= 0) || (min_dist <= 0) || (canny_thresh <= 0) || (acc_thresh <= 0))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "dp, min_dist, canny_thresh and acc_thresh should be positive");
        return Status::ERROR;
    }

    m_src          = mat;
    m_circles      = &circles;
    m_method       = method;
    m_dp           = dp;
    m_min_dist     = min_dist;
    m_canny_thresh = canny_thresh;
    m_acc_thresh   = acc_thresh;
    m_min_radius   = min_radius;
    m_max_radius   = max_radius;

    return Status::OK;
}

std::vector<const Array*> HoughCirclesImpl::GetInputArrays() const
{
    return {m_src};
}

std::string HoughCirclesImpl::ToString() const
{
    std::string str;

    str = "op(HoughCircles)";
    str += " target(" + GetOpTarget().ToString() + ")";
    str += " param(" + HoughCirclesMethodToString(m_method) + " | " + 
           "dp:" + std::to_string(m_dp) + " | " + "min_dist:" + std::to_string(m_min_dist) + " | "
           "canny_thresh:" + std::to_string(m_canny_thresh) + " | " + "acc_thresh:" + std::to_string(m_acc_thresh) + " | "
           "min_radius:" + std::to_string(m_min_radius) + " | " + "max_radius:" + std::to_string(m_max_radius)+ ")\n";

    return str;
}

DT_VOID HoughCirclesImpl::Dump(const std::string &prefix) const
{
    JsonWrapper json_wrapper(m_ctx, prefix, m_name);

    if (json_wrapper.SetArray("src", m_src) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "SetArray src failed");
        return;
    }

    AURA_JSON_SERIALIZE(m_ctx, json_wrapper, m_src, m_method, m_dp, m_min_dist,
                        m_canny_thresh, m_acc_thresh, m_min_radius, m_max_radius, *m_circles);
}

} // namespace aura