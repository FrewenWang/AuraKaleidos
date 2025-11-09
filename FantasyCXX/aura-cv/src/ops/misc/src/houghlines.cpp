#include "houghlines_impl.hpp"
#include "aura/runtime/logger.h"
#include "aura/tools/json.h"

namespace aura
{

static std::shared_ptr<HoughLinesImpl> CreateHoughLinesImpl(Context *ctx, const OpTarget &target)
{
    std::shared_ptr<HoughLinesImpl> impl;

    switch (target.m_type)
    {
        case TargetType::NONE:
        {
            impl.reset(new HoughLinesNone(ctx, target));
            break;
        }

        default :
        {
            break;
        }
    }

    return impl;
}

HoughLines::HoughLines(Context *ctx, const OpTarget &target) : Op(ctx, target)
{}

Status HoughLines::SetArgs(const Array *src, std::vector<Scalar> &lines, LinesType line_type, DT_F64 rho, DT_F64 theta, DT_S32 threshold,
                           DT_F64 srn, DT_F64 stn, DT_F64 min_theta, DT_F64 max_theta)
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
        m_impl = CreateHoughLinesImpl(m_ctx, impl_target);
    }

    // run SetArgs
    HoughLinesImpl *hough_lines_impl = dynamic_cast<HoughLinesImpl *>(m_impl.get());
    if (DT_NULL == hough_lines_impl)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "hough_lines_impl is null ptr");
        return Status::ERROR;
    }

    Status ret = hough_lines_impl->SetArgs(src, lines, line_type, rho, theta, threshold, srn, stn, min_theta, max_theta);

    AURA_RETURN(m_ctx, ret);
}

AURA_EXPORTS Status IHoughLines(Context *ctx, const Mat &mat, std::vector<Scalar> &lines, LinesType line_type, DT_F64 rho, DT_F64 theta, DT_S32 threshold,
                                DT_F64 srn, DT_F64 stn, DT_F64 min_theta, DT_F64 max_theta, const OpTarget &target)
{
    HoughLines hough_lines(ctx, target);

    return OpCall(ctx, hough_lines, &mat, lines, line_type, rho, theta, threshold, srn, stn, min_theta, max_theta);
}

HoughLinesImpl::HoughLinesImpl(Context *ctx, const OpTarget &target) : OpImpl(ctx, "HoughLines", target),
                                                                       m_line_type(LinesType::VEC2F),
                                                                       m_rho(0.0), m_theta(0.0), m_threshold(0),
                                                                       m_srn(0.0), m_stn(0.0),   m_min_theta(0.0),
                                                                       m_max_theta(AURA_PI), m_src(DT_NULL)
{}

Status HoughLinesImpl::SetArgs(const Array *mat, std::vector<Scalar> &lines, LinesType line_type, DT_F64 rho, DT_F64 theta, DT_S32 threshold,
                               DT_F64 srn, DT_F64 stn, DT_F64 min_theta, DT_F64 max_theta)
{
    if (DT_NULL == m_ctx)
    {
        return Status::ERROR;
    }

    if (!mat->IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "invalid mat");
        return Status::ERROR;
    }

    if (mat->GetElemType() != ElemType::U8)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "mat elem type error, should be u8");
        return Status::ERROR;
    }

    if (mat->GetSizes().m_channel != 1)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "channel number err, should be 1");
        return Status::ERROR;
    }

    if ((LinesType::VEC2F != line_type) && (LinesType::VEC3F != line_type))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "line_type type error, should be LinesType::VEC2F or LinesType::VEC3F");
        return Status::ERROR;
    }

    m_src       = mat;
    m_lines     = &lines;
    m_rho       = rho;
    m_theta     = theta;
    m_threshold = threshold;
    m_srn       = srn;
    m_stn       = stn;
    m_min_theta = min_theta;
    m_max_theta = max_theta;

    return Status::OK;
}

std::vector<const Array*> HoughLinesImpl::GetInputArrays() const
{
    return {m_src};
}

std::string HoughLinesImpl::ToString() const
{
    std::string str;

    str = "op(HoughCircles)";
    str += " target(" + GetOpTarget().ToString() + ")";
    str += " param(" + LinesTypeToString(m_line_type) + " | " + 
           "rho:" + std::to_string(m_rho) + " | " + "theta:" + std::to_string(m_theta) + " | "
           "threshold:" + std::to_string(m_threshold) + " | " + "srn:" + std::to_string(m_srn) + " | "
           "stn:" + std::to_string(m_stn) + " | " + "min_theta:" + std::to_string(m_min_theta) + " | "
           "max_theta:" + std::to_string(m_max_theta) + ")\n";

    return str;
}

DT_VOID HoughLinesImpl::Dump(const std::string &prefix) const
{
    JsonWrapper json_wrapper(m_ctx, prefix, m_name);

    if (json_wrapper.SetArray("src", m_src) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "SetArray src failed");
        return;
    }

    AURA_JSON_SERIALIZE(m_ctx, json_wrapper, m_src, m_line_type, m_rho, m_theta,
                        m_threshold, m_srn, m_stn, m_min_theta, m_max_theta, *m_lines);
}

static std::shared_ptr<HoughLinesPImpl> CreateHoughLinesPImpl(Context *ctx, const OpTarget &target)
{
    std::shared_ptr<HoughLinesPImpl> impl;

    switch (target.m_type)
    {
        case TargetType::NONE:
        {
            impl.reset(new HoughLinesPNone(ctx, target));
            break;
        }

        default :
        {
            break;
        }
    }

    return impl;
}

HoughLinesP::HoughLinesP(Context *ctx, const OpTarget &target) : Op(ctx, target)
{}

Status HoughLinesP::SetArgs(const Array *src, std::vector<Scalari> &lines, DT_F64 rho, DT_F64 theta, DT_S32 threshold,
                            DT_F64 min_line_length, DT_F64 max_gap)
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
        m_impl = CreateHoughLinesPImpl(m_ctx, impl_target);
    }

    // run SetArgs
    HoughLinesPImpl *hough_linesp_impl = dynamic_cast<HoughLinesPImpl *>(m_impl.get());
    if (DT_NULL == hough_linesp_impl)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "hough_linesp_impl is null ptr");
        return Status::ERROR;
    }

    Status ret = hough_linesp_impl->SetArgs(src, lines, rho, theta, threshold, min_line_length, max_gap);

    AURA_RETURN(m_ctx, ret);
}

AURA_EXPORTS Status IHoughLinesP(Context *ctx, const Mat &mat, std::vector<Scalari> &lines, DT_F64 rho, DT_F64 theta, DT_S32 threshold,
                                 DT_F64 min_line_length, DT_F64 max_gap, const OpTarget &target)
{
    HoughLinesP hough_linesp(ctx, target);

    return OpCall(ctx, hough_linesp, &mat, lines, rho, theta, threshold, min_line_length, max_gap);
}

HoughLinesPImpl::HoughLinesPImpl(Context *ctx, const OpTarget &target) : OpImpl(ctx, "HoughLinesP", target),
                                                                         m_rho(0.0), m_theta(0.0), m_threshold(0),
                                                                         m_min_line_length(0.0), m_max_gap(0.0), m_src(DT_NULL)
{}

Status HoughLinesPImpl::SetArgs(const Array *mat, std::vector<Scalari> &lines, DT_F64 rho, DT_F64 theta, DT_S32 threshold,
                                DT_F64 min_line_length, DT_F64 max_gap)
{
    if (DT_NULL == m_ctx)
    {
        return Status::ERROR;
    }

    if (!mat->IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "invalid mat");
        return Status::ERROR;
    }

    if (mat->GetElemType() != ElemType::U8)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "mat elem type error, should be u8");
        return Status::ERROR;
    }

    if (mat->GetSizes().m_channel != 1)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "channel number err, should be 1");
        return Status::ERROR;
    }

    m_src             = mat;
    m_lines           = &lines;
    m_rho             = rho;
    m_theta           = theta;
    m_threshold       = threshold;
    m_min_line_length = min_line_length;
    m_max_gap         = max_gap;

    return Status::OK;
}

std::vector<const Array*> HoughLinesPImpl::GetInputArrays() const
{
    return {m_src};
}

std::string HoughLinesPImpl::ToString() const
{
    std::string str;

    str = "op(HoughCircles)";
    str += " target(" + GetOpTarget().ToString() + ")";
    str += " param(" + std::string("rho:") + std::to_string(m_rho) + " | " + 
            "theta:" + std::to_string(m_theta) + " | "
            "threshold:" + std::to_string(m_threshold) + " | "
            "min_line_length:" + std::to_string(m_min_line_length) + " | "
            "max_gap:" + std::to_string(m_max_gap) + ")\n";

    return str;
}

DT_VOID HoughLinesPImpl::Dump(const std::string &prefix) const
{
    JsonWrapper json_wrapper(m_ctx, prefix, m_name);

    if (json_wrapper.SetArray("src", m_src) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "SetArray src failed");
        return;
    }

    AURA_JSON_SERIALIZE(m_ctx, json_wrapper, m_src, m_rho, m_theta,
                        m_threshold, m_min_line_length, m_max_gap, *m_lines);
}

} // namespace aura