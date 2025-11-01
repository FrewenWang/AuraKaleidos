#include "aura/ops/misc/find_contours.hpp"
#include "find_contours_impl.hpp"
#include "aura/runtime/logger.h"
#include "aura/tools/json.h"

namespace aura
{

static std::shared_ptr<FindContoursImpl> CreateFindContoursImpl(Context *ctx, const OpTarget &target)
{
    std::shared_ptr<FindContoursImpl> impl;

    switch (target.m_type)
    {
        case TargetType::NONE:
        {
            impl.reset(new FindContoursNone(ctx, target));
            break;
        }

        default:
        {
            break;
        }
    }

    return impl;
}

FindContours::FindContours(Context *ctx, const OpTarget &target) : Op(ctx, target)
{}

Status FindContours::SetArgs(const Array *src, std::vector<std::vector<Point2i>> &contours, std::vector<Scalari> &hierarchy, 
                             ContoursMode mode, ContoursMethod method, Point2i offset)
{
    if ((MI_NULL == m_ctx))
    {
        return Status::ERROR;
    }

    if (MI_NULL == src)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src is null ptr");
        return Status::ERROR;
    }

    OpTarget impl_target = m_target;

    // set m_impl
    if (MI_NULL == m_impl.get() || impl_target != m_impl->GetOpTarget())
    {
        m_impl = CreateFindContoursImpl(m_ctx, impl_target);
    }

    // run initialize
    FindContoursImpl *find_contours_impl = dynamic_cast<FindContoursImpl *>(m_impl.get());
    if (MI_NULL == find_contours_impl)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "find_contours_impl is null ptr");
        return Status::ERROR;
    }

    Status ret = find_contours_impl->SetArgs(src, contours, hierarchy, mode, method, offset);

    AURA_RETURN(m_ctx, ret);
}

AURA_EXPORTS Status IFindContours(Context *ctx, const Mat &src, std::vector<std::vector<Point2i>> &contours, std::vector<Scalari> &hierarchy,
                                  ContoursMode mode, ContoursMethod method, Point2i offset, const OpTarget &target)
{
    FindContours find_contours(ctx, target);

    return OpCall(ctx, find_contours, &src, contours, hierarchy, mode, method, offset);
}

FindContoursImpl::FindContoursImpl(Context *ctx, const OpTarget &target) : OpImpl(ctx, "FindContours", target),
                                                                           m_mode(ContoursMode::RETR_EXTERNAL),
                                                                           m_method(ContoursMethod::CHAIN_APPROX_SIMPLE),
                                                                           m_src(MI_NULL)
{}

Status FindContoursImpl::SetArgs(const Array *src, std::vector<std::vector<Point2i>> &contours, std::vector<Scalari> &hierarchy,
                                 const ContoursMode mode, const ContoursMethod method, Point2i offset)
{
    if (MI_NULL == m_ctx)
    {
        return Status::ERROR;
    }

    if (!src->IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "invalid src");
        return Status::ERROR;
    }

    contours.clear();
    hierarchy.clear();

    m_src       = src;
    m_mode      = mode;
    m_method    = method;
    m_offset    = offset;
    m_contours  = &contours;
    m_hierarchy = &hierarchy;

    return Status::OK;
}

std::vector<const Array*> FindContoursImpl::GetInputArrays() const
{
    return {m_src};
}

std::string FindContoursImpl::ToString() const
{
    std::string str;

    str = "op(FindContours)";
    str += " target(" + GetOpTarget().ToString() + ")";
    str += " param(" + FindContoursModeToString(m_mode) + FindContoursMethodToString(m_method) + ")\n";

    return str;
}

AURA_VOID FindContoursImpl::Dump(const std::string &prefix) const
{
    JsonWrapper json_wrapper(m_ctx, prefix, m_name);

    if (json_wrapper.SetArray("src", m_src) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "SetArray src failed");
        return;
    }

    AURA_JSON_SERIALIZE(m_ctx, json_wrapper, m_src, m_mode, m_method, m_offset, *m_contours, *m_hierarchy);
}

} // namespace aura