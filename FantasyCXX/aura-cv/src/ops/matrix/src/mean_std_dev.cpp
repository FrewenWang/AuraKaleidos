#include "aura/ops/matrix/mean_std_dev.hpp"
#include "mean_std_dev_impl.hpp"
#include "aura/runtime/logger.h"
#include "aura/tools/json.h"

namespace aura
{

static std::shared_ptr<MeanStdDevImpl> CreateMeanStdDevImpl(Context *ctx, const OpTarget &target)
{
    std::shared_ptr<MeanStdDevImpl> impl;

    switch (target.m_type)
    {
        case TargetType::NONE:
        {
            impl.reset(new MeanStdDevNone(ctx, target));
            break;
        }

        case TargetType::NEON:
        {
#if defined(AURA_ENABLE_NEON)
            impl.reset(new MeanStdDevNeon(ctx, target));
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

MeanStdDev::MeanStdDev(Context *ctx, const OpTarget &target) : Op(ctx, target)
{}

Status MeanStdDev::SetArgs(const Array *src, Scalar &mean, Scalar &std_dev)
{
    if ((DT_NULL == m_ctx))
    {
        return Status::ERROR;
    }

    if (DT_NULL == src)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src is null ptr");
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
        m_impl = CreateMeanStdDevImpl(m_ctx, impl_target);
    }

    // run initialize
    MeanStdDevImpl *mean_std_dev_impl = dynamic_cast<MeanStdDevImpl*>(m_impl.get());
    if (DT_NULL == mean_std_dev_impl)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "mean_std_dev_impl is null ptr");
        return Status::ERROR;
    }

    Status ret = mean_std_dev_impl->SetArgs(src, &mean, &std_dev);

    AURA_RETURN(m_ctx, ret);
}

AURA_EXPORTS Status IMeanStdDev(Context *ctx, const Mat &src, Scalar &mean, Scalar &std_dev, const OpTarget &target)
{
    MeanStdDev mean_std_dev(ctx, target);

    return OpCall(ctx, mean_std_dev, &src, mean, std_dev);
}

MeanStdDevImpl::MeanStdDevImpl(Context *ctx, const OpTarget &target) : OpImpl(ctx, "MeanStdDev", target), m_src(DT_NULL), 
                                                                       m_mean(DT_NULL), m_std_dev(DT_NULL)
{}

Status MeanStdDevImpl::SetArgs(const Array *src, Scalar *mean, Scalar *std_dev)
{
    if (DT_NULL == m_ctx)
    {
        return Status::ERROR;
    }

    if (DT_NULL == src)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src is null");
        return Status::ERROR;
    }

    if ((DT_NULL == mean) || (DT_NULL == std_dev))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "mean or std_dev is null");
        return Status::ERROR;
    }

    if (!src->IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src or dst mat is invalid.");
        return Status::ERROR;
    }

    if (src->GetSizes().m_channel > 3)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "MeanStdDev only support channels <=3.");
        return Status::ERROR;
    }

    m_src     = src;
    m_mean    = mean;
    m_std_dev = std_dev;

    return Status::OK;
}

std::vector<const Array*> MeanStdDevImpl::GetInputArrays() const
{
    return {m_src};
}

std::string MeanStdDevImpl::ToString() const
{
    std::string str;

    str = "op(MeanStdDev)";
    str += " target(" + GetOpTarget().ToString() + ")\n";

    return str;
}

DT_VOID MeanStdDevImpl::Dump(const std::string &prefix) const
{
    JsonWrapper json_wrapper(m_ctx, prefix, m_name);

    if (json_wrapper.SetArray("src", m_src) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "SetArray src failed");
        return;
    }

    AURA_JSON_SERIALIZE(m_ctx, json_wrapper, m_src, *m_mean, *m_std_dev);
}

} // namespace aura