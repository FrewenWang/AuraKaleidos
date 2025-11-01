#include "norm_impl.hpp"
#include "aura/ops/matrix/sum.hpp"
#include "mean_std_dev_impl.hpp"
#include "aura/runtime/logger.h"

namespace aura
{

MeanStdDevNeon::MeanStdDevNeon(Context *ctx, const OpTarget &target) : MeanStdDevImpl(ctx, target)
{}

Status MeanStdDevNeon::SetArgs(const Array *src, Scalar *mean, Scalar *std_dev)
{
    if (MeanStdDevImpl::SetArgs(src, mean, std_dev) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "MeanStdDevImpl::SetArgs failed");
        return Status::ERROR;
    }

    if (src->GetArrayType() != ArrayType::MAT)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src must be mat type");
        return Status::ERROR;
    }

    return Status::OK;
}

Status MeanStdDevNeon::Run()
{
    const Mat *src = dynamic_cast<const Mat*>(m_src);

    if (MI_NULL == src)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src is null");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    Scalar sums    = Scalar::All(0.0);
    Scalar sq_sums = Scalar::All(0.0);

    ret = ISum(m_ctx, *src, sums, OpTarget::Neon());
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "MeanStdDevNeon call SumNeon failed.");
        return Status::ERROR;
    }

    ret = SqSumNeon(m_ctx, *src, sq_sums, m_target);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "MeanStdDevNeon call SqSumNeon failed.");
        return Status::ERROR;
    }

    Sizes3 sz     = src->GetSizes();
    MI_S32 width  = sz.m_width;
    MI_S32 height = sz.m_height;

    MI_S32 elem_count = width * height;

    for (MI_S32 i = 0; i < 4; ++i)
    {
        m_mean->m_val[i]    = sums.m_val[i] / elem_count;
        m_std_dev->m_val[i] = Sqrt(sq_sums.m_val[i] / elem_count - m_mean->m_val[i] * m_mean->m_val[i]);
    }
    AURA_RETURN(m_ctx, ret);
}

} // namespace aura