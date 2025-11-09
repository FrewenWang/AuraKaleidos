#include "normalize_impl.hpp"
#include "aura/ops/matrix/norm.hpp"
#include "aura/ops/matrix/convert_to.hpp"
#include "aura/ops/matrix/min_max_loc.hpp"
#include "aura/runtime/logger.h"

namespace aura
{

static Status NormalizeMinMaxNeon(Context *ctx, const Mat &src, Mat &dst, DT_F32 alpha, DT_F32 beta, const OpTarget &target)
{
    Status ret = Status::OK;

    DT_F32 dmin = alpha;
    DT_F32 dmax = beta;

    MinMax(dmin, dmax);

    Point3i min_pos, max_pos;
    DT_F64 min_val = 0.0;
    DT_F64 max_val = 0.0;

    ret = IMinMaxLoc(ctx, src, &min_val, &max_val, &min_pos, &max_pos, target);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "Normalize call MinMaxLoc failed.");
        return ret;
    }

    DT_F32 scale = 1.0f;
    DT_F32 shift = 0.0f;
    DT_F32 delta = max_val - min_val;

    scale = (dmax - dmin) * (delta > FLT_EPSILON ? 1.0f / (delta) : 0.0f);
    shift = dmin - min_val * scale;

    ret = IConvertTo(ctx, src, dst, scale, shift, target);

    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "ConvertToNeon failed.");
    }

    AURA_RETURN(ctx, ret);
}

static Status NormalizeNormNeon(Context *ctx, const Mat &src, Mat &dst, DT_F32 alpha, NormType type, const OpTarget &target)
{
    Status ret = Status::ERROR;
    DT_F64 norm_value = 0.0;

    ret = INorm(ctx, src, &norm_value, type, OpTarget::Neon());
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "Normalize call Norm failed.");
        return ret;
    }

    DT_F32 scale = norm_value > DBL_EPSILON ? alpha / norm_value : 0.0;
    DT_F32 shift = 0.0;

    ret = IConvertTo(ctx, src, dst, scale, shift, target);

    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "ConvertToNeon failed.");
    }

    AURA_RETURN(ctx, ret);
}

NormalizeNeon::NormalizeNeon(Context *ctx, const OpTarget &target) : NormalizeImpl(ctx, target)
{}

Status NormalizeNeon::SetArgs(const Array *src, Array *dst, DT_F32 alpha, DT_F32 beta, NormType type)
{
    if (NormalizeImpl::SetArgs(src, dst, alpha, beta, type) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "NormalizeImpl::SetArgs failed");
        return Status::ERROR;
    }

    if ((src->GetArrayType() != ArrayType::MAT) || (dst->GetArrayType() != ArrayType::MAT))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src dst must be mat type");
        return Status::ERROR;
    }

    return Status::OK;
}

Status NormalizeNeon::Run()
{
    const Mat *src = dynamic_cast<const Mat*>(m_src);
    Mat *dst       = dynamic_cast<Mat*>(m_dst);

    if ((DT_NULL == src) || (DT_NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src or dst is null");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    if (NormType::NORM_MINMAX == m_type)
    {
        ret = NormalizeMinMaxNeon(m_ctx, *src, *dst, m_alpha, m_beta, m_target);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "NormalizeMinMaxNeon failed.");
        }
    }
    else
    {
        ret = NormalizeNormNeon(m_ctx, *src, *dst, m_alpha, m_type, m_target);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "NormalizeNormNeon failed.");
        }
    }

    AURA_RETURN(m_ctx, ret);
}

} // namespace aura