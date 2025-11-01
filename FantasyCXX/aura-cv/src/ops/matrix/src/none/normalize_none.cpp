#include "normalize_impl.hpp"
#include "aura/ops/matrix/norm.hpp"
#include "aura/ops/matrix/convert_to.hpp"
#include "aura/ops/matrix/min_max_loc.hpp"
#include "aura/runtime/logger.h"

namespace aura
{

static Status NormalizeMinMaxNone(Context *ctx, const Mat &src, Mat &dst, MI_F32 alpha, MI_F32 beta)
{
    Status ret = Status::OK;

    MI_F32 dmin = alpha;
    MI_F32 dmax = beta;

    MinMax(dmin, dmax);

    Point3i min_pos, max_pos;
    MI_F64 min_val = 0.0;
    MI_F64 max_val = 0.0;

    ret = IMinMaxLoc(ctx, src, &min_val, &max_val, &min_pos, &max_pos, OpTarget::None());

    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "NormalizeMinMaxNone call MinMaxLocNone failed.");
        return ret;
    }

    MI_F32 scale = 1.0f;
    MI_F32 shift = 0.0f;
    MI_F32 delta = max_val - min_val;

    scale = (dmax - dmin) * (delta > FLT_EPSILON ? 1.0f / (delta) : 0.0f);
    shift = dmin - min_val * scale;

    ret = IConvertTo(ctx, src, dst, scale, shift, OpTarget::None());

    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "NormalizeMinMaxNone call ConvertToNone failed.");
        return ret;
    }

    AURA_RETURN(ctx, ret);
}

static Status NormalizeNormNone(Context *ctx, const Mat &src, Mat &dst, MI_F32 alpha, NormType type)
{
    Status ret = Status::OK;

    MI_F64 norm_value = 0.0;

    ret = INorm(ctx, src, &norm_value, type, OpTarget::None());

    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "Normalize call Norm failed.");
        return Status::ERROR;
    }

    MI_F32 scale = norm_value > DBL_EPSILON ? alpha / norm_value : 0.0;
    MI_F32 shift = 0.0;

    ret = IConvertTo(ctx, src, dst, scale, shift, OpTarget::None());

    AURA_RETURN(ctx, ret);
}

NormalizeNone::NormalizeNone(Context *ctx, const OpTarget &target) : NormalizeImpl(ctx, target)
{}

Status NormalizeNone::SetArgs(const Array *src, Array *dst, MI_F32 alpha, MI_F32 beta, NormType type)
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

Status NormalizeNone::Run()
{
    const Mat *src = dynamic_cast<const Mat*>(m_src);
    Mat *dst       = dynamic_cast<Mat*>(m_dst);

    if ((MI_NULL == src) || (MI_NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src or dst is null");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    if (NormType::NORM_MINMAX == m_type)
    {
        ret = NormalizeMinMaxNone(m_ctx, *src, *dst, m_alpha, m_beta);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "NormalizeMinMaxNone failed.");
        }
    }
    else
    {
        ret = NormalizeNormNone(m_ctx, *src, *dst, m_alpha, m_type);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "NormalizeNormNone failed.");
        }
    }

    AURA_RETURN(m_ctx, ret);
}

} // namespace aura