#include "sobel_impl.hpp"
#include "aura/runtime/logger.h"

namespace aura
{

SobelNeon::SobelNeon(Context *ctx, const OpTarget &target) : SobelImpl(ctx, target)
{}

Status SobelNeon::SetArgs(const Array *src, Array *dst, DT_S32 dx, DT_S32 dy, DT_S32 ksize, DT_F32 scale,
                          BorderType border_type, const Scalar &border_value)
{
    if (SobelImpl::SetArgs(src, dst, dx, dy, ksize, scale, border_type, border_value) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "SobelImpl::SetArgs failed");
        return Status::ERROR;
    }

    if ((src->GetArrayType() != ArrayType::MAT) || (dst->GetArrayType() != ArrayType::MAT))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src dst must be mat type");
        return Status::ERROR;
    }

    if (m_ksize <= 0)
    {
        m_dx = m_dx > 0 ? m_dx : 3;
        m_dy = m_dy > 0 ? m_dy : 3;
        m_ksize = 3;
    }

    if ((m_dx > 0) && (m_dy > 0) && (m_ksize == 1))
    {
        m_ksize = 3;
    }

    if (m_ksize != 1 && m_ksize != 3 && m_ksize != 5)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "ksize is only suppose 1/3/5");
        return Status::ERROR;
    }

    DT_S32 ch = src->GetSizes().m_channel;

    if (ch != 1 && ch != 2 && ch != 3)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "channel is only suppose 1/2/3");
        return Status::ERROR;
    }

    return Status::OK;
}

Status SobelNeon::Run()
{
    const Mat *src = dynamic_cast<const Mat*>(m_src);
    Mat *dst       = dynamic_cast<Mat*>(m_dst);

    if ((DT_NULL == src) || (DT_NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src dst is null");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    switch (m_ksize)
    {
        case 1:
        {
            ret = Sobel1x1Neon(m_ctx, *src, *dst, m_dx, m_dy, m_scale, m_border_type, m_border_value, m_target);
            break;
        }
        case 3:
        {
            ret = Sobel3x3Neon(m_ctx, *src, *dst, m_dx, m_dy, m_scale, m_border_type, m_border_value, m_target);
            break;
        }
        case 5:
        {
            ret = Sobel5x5Neon(m_ctx, *src, *dst, m_dx, m_dy, m_scale, m_border_type, m_border_value, m_target);
            break;
        }
        default:
        {
            AURA_ADD_ERROR_STRING(m_ctx, "unsupported ksize");
            return Status::ERROR;
        }
    }

    AURA_RETURN(m_ctx, ret);
}

} // namespace aura
