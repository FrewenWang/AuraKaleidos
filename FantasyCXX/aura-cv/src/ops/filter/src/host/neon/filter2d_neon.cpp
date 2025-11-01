#include "filter2d_impl.hpp"
#include "aura/runtime/logger.h"

namespace aura
{

Filter2dNeon::Filter2dNeon(Context *ctx, const OpTarget &target) : Filter2dImpl(ctx, target)
{}

Status Filter2dNeon::SetArgs(const Array *src, Array *dst, const Array *kmat,
                             BorderType border_type, const Scalar &border_value)
{
    if (Filter2dImpl::SetArgs(src, dst, kmat, border_type, border_value) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Filter2dImpl::SetArgs failed");
        return Status::ERROR;
    }

    if ((src->GetArrayType()  != ArrayType::MAT) || (dst->GetArrayType() != ArrayType::MAT) ||
        (kmat->GetArrayType() != ArrayType::MAT))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src dst kmat must be mat type");
        return Status::ERROR;
    }

    if (m_ksize != 3 && m_ksize != 5 && m_ksize != 7)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "ksize is only suppose 3/5/7");
        return Status::ERROR;
    }

    MI_S32 ch = src->GetSizes().m_channel;
    if (ch != 1 && ch != 2 && ch != 3)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "channel is only suppose 1/2/3");
        return Status::ERROR;
    }

    return Status::OK;
}

Status Filter2dNeon::Run()
{
    const Mat *src  = dynamic_cast<const Mat*>(m_src);
    Mat *dst        = dynamic_cast<Mat*>(m_dst);
    const Mat *kmat = dynamic_cast<const Mat*>(m_kmat);

    if ((MI_NULL == src) || (MI_NULL == dst) || (MI_NULL == kmat))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src dst kmat is null");
        return Status::ERROR;
    }

    std::vector<MI_F32> kernel(m_ksize * m_ksize, 0.f);
    for (MI_S32 y = 0; y < m_ksize; y++)
    {
        const MI_F32 *k_row = kmat->Ptr<MI_F32>(y);
        for (MI_S32 x = 0; x < m_ksize; x++)
        {
            kernel[y * m_ksize + x] = k_row[x];
        }
    }

    Status ret = Status::ERROR;
    switch (m_ksize)
    {
        case 3:
        {
            ret = Filter2d3x3Neon(m_ctx, *src, *dst, kernel, m_border_type, m_border_value, m_target);
            break;
        }
        case 5:
        {
            ret = Filter2d5x5Neon(m_ctx, *src, *dst, kernel, m_border_type, m_border_value, m_target);
            break;
        }
        case 7:
        {
            ret = Filter2d7x7Neon(m_ctx, *src, *dst, kernel, m_border_type, m_border_value, m_target);
            break;
        }
        default:
        {
            AURA_ADD_ERROR_STRING(m_ctx, "Unsupported ksize");
            return Status::ERROR;
        }
    }

    AURA_RETURN(m_ctx, ret);
}

} // namespace aura
