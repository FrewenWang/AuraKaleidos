#include "laplacian_impl.hpp"
#include "aura/runtime/logger.h"

namespace aura
{

LaplacianNeon::LaplacianNeon(Context *ctx, const OpTarget &target) : LaplacianImpl(ctx, target)
{}

Status LaplacianNeon::SetArgs(const Array *src, Array *dst, DT_S32 ksize,
                              BorderType border_type, const Scalar &border_value)
{
    if (LaplacianImpl::SetArgs(src, dst, ksize, border_type, border_value) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "LaplacianImpl::Iinitialize failed");
        return Status::ERROR;
    }

    if ((src->GetArrayType() != ArrayType::MAT) || (dst->GetArrayType() != ArrayType::MAT))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src and dst must be mat type");
        return Status::ERROR;
    }

    if (ksize != 1 && ksize != 3 && ksize != 5 && ksize != 7)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "ksize only supports 1/3/5/7");
        return Status::ERROR;
    }

    DT_S32 ch = src->GetSizes().m_channel;

    if (ch != 1 && ch != 2 && ch != 3)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "channel only supports 1/2/3");
        return Status::ERROR;
    }

    return Status::OK;
}

Status LaplacianNeon::Run()
{
    const Mat *src = dynamic_cast<const Mat*>(m_src);
    Mat       *dst = dynamic_cast<Mat*>(m_dst);

    if (DT_NULL == src || DT_NULL == dst)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src dst is null");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    switch (m_ksize)
    {
        case 1:
        {
            ret = Laplacian1x1Neon(m_ctx, *src, *dst, m_border_type, m_border_value, m_target);
            break;
        }

        case 3:
        {
            ret = Laplacian3x3Neon(m_ctx, *src, *dst, m_border_type, m_border_value, m_target);
            break;
        }

        case 5:
        {
            ret = Laplacian5x5Neon(m_ctx, *src, *dst, m_border_type, m_border_value, m_target);
            break;
        }

        case 7:
        {
            ret = Laplacian7x7Neon(m_ctx, *src, *dst, m_border_type, m_border_value, m_target);
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(m_ctx, "Unsupport ksize");
            return Status::ERROR;
        }
    }

    AURA_RETURN(m_ctx, ret);
}

} // namespace aura
