#include "median_impl.hpp"
#include "aura/runtime/logger.h"

namespace aura
{

MedianNeon::MedianNeon(Context *ctx, const OpTarget &target) : MedianImpl(ctx, target)
{}

Status MedianNeon::SetArgs(const Array *src, Array *dst, DT_S32 ksize)
{
    if (MedianImpl::SetArgs(src, dst, ksize) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "MedianImpl::Initialize failed");
        return Status::ERROR;
    }

    if ((src->GetArrayType() != ArrayType::MAT) || (dst->GetArrayType() != ArrayType::MAT))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src dst must be mat type");
        return Status::ERROR;
    }

    if (ksize != 3 && ksize != 5 && ksize != 7)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "ksize must be 3/5/7");
        return Status::ERROR;
    }

    DT_S32 ch = src->GetSizes().m_channel;

    if (ch != 1 && ch != 2 && ch != 3)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "channel must be 1/2/3");
        return Status::ERROR;
    }

    return Status::OK;
}

Status MedianNeon::Run()
{
    const Mat *src = dynamic_cast<const Mat*>(m_src);
    Mat *dst = dynamic_cast<Mat*>(m_dst);

    if ((DT_NULL == src) || (DT_NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src dst is null");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    switch (m_ksize)
    {
        case 3:
        {
            ret = Median3x3Neon(m_ctx, *src, *dst, m_target);
            break;
        }
        case 5:
        {
            ret = Median5x5Neon(m_ctx, *src, *dst, m_target);
            break;
        }
        case 7:
        {
            ret = Median7x7Neon(m_ctx, *src, *dst, m_target);
            break;
        }
        default:
        {
            AURA_ADD_ERROR_STRING(m_ctx, "Unsuported ksize");
            return Status::ERROR;
        }
    }

    AURA_RETURN(m_ctx, ret);
}

} // namespace aura
