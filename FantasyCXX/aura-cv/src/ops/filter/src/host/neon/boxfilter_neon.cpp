#include "boxfilter_impl.hpp"
#include "aura/runtime/logger.h"

namespace aura
{

BoxFilterNeon::BoxFilterNeon(Context *ctx, const OpTarget &target) : BoxFilterImpl(ctx, target)
{}

Status BoxFilterNeon::SetArgs(const Array *src, Array *dst, DT_S32 ksize,
                              BorderType border_type, const Scalar &border_value)
{
    if (BoxFilterImpl::SetArgs(src, dst, ksize, border_type, border_value) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "BoxFilterImpl::SetArgs failed");
        return Status::ERROR;
    }

    if ((src->GetArrayType() != ArrayType::MAT) || (dst->GetArrayType() != ArrayType::MAT))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src or dst must be mat type");
        return Status::ERROR;
    }

    if ((ksize & 1) != 1)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "ksize must be odd");
        return Status::ERROR;
    }

    if (((src->GetElemType() == ElemType::U8)  || (src->GetElemType() == ElemType::S8) ||
         (src->GetElemType() == ElemType::S16) || (src->GetElemType() == ElemType::S16)) &&
        (ksize > 255))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "elem_type u8/s8/u16/s16 only support kernel size <= 255");
        return Status::ERROR;
    }

    DT_S32 ch = src->GetSizes().m_channel;

    if (ch != 1 && ch != 2 && ch != 3)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "only support channel 1/2/3");
        return Status::ERROR;
    }

    return Status::OK;
}

Status BoxFilterNeon::Run()
{
    const Mat *src = dynamic_cast<const Mat*>(m_src);
    Mat *dst       = dynamic_cast<Mat*>(m_dst);

    if ((DT_NULL == src) || (DT_NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src or dst is null");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    switch (m_ksize)
    {
        case 3:
        {
            ret = BoxFilter3x3Neon(m_ctx, *src, *dst, m_border_type, m_border_value, m_target);
            break;
        }

        case 5:
        {
            ret = BoxFilter5x5Neon(m_ctx, *src, *dst, m_border_type, m_border_value, m_target);
            break;
        }

        case 7:
        {
            ret = BoxFilter7x7Neon(m_ctx, *src, *dst, m_border_type, m_border_value, m_target);
            break;
        }

        default:
        {
            ret = BoxFilterKxKNeon(m_ctx, *src, *dst, m_ksize, m_border_type, m_border_value, m_target);
            break;
        }
    }

    AURA_RETURN(m_ctx, ret);
}

} // namespace aura