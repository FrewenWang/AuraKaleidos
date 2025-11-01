#include "bilateral_impl.hpp"
#include "aura/runtime/logger.h"

namespace aura
{

BilateralNeon::BilateralNeon(Context *ctx, const OpTarget &target) : BilateralImpl(ctx, target)
{}

Status BilateralNeon::SetArgs(const Array *src, Array *dst, MI_F32 sigma_color, MI_F32 sigma_space,
                    MI_S32 ksize, BorderType border_type, const Scalar &border_value)
{
    if (BilateralImpl::SetArgs(src, dst, sigma_color, sigma_space, ksize, border_type, border_value) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "BilateralImpl::SetArgs failed");
        return Status::ERROR;
    }

    if ((src->GetArrayType() != ArrayType::MAT) || (dst->GetArrayType() != ArrayType::MAT))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src dst must be mat type");
        return Status::ERROR;
    }

    if (ksize != 3)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "ksize only support 3");
        return Status::ERROR;
    }

    MI_S32 ch = src->GetSizes().m_channel;
    if (ch != 1 && ch != 3)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "channel only support 1/3");
        return Status::ERROR;
    }

    ElemType elem_type = src->GetElemType();
    if (elem_type != ElemType::U8)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "elem_type only support u8");
        return Status::ERROR;
    }

    return Status::OK;
}

Status BilateralNeon::Run()
{
    const Mat *src = dynamic_cast<const Mat*>(m_src);
    Mat *dst = dynamic_cast<Mat*>(m_dst);

    if ((MI_NULL == src) || (MI_NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src dst is null");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    switch (m_ksize)
    {
        case 3:
        {
            ret = Bilateral3x3Neon(m_ctx, *src, *dst, m_space_weight, m_color_weight, m_valid_num, m_border_type, m_border_value, m_target);
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