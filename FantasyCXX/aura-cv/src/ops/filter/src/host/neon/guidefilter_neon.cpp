#include "guidefilter_impl.hpp"
#include "aura/runtime/logger.h"

namespace aura
{
GuideFilterNeon::GuideFilterNeon(Context *ctx, const OpTarget &target) : GuideFilterImpl(ctx, target)
{}

Status GuideFilterNeon::SetArgs(const Array *src0, const Array *src1, Array *dst, MI_S32 ksize, MI_F32 eps,
                                GuideFilterType type, BorderType border_type, const Scalar &border_value)
{
    if (GuideFilterImpl::SetArgs(src0, src1, dst, ksize, eps, type, border_type, border_value) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "GuideFilterImpl::SetArgs failed");
        return Status::ERROR;
    }

    if ((src0->GetArrayType() != ArrayType::MAT) || (src1->GetArrayType() != ArrayType::MAT) || (dst->GetArrayType() != ArrayType::MAT))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src0 src1 dst must be mat type");
        return Status::ERROR;
    }

    if (((src0->GetElemType() == ElemType::U8)  || (src0->GetElemType() == ElemType::S8) ||
         (src0->GetElemType() == ElemType::S16) || (src0->GetElemType() == ElemType::S16)) &&
        (ksize > 255))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "elem_type u8/s8/u16/s16 only support kernel size <= 255");
        return Status::ERROR;
    }

    return Status::OK;
}

Status GuideFilterNeon::Run()
{
    const Mat *src0 = dynamic_cast<const Mat*>(m_src0);
    const Mat *src1 = dynamic_cast<const Mat*>(m_src1);
    Mat *dst        = dynamic_cast<Mat*>(m_dst);

    if ((MI_NULL == src0) || (MI_NULL == src1) || (MI_NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src0 or src1 or dst is null");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    switch (m_type)
    {
        case GuideFilterType::NORMAL:
        {
            ret = GuideFilterNormalNeon(m_ctx, *src0, *src1, *dst, m_ksize, m_eps, m_border_type, m_border_value, m_target);
            break;
        }

        case GuideFilterType::FAST:
        {
            ret = GuideFilterFastNeon(m_ctx, *src0, *src1, *dst, m_ksize, m_eps, m_border_type, m_border_value, m_target);
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(m_ctx, "GuideFilterType is not supported");
            return Status::ERROR;
        }
    }

    AURA_RETURN(m_ctx, ret);
}

} // namespace aura
