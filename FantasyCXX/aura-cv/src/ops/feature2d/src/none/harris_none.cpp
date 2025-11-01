#include "harris_impl.hpp"
#include "aura/ops/filter.h"
#include "aura/runtime/logger.h"

namespace aura
{

static AURA_VOID CalcHarrisNone(Context *ctx, const Mat &src, Mat &dst, MI_F64 k)
{
    AURA_UNUSED(ctx);
    Sizes3 size = src.GetSizes();

    for (MI_S32 y = 0; y < size.m_height; y++)
    {
        const MI_F32 *src_data = src.Ptr<MI_F32>(y);
        MI_F32       *dst_data = dst.Ptr<MI_F32>(y);

        for (MI_S32 x = 0; x < size.m_width; x++)
        {
            MI_F32 a = src_data[x * 3];
            MI_F32 b = src_data[x * 3 + 1];
            MI_F32 c = src_data[x * 3 + 2];
            dst_data[x] = a * c - b * b - k * (a + c) * (a + c);
        }
    }
}

static AURA_VOID CalcMinEigenValNone(Context *ctx, const Mat &src, Mat &dst)
{
    AURA_UNUSED(ctx);
    Sizes3 size = src.GetSizes();

    for (MI_S32 y = 0; y < size.m_height; y++)
    {
        const MI_F32 *src_data = src.Ptr<MI_F32>(y);
        MI_F32       *dst_data = dst.Ptr<MI_F32>(y);

        for (MI_S32 x = 0; x < size.m_width; x++)
        {
            MI_F32 a = src_data[x * 3] * 0.5f;
            MI_F32 b = src_data[x * 3 + 1];
            MI_F32 c = src_data[x * 3 + 2] * 0.5f;
            dst_data[x] = (a + c) - Sqrt(Pow((a - c), 2) + Pow(b, 2));
        }
    }
}

Status CornerEigenValsVecsNone(Context *ctx, const Mat &src, Mat &dst, MI_S32 block_size, MI_S32 aperture_size,
                               MI_BOOL use_harris, MI_F64 k, BorderType border_type, const Scalar &border_value, const OpTarget &target)
{
    MI_F64 scale = (1 << ((aperture_size > 0 ? aperture_size : 3) - 1)) * block_size;
    if (aperture_size < 0)
    {
        scale *= 2.0;
    }

    if (src.GetElemType() == ElemType::U8)
    {
        scale *= 255.0;
    }

    scale = 1.0 / scale;
    const Sizes3 &src_size  = src.GetSizes();

    Mat dx(ctx, ElemType::F32, src_size);
    Mat dy(ctx, ElemType::F32, src_size);
    Status ret = Status::ERROR;

    ret  = ISobel(ctx, src, dx, 1, 0, aperture_size, scale, border_type, border_value, target);
    ret |= ISobel(ctx, src, dy, 0, 1, aperture_size, scale, border_type, border_value, target);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "sobel excute failed");
        return Status::ERROR;
    }

    Sizes3 cov_size(src_size.m_height, src_size.m_width, 3);
    Mat cov(ctx, ElemType::F32, cov_size);
    if (!cov.IsValid())
    {
        AURA_ADD_ERROR_STRING(ctx, "cov is invalid");
        return Status::ERROR;
    }

    for (MI_S32 y = 0; y < cov_size.m_height; y++)
    {
        MI_F32 *cov_data = cov.Ptr<MI_F32>(y);
        MI_F32 *dx_data  = dx.Ptr<MI_F32>(y);
        MI_F32 *dy_data  = dy.Ptr<MI_F32>(y);

        for (MI_S32 x = 0; x < cov_size.m_width; x++)
        {
            MI_F32 dx = dx_data[x];
            MI_F32 dy = dy_data[x];

            cov_data[x * 3] = dx * dx;
            cov_data[x * 3 + 1] = dx * dy;
            cov_data[x * 3 + 2] = dy * dy;
        }
    }

    Mat cov_dst(ctx, ElemType::F32, cov_size);
    if (!cov_dst.IsValid())
    {
        AURA_ADD_ERROR_STRING(ctx, "cov_dst is invalid");
        return Status::ERROR;
    }

    ret = IBoxfilter(ctx, cov, cov_dst, block_size, border_type, border_value, target);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "Boxfilter excute failed");
        return Status::ERROR;
    }

    MI_F32 scale_tmp = block_size * block_size;
    for (MI_S32 y = 0; y < cov_size.m_height; y++)
    {
        MI_F32 *dst_data = cov_dst.Ptr<MI_F32>(y);
        for (MI_S32 x = 0; x < cov_size.m_width; x++)
        {
            dst_data[x * 3] *= scale_tmp;
            dst_data[x * 3 + 1] *= scale_tmp;
            dst_data[x * 3 + 2] *= scale_tmp;
        }
    }

    if (use_harris)
    {
        CalcHarrisNone(ctx, cov_dst, dst, k);
    }
    else
    {
        CalcMinEigenValNone(ctx, cov_dst, dst);
    }

    return Status::OK;
}

HarrisNone::HarrisNone(Context *ctx, const OpTarget &target) : HarrisImpl(ctx, target)
{}

Status HarrisNone::SetArgs(const Array *src, Array *dst, MI_S32 block_size, MI_S32 ksize, MI_F64 k,
                           BorderType border_type, const Scalar &border_value)
{
    if (HarrisImpl::SetArgs(src, dst, block_size, ksize, k, border_type, border_value) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "HarrisImpl::SetArgs failed");
        return Status::ERROR;
    }

    if ((src->GetArrayType() != ArrayType::MAT) || (dst->GetArrayType() != ArrayType::MAT))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src dst must be mat type");
        return Status::ERROR;
    }

    return Status::OK;
}

Status HarrisNone::Run()
{
    const Mat *src = dynamic_cast<const Mat*>(m_src);
    Mat *dst = dynamic_cast<Mat*>(m_dst);
    if ((MI_NULL == src) || (MI_NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src dst kmat is null");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    switch (src->GetElemType())
    {
        case ElemType::U8:
        case ElemType::F32:
        {
            ret = CornerEigenValsVecsNone(m_ctx, *src, *dst, m_block_size, m_ksize, MI_TRUE,
                                          m_k, m_border_type, m_border_value, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "CornerEigenValsVecsNone failed, ElemType: U8 or F32");
            }
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(m_ctx, "elem type error");
            return Status::ERROR;
        }
    }

    AURA_RETURN(m_ctx, ret);
}

} // namespace aura