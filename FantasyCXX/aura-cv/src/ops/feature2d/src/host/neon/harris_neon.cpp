#include "harris_impl.hpp"
#include "aura/ops/filter.h"
#include "aura/runtime/logger.h"

namespace aura
{

static DT_VOID CalcHarrisNeon(Context *ctx, const Mat &src, Mat &dst, DT_F64 k)
{
    AURA_UNUSED(ctx);
    Sizes3 size = src.GetSizes();
    float32x4_t vqf32_const_k;
    neon::vdup(vqf32_const_k, static_cast<DT_F32>(k));

    DT_S32 width_align4 = size.m_width & (-4);
    for (DT_S32 y = 0; y < size.m_height; y++)
    {
        const DT_F32 *src_data = src.Ptr<DT_F32>(y);
        DT_F32       *dst_data = dst.Ptr<DT_F32>(y);

        DT_S32 x = 0;
        for (; x < width_align4; x += 4)
        {
            float32x4x3_t v3qf32_src   = neon::vload3q(src_data);
            float32x4_t   vqf32_sub    = neon::vsub(neon::vmul(v3qf32_src.val[0], v3qf32_src.val[2]), neon::vmul(v3qf32_src.val[1], v3qf32_src.val[1]));
            float32x4_t   vqf32_result = neon::vadd(v3qf32_src.val[0], v3qf32_src.val[2]);
            vqf32_result               = neon::vmul(neon::vmul(vqf32_result, vqf32_result), vqf32_const_k);
            vqf32_result               = neon::vsub(vqf32_sub, vqf32_result);
            neon::vstore(dst_data, vqf32_result);

            src_data += 12;
            dst_data += 4;
        }
        for (; x < size.m_width; x++)
        {
            DT_F32 a = *src_data++;
            DT_F32 b = *src_data++;
            DT_F32 c = *src_data++;
            *dst_data++ = a * c - b * b - k * (a + c) * (a + c);
        }
    }
}

static DT_VOID CalcMinEigenValNeon(Context *ctx, const Mat &src, Mat &dst)
{
    AURA_UNUSED(ctx);
    Sizes3 size = src.GetSizes();
    float32x4_t vqf32_const;
    neon::vdup(vqf32_const, 0.5f);

    DT_S32 width_align4 = size.m_width & (-4);
    for (DT_S32 y = 0; y < size.m_height; y++)
    {
        const DT_F32 *src_data = src.Ptr<DT_F32>(y);
        DT_F32       *dst_data = dst.Ptr<DT_F32>(y);

        DT_S32 x = 0;
        for (; x < width_align4; x += 4)
        {
            float32x4x3_t v3qf32_src = neon::vload3q(src_data);
            float32x4_t   vqf32_mul0 = neon::vmul(neon::vadd(v3qf32_src.val[0], v3qf32_src.val[2]), vqf32_const);
            float32x4_t   vqf32_mul1 = neon::vmul(neon::vsub(v3qf32_src.val[0], v3qf32_src.val[2]), vqf32_const);
            float32x4_t   vqf32_mul2 = neon::vmul(v3qf32_src.val[1], v3qf32_src.val[1]);

            vqf32_mul1 = neon::vadd(neon::vmul(vqf32_mul1, vqf32_mul1), vqf32_mul2);
            vqf32_mul1 = neon::vsetlane<0>(Sqrt(neon::vgetlane<0>(vqf32_mul1)), vqf32_mul1);
            vqf32_mul1 = neon::vsetlane<1>(Sqrt(neon::vgetlane<1>(vqf32_mul1)), vqf32_mul1);
            vqf32_mul1 = neon::vsetlane<2>(Sqrt(neon::vgetlane<2>(vqf32_mul1)), vqf32_mul1);
            vqf32_mul1 = neon::vsetlane<3>(Sqrt(neon::vgetlane<3>(vqf32_mul1)), vqf32_mul1);

            float32x4_t vqf32_result = neon::vsub(vqf32_mul0, vqf32_mul1);
            neon::vstore(dst_data, vqf32_result);

            src_data += 12;
            dst_data += 4;
        }
        for (; x < size.m_width; x++)
        {
            DT_F32 a = *src_data++ * 0.5f;
            DT_F32 b = *src_data++;
            DT_F32 c = *src_data++ * 0.5f;
            *dst_data++ = (a + c) - Sqrt(Pow((a - c), 2) + Pow(b, 2));
        }
    }
}

Status CornerEigenValsVecsNeon(Context *ctx, const Mat &src, Mat &dst, DT_S32 block_size, DT_S32 aperture_size,
                               DT_BOOL use_harris, DT_F64 k, BorderType border_type, const Scalar &border_value, const OpTarget &target)
{
    DT_F64 scale = (1 << ((aperture_size > 0 ? aperture_size : 3) - 1)) * block_size;
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

    DT_S32 width_align4 = src_size.m_width & (-4);
    for (DT_S32 y = 0; y < cov_size.m_height; y++)
    {
        DT_F32 *cov_data = cov.Ptr<DT_F32>(y);
        const DT_F32 *dx_data  = dx.Ptr<DT_F32>(y);
        const DT_F32 *dy_data  = dy.Ptr<DT_F32>(y);

        DT_S32 x = 0;
        for (; x < width_align4; x += 4)
        {
            float32x4_t vqf32_dx = neon::vload1q(dx_data);
            float32x4_t vqf32_dy = neon::vload1q(dy_data);

            float32x4x3_t v3qf32_cov;
            v3qf32_cov.val[0] = neon::vmul(vqf32_dx, vqf32_dx);
            v3qf32_cov.val[1] = neon::vmul(vqf32_dx, vqf32_dy);
            v3qf32_cov.val[2] = neon::vmul(vqf32_dy, vqf32_dy);

            neon::vstore(cov_data, v3qf32_cov);
            dx_data  += 4;
            dy_data  += 4;
            cov_data += 12;
        }
        for (; x < cov_size.m_width; x++)
        {
            DT_F32 dx = *dx_data++;
            DT_F32 dy = *dy_data++;

            *cov_data++ = dx * dx;
            *cov_data++ = dx * dy;
            *cov_data++ = dy * dy;
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

    DT_F32 kxk = block_size * block_size;
    float32x4_t vqf32_const_kxk;
    neon::vdup(vqf32_const_kxk, kxk);

    DT_S32 widthx3 = cov_size.m_width * 3;
    DT_S32 widthx3_align4 = widthx3 & (-4);
    for (DT_S32 y = 0; y < cov_size.m_height; y++)
    {
        DT_F32 *dst_data = cov_dst.Ptr<DT_F32>(y);
        DT_S32 x = 0;
        for (; x < widthx3_align4; x += 4)
        {
            float32x4_t vqf32_data = neon::vload1q(dst_data);
            vqf32_data = neon::vmul(vqf32_data, vqf32_const_kxk);
            neon::vstore(dst_data, vqf32_data);
            dst_data += 4;
        }
        for (; x < widthx3; x++)
        {
            *dst_data++ *= kxk;
        }
    }

    if (use_harris)
    {
        CalcHarrisNeon(ctx, cov_dst, dst, k);
    }
    else
    {
        CalcMinEigenValNeon(ctx, cov_dst, dst);
    }

    return Status::OK;
}

HarrisNeon::HarrisNeon(Context *ctx, const OpTarget &target) : HarrisImpl(ctx, target)
{}

Status HarrisNeon::SetArgs(const Array *src, Array *dst, DT_S32 block_size, DT_S32 ksize, DT_F64 k,
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

Status HarrisNeon::Run()
{
    const Mat *src = dynamic_cast<const Mat*>(m_src);
    Mat *dst = dynamic_cast<Mat*>(m_dst);
    if ((DT_NULL == src) || (DT_NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src dst is null");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    switch (src->GetElemType())
    {
        case ElemType::U8:
        case ElemType::F32:
        {
            ret = CornerEigenValsVecsNeon(m_ctx, *src, *dst, m_block_size, m_ksize, DT_TRUE,
                                          m_k, m_border_type, m_border_value, m_target);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "CornerEigenValsVecsNeon failed, ElemType: U8 or F32");
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