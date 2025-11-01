#include "gemm_impl.hpp"
#include "aura/ops/core.h"
#include "aura/runtime/mat.h"
#include "aura/runtime/logger.h"

namespace aura
{

AURA_ALWAYS_INLINE AURA_VOID AddDot4x4(MI_S32 k, MI_F32 *a, MI_S32 lda, MI_F32 *b, MI_S32 ldb,
                                     MI_F32 *c, MI_S32 ldc)
{
    MI_S32 p = 0;
    /** hold contributions to
        C( 0, 0 ), C( 0, 1 ), C( 0, 2 ), C( 0, 3 )
        C( 1, 0 ), C( 1, 1 ), C( 1, 2 ), C( 1, 3 )
        C( 2, 0 ), C( 2, 1 ), C( 2, 2 ), C( 2, 3 )
        C( 3, 0 ), C( 3, 1 ), C( 3, 2 ), C( 3, 3 )  
    */
    MI_F32 c00_reg = 0.0f, c01_reg = 0.0f, c02_reg = 0.0f, c03_reg = 0.0f;
    MI_F32 c10_reg = 0.0f, c11_reg = 0.0f, c12_reg = 0.0f, c13_reg = 0.0f;
    MI_F32 c20_reg = 0.0f, c21_reg = 0.0f, c22_reg = 0.0f, c23_reg = 0.0f;
    MI_F32 c30_reg = 0.0f, c31_reg = 0.0f, c32_reg = 0.0f, c33_reg = 0.0f;
    /** hold
        A( 0, p )
        A( 1, p )
        A( 2, p )
        A( 3, p )
    */
    MI_F32 a0r_reg = 0.0f;
    MI_F32 a1r_reg = 0.0f;
    MI_F32 a2r_reg = 0.0f;
    MI_F32 a3r_reg = 0.0f;
    MI_F32 b0c_reg = 0.0f;
    MI_F32 b1c_reg = 0.0f;
    MI_F32 b2c_reg = 0.0f;
    MI_F32 b3c_reg = 0.0f;

    MI_F32 *b0_pntr = &b[0];
    MI_F32 *b1_pntr = &b[1 * ldb];
    MI_F32 *b2_pntr = &b[2 * ldb];
    MI_F32 *b3_pntr = &b[3 * ldb];

    for (p = 0; p < k; p++)
    {
        a0r_reg = a[0 * lda + p];
        a1r_reg = a[1 * lda + p];
        a2r_reg = a[2 * lda + p];
        a3r_reg = a[3 * lda + p];

        b0c_reg = *b0_pntr++;
        b1c_reg = *b1_pntr++;
        b2c_reg = *b2_pntr++;
        b3c_reg = *b3_pntr++;

        c00_reg += a0r_reg * b0c_reg;
        c10_reg += a1r_reg * b0c_reg;

        c01_reg += a0r_reg * b1c_reg;
        c11_reg += a1r_reg * b1c_reg;

        c02_reg += a0r_reg * b2c_reg;
        c12_reg += a1r_reg * b2c_reg;

        c03_reg += a0r_reg * b3c_reg;
        c13_reg += a1r_reg * b3c_reg;

        c20_reg += a2r_reg * b0c_reg;
        c30_reg += a3r_reg * b0c_reg;

        c21_reg += a2r_reg * b1c_reg;
        c31_reg += a3r_reg * b1c_reg;

        c22_reg += a2r_reg * b2c_reg;
        c32_reg += a3r_reg * b2c_reg;

        c23_reg += a2r_reg * b3c_reg;
        c33_reg += a3r_reg * b3c_reg;
    }

    c[0 * ldc + 0] = c00_reg;   c[0 * ldc + 1] = c01_reg;   c[0 * ldc + 2] = c02_reg;   c[0 * ldc + 3] = c03_reg;
    c[1 * ldc + 0] = c10_reg;   c[1 * ldc + 1] = c11_reg;   c[1 * ldc + 2] = c12_reg;   c[1 * ldc + 3] = c13_reg;
    c[2 * ldc + 0] = c20_reg;   c[2 * ldc + 1] = c21_reg;   c[2 * ldc + 2] = c22_reg;   c[2 * ldc + 3] = c23_reg;
    c[3 * ldc + 0] = c30_reg;   c[3 * ldc + 1] = c31_reg;   c[3 * ldc + 2] = c32_reg;   c[3 * ldc + 3] = c33_reg;
}

AURA_ALWAYS_INLINE AURA_VOID AddDot1x4(MI_S32 k, MI_F32 *a, MI_S32 lda, MI_F32 *b, MI_S32 ldb,
                                     MI_F32 *c, MI_S32 ldc)
{
    AURA_UNUSED(lda);
    AURA_UNUSED(ldc);

    MI_S32 p = 0;
    /** hold contributions to
        C( 0, 0 ), C( 0, 1 ), C( 0, 2 ), C( 0, 3 )
    */
    MI_F32 c00_reg = 0.0f;
    MI_F32 c01_reg = 0.0f;
    MI_F32 c02_reg = 0.0f;
    MI_F32 c03_reg = 0.0f;

    // hold A( 0, p )
    MI_F32 a0r_reg = 0.0f;
    MI_F32 b0c_reg = 0.0f;
    MI_F32 b1c_reg = 0.0f;
    MI_F32 b2c_reg = 0.0f;
    MI_F32 b3c_reg = 0.0f;

    MI_F32 *b0_pntr = &b[0];
    MI_F32 *b1_pntr = &b[1 * ldb];
    MI_F32 *b2_pntr = &b[2 * ldb];
    MI_F32 *b3_pntr = &b[3 * ldb];

    for (p = 0; p < k; p++)
    {
        a0r_reg = a[p];

        b0c_reg = *b0_pntr++;
        b1c_reg = *b1_pntr++;
        b2c_reg = *b2_pntr++;
        b3c_reg = *b3_pntr++;

        c00_reg += a0r_reg * b0c_reg;
        c01_reg += a0r_reg * b1c_reg;
        c02_reg += a0r_reg * b2c_reg;
        c03_reg += a0r_reg * b3c_reg;
    }

    c[0] = c00_reg;
    c[1] = c01_reg;
    c[2] = c02_reg;
    c[3] = c03_reg;
}

AURA_ALWAYS_INLINE AURA_VOID AddDot4x1(MI_S32 k, MI_F32 *a, MI_S32 lda, MI_F32 *b, MI_S32 ldb,
                                     MI_F32 *c, MI_S32 ldc)
{
    AURA_UNUSED(ldb);

    MI_S32 p = 0;
    /** hold contributions to
        C( 0, 0 ), C( 1, 0 ), C( 2, 0 ), C( 3, 0 ) 
    */
    MI_F32 c00_reg = 0.0f;
    MI_F32 c10_reg = 0.0f;
    MI_F32 c20_reg = 0.0f;
    MI_F32 c30_reg = 0.0f;

    /** hold
        A( 0, p )
        A( 1, p )
        A( 2, p )
        A( 3, p )
    */
    MI_F32 a0r_reg = 0.0f;
    MI_F32 a1r_reg = 0.0f;
    MI_F32 a2r_reg = 0.0f;
    MI_F32 a3r_reg = 0.0f;
    MI_F32 b0c_reg = 0.0f;

    MI_F32 *b0_pntr = b;

    for (p = 0; p < k; p++)
    {
        a0r_reg = a[0 * lda + p];
        a1r_reg = a[1 * lda + p];
        a2r_reg = a[2 * lda + p];
        a3r_reg = a[3 * lda + p];

        b0c_reg = *b0_pntr++;

        c00_reg += a0r_reg * b0c_reg;
        c10_reg += a1r_reg * b0c_reg;
        c20_reg += a2r_reg * b0c_reg;
        c30_reg += a3r_reg * b0c_reg;
    }

    c[0 * ldc] = c00_reg;
    c[1 * ldc] = c10_reg;
    c[2 * ldc] = c20_reg;
    c[3 * ldc] = c30_reg;
}

AURA_ALWAYS_INLINE AURA_VOID AddDot1x1(MI_S32 k, MI_F32 *a, MI_S32 lda, MI_F32 *b, MI_S32 ldb,
                                     MI_F32 *c, MI_S32 ldc)
{
    AURA_UNUSED(lda);
    AURA_UNUSED(ldb);
    AURA_UNUSED(ldc);

    MI_S32 p = 0;
    MI_F32 a_reg = 0.0f;
    MI_F32 b_reg = 0.0f;
    MI_F32 c_reg = 0.0f;
    MI_F32 *b_pntr = b;

    for (p = 0; p < k; p++)
    {
        a_reg = a[p];
        b_reg = *b_pntr++;
        c_reg += a_reg * b_reg;
    }

    c[0] = c_reg;
}

Status GemmNoneImpl(Context *ctx, const Mat &src0, const Mat &src1, Mat &dst, const OpTarget &target)
{
    AURA_UNUSED(ctx);
    AURA_UNUSED(target);

    const MI_S32 m = src0.GetSizes().m_height;
    const MI_S32 n = src1.GetSizes().m_width;
    const MI_S32 k = src0.GetSizes().m_width;

    const MI_S32 stride_src0 = src0.GetRowPitch() / sizeof(MI_F32);
    const MI_S32 stride_src1 = src1.GetRowPitch() / sizeof(MI_F32);
    const MI_S32 stride_dst  = dst.GetRowPitch() / sizeof(MI_F32);
    MI_F32 *b_data = (MI_F32 *)src1.GetData();

    MI_F32 *tmp_b = static_cast<MI_F32*>(AURA_ALLOC(ctx, k * 4 * sizeof(MI_F32)));
    if (MI_NULL == tmp_b)
    {
        AURA_ADD_ERROR_STRING(ctx, "AURA_ALLOC_PARAM fail");
        return Status::ERROR;
    }

    MI_S32 i, l, j = 0;
    for (; j < n - 3; j += 4)
    {
        for (i = 0; i < k; i++)
        {
            for (l = 0; l < 4; l++)
            {
                tmp_b[l * k + i] = *(b_data + i * stride_src1 + j + l);
            }
        }

        i = 0;
        for (; i < m - 3; i += 4)
        {
            AddDot4x4(k, const_cast<MI_F32*>(src0.Ptr<MI_F32>(i)), stride_src0, tmp_b,
                      k, &dst.At<MI_F32>(i, j), stride_dst);
        }
        for (; i < m; i++)
        {
            AddDot1x4(k, const_cast<MI_F32*>(src0.Ptr<MI_F32>(i)), stride_src0, tmp_b,
                      k, &dst.At<MI_F32>(i, j), stride_dst);
        }
    }

    for (; j < n; j++)
    {
        for (i = 0; i < k; i++)
        {
            tmp_b[i] = *(b_data + i * stride_src1 + j);
        }

        i = 0;
        for (; i < m - 3; i += 4)
        {
            AddDot4x1(k, const_cast<MI_F32*>(src0.Ptr<MI_F32>(i)), stride_src0, tmp_b,
                      k, &dst.At<MI_F32>(i, j), stride_dst);
        }
        for (; i < m; i++)
        {
            AddDot1x1(k, const_cast<MI_F32*>(src0.Ptr<MI_F32>(i)), stride_src0, tmp_b,
                      k, &dst.At<MI_F32>(i, j), stride_dst);
        }
    }

    if (tmp_b != MI_NULL)
    {
        AURA_FREE(ctx, tmp_b);
    }

    return Status::OK;
}

GemmNone::GemmNone(Context *ctx, const OpTarget &target) : GemmImpl(ctx, target)
{}

Status GemmNone::SetArgs(const Array *src0, const Array *src1, Array *dst)
{
    if (GemmImpl::SetArgs(src0, src1, dst) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "GemmImpl::SetArgs failed");
        return Status::ERROR;
    }

    if ((src0->GetArrayType() != ArrayType::MAT) || (src1->GetArrayType() != ArrayType::MAT) || (dst->GetArrayType() != ArrayType::MAT))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src dst must be mat type");
        return Status::ERROR;
    }

    return Status::OK;
}

Status GemmNone::Run()
{
    const Mat *src0 = dynamic_cast<const Mat*>(m_src0);
    const Mat *src1 = dynamic_cast<const Mat*>(m_src1);
    Mat *dst        = dynamic_cast<Mat*>(m_dst);

    if ((NULL == src0) || (NULL == src1) || (NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src or dst is null");
        return Status::ERROR;
    }

    Status ret = GemmNoneImpl(m_ctx, *src0, *src1, *dst, m_target);
    AURA_RETURN(m_ctx, ret);
}

} // namespace aura
