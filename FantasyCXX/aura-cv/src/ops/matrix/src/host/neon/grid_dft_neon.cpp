#include "grid_dft_impl.hpp"
#include "dft_impl.hpp"
#include "aura/runtime/logger.h"
#include "aura/runtime/worker_pool.h"

namespace aura
{

#define COMPLEX_MUL(v2df32_x, v2df32_w, v2df32_wx)                                          \
{                                                                                           \
    float32x2_t vdf32_wrx1r = neon::vmul(v2df32_w.val[0], v2df32_x.val[0]);                 \
    float32x2_t vdf32_wrx1i = neon::vmul(v2df32_w.val[0], v2df32_x.val[1]);                 \
                                                                                            \
    v2df32_wx.val[0] = neon::vmls(vdf32_wrx1r, v2df32_w.val[1], v2df32_x.val[1]);           \
    v2df32_wx.val[1] = neon::vmla(vdf32_wrx1i, v2df32_w.val[1], v2df32_x.val[0]);           \
}

template <typename Tp>
static Status GridDft4x4NeonImpl(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target)
{
    AURA_UNUSED(target);
    const Sizes3 sz = src.GetSizes();
    const MI_S32 width = sz.m_width;
    const MI_S32 height = sz.m_height;

    WorkerPool *wp = ctx->GetWorkerPool();
    if (MI_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get WorkerPool Failed.");
        return Status::ERROR;
    }

    MI_F32 conj[2] = {1.0f, -1.0f};
    float32x2_t vdf32_conj = neon::vload1(conj);
    float32x2_t vdf32_zero;
    neon::vdup(vdf32_zero, 0.0f);

    auto row_process_func = [&](MI_S32 start, MI_S32 end)->Status
    {
        MI_F32 buffer_roi[16] = {0};
        for (MI_S32 y = start * 4; y < end * 4; y += 4)
        {
            const std::complex<MI_F32> *dst_row0 = dst.Ptr<std::complex<MI_F32>>(y);
            const std::complex<MI_F32> *dst_row1 = dst.Ptr<std::complex<MI_F32>>(y + 1);
            const std::complex<MI_F32> *dst_row2 = dst.Ptr<std::complex<MI_F32>>(y + 2);
            const std::complex<MI_F32> *dst_row3 = dst.Ptr<std::complex<MI_F32>>(y + 3);
            for (MI_S32 x = 0; x < width; x += 4)
            {
                MI_F32 *dst_shift0 = (MI_F32 *)(dst_row0 + x);
                MI_F32 *dst_shift1 = (MI_F32 *)(dst_row1 + x);
                MI_F32 *dst_shift2 = (MI_F32 *)(dst_row2 + x);
                MI_F32 *dst_shift3 = (MI_F32 *)(dst_row3 + x);

                for (MI_S32 y_shift = 0; y_shift < 4; y_shift++)
                {
                    const Tp *src_row = src.Ptr<Tp>(y + y_shift);
                    for (MI_S32 x_shift = 0; x_shift < 4; x_shift++)
                    {
                        buffer_roi[y_shift * 4 + x_shift] = SaturateCast<MI_F32>(src_row[x + x_shift]);
                    }
                }

                float32x4_t vqf32_src0 = neon::vload1q(buffer_roi);
                float32x4_t vqf32_src1 = neon::vload1q(buffer_roi + 4);
                float32x4_t vqf32_src2 = neon::vload1q(buffer_roi + 8);
                float32x4_t vqf32_src3 = neon::vload1q(buffer_roi + 12);

                float32x2_t vdf32_dst00, vdf32_dst01, vdf32_dst02, vdf32_dst03;
                float32x2_t vdf32_dst10, vdf32_dst11, vdf32_dst12, vdf32_dst13;
                float32x2_t vdf32_dst20, vdf32_dst21, vdf32_dst22, vdf32_dst23;
                float32x2_t vdf32_dst30, vdf32_dst31, vdf32_dst32, vdf32_dst33;

                float32x2_t vdf32_src0_lo = neon::vgetlow(vqf32_src0);
                float32x2_t vdf32_src0_hi = neon::vgethigh(vqf32_src0);
                float32x2_t vdf32_src1_lo = neon::vgetlow(vqf32_src1);
                float32x2_t vdf32_src1_hi = neon::vgethigh(vqf32_src1);
                float32x2_t vdf32_src2_lo = neon::vgetlow(vqf32_src2);
                float32x2_t vdf32_src2_hi = neon::vgethigh(vqf32_src2);
                float32x2_t vdf32_src3_lo = neon::vgetlow(vqf32_src3);
                float32x2_t vdf32_src3_hi = neon::vgethigh(vqf32_src3);

                float32x2_t vdf32_sumx0 = neon::vadd(vdf32_src0_lo, vdf32_src0_hi);
                float32x2_t vdf32_subx0 = neon::vsub(vdf32_src0_lo, vdf32_src0_hi);
                float32x2_t vdf32_sumx1 = neon::vadd(vdf32_src1_lo, vdf32_src1_hi);
                float32x2_t vdf32_subx1 = neon::vsub(vdf32_src1_lo, vdf32_src1_hi);
                float32x2_t vdf32_sumx2 = neon::vadd(vdf32_src2_lo, vdf32_src2_hi);
                float32x2_t vdf32_subx2 = neon::vsub(vdf32_src2_lo, vdf32_src2_hi);
                float32x2_t vdf32_sumx3 = neon::vadd(vdf32_src3_lo, vdf32_src3_hi);
                float32x2_t vdf32_subx3 = neon::vsub(vdf32_src3_lo, vdf32_src3_hi);

                float32x2x2_t v2df32_zip;
                {
                    float32x2_t vdf32_tmp0, vdf32_tmp1, vdf32_sum, vdf32_sub;
                    vdf32_tmp0 = neon::vadd(vdf32_sumx0, vdf32_sumx2);
                    vdf32_tmp1 = neon::vadd(vdf32_sumx1, vdf32_sumx3);
                    v2df32_zip = neon::vzip(vdf32_tmp0, vdf32_tmp1);
                    vdf32_tmp0 = v2df32_zip.val[0];
                    vdf32_tmp1 = v2df32_zip.val[1];
                    vdf32_sum  = neon::vadd(vdf32_tmp0, vdf32_tmp1);
                    vdf32_sub  = neon::vsub(vdf32_tmp0, vdf32_tmp1);
                    v2df32_zip = neon::vzip(vdf32_sum, vdf32_sub);
                    vdf32_tmp0 = neon::vadd(v2df32_zip.val[0], v2df32_zip.val[1]);
                    vdf32_tmp1 = neon::vsub(v2df32_zip.val[0], v2df32_zip.val[1]);

                    v2df32_zip  = neon::vzip(vdf32_tmp0, vdf32_zero);
                    vdf32_dst00 = v2df32_zip.val[0];
                    vdf32_dst02 = v2df32_zip.val[1];
                    v2df32_zip  = neon::vzip(vdf32_tmp1, vdf32_zero);
                    vdf32_dst20 = v2df32_zip.val[0];
                    vdf32_dst22 = v2df32_zip.val[1];
                }
                {
                    float32x2_t vdf32_tmp0 = neon::vadd(vdf32_subx0, vdf32_subx2);
                    float32x2_t vdf32_tmp1 = neon::vadd(vdf32_subx1, vdf32_subx3);
                    vdf32_dst03 = neon::vadd(vdf32_tmp0, vdf32_tmp1);
                    vdf32_dst01 = neon::vmul(vdf32_dst03, vdf32_conj);
                    vdf32_dst23 = neon::vsub(vdf32_tmp0, vdf32_tmp1);
                    vdf32_dst21 = neon::vmul(vdf32_dst23, vdf32_conj);
                }
                {
                    float32x2_t vdf32_tmp0 = neon::vsub(vdf32_sumx0, vdf32_sumx2);
                    float32x2_t vdf32_tmp1 = neon::vsub(vdf32_sumx1, vdf32_sumx3);
                    v2df32_zip  = neon::vzip(vdf32_tmp0, vdf32_tmp1);
                    vdf32_tmp0  = v2df32_zip.val[0];
                    vdf32_tmp1  = v2df32_zip.val[1];
                    vdf32_dst30 = neon::vadd(vdf32_tmp0, vdf32_tmp1);
                    vdf32_dst32 = neon::vsub(vdf32_tmp0, vdf32_tmp1);
                    vdf32_dst10 = neon::vmul(vdf32_dst30, vdf32_conj);
                    vdf32_dst12 = neon::vmul(vdf32_dst32, vdf32_conj);
                }
                {
                    float32x2_t vdf32_tmp0  = neon::vsub(vdf32_subx0, vdf32_subx2);
                    float32x2_t vdf32_tmp1  = neon::vsub(vdf32_subx1, vdf32_subx3);
                    float32x2_t vdf32_tmp3  = neon::vrev64(vdf32_tmp1);
                    float32x2_t vdf32_add03 = neon::vadd(vdf32_tmp0, vdf32_tmp3);
                    float32x2_t vdf32_sub03 = neon::vsub(vdf32_tmp0, vdf32_tmp3);
                    v2df32_zip  = neon::vzip(vdf32_add03, neon::vrev64(vdf32_sub03));
                    vdf32_dst13 = v2df32_zip.val[0];
                    vdf32_dst33 = neon::vrev64(v2df32_zip.val[1]);
                    vdf32_dst31 = neon::vmul(vdf32_dst13, vdf32_conj);
                    vdf32_dst11 = neon::vmul(vdf32_dst33, vdf32_conj);
                }

                neon::vstore(dst_shift0,     vdf32_dst00); neon::vstore(dst_shift0 + 2, vdf32_dst01);
                neon::vstore(dst_shift0 + 4, vdf32_dst02); neon::vstore(dst_shift0 + 6, vdf32_dst03);
                neon::vstore(dst_shift1,     vdf32_dst10); neon::vstore(dst_shift1 + 2, vdf32_dst11);
                neon::vstore(dst_shift1 + 4, vdf32_dst12); neon::vstore(dst_shift1 + 6, vdf32_dst13);
                neon::vstore(dst_shift2,     vdf32_dst20); neon::vstore(dst_shift2 + 2, vdf32_dst21);
                neon::vstore(dst_shift2 + 4, vdf32_dst22); neon::vstore(dst_shift2 + 6, vdf32_dst23);
                neon::vstore(dst_shift3,     vdf32_dst30); neon::vstore(dst_shift3 + 2, vdf32_dst31);
                neon::vstore(dst_shift3 + 4, vdf32_dst32); neon::vstore(dst_shift3 + 6, vdf32_dst33);
            }
        }
        return Status::OK;
    };

    if (wp->ParallelFor(0, height / 4, row_process_func) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "GridDft4x4NeonImpl ParallelFor failed.");
        return Status::ERROR;
    }

    return Status::OK;
}

template <typename T>
static AURA_VOID RowDftNeon1x8(T *src, std::complex<MI_F32> *dst, std::complex<MI_F32> *row_real_exp_table)
{
    MI_F32 conj[2] = {1.0f, -1.0f};
    float32x2_t vdf32_conj = neon::vload1(conj);

    std::complex<MI_F32> src0, src1, src2, src3;
    src0.real((MI_F32)src[0]); src0.imag((MI_F32)src[1]);
    src1.real((MI_F32)src[4]); src1.imag((MI_F32)src[5]);
    src2.real((MI_F32)src[2]); src2.imag((MI_F32)src[3]);
    src3.real((MI_F32)src[6]); src3.imag((MI_F32)src[7]);

    float32x2_t vdf32_src0 = neon::vload1((MI_F32 *)(&src0));
    float32x2_t vdf32_src1 = neon::vload1((MI_F32 *)(&src1));
    float32x2_t vdf32_src2 = neon::vload1((MI_F32 *)(&src2));
    float32x2_t vdf32_src3 = neon::vload1((MI_F32 *)(&src3));

    std::complex<MI_F32>fk_scale(0.5f, 0.0f);
    std::complex<MI_F32>gk_scale(0.0f, 0.5f);

    float32x2_t vdf32_fk_scale = neon::vload1((MI_F32 *)(&fk_scale));
    float32x2_t vdf32_gk_scale = neon::vload1((MI_F32 *)(&gk_scale));
    float32x2_t vdf32_exp1     = neon::vload1((MI_F32 *)(row_real_exp_table + 1));
    float32x2_t vdf32_exp3     = neon::vload1((MI_F32 *)(row_real_exp_table + 3));

    float32x2x2_t v2df32_scale0 = neon::vzip(vdf32_fk_scale, vdf32_fk_scale);
    float32x2x2_t v2df32_scale1 = neon::vzip(vdf32_gk_scale, vdf32_gk_scale);
    float32x2x2_t v2df32_scale2 = neon::vzip(vdf32_exp1, vdf32_exp3);

    float32x2_t vdf32_temp;
    vdf32_temp = vdf32_src1;
    vdf32_src1 = neon::vsub(vdf32_src0, vdf32_temp);
    vdf32_src0 = neon::vadd(vdf32_src0, vdf32_temp);
    vdf32_temp = vdf32_src3;
    vdf32_src3 = neon::vsub(vdf32_src2, vdf32_temp);
    vdf32_src2 = neon::vadd(vdf32_src2, vdf32_temp);

    vdf32_temp = vdf32_src2;
    vdf32_src2 = neon::vsub(vdf32_src0, vdf32_temp);
    vdf32_src0 = neon::vadd(vdf32_src0, vdf32_temp);
    vdf32_temp = neon::vmul(neon::vrev64(vdf32_src3), vdf32_conj);
    vdf32_src3 = neon::vsub(vdf32_src1, vdf32_temp);
    vdf32_src1 = neon::vadd(vdf32_src1, vdf32_temp);

    float32x2_t vdf32_src_conj  = neon::vmul(vdf32_src3,     vdf32_conj);
    float32x2_t vdf32_src13_add = neon::vadd(vdf32_src1,     vdf32_src_conj);
    float32x2_t vdf32_src13_sub = neon::vsub(vdf32_src_conj, vdf32_src1);

    vdf32_src_conj = neon::vmul(vdf32_src1, vdf32_conj);
    float32x2_t vdf32_src31_add  = neon::vadd(vdf32_src3,      vdf32_src_conj);
    float32x2_t vdf32_src31_sub  = neon::vsub(vdf32_src_conj,  vdf32_src3);
    float32x2x2_t v2df32_add_zip = neon::vzip(vdf32_src13_add, vdf32_src31_add);

    float32x2x2_t v2df32_fk_zip, v2df32_sub_zip, v2df32_gk_zip, v2df32_result_zip;
    COMPLEX_MUL(v2df32_scale0, v2df32_add_zip, v2df32_fk_zip);
    v2df32_sub_zip = neon::vzip(vdf32_src13_sub, vdf32_src31_sub);
    COMPLEX_MUL(v2df32_scale1, v2df32_sub_zip, v2df32_gk_zip);
    COMPLEX_MUL(v2df32_scale2, v2df32_gk_zip,  v2df32_result_zip);

    v2df32_result_zip.val[0] = neon::vadd(v2df32_fk_zip.val[0], v2df32_result_zip.val[0]);
    v2df32_result_zip.val[1] = neon::vadd(v2df32_fk_zip.val[1], v2df32_result_zip.val[1]);
    float32x2x2_t v2df32_uzip = neon::vuzp(v2df32_result_zip.val[0], v2df32_result_zip.val[1]);

    float32x2_t vdf32_dst1 = v2df32_uzip.val[0];
    float32x2_t vdf32_dst3 = v2df32_uzip.val[1];
    float32x2_t vdf32_dst7 = neon::vmul(vdf32_dst1, vdf32_conj);
    float32x2_t vdf32_dst5 = neon::vmul(vdf32_dst3, vdf32_conj);

    vdf32_src_conj  = neon::vmul(vdf32_src2,     vdf32_conj);
    vdf32_src13_add = neon::vadd(vdf32_src2,     vdf32_src_conj);
    vdf32_src13_sub = neon::vsub(vdf32_src_conj, vdf32_src2);

    vdf32_src_conj  = neon::vmul(vdf32_src0,      vdf32_conj);
    vdf32_src31_add = neon::vadd(vdf32_src0,      vdf32_src_conj);
    vdf32_src31_sub = neon::vsub(vdf32_src_conj,  vdf32_src0);
    v2df32_add_zip  = neon::vzip(vdf32_src13_add, vdf32_src31_add);

    COMPLEX_MUL(v2df32_scale0, v2df32_add_zip, v2df32_fk_zip);
    v2df32_sub_zip = neon::vzip(vdf32_src13_sub, vdf32_src31_sub);
    COMPLEX_MUL(v2df32_scale1, v2df32_sub_zip, v2df32_gk_zip);
    v2df32_uzip = neon::vuzp(v2df32_fk_zip.val[0], v2df32_fk_zip.val[1]);

    float32x2_t vdf32_fk = v2df32_uzip.val[0];
    float32x2_t vdf32_f0 = v2df32_uzip.val[1];
    v2df32_uzip = neon::vuzp(v2df32_gk_zip.val[0], v2df32_gk_zip.val[1]);
    float32x2_t vdf32_gk = v2df32_uzip.val[0];
    float32x2_t vdf32_g0 = v2df32_uzip.val[1];

    float32x2_t vdf32_dst0 = neon::vadd(vdf32_f0, vdf32_g0);
    float32x2_t vdf32_dst4 = neon::vsub(vdf32_f0, vdf32_g0);
    float32x2_t vdf32_dst2 = neon::vadd(vdf32_fk, neon::vmul(neon::vrev64(vdf32_gk), vdf32_conj));
    float32x2_t vdf32_dst6 = neon::vmul(vdf32_dst2, vdf32_conj);

    MI_F32 *dst_ptr = (MI_F32 *)dst;
    neon::vstore(dst_ptr,      vdf32_dst0); neon::vstore(dst_ptr + 2,  vdf32_dst1);
    neon::vstore(dst_ptr + 4,  vdf32_dst2); neon::vstore(dst_ptr + 6,  vdf32_dst3);
    neon::vstore(dst_ptr + 8,  vdf32_dst4); neon::vstore(dst_ptr + 10, vdf32_dst5);
    neon::vstore(dst_ptr + 12, vdf32_dst6); neon::vstore(dst_ptr + 14, vdf32_dst7);
    // clear
    dst[0].imag(0.0f);
    dst[4].imag(0.0f);
}

static AURA_VOID ColButterflyCalc(std::complex<MI_F32> src[][8], MI_BOOL is_inverse)
{
    MI_F32 conj0[4] = { 1.0f, -1.0f,  1.0f, -1.0f};
    MI_F32 conj1[4] = {-1.0f,  1.0f, -1.0f,  1.0f};

    float32x4_t vqf32_conj;
    vqf32_conj = is_inverse ? neon::vload1q(conj1) : neon::vload1q(conj0);

    MI_F32 *line0 = (MI_F32 *)(&src[0][0]);
    MI_F32 *line1 = (MI_F32 *)(&src[1][0]);
    MI_F32 *line2 = (MI_F32 *)(&src[2][0]);
    MI_F32 *line3 = (MI_F32 *)(&src[3][0]);
    MI_F32 *line4 = (MI_F32 *)(&src[4][0]);
    MI_F32 *line5 = (MI_F32 *)(&src[5][0]);
    MI_F32 *line6 = (MI_F32 *)(&src[6][0]);
    MI_F32 *line7 = (MI_F32 *)(&src[7][0]);

    float32x4_t vqf32_src00 = neon::vload1q(line0);     float32x4_t vqf32_src01 = neon::vload1q(line0 + 4);
    float32x4_t vqf32_src02 = neon::vload1q(line0 + 8); float32x4_t vqf32_src03 = neon::vload1q(line0 + 12);
    float32x4_t vqf32_src10 = neon::vload1q(line1);     float32x4_t vqf32_src11 = neon::vload1q(line1 + 4);
    float32x4_t vqf32_src12 = neon::vload1q(line1 + 8); float32x4_t vqf32_src13 = neon::vload1q(line1 + 12);

    float32x4_t vqf32_src20 = neon::vload1q(line2);     float32x4_t vqf32_src21 = neon::vload1q(line2 + 4);
    float32x4_t vqf32_src22 = neon::vload1q(line2 + 8); float32x4_t vqf32_src23 = neon::vload1q(line2 + 12);
    float32x4_t vqf32_src30 = neon::vload1q(line3);     float32x4_t vqf32_src31 = neon::vload1q(line3 + 4);
    float32x4_t vqf32_src32 = neon::vload1q(line3 + 8); float32x4_t vqf32_src33 = neon::vload1q(line3 + 12);

    float32x4_t vqf32_temp0, vqf32_temp1, vqf32_temp2, vqf32_temp3;
    vqf32_temp0 = vqf32_src10, vqf32_temp1 = vqf32_src11;
    vqf32_temp2 = vqf32_src12, vqf32_temp3 = vqf32_src13;

    vqf32_src10 = neon::vsub(vqf32_src00, vqf32_temp0);
    vqf32_src11 = neon::vsub(vqf32_src01, vqf32_temp1);
    vqf32_src12 = neon::vsub(vqf32_src02, vqf32_temp2);
    vqf32_src13 = neon::vsub(vqf32_src03, vqf32_temp3);
    vqf32_src00 = neon::vadd(vqf32_src00, vqf32_temp0);
    vqf32_src01 = neon::vadd(vqf32_src01, vqf32_temp1);
    vqf32_src02 = neon::vadd(vqf32_src02, vqf32_temp2);
    vqf32_src03 = neon::vadd(vqf32_src03, vqf32_temp3);

    vqf32_temp0 = vqf32_src30, vqf32_temp1 = vqf32_src31;
    vqf32_temp2 = vqf32_src32, vqf32_temp3 = vqf32_src33;

    vqf32_src30 = neon::vsub(vqf32_src20, vqf32_temp0);
    vqf32_src31 = neon::vsub(vqf32_src21, vqf32_temp1);
    vqf32_src32 = neon::vsub(vqf32_src22, vqf32_temp2);
    vqf32_src33 = neon::vsub(vqf32_src23, vqf32_temp3);
    vqf32_src20 = neon::vadd(vqf32_src20, vqf32_temp0);
    vqf32_src21 = neon::vadd(vqf32_src21, vqf32_temp1);
    vqf32_src22 = neon::vadd(vqf32_src22, vqf32_temp2);
    vqf32_src23 = neon::vadd(vqf32_src23, vqf32_temp3);

    //layer4
    vqf32_temp0 = vqf32_src20, vqf32_temp1 = vqf32_src21;
    vqf32_temp2 = vqf32_src22, vqf32_temp3 = vqf32_src23;

    vqf32_src20 = neon::vsub(vqf32_src00, vqf32_temp0);
    vqf32_src21 = neon::vsub(vqf32_src01, vqf32_temp1);
    vqf32_src22 = neon::vsub(vqf32_src02, vqf32_temp2);
    vqf32_src23 = neon::vsub(vqf32_src03, vqf32_temp3);
    vqf32_src00 = neon::vadd(vqf32_src00, vqf32_temp0);
    vqf32_src01 = neon::vadd(vqf32_src01, vqf32_temp1);
    vqf32_src02 = neon::vadd(vqf32_src02, vqf32_temp2);
    vqf32_src03 = neon::vadd(vqf32_src03, vqf32_temp3);

    vqf32_temp0 = neon::vmul(neon::vrev64(vqf32_src30), vqf32_conj);
    vqf32_temp1 = neon::vmul(neon::vrev64(vqf32_src31), vqf32_conj);
    vqf32_temp2 = neon::vmul(neon::vrev64(vqf32_src32), vqf32_conj);
    vqf32_temp3 = neon::vmul(neon::vrev64(vqf32_src33), vqf32_conj);

    vqf32_src30 = neon::vsub(vqf32_src10, vqf32_temp0);
    vqf32_src31 = neon::vsub(vqf32_src11, vqf32_temp1);
    vqf32_src32 = neon::vsub(vqf32_src12, vqf32_temp2);
    vqf32_src33 = neon::vsub(vqf32_src13, vqf32_temp3);
    vqf32_src10 = neon::vadd(vqf32_src10, vqf32_temp0);
    vqf32_src11 = neon::vadd(vqf32_src11, vqf32_temp1);
    vqf32_src12 = neon::vadd(vqf32_src12, vqf32_temp2);
    vqf32_src13 = neon::vadd(vqf32_src13, vqf32_temp3);

    neon::vstore(line0,     vqf32_src00); neon::vstore(line0 + 4,  vqf32_src01);
    neon::vstore(line0 + 8, vqf32_src02); neon::vstore(line0 + 12, vqf32_src03);
    neon::vstore(line1,     vqf32_src10); neon::vstore(line1 + 4,  vqf32_src11);
    neon::vstore(line1 + 8, vqf32_src12); neon::vstore(line1 + 12, vqf32_src13);
    neon::vstore(line2,     vqf32_src20); neon::vstore(line2 + 4,  vqf32_src21);
    neon::vstore(line2 + 8, vqf32_src22); neon::vstore(line2 + 12, vqf32_src23);
    neon::vstore(line3,     vqf32_src30); neon::vstore(line3 + 4,  vqf32_src31);
    neon::vstore(line3 + 8, vqf32_src32); neon::vstore(line3 + 12, vqf32_src33);

    vqf32_src00 = neon::vload1q(line4);     vqf32_src01 = neon::vload1q(line4 + 4);
    vqf32_src02 = neon::vload1q(line4 + 8); vqf32_src03 = neon::vload1q(line4 + 12);
    vqf32_src10 = neon::vload1q(line5);     vqf32_src11 = neon::vload1q(line5 + 4);
    vqf32_src12 = neon::vload1q(line5 + 8); vqf32_src13 = neon::vload1q(line5 + 12);
    vqf32_src20 = neon::vload1q(line6);     vqf32_src21 = neon::vload1q(line6 + 4);
    vqf32_src22 = neon::vload1q(line6 + 8); vqf32_src23 = neon::vload1q(line6 + 12);
    vqf32_src30 = neon::vload1q(line7);     vqf32_src31 = neon::vload1q(line7 + 4);
    vqf32_src32 = neon::vload1q(line7 + 8); vqf32_src33 = neon::vload1q(line7 + 12);

    vqf32_temp0 = vqf32_src10; vqf32_temp1 = vqf32_src11;
    vqf32_temp2 = vqf32_src12; vqf32_temp3 = vqf32_src13;

    vqf32_src10 = neon::vsub(vqf32_src00, vqf32_temp0);
    vqf32_src11 = neon::vsub(vqf32_src01, vqf32_temp1);
    vqf32_src12 = neon::vsub(vqf32_src02, vqf32_temp2);
    vqf32_src13 = neon::vsub(vqf32_src03, vqf32_temp3);
    vqf32_src00 = neon::vadd(vqf32_src00, vqf32_temp0);
    vqf32_src01 = neon::vadd(vqf32_src01, vqf32_temp1);
    vqf32_src02 = neon::vadd(vqf32_src02, vqf32_temp2);
    vqf32_src03 = neon::vadd(vqf32_src03, vqf32_temp3);

    vqf32_temp0 = vqf32_src30; vqf32_temp1 = vqf32_src31;
    vqf32_temp2 = vqf32_src32; vqf32_temp3 = vqf32_src33;

    vqf32_src30 = neon::vsub(vqf32_src20, vqf32_temp0);
    vqf32_src31 = neon::vsub(vqf32_src21, vqf32_temp1);
    vqf32_src32 = neon::vsub(vqf32_src22, vqf32_temp2);
    vqf32_src33 = neon::vsub(vqf32_src23, vqf32_temp3);
    vqf32_src20 = neon::vadd(vqf32_src20, vqf32_temp0);
    vqf32_src21 = neon::vadd(vqf32_src21, vqf32_temp1);
    vqf32_src22 = neon::vadd(vqf32_src22, vqf32_temp2);
    vqf32_src23 = neon::vadd(vqf32_src23, vqf32_temp3);

    vqf32_temp0 = vqf32_src20; vqf32_temp1 = vqf32_src21;
    vqf32_temp2 = vqf32_src22; vqf32_temp3 = vqf32_src23;

    vqf32_src20 = neon::vsub(vqf32_src00, vqf32_temp0);
    vqf32_src21 = neon::vsub(vqf32_src01, vqf32_temp1);
    vqf32_src22 = neon::vsub(vqf32_src02, vqf32_temp2);
    vqf32_src23 = neon::vsub(vqf32_src03, vqf32_temp3);
    vqf32_src00 = neon::vadd(vqf32_src00, vqf32_temp0);
    vqf32_src01 = neon::vadd(vqf32_src01, vqf32_temp1);
    vqf32_src02 = neon::vadd(vqf32_src02, vqf32_temp2);
    vqf32_src03 = neon::vadd(vqf32_src03, vqf32_temp3);

    vqf32_temp0 = neon::vmul(neon::vrev64(vqf32_src30), vqf32_conj);
    vqf32_temp1 = neon::vmul(neon::vrev64(vqf32_src31), vqf32_conj);
    vqf32_temp2 = neon::vmul(neon::vrev64(vqf32_src32), vqf32_conj);
    vqf32_temp3 = neon::vmul(neon::vrev64(vqf32_src33), vqf32_conj);

    vqf32_src30 = neon::vsub(vqf32_src10, vqf32_temp0);
    vqf32_src31 = neon::vsub(vqf32_src11, vqf32_temp1);
    vqf32_src32 = neon::vsub(vqf32_src12, vqf32_temp2);
    vqf32_src33 = neon::vsub(vqf32_src13, vqf32_temp3);
    vqf32_src10 = neon::vadd(vqf32_src10, vqf32_temp0);
    vqf32_src11 = neon::vadd(vqf32_src11, vqf32_temp1);
    vqf32_src12 = neon::vadd(vqf32_src12, vqf32_temp2);
    vqf32_src13 = neon::vadd(vqf32_src13, vqf32_temp3);

    neon::vstore(line4,     vqf32_src00); neon::vstore(line4 + 4,  vqf32_src01);
    neon::vstore(line4 + 8, vqf32_src02); neon::vstore(line4 + 12, vqf32_src03);
    neon::vstore(line5,     vqf32_src10); neon::vstore(line5 + 4,  vqf32_src11);
    neon::vstore(line5 + 8, vqf32_src12); neon::vstore(line5 + 12, vqf32_src13);
    neon::vstore(line6,     vqf32_src20); neon::vstore(line6 + 4,  vqf32_src21);
    neon::vstore(line6 + 8, vqf32_src22); neon::vstore(line6 + 12, vqf32_src23);
    neon::vstore(line7,     vqf32_src30); neon::vstore(line7 + 4,  vqf32_src31);
    neon::vstore(line7 + 8, vqf32_src32); neon::vstore(line7 + 12, vqf32_src33);
}

static AURA_VOID ColGridDft8x8(std::complex<MI_F32> src[][8], std::complex<MI_F32> *dst, MI_S32 ostep)
{
    // butter fly size is 2 and 4, and only update src
    ColButterflyCalc(src, MI_FALSE);

    // butterfly size is 8, and store result to dst
    MI_F32 exp1[8] = { 0.707107f, -0.707107f,  0.707107f, -0.707107f,  0.707107f, -0.707107f,  0.707107f, -0.707107f};
    MI_F32 exp3[8] = {-0.707107f, -0.707107f, -0.707107f, -0.707107f, -0.707107f, -0.707107f, -0.707107f, -0.707107f};

    float32x4_t vqf32_conj;
    neon::vdup(vqf32_conj, -1.0f);
    float32x4x2_t v2qf32_table1 = neon::vload2q(exp1);
    float32x4x2_t v2qf32_table3 = neon::vload2q(exp3);

    MI_F32 *line0 = (MI_F32 *)(&src[0][0]);
    MI_F32 *line1 = (MI_F32 *)(&src[1][0]);
    MI_F32 *line2 = (MI_F32 *)(&src[2][0]);
    MI_F32 *line3 = (MI_F32 *)(&src[3][0]);
    MI_F32 *line4 = (MI_F32 *)(&src[4][0]);
    MI_F32 *line5 = (MI_F32 *)(&src[5][0]);
    MI_F32 *line6 = (MI_F32 *)(&src[6][0]);
    MI_F32 *line7 = (MI_F32 *)(&src[7][0]);

    MI_F32 *dst0 = (MI_F32 *)(&dst[0]);
    MI_F32 *dst1 = (MI_F32 *)(&dst[1 * ostep]);
    MI_F32 *dst2 = (MI_F32 *)(&dst[2 * ostep]);
    MI_F32 *dst3 = (MI_F32 *)(&dst[3 * ostep]);
    MI_F32 *dst4 = (MI_F32 *)(&dst[4 * ostep]);
    MI_F32 *dst5 = (MI_F32 *)(&dst[5 * ostep]);
    MI_F32 *dst6 = (MI_F32 *)(&dst[6 * ostep]);
    MI_F32 *dst7 = (MI_F32 *)(&dst[7 * ostep]);

    float32x4x2_t v2qf32_temp0, v2qf32_temp1;
    float32x4x2_t v2qf32_dst00, v2qf32_dst01, v2qf32_dst40, v2qf32_dst41;
    float32x4x2_t v2qf32_dst20, v2qf32_dst21, v2qf32_dst60, v2qf32_dst61;

    float32x4x2_t v2qf32_src00 = neon::vload2q(line0);
    float32x4x2_t v2qf32_src01 = neon::vload2q(line0 + 8);
    float32x4x2_t v2qf32_src40 = neon::vload2q(line4);
    float32x4x2_t v2qf32_src41 = neon::vload2q(line4 + 8);
    float32x4x2_t v2qf32_src20 = neon::vload2q(line2);
    float32x4x2_t v2qf32_src21 = neon::vload2q(line2 + 8);
    float32x4x2_t v2qf32_src60 = neon::vload2q(line6);
    float32x4x2_t v2qf32_src61 = neon::vload2q(line6 + 8);

    v2qf32_temp0 = v2qf32_src40; v2qf32_temp1 = v2qf32_src41;
    v2qf32_dst40.val[0] = neon::vsub(v2qf32_src00.val[0], v2qf32_temp0.val[0]);
    v2qf32_dst40.val[1] = neon::vsub(v2qf32_src00.val[1], v2qf32_temp0.val[1]);
    v2qf32_dst41.val[0] = neon::vsub(v2qf32_src01.val[0], v2qf32_temp1.val[0]);
    v2qf32_dst41.val[1] = neon::vsub(v2qf32_src01.val[1], v2qf32_temp1.val[1]);
    v2qf32_dst00.val[0] = neon::vadd(v2qf32_src00.val[0], v2qf32_temp0.val[0]);
    v2qf32_dst00.val[1] = neon::vadd(v2qf32_src00.val[1], v2qf32_temp0.val[1]);
    v2qf32_dst01.val[0] = neon::vadd(v2qf32_src01.val[0], v2qf32_temp1.val[0]);
    v2qf32_dst01.val[1] = neon::vadd(v2qf32_src01.val[1], v2qf32_temp1.val[1]);

    v2qf32_temp0.val[0] = v2qf32_src60.val[1];
    v2qf32_temp0.val[1] = neon::vmul(v2qf32_src60.val[0], vqf32_conj);
    v2qf32_temp1.val[0] = v2qf32_src61.val[1];
    v2qf32_temp1.val[1] = neon::vmul(v2qf32_src61.val[0], vqf32_conj);
    v2qf32_dst60.val[0] = neon::vsub(v2qf32_src20.val[0], v2qf32_temp0.val[0]);
    v2qf32_dst60.val[1] = neon::vsub(v2qf32_src20.val[1], v2qf32_temp0.val[1]);
    v2qf32_dst61.val[0] = neon::vsub(v2qf32_src21.val[0], v2qf32_temp1.val[0]);
    v2qf32_dst61.val[1] = neon::vsub(v2qf32_src21.val[1], v2qf32_temp1.val[1]);
    v2qf32_dst20.val[0] = neon::vadd(v2qf32_src20.val[0], v2qf32_temp0.val[0]);
    v2qf32_dst20.val[1] = neon::vadd(v2qf32_src20.val[1], v2qf32_temp0.val[1]);
    v2qf32_dst21.val[0] = neon::vadd(v2qf32_src21.val[0], v2qf32_temp1.val[0]);
    v2qf32_dst21.val[1] = neon::vadd(v2qf32_src21.val[1], v2qf32_temp1.val[1]);

    neon::vstore(dst0, v2qf32_dst00); neon::vstore(dst0 + 8, v2qf32_dst01);
    neon::vstore(dst4, v2qf32_dst40); neon::vstore(dst4 + 8, v2qf32_dst41);
    neon::vstore(dst2, v2qf32_dst20); neon::vstore(dst2 + 8, v2qf32_dst21);
    neon::vstore(dst6, v2qf32_dst60); neon::vstore(dst6 + 8, v2qf32_dst61);

    v2qf32_src00 = neon::vload2q(line1); v2qf32_src01 = neon::vload2q(line1 + 8);
    v2qf32_src40 = neon::vload2q(line5); v2qf32_src41 = neon::vload2q(line5 + 8);
    v2qf32_src20 = neon::vload2q(line3); v2qf32_src21 = neon::vload2q(line3 + 8);
    v2qf32_src60 = neon::vload2q(line7); v2qf32_src61 = neon::vload2q(line7 + 8);

    v2qf32_temp0.val[0] = neon::vsub(neon::vmul(v2qf32_src40.val[0], v2qf32_table1.val[0]), neon::vmul(v2qf32_src40.val[1], v2qf32_table1.val[1]));
    v2qf32_temp0.val[1] = neon::vadd(neon::vmul(v2qf32_src40.val[1], v2qf32_table1.val[0]), neon::vmul(v2qf32_src40.val[0], v2qf32_table1.val[1]));
    v2qf32_temp1.val[0] = neon::vsub(neon::vmul(v2qf32_src41.val[0], v2qf32_table1.val[0]), neon::vmul(v2qf32_src41.val[1], v2qf32_table1.val[1]));
    v2qf32_temp1.val[1] = neon::vadd(neon::vmul(v2qf32_src41.val[1], v2qf32_table1.val[0]), neon::vmul(v2qf32_src41.val[0], v2qf32_table1.val[1]));

    v2qf32_dst40.val[0] = neon::vsub(v2qf32_src00.val[0], v2qf32_temp0.val[0]);
    v2qf32_dst40.val[1] = neon::vsub(v2qf32_src00.val[1], v2qf32_temp0.val[1]);
    v2qf32_dst41.val[0] = neon::vsub(v2qf32_src01.val[0], v2qf32_temp1.val[0]);
    v2qf32_dst41.val[1] = neon::vsub(v2qf32_src01.val[1], v2qf32_temp1.val[1]);
    v2qf32_dst00.val[0] = neon::vadd(v2qf32_src00.val[0], v2qf32_temp0.val[0]);
    v2qf32_dst00.val[1] = neon::vadd(v2qf32_src00.val[1], v2qf32_temp0.val[1]);
    v2qf32_dst01.val[0] = neon::vadd(v2qf32_src01.val[0], v2qf32_temp1.val[0]);
    v2qf32_dst01.val[1] = neon::vadd(v2qf32_src01.val[1], v2qf32_temp1.val[1]);

    v2qf32_temp0.val[0] = neon::vsub(neon::vmul(v2qf32_src60.val[0], v2qf32_table3.val[0]), neon::vmul(v2qf32_src60.val[1], v2qf32_table3.val[1]));
    v2qf32_temp0.val[1] = neon::vadd(neon::vmul(v2qf32_src60.val[1], v2qf32_table3.val[0]), neon::vmul(v2qf32_src60.val[0], v2qf32_table3.val[1]));
    v2qf32_temp1.val[0] = neon::vsub(neon::vmul(v2qf32_src61.val[0], v2qf32_table3.val[0]), neon::vmul(v2qf32_src61.val[1], v2qf32_table3.val[1]));
    v2qf32_temp1.val[1] = neon::vadd(neon::vmul(v2qf32_src61.val[1], v2qf32_table3.val[0]), neon::vmul(v2qf32_src61.val[0], v2qf32_table3.val[1]));

    v2qf32_dst60.val[0] = neon::vsub(v2qf32_src20.val[0], v2qf32_temp0.val[0]);
    v2qf32_dst60.val[1] = neon::vsub(v2qf32_src20.val[1], v2qf32_temp0.val[1]);
    v2qf32_dst61.val[0] = neon::vsub(v2qf32_src21.val[0], v2qf32_temp1.val[0]);
    v2qf32_dst61.val[1] = neon::vsub(v2qf32_src21.val[1], v2qf32_temp1.val[1]);
    v2qf32_dst20.val[0] = neon::vadd(v2qf32_src20.val[0], v2qf32_temp0.val[0]);
    v2qf32_dst20.val[1] = neon::vadd(v2qf32_src20.val[1], v2qf32_temp0.val[1]);
    v2qf32_dst21.val[0] = neon::vadd(v2qf32_src21.val[0], v2qf32_temp1.val[0]);
    v2qf32_dst21.val[1] = neon::vadd(v2qf32_src21.val[1], v2qf32_temp1.val[1]);

    neon::vstore(dst1, v2qf32_dst00); neon::vstore(dst1 + 8, v2qf32_dst01);
    neon::vstore(dst5, v2qf32_dst40); neon::vstore(dst5 + 8, v2qf32_dst41);
    neon::vstore(dst3, v2qf32_dst20); neon::vstore(dst3 + 8, v2qf32_dst21);
    neon::vstore(dst7, v2qf32_dst60); neon::vstore(dst7 + 8, v2qf32_dst61);
}

template <typename Tp>
static Status GridDft8x8NeonImpl(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target)
{
    AURA_UNUSED(target);
    const Sizes3 sz = src.GetSizes();
    const MI_S32 width = sz.m_width;
    const MI_S32 height = sz.m_height;
    const MI_S32 grid_len = 8;
    const MI_S32 ostep = dst.GetRowStep();

    std::complex<MI_F32> row_real_exp_table[4];
    GetDftExpTable<0>(row_real_exp_table, grid_len);

    WorkerPool *wp = ctx->GetWorkerPool();
    if (MI_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get WorkerPool Failed.");
        return Status::ERROR;
    }

    auto row_process_func = [&](MI_S32 start, MI_S32 end)->Status
    {
        std::complex<MI_F32> tmp_conv[8][8];
        for (MI_S32 y = start * 8; y < end * 8; y += 8)
        {
            const Tp *src_row0 = src.Ptr<Tp>(y);
            const Tp *src_row1 = src.Ptr<Tp>(y + 1);
            const Tp *src_row2 = src.Ptr<Tp>(y + 2);
            const Tp *src_row3 = src.Ptr<Tp>(y + 3);
            const Tp *src_row4 = src.Ptr<Tp>(y + 4);
            const Tp *src_row5 = src.Ptr<Tp>(y + 5);
            const Tp *src_row6 = src.Ptr<Tp>(y + 6);
            const Tp *src_row7 = src.Ptr<Tp>(y + 7);
            std::complex<MI_F32> *dst_ptr = dst.Ptr<std::complex<MI_F32>>(y);
            for (MI_S32 x = 0; x < width; x += 8)
            {
                RowDftNeon1x8(src_row0 + x, tmp_conv[0], row_real_exp_table);
                RowDftNeon1x8(src_row1 + x, tmp_conv[4], row_real_exp_table);
                RowDftNeon1x8(src_row2 + x, tmp_conv[2], row_real_exp_table);
                RowDftNeon1x8(src_row3 + x, tmp_conv[6], row_real_exp_table);
                RowDftNeon1x8(src_row4 + x, tmp_conv[1], row_real_exp_table);
                RowDftNeon1x8(src_row5 + x, tmp_conv[5], row_real_exp_table);
                RowDftNeon1x8(src_row6 + x, tmp_conv[3], row_real_exp_table);
                RowDftNeon1x8(src_row7 + x, tmp_conv[7], row_real_exp_table);

                ColGridDft8x8(tmp_conv, dst_ptr + x, ostep);
            }
        }
        return Status::OK;
    };

    if (wp->ParallelFor(0, height / 8, row_process_func) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "GridDft8x8NeonImpl ParallelFor failed.");
        return Status::ERROR;
    }
    // exit
    return Status::OK;
}

template <typename Tp, MI_S32 GRID_LEN>
static Status GridDftNeonNxN(const Mat &src, Mat &dst, MI_U16 *idx_x_table, MI_U16 *idx_y_table,
                      std::complex<MI_F32> *dft_row_exp_table, std::complex<MI_F32> *row_real_exp_table,
                      MI_S32 width_offset, MI_S32 height_offset)
{
    constexpr MI_S32 HALF_GRID = GRID_LEN / 2;
    const MI_S32 ostep = dst.GetRowStep();
    std::complex<MI_F32> *dst_ptr = dst.Ptr<std::complex<MI_F32>>(height_offset);
    std::complex<MI_F32> buffer[HALF_GRID];
    std::complex<MI_F32> dst_complex[GRID_LEN][GRID_LEN];

    for (MI_S32 y_grid = 0; y_grid < GRID_LEN; y_grid++)
    {
        const Tp *src_row = src.Ptr<Tp>(y_grid + height_offset);
        MI_S32 y_grid_index = idx_y_table[y_grid];

        for (MI_S32 x_grid = 0; x_grid < HALF_GRID; x_grid++)
        {
            MI_S32 idx = idx_x_table[x_grid];
            buffer[x_grid].real(SaturateCast<MI_F32>(src_row[width_offset + 2 * idx]));
            buffer[x_grid].imag(SaturateCast<MI_F32>(src_row[width_offset + 2 * idx + 1]));
        }

        ButterflyTransformNeon(buffer, 2, HALF_GRID, MI_FALSE, dft_row_exp_table);

        for (MI_S32 x_grid = 1; x_grid < HALF_GRID; x_grid++)
        {
            std::complex<MI_F32> yk = buffer[x_grid];
            std::complex<MI_F32> yk_conj = std::conj(buffer[HALF_GRID - x_grid]);

            std::complex<MI_F32> fk = std::complex<MI_F32>(0.5f, 0.0f) * (yk + yk_conj);
            std::complex<MI_F32> gk = std::complex<MI_F32>(0.0f, 0.5f) * (yk_conj - yk);

            std::complex<MI_F32> result = fk + row_real_exp_table[x_grid] * gk;
            dst_complex[y_grid_index][x_grid] = result;
            dst_complex[y_grid_index][GRID_LEN - x_grid] = std::conj(result);
        }

        std::complex<MI_F32> y0 = buffer[0];
        std::complex<MI_F32> y0_conj = std::conj(y0);
        std::complex<MI_F32> f0 = std::complex<MI_F32>(0.5f, 0.0f) * (y0 + y0_conj);
        std::complex<MI_F32> g0 = std::complex<MI_F32>(0.0f, 0.5f) * (y0_conj - y0);
        dst_complex[y_grid_index][0] = f0 + g0;
        dst_complex[y_grid_index][HALF_GRID] = f0 - g0;
        // clear
        dst_complex[y_grid_index][0].imag(0.0f);
        dst_complex[y_grid_index][HALF_GRID].imag(0.0f);
    }

    std::complex<MI_F32> *row_result = &dst_complex[0][0];
    std::complex<MI_F32> exp_table[HALF_GRID];
    GetDftExpTable<0>(exp_table, GRID_LEN);

    for (MI_S32 size = 2; size < GRID_LEN; size *= 2)
    {
        MI_S32 half_size = size / 2;
        MI_S32 table_step = GRID_LEN / size;

        for (MI_S32 i = 0; i < GRID_LEN; i += size)
        {
            float32x4x2_t v2qf32_temp, v2qf32_src0, v2qf32_src1, v2qf32_dst0, v2qf32_dst1;
            float32x4_t vqf32_table_real, vqf32_table_imag;
            MI_S32 j = i;
            MI_S32 k = 0;
            {
                neon::vdup(vqf32_table_real, exp_table[k].real());
                neon::vdup(vqf32_table_imag, exp_table[k].imag());

                for (MI_S32 row_index = 0; row_index < GRID_LEN; row_index += 4)
                {
                    MI_F32 *src_complex0 = (MI_F32 *)(&dst_complex[j][row_index]);
                    MI_F32 *src_complex1 = (MI_F32 *)(&dst_complex[j + half_size][row_index]);
                    v2qf32_src0 = neon::vload2q(src_complex0);
                    v2qf32_src1 = neon::vload2q(src_complex1);

                    v2qf32_temp = v2qf32_src1;
                    v2qf32_dst1.val[0] = neon::vsub(v2qf32_src0.val[0], v2qf32_temp.val[0]);
                    v2qf32_dst1.val[1] = neon::vsub(v2qf32_src0.val[1], v2qf32_temp.val[1]);
                    v2qf32_dst0.val[0] = neon::vadd(v2qf32_src0.val[0], v2qf32_temp.val[0]);
                    v2qf32_dst0.val[1] = neon::vadd(v2qf32_src0.val[1], v2qf32_temp.val[1]);
                    neon::vstore(src_complex0, v2qf32_dst0);
                    neon::vstore(src_complex1, v2qf32_dst1);
                }
                j++;
                k += table_step;
            }

            for (; j < i + half_size; j++, k += table_step)
            {
                neon::vdup(vqf32_table_real, exp_table[k].real());
                neon::vdup(vqf32_table_imag, exp_table[k].imag());

                for (MI_S32 row_index = 0; row_index < GRID_LEN; row_index += 4)
                {
                    MI_F32 *src_complex0 = (MI_F32 *)(&dst_complex[j][row_index]);
                    MI_F32 *src_complex1 = (MI_F32 *)(&dst_complex[j + half_size][row_index]);

                    v2qf32_src0 = neon::vload2q(src_complex0);
                    v2qf32_src1 = neon::vload2q(src_complex1);

                    v2qf32_temp.val[0] = neon::vsub(neon::vmul(v2qf32_src1.val[0], vqf32_table_real), neon::vmul(v2qf32_src1.val[1], vqf32_table_imag));
                    v2qf32_temp.val[1] = neon::vadd(neon::vmul(v2qf32_src1.val[1], vqf32_table_real), neon::vmul(v2qf32_src1.val[0], vqf32_table_imag));

                    v2qf32_dst1.val[0] = neon::vsub(v2qf32_src0.val[0], v2qf32_temp.val[0]);
                    v2qf32_dst1.val[1] = neon::vsub(v2qf32_src0.val[1], v2qf32_temp.val[1]);
                    v2qf32_dst0.val[0] = neon::vadd(v2qf32_src0.val[0], v2qf32_temp.val[0]);
                    v2qf32_dst0.val[1] = neon::vadd(v2qf32_src0.val[1], v2qf32_temp.val[1]);
                    neon::vstore(src_complex0, v2qf32_dst0);
                    neon::vstore(src_complex1, v2qf32_dst1);
                }
            }
        }
    }

    // last butterfly
    for (MI_S32 j = 0; j < HALF_GRID; j++)
    {
        std::complex<MI_F32> *dst_shift        = dst_ptr + j * ostep;
        std::complex<MI_F32> *row_result_shift = row_result + j * GRID_LEN;

        std::complex<MI_F32> *dst_half_shift        = dst_ptr + (j + HALF_GRID) * ostep;
        std::complex<MI_F32> *row_result_half_shift = row_result + (j + HALF_GRID) * GRID_LEN;

        float32x4x2_t v2qf32_temp, v2qf32_src0, v2qf32_src1, v2qf32_dst0, v2qf32_dst1;
        float32x4_t vqf32_table_real, vqf32_table_imag;
        neon::vdup(vqf32_table_real, exp_table[j].real());
        neon::vdup(vqf32_table_imag, exp_table[j].imag());

        for (MI_S32 row_index = 0; row_index < GRID_LEN; row_index += 4)
        {
            MI_F32 *src_complex0 = (MI_F32 *)(row_result_shift      + row_index);
            MI_F32 *src_complex1 = (MI_F32 *)(row_result_half_shift + row_index);
            MI_F32 *dst_data0    = (MI_F32 *)(dst_shift      + width_offset + row_index);
            MI_F32 *dst_data1    = (MI_F32 *)(dst_half_shift + width_offset + row_index);

            v2qf32_src0 = neon::vload2q(src_complex0);
            v2qf32_src1 = neon::vload2q(src_complex1);
            if (0 == j)
            {
                v2qf32_temp = v2qf32_src1;
            }
            else
            {
                v2qf32_temp.val[0] = neon::vsub(neon::vmul(v2qf32_src1.val[0], vqf32_table_real), neon::vmul(v2qf32_src1.val[1], vqf32_table_imag));
                v2qf32_temp.val[1] = neon::vadd(neon::vmul(v2qf32_src1.val[1], vqf32_table_real), neon::vmul(v2qf32_src1.val[0], vqf32_table_imag));
            }

            v2qf32_dst1.val[0] = neon::vsub(v2qf32_src0.val[0], v2qf32_temp.val[0]);
            v2qf32_dst1.val[1] = neon::vsub(v2qf32_src0.val[1], v2qf32_temp.val[1]);
            v2qf32_dst0.val[0] = neon::vadd(v2qf32_src0.val[0], v2qf32_temp.val[0]);
            v2qf32_dst0.val[1] = neon::vadd(v2qf32_src0.val[1], v2qf32_temp.val[1]);
            neon::vstore(dst_data0, v2qf32_dst0);
            neon::vstore(dst_data1, v2qf32_dst1);
        }
    }

    return Status::OK;
}

template <typename Tp, MI_S32 GRID_LEN>
static Status GridDftNeonImpl(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target)
{
    AURA_UNUSED(target);
    const Sizes3 sz = src.GetSizes();
    const MI_S32 width = sz.m_width;
    const MI_S32 height = sz.m_height;

    constexpr MI_S32 HALF_GRID = GRID_LEN / 2;

    MI_U16 idx_x_table[HALF_GRID] = {0};
    MI_U16 idx_y_table[GRID_LEN]  = {0};
    GetReverseIndex(idx_x_table, HALF_GRID);
    GetReverseIndex(idx_y_table, GRID_LEN);

    std::complex<MI_F32> dft_row_exp_table[HALF_GRID];
    std::complex<MI_F32> row_real_exp_table[GRID_LEN];
    GetDftExpTable<0>(dft_row_exp_table,  HALF_GRID);
    GetDftExpTable<0>(row_real_exp_table, GRID_LEN);

    WorkerPool *wp = ctx->GetWorkerPool();
    if (MI_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get WorkerPool Failed.");
        return Status::ERROR;
    }

    auto row_process_func = [&](MI_S32 start, MI_S32 end)->Status
    {
        for (MI_S32 y = start * GRID_LEN; y < end * GRID_LEN; y += GRID_LEN)
        {
            for (MI_S32 x = 0; x < width; x += GRID_LEN)
            {
                GridDftNeonNxN<Tp, GRID_LEN>(src, dst, idx_x_table, idx_y_table, dft_row_exp_table, row_real_exp_table, x, y);
            }
        }
        return Status::OK;
    };

    if (wp->ParallelFor(0, height / GRID_LEN, row_process_func) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "GridDftNeonImpl ParallelFor failed.");
        return Status::ERROR;
    }
    // exit
    return Status::OK;
}

template <typename Tp>
static Status GridDft16x16NeonImpl(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target)
{
    return GridDftNeonImpl<Tp, 16>(ctx, src, dst, target);
}

template <typename Tp>
static Status GridDft32x32NeonImpl(Context *ctx, const Mat &src, Mat &dst, const OpTarget &target)
{
    return GridDftNeonImpl<Tp, 32>(ctx, src, dst, target);
}

template <typename Tp>
static Status GridDftNeonImpl(Context *ctx, const Mat &src, Mat &dst, MI_S32 grid_len, const OpTarget &target)
{
    Status ret = Status::OK;
    switch (grid_len)
    {
        case 4:
        {
            ret = GridDft4x4NeonImpl<Tp>(ctx, src, dst, target);
            break;
        }
        case 8:
        {
            ret = GridDft8x8NeonImpl<Tp>(ctx, src, dst, target);
            break;
        }
        case 16:
        {
            ret = GridDft16x16NeonImpl<Tp>(ctx, src, dst, target);
            break;
        }
        case 32:
        {
            ret = GridDft32x32NeonImpl<Tp>(ctx, src, dst, target);
            break;
        }
        default:
        {
            ret = Status::ERROR;
            AURA_ADD_ERROR_STRING(ctx, "unsupported grid length");
            break;
        }
    }

    AURA_RETURN(ctx, ret);
}

GridDftNeon::GridDftNeon(Context *ctx, const OpTarget &target) : GridDftImpl(ctx, target)
{}

Status GridDftNeon::SetArgs(const Array *src, Array *dst, MI_S32 grid_len)
{
    if (GridDftImpl::SetArgs(src, dst, grid_len) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "GridDftImpl::SetArgs failed");
        return Status::ERROR;
    }

    if ((src->GetArrayType() != ArrayType::MAT) || (dst->GetArrayType() != ArrayType::MAT))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src dst must be mat type");
        return Status::ERROR;
    }

    if (ElemType::S32 == src->GetElemType() || ElemType::U32 == src->GetElemType() || ElemType::F64 == src->GetElemType())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "current src does not support MI_S32/MI_U32/MI_F64 type.");
        return Status::ERROR;
    }

    return Status::OK;
}

Status GridDftNeon::Run()
{
    const Mat *src = dynamic_cast<const Mat*>(m_src);
    Mat       *dst = dynamic_cast<Mat*>(m_dst);

    if ((MI_NULL == src) || (MI_NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src or dst is null");
        return Status::ERROR;
    }

    Status ret = Status::OK;
    switch (src->GetElemType())
    {
        case ElemType::U8:
        {
            ret = GridDftNeonImpl<MI_U8>(m_ctx, *src, *dst, m_grid_len, m_target);
            break;
        }
        case ElemType::S8:
        {
            ret = GridDftNeonImpl<MI_S8>(m_ctx, *src, *dst, m_grid_len, m_target);
            break;
        }
        case ElemType::U16:
        {
            ret = GridDftNeonImpl<MI_U16>(m_ctx, *src, *dst, m_grid_len, m_target);
            break;
        }
        case ElemType::S16:
        {
            ret = GridDftNeonImpl<MI_S16>(m_ctx, *src, *dst, m_grid_len, m_target);
            break;
        }
#if defined(AURA_ENABLE_NEON_FP16)
        case ElemType::F16:
        {
            ret = GridDftNeonImpl<MI_F16>(m_ctx, *src, *dst, m_grid_len, m_target);
            break;
        }
#endif
        case ElemType::F32:
        {
            ret = GridDftNeonImpl<MI_F32>(m_ctx, *src, *dst, m_grid_len, m_target);
            break;
        }
        default:
        {
            ret = Status::ERROR;
            AURA_ADD_ERROR_STRING(m_ctx, "unsupported element type");
            break;
        }
    }

    AURA_RETURN(m_ctx, ret);
}

#define ROW_IDFT_1X4_HELP(vqf32_src00, vqf32_src01, vqf32_dft00, vqf32_dft01, vdf32_conj, vqf32_n)          \
{                                                                                                           \
    float32x2_t vdf32_temp;                                                                                 \
    float32x4_t vqf32_temp0, vqf32_temp1;                                                                   \
                                                                                                            \
    vqf32_temp0 = neon::vadd(vqf32_src00, vqf32_src01);                                                     \
    vqf32_temp1 = neon::vsub(vqf32_src00, vqf32_src01);                                                     \
    vqf32_dft00 = neon::vcombine(neon::vgetlow(vqf32_temp0), neon::vgetlow(vqf32_temp1));                   \
    vqf32_dft01 = neon::vcombine(neon::vgethigh(vqf32_temp0), neon::vgethigh(vqf32_temp1));                 \
                                                                                                            \
    vdf32_temp  = neon::vmul(neon::vrev64(neon::vgethigh(vqf32_dft01)), vdf32_conj);                        \
    vqf32_temp0 = neon::vcombine(neon::vgetlow(vqf32_dft01), vdf32_temp);                                   \
    vqf32_dft01 = neon::vmul(neon::vsub(vqf32_dft00, vqf32_temp0), vqf32_n);                                \
    vqf32_dft00 = neon::vmul(neon::vadd(vqf32_dft00, vqf32_temp0), vqf32_n);                                \
}

template <typename Tp, MI_S32 C>
AURA_ALWAYS_INLINE AURA_VOID IDft4x4NeonImpl(const std::complex<MI_F32> *src0, const std::complex<MI_F32> *src1,
                                           const std::complex<MI_F32> *src2, const std::complex<MI_F32> *src3,
                                           MI_BOOL with_scale, Tp *dst, MI_S32 ostep)
{
    float32x4_t vqf32_n;
    if (with_scale)
    {
        float32x4_t vqf32_scale;
        neon::vdup(vqf32_scale, 4.0f);
        vqf32_n = neon::vreciprocal_newton(vqf32_scale);
    }
    else
    {
        neon::vdup(vqf32_n, 1.0f);
    }

    MI_F32 conj[2] = {-1.0f, 1.0f};
    float32x2_t vdf32_conj = neon::vload1(conj);
    float32x4_t vqf32_conj = neon::vcombine(vdf32_conj, vdf32_conj);

    // load src data
    float32x4_t vqf32_src00, vqf32_src01;
    float32x4_t vqf32_src10, vqf32_src11;
    float32x4_t vqf32_src20, vqf32_src21;
    float32x4_t vqf32_src30, vqf32_src31;
    vqf32_src00 = neon::vload1q((MI_F32 *)src0); vqf32_src01 = neon::vload1q((MI_F32 *)(src0 + 2));
    vqf32_src10 = neon::vload1q((MI_F32 *)src1); vqf32_src11 = neon::vload1q((MI_F32 *)(src1 + 2));
    vqf32_src20 = neon::vload1q((MI_F32 *)src2); vqf32_src21 = neon::vload1q((MI_F32 *)(src2 + 2));
    vqf32_src30 = neon::vload1q((MI_F32 *)src3); vqf32_src31 = neon::vload1q((MI_F32 *)(src3 + 2));

    // row idft
    float32x4_t vqf32_dft00, vqf32_dft01;
    float32x4_t vqf32_dft10, vqf32_dft11;
    float32x4_t vqf32_dft20, vqf32_dft21;
    float32x4_t vqf32_dft30, vqf32_dft31;
    ROW_IDFT_1X4_HELP(vqf32_src00, vqf32_src01, vqf32_dft00, vqf32_dft01, vdf32_conj, vqf32_n);
    ROW_IDFT_1X4_HELP(vqf32_src10, vqf32_src11, vqf32_dft10, vqf32_dft11, vdf32_conj, vqf32_n);
    ROW_IDFT_1X4_HELP(vqf32_src20, vqf32_src21, vqf32_dft20, vqf32_dft21, vdf32_conj, vqf32_n);
    ROW_IDFT_1X4_HELP(vqf32_src30, vqf32_src31, vqf32_dft30, vqf32_dft31, vdf32_conj, vqf32_n);

    // col idft
    float32x4_t vqf32_temp0, vqf32_temp1;
    vqf32_temp0 = vqf32_dft10;
    vqf32_temp1 = vqf32_dft11;
    vqf32_dft10 = neon::vsub(vqf32_dft00, vqf32_temp0);
    vqf32_dft11 = neon::vsub(vqf32_dft01, vqf32_temp1);
    vqf32_dft00 = neon::vadd(vqf32_dft00, vqf32_temp0);
    vqf32_dft01 = neon::vadd(vqf32_dft01, vqf32_temp1);

    vqf32_temp0 = vqf32_dft30;
    vqf32_temp1 = vqf32_dft31;
    vqf32_dft30 = neon::vsub(vqf32_dft20, vqf32_temp0);
    vqf32_dft31 = neon::vsub(vqf32_dft21, vqf32_temp1);
    vqf32_dft20 = neon::vadd(vqf32_dft20, vqf32_temp0);
    vqf32_dft21 = neon::vadd(vqf32_dft21, vqf32_temp1);

    vqf32_temp0 = vqf32_dft20;
    vqf32_temp1 = vqf32_dft21;
    vqf32_dft20 = neon::vmul(neon::vsub(vqf32_dft00, vqf32_temp0), vqf32_n);
    vqf32_dft21 = neon::vmul(neon::vsub(vqf32_dft01, vqf32_temp1), vqf32_n);
    vqf32_dft00 = neon::vmul(neon::vadd(vqf32_dft00, vqf32_temp0), vqf32_n);
    vqf32_dft01 = neon::vmul(neon::vadd(vqf32_dft01, vqf32_temp1), vqf32_n);

    vqf32_temp0 = neon::vmul(neon::vrev64(vqf32_dft30), vqf32_conj);
    vqf32_temp1 = neon::vmul(neon::vrev64(vqf32_dft31), vqf32_conj);
    vqf32_dft30 = neon::vmul(neon::vsub(vqf32_dft10, vqf32_temp0), vqf32_n);
    vqf32_dft31 = neon::vmul(neon::vsub(vqf32_dft11, vqf32_temp1), vqf32_n);
    vqf32_dft10 = neon::vmul(neon::vadd(vqf32_dft10, vqf32_temp0), vqf32_n);
    vqf32_dft11 = neon::vmul(neon::vadd(vqf32_dft11, vqf32_temp1), vqf32_n);

    Tp *dst0 = dst;
    Tp *dst1 = dst0 + ostep;
    Tp *dst2 = dst1 + ostep;
    Tp *dst3 = dst2 + ostep;

    neon::vstore(dst0, vqf32_dft00); neon::vstore(dst0 + 4, vqf32_dft01);
    neon::vstore(dst1, vqf32_dft10); neon::vstore(dst1 + 4, vqf32_dft11);
    neon::vstore(dst2, vqf32_dft20); neon::vstore(dst2 + 4, vqf32_dft21);
    neon::vstore(dst3, vqf32_dft30); neon::vstore(dst3 + 4, vqf32_dft31);
}

template <typename Tp, MI_S32 C>
static Status GridIDft4x4NeonImpl(Context *ctx, const Mat &src, Mat &dst, MI_BOOL with_scale, const OpTarget &target)
{
    AURA_UNUSED(target);
    Sizes3 size = dst.GetSizes();
    MI_S32 width = size.m_width;
    MI_S32 height = size.m_height;

    if (2 != C)
    {
        AURA_ADD_ERROR_STRING(ctx, "dst channel must be 2, because this function do not support real output.");
        return Status::ERROR;
    }

    MI_S32 ostep = dst.GetRowPitch() / sizeof (Tp);

    WorkerPool *wp = ctx->GetWorkerPool();
    if (MI_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get WorkerPool Failed.");
        return Status::ERROR;
    }

    auto row_process_func = [&](MI_S32 start, MI_S32 end)->Status
    {
        for (MI_S32 h = start * 4; h < end * 4; h += 4)
        {
            const std::complex<MI_F32> *src_row0 = src.Ptr<std::complex<MI_F32>>(h);
            const std::complex<MI_F32> *src_row1 = src.Ptr<std::complex<MI_F32>>(h + 1);
            const std::complex<MI_F32> *src_row2 = src.Ptr<std::complex<MI_F32>>(h + 2);
            const std::complex<MI_F32> *src_row3 = src.Ptr<std::complex<MI_F32>>(h + 3);

            Tp *dst_row = dst.Ptr<Tp>(h);

            for (MI_S32 w = 0; w < width; w += 4)
            {
                IDft4x4NeonImpl<Tp, C>(src_row0 + w, src_row2 + w, src_row1 + w, src_row3 + w, with_scale,
                                       dst_row + w * C, ostep);
            }
        }

        return Status::OK;
    };

    if (wp->ParallelFor(0, height / 4, row_process_func) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "GridIDft4x4NeonImpl ParallelFor failed.");
        return Status::ERROR;
    }

    return Status::OK;
}

static AURA_VOID RowGridIDft1x8(const std::complex<MI_F32> *src, std::complex<MI_F32> *dst, MI_BOOL with_scale)
{
    float32x4_t vqf32_n;
    if (with_scale)
    {
        float32x4_t vqf32_scale;
        neon::vdup(vqf32_scale, 8.0f);
        vqf32_n = neon::vreciprocal_newton(vqf32_scale);
    }
    else
    {
        neon::vdup(vqf32_n, 1.0f);
    }

    MI_F32 conj[2] = {-1.0f, 1.0f};
    MI_F32 exp0[2] = {0.707107f, 0.707107f};
    MI_F32 exp1[2] = {-0.707107f, 0.707107f};

    float32x2_t vdf32_exp0 = neon::vload1(exp0);
    float32x2_t vdf32_exp1 = neon::vload1(exp1);
    float32x2_t vdf32_conj = neon::vload1(conj);
    // load src data
    float32x4_t vqf32_src0, vqf32_src1, vqf32_src2, vqf32_src3;
    vqf32_src0 = neon::vload1q((MI_F32 *)src);
    vqf32_src1 = neon::vload1q((MI_F32 *)(src + 2));
    vqf32_src2 = neon::vload1q((MI_F32 *)(src + 4));
    vqf32_src3 = neon::vload1q((MI_F32 *)(src + 6));

    // suffer data and butterfly size 2
    float32x4_t vqf32_tmp0, vqf32_tmp1, vqf32_tmp2, vqf32_tmp3;
    vqf32_tmp0 = neon::vadd(vqf32_src0, vqf32_src2);// 0, 1
    vqf32_tmp1 = neon::vsub(vqf32_src0, vqf32_src2);// 4, 5
    vqf32_tmp2 = neon::vadd(vqf32_src1, vqf32_src3);// 2, 3
    vqf32_tmp3 = neon::vsub(vqf32_src1, vqf32_src3);// 6, 7

    float32x4_t vqf32_dst0, vqf32_dst1, vqf32_dst2, vqf32_dst3;
    vqf32_dst0 = neon::vcombine(neon::vgetlow(vqf32_tmp0), neon::vgetlow(vqf32_tmp1));
    vqf32_dst1 = neon::vcombine(neon::vgetlow(vqf32_tmp2), neon::vgetlow(vqf32_tmp3));
    vqf32_dst2 = neon::vcombine(neon::vgethigh(vqf32_tmp0), neon::vgethigh(vqf32_tmp1));
    vqf32_dst3 = neon::vcombine(neon::vgethigh(vqf32_tmp2), neon::vgethigh(vqf32_tmp3));

    // butterfly size 4
    float32x2_t vdf32_tmp0, vdf32_tmp1;
    vdf32_tmp0 = neon::vgetlow(vqf32_dst1);
    vdf32_tmp1 = neon::vmul(neon::vrev64(neon::vgethigh(vqf32_dst1)), vdf32_conj);
    vqf32_tmp0 = neon::vcombine(vdf32_tmp0, vdf32_tmp1);
    vqf32_dst1 = neon::vsub(vqf32_dst0, vqf32_tmp0);
    vqf32_dst0 = neon::vadd(vqf32_dst0, vqf32_tmp0);

    vdf32_tmp0 = neon::vgetlow(vqf32_dst3);
    vdf32_tmp1 = neon::vmul(neon::vrev64(neon::vgethigh(vqf32_dst3)), vdf32_conj);
    vqf32_tmp1 = neon::vcombine(vdf32_tmp0, vdf32_tmp1);
    vqf32_dst3 = neon::vsub(vqf32_dst2, vqf32_tmp1);
    vqf32_dst2 = neon::vadd(vqf32_dst2, vqf32_tmp1);

    // butterfly size 8
    float32x2x2_t v2df32_dst23 = neon::vzip(neon::vgethigh(vqf32_dst2), neon::vgethigh(vqf32_dst3));
    float32x2x2_t v2df32_exp01 = neon::vzip(vdf32_exp0, vdf32_exp1);
    float32x2x2_t v2df32_mul;
    COMPLEX_MUL(v2df32_dst23, v2df32_exp01, v2df32_mul);
    float32x2x2_t v2df32_tmp = neon::vuzp(v2df32_mul.val[0], v2df32_mul.val[1]);

    vqf32_tmp0 = neon::vcombine(neon::vgetlow(vqf32_dst2), v2df32_tmp.val[0]);
    vdf32_tmp0 = neon::vmul(neon::vrev64(neon::vgetlow(vqf32_dst3)), vdf32_conj);
    vqf32_tmp1 = neon::vcombine(vdf32_tmp0, v2df32_tmp.val[1]);

    vqf32_dst2 = neon::vmul(neon::vsub(vqf32_dst0, vqf32_tmp0), vqf32_n);
    vqf32_dst0 = neon::vmul(neon::vadd(vqf32_dst0, vqf32_tmp0), vqf32_n);
    vqf32_dst3 = neon::vmul(neon::vsub(vqf32_dst1, vqf32_tmp1), vqf32_n);
    vqf32_dst1 = neon::vmul(neon::vadd(vqf32_dst1, vqf32_tmp1), vqf32_n);

    neon::vstore((MI_F32 *)(dst),     vqf32_dst0);
    neon::vstore((MI_F32 *)(dst + 2), vqf32_dst1);
    neon::vstore((MI_F32 *)(dst + 4), vqf32_dst2);
    neon::vstore((MI_F32 *)(dst + 6), vqf32_dst3);
}

template <typename Tp, MI_S32 C>
static AURA_VOID ColGridIDft8x8(std::complex<MI_F32> src[][8], MI_BOOL with_scale, Tp *dst, MI_S32 ostep)
{
    float32x4_t vqf32_n;
    if (with_scale)
    {
        float32x4_t vqf32_scale;
        neon::vdup(vqf32_scale, 8.0f);
        vqf32_n = neon::vreciprocal_newton(vqf32_scale);
    }
    else
    {
        neon::vdup(vqf32_n, 1.0f);
    }

    // butterfly size is 2 and 4, and only update src
    ColButterflyCalc(src, MI_TRUE);

    // butterfly size is 8, and store result to dst
    MI_F32 exp1[8] = {0.707107f, 0.707107f, 0.707107f, 0.707107f, 0.707107f, 0.707107f, 0.707107f, 0.707107f};
    MI_F32 exp3[8] = {-0.707107f, 0.707107f, -0.707107f, 0.707107f, -0.707107f, 0.707107f, -0.707107f, 0.707107f};

    float32x4_t vqf32_conj;
    neon::vdup(vqf32_conj, -1.0f);
    float32x4x2_t v2qf32_table1 = neon::vload2q(exp1);
    float32x4x2_t v2qf32_table3 = neon::vload2q(exp3);

    MI_F32 *line0 = (MI_F32 *)(&src[0][0]);
    MI_F32 *line1 = (MI_F32 *)(&src[1][0]);
    MI_F32 *line2 = (MI_F32 *)(&src[2][0]);
    MI_F32 *line3 = (MI_F32 *)(&src[3][0]);
    MI_F32 *line4 = (MI_F32 *)(&src[4][0]);
    MI_F32 *line5 = (MI_F32 *)(&src[5][0]);
    MI_F32 *line6 = (MI_F32 *)(&src[6][0]);
    MI_F32 *line7 = (MI_F32 *)(&src[7][0]);

    float32x4x2_t v2qf32_temp0, v2qf32_temp1;
    float32x4x2_t v2qf32_dst00, v2qf32_dst01, v2qf32_dst40, v2qf32_dst41;
    float32x4x2_t v2qf32_dst20, v2qf32_dst21, v2qf32_dst60, v2qf32_dst61;

    float32x4x2_t v2qf32_src00 = neon::vload2q(line0);
    float32x4x2_t v2qf32_src01 = neon::vload2q(line0 + 8);
    float32x4x2_t v2qf32_src40 = neon::vload2q(line4);
    float32x4x2_t v2qf32_src41 = neon::vload2q(line4 + 8);
    float32x4x2_t v2qf32_src20 = neon::vload2q(line2);
    float32x4x2_t v2qf32_src21 = neon::vload2q(line2 + 8);
    float32x4x2_t v2qf32_src60 = neon::vload2q(line6);
    float32x4x2_t v2qf32_src61 = neon::vload2q(line6 + 8);

    v2qf32_temp0 = v2qf32_src40;
    v2qf32_temp1 = v2qf32_src41;
    v2qf32_dst40.val[0] = neon::vmul(neon::vsub(v2qf32_src00.val[0], v2qf32_temp0.val[0]), vqf32_n);
    v2qf32_dst40.val[1] = neon::vmul(neon::vsub(v2qf32_src00.val[1], v2qf32_temp0.val[1]), vqf32_n);
    v2qf32_dst41.val[0] = neon::vmul(neon::vsub(v2qf32_src01.val[0], v2qf32_temp1.val[0]), vqf32_n);
    v2qf32_dst41.val[1] = neon::vmul(neon::vsub(v2qf32_src01.val[1], v2qf32_temp1.val[1]), vqf32_n);
    v2qf32_dst00.val[0] = neon::vmul(neon::vadd(v2qf32_src00.val[0], v2qf32_temp0.val[0]), vqf32_n);
    v2qf32_dst00.val[1] = neon::vmul(neon::vadd(v2qf32_src00.val[1], v2qf32_temp0.val[1]), vqf32_n);
    v2qf32_dst01.val[0] = neon::vmul(neon::vadd(v2qf32_src01.val[0], v2qf32_temp1.val[0]), vqf32_n);
    v2qf32_dst01.val[1] = neon::vmul(neon::vadd(v2qf32_src01.val[1], v2qf32_temp1.val[1]), vqf32_n);

    v2qf32_temp0.val[0] = neon::vmul(v2qf32_src60.val[1], vqf32_conj);
    v2qf32_temp0.val[1] = v2qf32_src60.val[0];
    v2qf32_temp1.val[0] = neon::vmul(v2qf32_src61.val[1], vqf32_conj);
    v2qf32_temp1.val[1] = v2qf32_src61.val[0];
    v2qf32_dst60.val[0] = neon::vmul(neon::vsub(v2qf32_src20.val[0], v2qf32_temp0.val[0]), vqf32_n);
    v2qf32_dst60.val[1] = neon::vmul(neon::vsub(v2qf32_src20.val[1], v2qf32_temp0.val[1]), vqf32_n);
    v2qf32_dst61.val[0] = neon::vmul(neon::vsub(v2qf32_src21.val[0], v2qf32_temp1.val[0]), vqf32_n);
    v2qf32_dst61.val[1] = neon::vmul(neon::vsub(v2qf32_src21.val[1], v2qf32_temp1.val[1]), vqf32_n);
    v2qf32_dst20.val[0] = neon::vmul(neon::vadd(v2qf32_src20.val[0], v2qf32_temp0.val[0]), vqf32_n);
    v2qf32_dst20.val[1] = neon::vmul(neon::vadd(v2qf32_src20.val[1], v2qf32_temp0.val[1]), vqf32_n);
    v2qf32_dst21.val[0] = neon::vmul(neon::vadd(v2qf32_src21.val[0], v2qf32_temp1.val[0]), vqf32_n);
    v2qf32_dst21.val[1] = neon::vmul(neon::vadd(v2qf32_src21.val[1], v2qf32_temp1.val[1]), vqf32_n);

    if (1 == C)
    {
        Tp *dst0 = dst;
        Tp *dst2 = dst0 + ostep * 2;
        Tp *dst4 = dst0 + ostep * 4;
        Tp *dst6 = dst0 + ostep * 6;

        float32x4x2_t v2qf32_real0, v2qf32_real2, v2qf32_real4, v2qf32_real6;
        v2qf32_real0.val[0] = v2qf32_dst00.val[0]; v2qf32_real0.val[1] = v2qf32_dst01.val[0];
        v2qf32_real2.val[0] = v2qf32_dst20.val[0]; v2qf32_real2.val[1] = v2qf32_dst21.val[0];
        v2qf32_real4.val[0] = v2qf32_dst40.val[0]; v2qf32_real4.val[1] = v2qf32_dst41.val[0];
        v2qf32_real6.val[0] = v2qf32_dst60.val[0]; v2qf32_real6.val[1] = v2qf32_dst61.val[0];

        StoreF32PairAs<Tp>(dst0, v2qf32_real0);
        StoreF32PairAs<Tp>(dst2, v2qf32_real2);
        StoreF32PairAs<Tp>(dst4, v2qf32_real4);
        StoreF32PairAs<Tp>(dst6, v2qf32_real6);
    }
    else
    {
        MI_F32 *dst0 = (MI_F32 *)dst;
        MI_F32 *dst2 = dst0 + 2 * ostep;
        MI_F32 *dst4 = dst0 + 4 * ostep;
        MI_F32 *dst6 = dst0 + 6 * ostep;

        neon::vstore(dst0, v2qf32_dst00); neon::vstore(dst0 + 8, v2qf32_dst01);
        neon::vstore(dst4, v2qf32_dst40); neon::vstore(dst4 + 8, v2qf32_dst41);
        neon::vstore(dst2, v2qf32_dst20); neon::vstore(dst2 + 8, v2qf32_dst21);
        neon::vstore(dst6, v2qf32_dst60); neon::vstore(dst6 + 8, v2qf32_dst61);
    }

    v2qf32_src00 = neon::vload2q(line1); v2qf32_src01 = neon::vload2q(line1 + 8);
    v2qf32_src40 = neon::vload2q(line5); v2qf32_src41 = neon::vload2q(line5 + 8);
    v2qf32_src20 = neon::vload2q(line3); v2qf32_src21 = neon::vload2q(line3 + 8);
    v2qf32_src60 = neon::vload2q(line7); v2qf32_src61 = neon::vload2q(line7 + 8);

    v2qf32_temp0.val[0] = neon::vsub(neon::vmul(v2qf32_src40.val[0], v2qf32_table1.val[0]), neon::vmul(v2qf32_src40.val[1], v2qf32_table1.val[1]));
    v2qf32_temp0.val[1] = neon::vadd(neon::vmul(v2qf32_src40.val[1], v2qf32_table1.val[0]), neon::vmul(v2qf32_src40.val[0], v2qf32_table1.val[1]));
    v2qf32_temp1.val[0] = neon::vsub(neon::vmul(v2qf32_src41.val[0], v2qf32_table1.val[0]), neon::vmul(v2qf32_src41.val[1], v2qf32_table1.val[1]));
    v2qf32_temp1.val[1] = neon::vadd(neon::vmul(v2qf32_src41.val[1], v2qf32_table1.val[0]), neon::vmul(v2qf32_src41.val[0], v2qf32_table1.val[1]));

    v2qf32_dst40.val[0] = neon::vmul(neon::vsub(v2qf32_src00.val[0], v2qf32_temp0.val[0]), vqf32_n);
    v2qf32_dst40.val[1] = neon::vmul(neon::vsub(v2qf32_src00.val[1], v2qf32_temp0.val[1]), vqf32_n);
    v2qf32_dst41.val[0] = neon::vmul(neon::vsub(v2qf32_src01.val[0], v2qf32_temp1.val[0]), vqf32_n);
    v2qf32_dst41.val[1] = neon::vmul(neon::vsub(v2qf32_src01.val[1], v2qf32_temp1.val[1]), vqf32_n);
    v2qf32_dst00.val[0] = neon::vmul(neon::vadd(v2qf32_src00.val[0], v2qf32_temp0.val[0]), vqf32_n);
    v2qf32_dst00.val[1] = neon::vmul(neon::vadd(v2qf32_src00.val[1], v2qf32_temp0.val[1]), vqf32_n);
    v2qf32_dst01.val[0] = neon::vmul(neon::vadd(v2qf32_src01.val[0], v2qf32_temp1.val[0]), vqf32_n);
    v2qf32_dst01.val[1] = neon::vmul(neon::vadd(v2qf32_src01.val[1], v2qf32_temp1.val[1]), vqf32_n);

    v2qf32_temp0.val[0] = neon::vsub(neon::vmul(v2qf32_src60.val[0], v2qf32_table3.val[0]), neon::vmul(v2qf32_src60.val[1], v2qf32_table3.val[1]));
    v2qf32_temp0.val[1] = neon::vadd(neon::vmul(v2qf32_src60.val[1], v2qf32_table3.val[0]), neon::vmul(v2qf32_src60.val[0], v2qf32_table3.val[1]));
    v2qf32_temp1.val[0] = neon::vsub(neon::vmul(v2qf32_src61.val[0], v2qf32_table3.val[0]), neon::vmul(v2qf32_src61.val[1], v2qf32_table3.val[1]));
    v2qf32_temp1.val[1] = neon::vadd(neon::vmul(v2qf32_src61.val[1], v2qf32_table3.val[0]), neon::vmul(v2qf32_src61.val[0], v2qf32_table3.val[1]));

    v2qf32_dst60.val[0] = neon::vmul(neon::vsub(v2qf32_src20.val[0], v2qf32_temp0.val[0]), vqf32_n);
    v2qf32_dst60.val[1] = neon::vmul(neon::vsub(v2qf32_src20.val[1], v2qf32_temp0.val[1]), vqf32_n);
    v2qf32_dst61.val[0] = neon::vmul(neon::vsub(v2qf32_src21.val[0], v2qf32_temp1.val[0]), vqf32_n);
    v2qf32_dst61.val[1] = neon::vmul(neon::vsub(v2qf32_src21.val[1], v2qf32_temp1.val[1]), vqf32_n);
    v2qf32_dst20.val[0] = neon::vmul(neon::vadd(v2qf32_src20.val[0], v2qf32_temp0.val[0]), vqf32_n);
    v2qf32_dst20.val[1] = neon::vmul(neon::vadd(v2qf32_src20.val[1], v2qf32_temp0.val[1]), vqf32_n);
    v2qf32_dst21.val[0] = neon::vmul(neon::vadd(v2qf32_src21.val[0], v2qf32_temp1.val[0]), vqf32_n);
    v2qf32_dst21.val[1] = neon::vmul(neon::vadd(v2qf32_src21.val[1], v2qf32_temp1.val[1]), vqf32_n);

    if (1 == C)
    {
        Tp *dst0 = dst;
        Tp *dst1 = dst0 + ostep;
        Tp *dst3 = dst0 + ostep * 3;
        Tp *dst5 = dst0 + ostep * 5;
        Tp *dst7 = dst0 + ostep * 7;

        float32x4x2_t v2qf32_real0, v2qf32_real2, v2qf32_real4, v2qf32_real6;
        v2qf32_real0.val[0] = v2qf32_dst00.val[0]; v2qf32_real0.val[1] = v2qf32_dst01.val[0];
        v2qf32_real2.val[0] = v2qf32_dst20.val[0]; v2qf32_real2.val[1] = v2qf32_dst21.val[0];
        v2qf32_real4.val[0] = v2qf32_dst40.val[0]; v2qf32_real4.val[1] = v2qf32_dst41.val[0];
        v2qf32_real6.val[0] = v2qf32_dst60.val[0]; v2qf32_real6.val[1] = v2qf32_dst61.val[0];

        StoreF32PairAs<Tp>(dst1, v2qf32_real0);
        StoreF32PairAs<Tp>(dst3, v2qf32_real2);
        StoreF32PairAs<Tp>(dst5, v2qf32_real4);
        StoreF32PairAs<Tp>(dst7, v2qf32_real6);
    }
    else
    {
        MI_F32 *dst0 = (MI_F32 *)dst;
        MI_F32 *dst1 = dst0 + 1 * ostep;
        MI_F32 *dst3 = dst0 + 3 * ostep;
        MI_F32 *dst5 = dst0 + 5 * ostep;
        MI_F32 *dst7 = dst0 + 7 * ostep;

        neon::vstore(dst1, v2qf32_dst00); neon::vstore(dst1 + 8, v2qf32_dst01);
        neon::vstore(dst5, v2qf32_dst40); neon::vstore(dst5 + 8, v2qf32_dst41);
        neon::vstore(dst3, v2qf32_dst20); neon::vstore(dst3 + 8, v2qf32_dst21);
        neon::vstore(dst7, v2qf32_dst60); neon::vstore(dst7 + 8, v2qf32_dst61);
    }
}

template <typename Tp, MI_S32 C>
static Status GridIDft8x8NeonImpl(Context *ctx, const Mat &src, Mat &dst, MI_BOOL with_scale, const OpTarget &target)
{
    AURA_UNUSED(target);
    Sizes3 size    = dst.GetSizes();
    MI_S32 width   = size.m_width;
    MI_S32 height  = size.m_height;

    MI_S32 ostep = dst.GetRowPitch() / sizeof (Tp);

    WorkerPool *wp = ctx->GetWorkerPool();
    if (MI_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get WorkerPool Failed.");
        return Status::ERROR;
    }

    auto row_process_func = [&](MI_S32 start, MI_S32 end)->Status
    {
        for (MI_S32 h = start * 8; h < end * 8; h += 8)
        {
            std::complex<MI_F32> row_dft_result[8][8];
            const std::complex<MI_F32> *src_row0 = src.Ptr<std::complex<MI_F32>>(h);
            const std::complex<MI_F32> *src_row1 = src.Ptr<std::complex<MI_F32>>(h + 1);
            const std::complex<MI_F32> *src_row2 = src.Ptr<std::complex<MI_F32>>(h + 2);
            const std::complex<MI_F32> *src_row3 = src.Ptr<std::complex<MI_F32>>(h + 3);
            const std::complex<MI_F32> *src_row4 = src.Ptr<std::complex<MI_F32>>(h + 4);
            const std::complex<MI_F32> *src_row5 = src.Ptr<std::complex<MI_F32>>(h + 5);
            const std::complex<MI_F32> *src_row6 = src.Ptr<std::complex<MI_F32>>(h + 6);
            const std::complex<MI_F32> *src_row7 = src.Ptr<std::complex<MI_F32>>(h + 7);

            Tp *dst_row = dst.Ptr<Tp>(h);

            for (MI_S32 w = 0; w < width; w += 8)
            {
                RowGridIDft1x8(src_row0 + w, row_dft_result[0], with_scale);
                RowGridIDft1x8(src_row1 + w, row_dft_result[4], with_scale);
                RowGridIDft1x8(src_row2 + w, row_dft_result[2], with_scale);
                RowGridIDft1x8(src_row3 + w, row_dft_result[6], with_scale);
                RowGridIDft1x8(src_row4 + w, row_dft_result[1], with_scale);
                RowGridIDft1x8(src_row5 + w, row_dft_result[5], with_scale);
                RowGridIDft1x8(src_row6 + w, row_dft_result[3], with_scale);
                RowGridIDft1x8(src_row7 + w, row_dft_result[7], with_scale);

                ColGridIDft8x8<Tp, C>(row_dft_result, with_scale, dst_row + w * C, ostep);
            }
        }

        return Status::OK;
    };

    if (wp->ParallelFor(0, height / 8, row_process_func) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "GridIDft4x4NeonImpl ParallelFor failed.");
        return Status::ERROR;
    }

    return Status::OK;
}

template <typename Tp, MI_S32 GRID_LEN, MI_S32 C>
static Status GridIDftNeonNxN(const Mat &src, Mat &dst, MI_U16 *idx_table,
                              std::complex<MI_F32> *exp_table, MI_BOOL with_scale,
                              MI_S32 width_offset, MI_S32 height_offset)
{
    const MI_S32 ostep   = dst.GetRowPitch() / sizeof(Tp);
    constexpr MI_S32 HALF_GRID = GRID_LEN / 2;

    float32x4_t vqf32_n;
    if (with_scale)
    {
        float32x4_t vqf32_scale;
        neon::vdup(vqf32_scale, (MI_F32)GRID_LEN);
        vqf32_n = neon::vreciprocal_newton(vqf32_scale);
    }
    else
    {
        neon::vdup(vqf32_n, 1.0f);
    }

    Tp *dst_ptr = dst.Ptr<Tp>(height_offset);
    std::complex<MI_F32> row_dft_result[GRID_LEN][GRID_LEN];

    for (MI_S32 y_grid = 0; y_grid < GRID_LEN; y_grid++)
    {
        const std::complex<MI_F32> *src_row = src.Ptr<std::complex<MI_F32>>(y_grid + height_offset);
        MI_S32 y_grid_index = idx_table[y_grid];

        for (MI_S32 x = 0; x < GRID_LEN; x += 2)
        {
            MI_U16 idx0 = idx_table[x];
            MI_U16 idx1 = idx_table[x + 1];
            row_dft_result[y_grid_index][x]     = src_row[width_offset + idx0] + src_row[width_offset + idx1];
            row_dft_result[y_grid_index][x + 1] = src_row[width_offset + idx0] - src_row[width_offset + idx1];
        }

        ButterflyTransformNeon(&row_dft_result[y_grid_index][0], 4, GRID_LEN, with_scale, exp_table);
    }

    std::complex<MI_F32> *row_result = &row_dft_result[0][0];

    for (MI_S32 size = 2; size < GRID_LEN; size *= 2)
    {
        MI_S32 half_size = size / 2;
        MI_S32 table_step = GRID_LEN / size;

        for (MI_S32 i = 0; i < GRID_LEN; i += size)
        {
            float32x4x2_t v2qf32_temp, v2qf32_src0, v2qf32_src1, v2qf32_dst0, v2qf32_dst1;
            float32x4_t vqf32_table_real, vqf32_table_imag;
            MI_S32 j = i;
            MI_S32 k = 0;
            {
                neon::vdup(vqf32_table_real, exp_table[k].real());
                neon::vdup(vqf32_table_imag, exp_table[k].imag());

                for (MI_S32 row_index = 0; row_index < GRID_LEN; row_index += 4)
                {
                    MI_F32 *src_complex0 = (MI_F32 *)(&row_dft_result[j][row_index]);
                    MI_F32 *src_complex1 = (MI_F32 *)(&row_dft_result[j + half_size][row_index]);

                    v2qf32_src0 = neon::vload2q(src_complex0);
                    v2qf32_src1 = neon::vload2q(src_complex1);
                    v2qf32_temp = v2qf32_src1;

                    v2qf32_dst1.val[0] = neon::vsub(v2qf32_src0.val[0], v2qf32_temp.val[0]);
                    v2qf32_dst1.val[1] = neon::vsub(v2qf32_src0.val[1], v2qf32_temp.val[1]);
                    v2qf32_dst0.val[0] = neon::vadd(v2qf32_src0.val[0], v2qf32_temp.val[0]);
                    v2qf32_dst0.val[1] = neon::vadd(v2qf32_src0.val[1], v2qf32_temp.val[1]);
                    neon::vstore(src_complex0, v2qf32_dst0);
                    neon::vstore(src_complex1, v2qf32_dst1);
                }

                j++;
                k += table_step;
            }

            for (; j < i + half_size; j++, k += table_step)
            {
                neon::vdup(vqf32_table_real, exp_table[k].real());
                neon::vdup(vqf32_table_imag, exp_table[k].imag());

                for (MI_S32 row_index = 0; row_index < GRID_LEN; row_index += 4)
                {
                    MI_F32 *src_complex0 = (MI_F32 *)(&row_dft_result[j][row_index]);
                    MI_F32 *src_complex1 = (MI_F32 *)(&row_dft_result[j + half_size][row_index]);

                    v2qf32_src0 = neon::vload2q(src_complex0);
                    v2qf32_src1 = neon::vload2q(src_complex1);

                    v2qf32_temp.val[0] = neon::vsub(neon::vmul(v2qf32_src1.val[0], vqf32_table_real), neon::vmul(v2qf32_src1.val[1], vqf32_table_imag));
                    v2qf32_temp.val[1] = neon::vadd(neon::vmul(v2qf32_src1.val[1], vqf32_table_real), neon::vmul(v2qf32_src1.val[0], vqf32_table_imag));

                    v2qf32_dst1.val[0] = neon::vsub(v2qf32_src0.val[0], v2qf32_temp.val[0]);
                    v2qf32_dst1.val[1] = neon::vsub(v2qf32_src0.val[1], v2qf32_temp.val[1]);
                    v2qf32_dst0.val[0] = neon::vadd(v2qf32_src0.val[0], v2qf32_temp.val[0]);
                    v2qf32_dst0.val[1] = neon::vadd(v2qf32_src0.val[1], v2qf32_temp.val[1]);
                    neon::vstore(src_complex0, v2qf32_dst0);
                    neon::vstore(src_complex1, v2qf32_dst1);
                }
            }
        }
    }

    // last butterfly
    for (MI_S32 j = 0; j < HALF_GRID; j++)
    {
        Tp *dst_shift      = dst_ptr + j * ostep;
        Tp *dst_half_shift = dst_ptr + (j + HALF_GRID) * ostep;

        std::complex<MI_F32> *row_result_shift      = row_result + j * GRID_LEN;
        std::complex<MI_F32> *row_result_half_shift = row_result + (j + HALF_GRID) * GRID_LEN;

        float32x4x2_t v2qf32_temp0, v2qf32_temp1, v2qf32_src00, v2qf32_src01, v2qf32_src10, v2qf32_src11;
        float32x4x2_t v2qf32_dst00, v2qf32_dst01, v2qf32_dst10, v2qf32_dst11;

        float32x4_t vqf32_table_real, vqf32_table_imag;
        neon::vdup(vqf32_table_real, exp_table[j].real());
        neon::vdup(vqf32_table_imag, exp_table[j].imag());

        for (MI_S32 row_index = 0; row_index < GRID_LEN; row_index += 8)
        {
            MI_F32 *src_complex0 = (MI_F32 *)(row_result_shift      + row_index);
            MI_F32 *src_complex1 = (MI_F32 *)(row_result_half_shift + row_index);

            v2qf32_src00 = neon::vload2q(src_complex0);
            v2qf32_src01 = neon::vload2q(src_complex0 + 8);
            v2qf32_src10 = neon::vload2q(src_complex1);
            v2qf32_src11 = neon::vload2q(src_complex1 + 8);
            if (0 == j)
            {
                v2qf32_temp0 = v2qf32_src10;
                v2qf32_temp1 = v2qf32_src11;
            }
            else
            {
                v2qf32_temp0.val[0] = neon::vsub(neon::vmul(v2qf32_src10.val[0], vqf32_table_real), neon::vmul(v2qf32_src10.val[1], vqf32_table_imag));
                v2qf32_temp0.val[1] = neon::vadd(neon::vmul(v2qf32_src10.val[1], vqf32_table_real), neon::vmul(v2qf32_src10.val[0], vqf32_table_imag));
                v2qf32_temp1.val[0] = neon::vsub(neon::vmul(v2qf32_src11.val[0], vqf32_table_real), neon::vmul(v2qf32_src11.val[1], vqf32_table_imag));
                v2qf32_temp1.val[1] = neon::vadd(neon::vmul(v2qf32_src11.val[1], vqf32_table_real), neon::vmul(v2qf32_src11.val[0], vqf32_table_imag));
            }

            v2qf32_dst10.val[0] = neon::vmul(neon::vsub(v2qf32_src00.val[0], v2qf32_temp0.val[0]), vqf32_n);
            v2qf32_dst10.val[1] = neon::vmul(neon::vsub(v2qf32_src00.val[1], v2qf32_temp0.val[1]), vqf32_n);
            v2qf32_dst00.val[0] = neon::vmul(neon::vadd(v2qf32_src00.val[0], v2qf32_temp0.val[0]), vqf32_n);
            v2qf32_dst00.val[1] = neon::vmul(neon::vadd(v2qf32_src00.val[1], v2qf32_temp0.val[1]), vqf32_n);

            v2qf32_dst11.val[0] = neon::vmul(neon::vsub(v2qf32_src01.val[0], v2qf32_temp1.val[0]), vqf32_n);
            v2qf32_dst11.val[1] = neon::vmul(neon::vsub(v2qf32_src01.val[1], v2qf32_temp1.val[1]), vqf32_n);
            v2qf32_dst01.val[0] = neon::vmul(neon::vadd(v2qf32_src01.val[0], v2qf32_temp1.val[0]), vqf32_n);
            v2qf32_dst01.val[1] = neon::vmul(neon::vadd(v2qf32_src01.val[1], v2qf32_temp1.val[1]), vqf32_n);

            if (1 == C)
            {
                Tp *dst_data0 = dst_shift      + (width_offset + row_index);
                Tp *dst_data1 = dst_half_shift + (width_offset + row_index);

                float32x4x2_t v2qf32_real0, v2qf32_real1;
                v2qf32_real0.val[0] = v2qf32_dst00.val[0]; v2qf32_real0.val[1] = v2qf32_dst01.val[0];
                v2qf32_real1.val[0] = v2qf32_dst10.val[0]; v2qf32_real1.val[1] = v2qf32_dst11.val[0];

                StoreF32PairAs<Tp>(dst_data0, v2qf32_real0);
                StoreF32PairAs<Tp>(dst_data1, v2qf32_real1);
            }
            else
            {
                MI_F32 *dst_data0 = (MI_F32 *)(dst_shift      + (width_offset + row_index) * 2);
                MI_F32 *dst_data1 = (MI_F32 *)(dst_half_shift + (width_offset + row_index) * 2);

                neon::vstore(dst_data0,     v2qf32_dst00);
                neon::vstore(dst_data0 + 8, v2qf32_dst01);
                neon::vstore(dst_data1,     v2qf32_dst10);
                neon::vstore(dst_data1 + 8, v2qf32_dst11);
            }
        }
    }

    return Status::OK;
}

template <typename Tp, MI_S32 GRID_LEN, MI_S32 C>
static Status GridIDftNeonNxNImpl(Context *ctx, const Mat &src, Mat &dst, MI_BOOL with_scale, const OpTarget &target)
{
    AURA_UNUSED(target);
    const Sizes3 sz = dst.GetSizes();
    const MI_S32 width = sz.m_width;
    const MI_S32 height = sz.m_height;

    MI_U16 idx_table[GRID_LEN] = {0};
    GetReverseIndex(idx_table, GRID_LEN);

    std::complex<MI_F32> dft_row_exp_table[GRID_LEN];
    GetDftExpTable<1>(dft_row_exp_table, GRID_LEN);

    WorkerPool *wp = ctx->GetWorkerPool();
    if (MI_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get WorkerPool Failed.");
        return Status::ERROR;
    }

    auto row_process_func = [&](MI_S32 start, MI_S32 end)->Status
    {
        for (MI_S32 y = start * GRID_LEN; y < end * GRID_LEN; y += GRID_LEN)
        {
            for (MI_S32 x = 0; x < width; x += GRID_LEN)
            {
                GridIDftNeonNxN<Tp, GRID_LEN, C>(src, dst, idx_table, dft_row_exp_table, with_scale, x, y);
            }
        }
        return Status::OK;
    };

    if (wp->ParallelFor(0, height / GRID_LEN, row_process_func) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "GridDftNeonImpl ParallelFor failed.");
        return Status::ERROR;
    }
    // exit
    return Status::OK;
}

template <typename Tp, MI_S32 C>
static Status GridIDftNeonImpl(Context *ctx, const Mat &src, Mat &dst, MI_S32 grid_len, MI_BOOL with_scale, const OpTarget &target)
{
    Status ret = Status::OK;
    switch (grid_len)
    {
        case 4:
        {
            ret = GridIDft4x4NeonImpl<Tp, C>(ctx, src, dst, with_scale, target);
            break;
        }
        case 8:
        {
            ret = GridIDft8x8NeonImpl<Tp, C>(ctx, src, dst, with_scale, target);
            break;
        }
        case 16:
        {
            ret = GridIDftNeonNxNImpl<Tp, 16, C>(ctx, src, dst, with_scale, target);
            break;
        }
        case 32:
        {
            ret = GridIDftNeonNxNImpl<Tp, 32, C>(ctx, src, dst, with_scale, target);
            break;
        }
        default:
        {
            ret = Status::ERROR;
            AURA_ADD_ERROR_STRING(ctx, "unsupported grid length");
            break;
        }
    }

    AURA_RETURN(ctx, ret);
}

template <typename Tp>
static Status GridIDftNeonHelper(Context *ctx, const Mat &src, Mat &dst, MI_S32 grid_len, MI_BOOL with_scale, const OpTarget &target)
{
    Status ret = Status::ERROR;

    if (1 == dst.GetSizes().m_channel)
    {
        ret = GridIDftNeonImpl<Tp, 1>(ctx, src, dst, grid_len, with_scale, target);
    }
    else if (2 == dst.GetSizes().m_channel)
    {
        ret = GridIDftNeonImpl<Tp, 2>(ctx, src, dst, grid_len, with_scale, target);
    }

    AURA_RETURN(ctx, ret);
}

GridIDftNeon::GridIDftNeon(Context *ctx, const OpTarget &target) : GridIDftImpl(ctx, target)
{}

Status GridIDftNeon::SetArgs(const Array *src, Array *dst, MI_S32 grid_len, MI_BOOL with_scale)
{
    if (GridIDftImpl::SetArgs(src, dst, grid_len, with_scale) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "GridIDftImpl::SetArgs failed");
        return Status::ERROR;
    }

    if ((src->GetArrayType() != ArrayType::MAT) || (dst->GetArrayType() != ArrayType::MAT))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src dst must be mat type");
        return Status::ERROR;
    }

    MI_S32 dst_channel    = dst->GetSizes().m_channel;
    ElemType dst_elemtype = dst->GetElemType();

    if (1 == dst_channel)
    {
        if (ElemType::S32 == dst_elemtype || ElemType::U32 == dst_elemtype || ElemType::F64 == dst_elemtype)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "when dst channel is 1, current dst does not support MI_S32/MI_U32/MI_F64 type.");
            return Status::ERROR;
        }
    }
    else if (2 == dst_channel)
    {
        if (dst_elemtype != ElemType::F32)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "when dst channel is 2, current dst only support MI_F32 type.");
            return Status::ERROR;
        }
    }
    else
    {
        AURA_ADD_ERROR_STRING(m_ctx, "dst channel must be 1 or 2");
        return Status::ERROR;
    }

    return Status::OK;
}

Status GridIDftNeon::Run()
{
    const Mat *src = dynamic_cast<const Mat*>(m_src);
    Mat *dst       = dynamic_cast<Mat*>(m_dst);

    if ((MI_NULL == src) || (MI_NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src dst is null");
        return Status::ERROR;
    }

    Status ret = Status::OK;
    switch (dst->GetElemType())
    {
        case ElemType::U8:
        {
            ret = GridIDftNeonHelper<MI_U8>(m_ctx, *src, *dst, m_grid_len, m_with_scale, m_target);
            break;
        }
        case ElemType::S8:
        {
            ret = GridIDftNeonHelper<MI_S8>(m_ctx, *src, *dst, m_grid_len, m_with_scale, m_target);
            break;
        }
        case ElemType::U16:
        {
            ret = GridIDftNeonHelper<MI_U16>(m_ctx, *src, *dst, m_grid_len, m_with_scale, m_target);
            break;
        }
        case ElemType::S16:
        {
            ret = GridIDftNeonHelper<MI_S16>(m_ctx, *src, *dst, m_grid_len, m_with_scale, m_target);
            break;
        }
# if defined(AURA_ENABLE_NEON_FP16)
        case ElemType::F16:
        {
            ret = GridIDftNeonHelper<MI_F16>(m_ctx, *src, *dst, m_grid_len, m_with_scale, m_target);
            break;
        }
# endif
        case ElemType::F32:
        {
            ret = GridIDftNeonHelper<MI_F32>(m_ctx, *src, *dst, m_grid_len, m_with_scale, m_target);
            break;
        }
        default:
        {
            ret = Status::ERROR;
            AURA_ADD_ERROR_STRING(m_ctx, "unsupported element type");
            break;
        }
    }

    AURA_RETURN(m_ctx, ret);
}

} // aura