#include "resize_impl.hpp"
#include "aura/runtime/logger.h"
#include "aura/runtime/worker_pool.h"

namespace aura
{

#define  HALF_AURA_HVLEN    64
#define  QUAR_AURA_HVLEN    32

struct CubicBufStruct
{
    MI_U8 *xofs;
    MI_U8 *yofs;
    MI_U8 *alpha0;
    MI_U8 *alpha1;
    MI_U8 *alpha2;
    MI_U8 *alpha3;
    MI_U8 *beta;
    MI_U8 *row;
    MI_U8 *vtcm;
    MI_U8 *l_gather_base;
    MI_U8 *r_gather_base;
};

template<typename Tp>
static typename std::enable_if<std::is_same<MI_U8, Tp>::value || std::is_same<MI_S8, Tp>::value, AURA_VOID>::type
GetCuOffset(CubicBufStruct *cubic_buf_struct, MI_S32 iwidth, MI_S32 owidth, MI_S32 iheight, MI_S32 oheight)
{
    MI_F64 scale_x = static_cast<MI_F64>(iwidth) / owidth;
    MI_F64 scale_y = static_cast<MI_F64>(iheight) / oheight;
    MI_U16 *xofs   = reinterpret_cast<MI_U16 *>(cubic_buf_struct->xofs);
    MI_U16 *yofs   = reinterpret_cast<MI_U16 *>(cubic_buf_struct->yofs);
    MI_S16 *alpha0 = reinterpret_cast<MI_S16 *>(cubic_buf_struct->alpha0);
    MI_S16 *alpha1 = reinterpret_cast<MI_S16 *>(cubic_buf_struct->alpha1);
    MI_S16 *alpha2 = reinterpret_cast<MI_S16 *>(cubic_buf_struct->alpha2);
    MI_S16 *alpha3 = reinterpret_cast<MI_S16 *>(cubic_buf_struct->alpha3);
    MI_S16 *beta   = reinterpret_cast<MI_S16 *>(cubic_buf_struct->beta);

    MI_F32 fx[4], fy[4];
    MI_S32 coe_x[4], coe_y[4];
    MI_S32 sx, sy;

    for (MI_S32 dx = 0; dx < owidth; dx++)
    {
        fx[0] = static_cast<MI_F32>(((dx + 0.5) * scale_x - 0.5));
        sx    = static_cast<MI_S32>(Floor(fx[0])) - 1;

        fx[0] -= sx;
        fx[1] = fx[0] - 1.0f;
        fx[2] = 2.0f - fx[0];

        fx[0] = GetCuOffsetCore(fx[0]);
        fx[1] = GetCuOffsetCore(fx[1]);
        fx[2] = GetCuOffsetCore(fx[2]);
        fx[3] = 1.0f - fx[0] - fx[1] - fx[2];

        coe_x[0] = SaturateCast<MI_S32>(fx[0] * 2048.0);
        coe_x[1] = SaturateCast<MI_S32>(fx[1] * 2048.0);
        coe_x[2] = SaturateCast<MI_S32>(fx[2] * 2048.0);
        coe_x[3] = SaturateCast<MI_S32>(fx[3] * 2048.0);

        if (sx >= 0 && sx <= (iwidth - 4))
        {
            xofs[dx]   = sx << 2;
            alpha0[dx] = static_cast<MI_S16>(coe_x[0]);
            alpha1[dx] = static_cast<MI_S16>(coe_x[1]);
            alpha2[dx] = static_cast<MI_S16>(coe_x[2]);
            alpha3[dx] = static_cast<MI_S16>(coe_x[3]);
        }
        else if ((-2) == sx)
        {
            xofs[dx]   = 0;
            alpha0[dx] = static_cast<MI_S16>((coe_x[0] + coe_x[1] + coe_x[2]));
            alpha1[dx] = static_cast<MI_S16>(coe_x[3]);
            alpha2[dx] = 0;
            alpha3[dx] = 0;
        }
        else if ((-1) == sx)
        {
            xofs[dx]   = 0;
            alpha0[dx] = static_cast<MI_S16>((coe_x[0] + coe_x[1]));
            alpha1[dx] = static_cast<MI_S16>(coe_x[2]);
            alpha2[dx] = static_cast<MI_S16>(coe_x[3]);
            alpha3[dx] = 0;
        }
        else if ((iwidth - 3) == sx)
        {
            xofs[dx]   = (iwidth - 4) << 2;
            alpha0[dx] = 0;
            alpha1[dx] = coe_x[0];
            alpha2[dx] = coe_x[1];
            alpha3[dx] = coe_x[2] + coe_x[3];
        }
        else if ((iwidth - 2) == sx)
        {
            xofs[dx]   = (iwidth - 4) << 2;
            alpha0[dx] = 0;
            alpha1[dx] = 0;
            alpha2[dx] = coe_x[0];
            alpha3[dx] = coe_x[1] + coe_x[2] + coe_x[3];
        }
    }

    for (MI_S32 dy = 0; dy < oheight; dy++)
    {
        fy[0] = static_cast<MI_F32>(((dy + 0.5) * scale_y - 0.5));
        sy    = static_cast<MI_S32>(Floor(fy[0]) - 1);

        fy[0] -= sy;
        fy[1] = fy[0] - 1.0f;
        fy[2] = 2.0f - fy[0];
        fy[3] = 3.0f - fy[0];

        fy[0] = GetCuOffsetCore(fy[0]);
        fy[1] = GetCuOffsetCore(fy[1]);
        fy[2] = GetCuOffsetCore(fy[2]);
        fy[3] = 1.0f - fy[0] - fy[1] - fy[2];

        coe_y[0] = SaturateCast<MI_S32>(fy[0] * 2048.0);
        coe_y[1] = SaturateCast<MI_S32>(fy[1] * 2048.0);
        coe_y[2] = SaturateCast<MI_S32>(fy[2] * 2048.0);
        coe_y[3] = SaturateCast<MI_S32>(fy[3] * 2048.0);

        if (sy >= 0 && sy <= (iheight - 4))
        {
            yofs[dy]            = static_cast<MI_U16>(sy);
            beta[(dy << 2)]     = static_cast<MI_S16>(coe_y[0]);
            beta[(dy << 2) + 1] = static_cast<MI_S16>(coe_y[1]);
            beta[(dy << 2) + 2] = static_cast<MI_S16>(coe_y[2]);
            beta[(dy << 2) + 3] = static_cast<MI_S16>(coe_y[3]);
        }
        else if ((-2) == sy)
        {
            yofs[dy]            = 0;
            beta[(dy << 2)]     = static_cast<MI_S16>((coe_y[0] + coe_y[1] + coe_y[2]));
            beta[(dy << 2) + 1] = static_cast<MI_S16>(coe_y[3]);
            beta[(dy << 2) + 2] = 0;
            beta[(dy << 2) + 3] = 0;
        }
        else if ((-1) == sy)
        {
            yofs[dy]            = 0;
            beta[(dy << 2)]     = static_cast<MI_S16>((coe_y[0] + coe_y[1]));
            beta[(dy << 2) + 1] = static_cast<MI_S16>(coe_y[2]);
            beta[(dy << 2) + 2] = static_cast<MI_S16>(coe_y[3]);
            beta[(dy << 2) + 3] = 0;
        }
        else if ((iheight - 3) == sy)
        {
            yofs[dy]            = static_cast<MI_U16>(iheight - 4);
            beta[(dy << 2)]     = 0;
            beta[(dy << 2) + 1] = static_cast<MI_S16>(coe_y[0]);
            beta[(dy << 2) + 2] = static_cast<MI_S16>(coe_y[1]);
            beta[(dy << 2) + 3] = static_cast<MI_S16>(coe_y[2] + coe_y[3]);
        }
        else if ((iheight - 2) == sy)
        {
            yofs[dy]            = static_cast<MI_U16>(iheight - 4);
            beta[(dy << 2)]     = 0;
            beta[(dy << 2) + 1] = 0;
            beta[(dy << 2) + 2] = static_cast<MI_S16>(coe_y[0]);
            beta[(dy << 2) + 3] = static_cast<MI_S16>(coe_y[1] + coe_y[2] + coe_y[3]);
        }
    }
}

template<typename Tp>
static typename std::enable_if<std::is_same<MI_U16, Tp>::value || std::is_same<MI_S16, Tp>::value, AURA_VOID>::type
GetCuOffset(CubicBufStruct *cubic_buf_struct, MI_S32 iwidth, MI_S32 owidth, MI_S32 iheight, MI_S32 oheight)
{
    MI_F64 scale_x = static_cast<MI_F64>(iwidth) / owidth;
    MI_F64 scale_y = static_cast<MI_F64>(iheight) / oheight;
    MI_U16 *xofs   = reinterpret_cast<MI_U16 *>(cubic_buf_struct->xofs);
    MI_U16 *yofs   = reinterpret_cast<MI_U16 *>(cubic_buf_struct->yofs);
    MI_F32 *alpha0 = reinterpret_cast<MI_F32 *>(cubic_buf_struct->alpha0);
    MI_F32 *alpha1 = reinterpret_cast<MI_F32 *>(cubic_buf_struct->alpha1);
    MI_F32 *alpha2 = reinterpret_cast<MI_F32 *>(cubic_buf_struct->alpha2);
    MI_F32 *alpha3 = reinterpret_cast<MI_F32 *>(cubic_buf_struct->alpha3);
    MI_F32 *beta   = reinterpret_cast<MI_F32 *>(cubic_buf_struct->beta);

    MI_F32 fx[4], fy[4];
    MI_S32 sx, sy;

    for (MI_S32 dx = 0; dx < owidth; dx++)
    {
        fx[0] = static_cast<MI_F32>(((dx + 0.5) * scale_x - 0.5));
        sx    = static_cast<MI_S32>(Floor(fx[0])) - 1;

        fx[0] -= sx;
        fx[1] = fx[0] - 1.0f;
        fx[2] = 2.0f - fx[0];

        fx[0] = GetCuOffsetCore(fx[0]);
        fx[1] = GetCuOffsetCore(fx[1]);
        fx[2] = GetCuOffsetCore(fx[2]);
        fx[3] = 1.0f - fx[0] - fx[1] - fx[2];

        if (sx >= 0 && sx <= (iwidth - 4))
        {
            xofs[dx]   = sx << 2;
            alpha0[dx] = fx[0];
            alpha1[dx] = fx[1];
            alpha2[dx] = fx[2];
            alpha3[dx] = fx[3];
        }
        else if ((-2) == sx)
        {
            xofs[dx]   = 0;
            alpha0[dx] = (fx[0] + fx[1] + fx[2]);
            alpha1[dx] = fx[3];
            alpha2[dx] = 0;
            alpha3[dx] = 0;
        }
        else if ((-1) == sx)
        {
            xofs[dx]   = 0;
            alpha0[dx] = (fx[0] + fx[1]);
            alpha1[dx] = fx[2];
            alpha2[dx] = fx[3];
            alpha3[dx] = 0;
        }
        else if ((iwidth - 3) == sx)
        {
            xofs[dx]   = (iwidth - 4) << 2;
            alpha0[dx] = 0;
            alpha1[dx] = fx[0];
            alpha2[dx] = fx[1];
            alpha3[dx] = fx[2] + fx[3];
        }
        else if ((iwidth - 2) == sx)
        {
            xofs[dx]   = (iwidth - 4) << 2;
            alpha0[dx] = 0;
            alpha1[dx] = 0;
            alpha2[dx] = fx[0];
            alpha3[dx] = fx[1] + fx[2] + fx[3];
        }
    }

    for (MI_S32 dy = 0; dy < oheight; dy++)
    {
        fy[0] = static_cast<MI_F32>(((dy + 0.5) * scale_y - 0.5));
        sy    = static_cast<MI_S32>(Floor(fy[0]) - 1);

        fy[0] -= sy;
        fy[1] = fy[0] - 1.0f;
        fy[2] = 2.0f - fy[0];
        fy[3] = 3.0f - fy[0];

        fy[0] = GetCuOffsetCore(fy[0]);
        fy[1] = GetCuOffsetCore(fy[1]);
        fy[2] = GetCuOffsetCore(fy[2]);
        fy[3] = 1.0f - fy[0] - fy[1] - fy[2];

        if (sy >= 0 && sy <= (iheight - 4))
        {
            yofs[dy]            = static_cast<MI_U16>(sy);
            beta[(dy << 2)]     = fy[0];
            beta[(dy << 2) + 1] = fy[1];
            beta[(dy << 2) + 2] = fy[2];
            beta[(dy << 2) + 3] = fy[3];
        }
        else if ((-2) == sy)
        {
            yofs[dy]            = 0;
            beta[(dy << 2)]     = (fy[0] + fy[1] + fy[2]);
            beta[(dy << 2) + 1] = fy[3];
            beta[(dy << 2) + 2] = 0;
            beta[(dy << 2) + 3] = 0;
        }
        else if ((-1) == sy)
        {
            yofs[dy]            = 0;
            beta[(dy << 2)]     = (fy[0] + fy[1]);
            beta[(dy << 2) + 1] = fy[2];
            beta[(dy << 2) + 2] = fy[3];
            beta[(dy << 2) + 3] = 0;
        }
        else if ((iheight - 3) == sy)
        {
            yofs[dy]            = static_cast<MI_U16>(iheight - 4);
            beta[(dy << 2)]     = 0;
            beta[(dy << 2) + 1] = fy[0];
            beta[(dy << 2) + 2] = fy[1];
            beta[(dy << 2) + 3] = fy[2] + fy[3];
        }
        else if ((iheight - 2) == sy)
        {
            yofs[dy]            = static_cast<MI_U16>(iheight - 4);
            beta[(dy << 2)]     = 0;
            beta[(dy << 2) + 1] = 0;
            beta[(dy << 2) + 2] = fy[0];
            beta[(dy << 2) + 3] = fy[1] + fy[2] + fy[3];
        }
    }
}

// Tp = MI_U8
template<typename Tp, MI_S32 C>
static typename std::enable_if<std::is_same<MI_U8, Tp>::value, AURA_VOID>::type
ResizeCuCommPerRow(const Tp *src_c, Tp *dst, MI_S32 *row_base, MI_S32 *vtcm_base, MI_U16 *xofs, MI_S16 *beta,
                   MI_S16 *alpha0, MI_S16 *alpha1, MI_S16 *alpha2, MI_S16 *alpha3, AURA_VOID *l_gather_base,
                   AURA_VOID *r_gather_base, MI_S32 iwidth, MI_S32 istride, MI_S32 istep, MI_S32 ostep)
{
    using MVType = typename MVHvxVector<C>::Type;
    MI_S32 *row  = reinterpret_cast<MI_S32 *>(row_base);
    constexpr MI_S32 elem_counts = AURA_HVLEN * C / sizeof(Tp);
    constexpr MI_S32 elem_counts_channel = elem_counts / C;
    MI_S32 iwidth_align = istep / elem_counts * elem_counts;

    // part1: calculate the median and save to vtcm
    MI_S32 i = 0, j = 0;
    {
        const Tp *src_n0   = src_c + istride;
        const Tp *src_n1   = src_n0 + istride;
        const Tp *src_n2   = src_n1 + istride;
        HVX_Vector v_beta0 = Q6_Vh_vsplat_R(beta[0]);
        HVX_Vector v_beta1 = Q6_Vh_vsplat_R(beta[1]);
        HVX_Vector v_beta2 = Q6_Vh_vsplat_R(beta[2]);
        HVX_Vector v_beta3 = Q6_Vh_vsplat_R(beta[3]);

        for (; i < iwidth_align; i += elem_counts, j += elem_counts_channel)
        {
            MVType mv_c_src, mv_n0_src, mv_n1_src, mv_n2_src;
            vload(src_c + i,  mv_c_src);
            vload(src_n0 + i, mv_n0_src);
            vload(src_n1 + i, mv_n1_src);
            vload(src_n2 + i, mv_n2_src);

            for (MI_S32 ch = 0; ch < C; ch++)
            {
                HVX_VectorPair w_c_src   = Q6_Wuh_vunpack_Vub(Q6_Vb_vshuff_Vb(mv_c_src.val[ch]));
                HVX_VectorPair w_n0_src  = Q6_Wuh_vunpack_Vub(Q6_Vb_vshuff_Vb(mv_n0_src.val[ch]));
                HVX_VectorPair w_n1_src  = Q6_Wuh_vunpack_Vub(Q6_Vb_vshuff_Vb(mv_n1_src.val[ch]));
                HVX_VectorPair w_n2_src  = Q6_Wuh_vunpack_Vub(Q6_Vb_vshuff_Vb(mv_n2_src.val[ch]));
                HVX_VectorPair w_result0 = Q6_Ww_vmpy_VhVh(Q6_V_lo_W(w_c_src), v_beta0);  // lo: 0  1  2  ... 31 hi: 64 65 66  ... 95
                HVX_VectorPair w_result1 = Q6_Ww_vmpy_VhVh(Q6_V_hi_W(w_c_src), v_beta0);  // lo:32 33 34  ... 63 hi: 96 97 98  ... 127
                HVX_VectorPair w_result2 = Q6_Ww_vmpy_VhVh(Q6_V_lo_W(w_n1_src), v_beta2);
                HVX_VectorPair w_result3 = Q6_Ww_vmpy_VhVh(Q6_V_hi_W(w_n1_src), v_beta2);

                w_result0 = Q6_Ww_vmpyacc_WwVhVh(w_result0, Q6_V_lo_W(w_n0_src), v_beta1);
                w_result1 = Q6_Ww_vmpyacc_WwVhVh(w_result1, Q6_V_hi_W(w_n0_src), v_beta1);
                w_result2 = Q6_Ww_vmpyacc_WwVhVh(w_result2, Q6_V_lo_W(w_n2_src), v_beta3);
                w_result3 = Q6_Ww_vmpyacc_WwVhVh(w_result3, Q6_V_hi_W(w_n2_src), v_beta3);
                w_result0 = Q6_Ww_vadd_WwWw(w_result0, w_result2);
                w_result1 = Q6_Ww_vadd_WwWw(w_result1, w_result3);

                HVX_Vector *v_dst = (HVX_Vector *)(row + j + ch * AURA_ALIGN(iwidth, QUAR_AURA_HVLEN));
                *v_dst++ = Q6_V_lo_W(w_result0);
                *v_dst++ = Q6_V_lo_W(w_result1);
                *v_dst++ = Q6_V_hi_W(w_result0);
                *v_dst   = Q6_V_hi_W(w_result1);
            }
        }

        if (istep > iwidth_align)
        {
            MVType mv_c_src, mv_n0_src, mv_n1_src, mv_n2_src;
            i = istep - elem_counts;
            j = i / C;

            vload(src_c + i,  mv_c_src);
            vload(src_n0 + i, mv_n0_src);
            vload(src_n1 + i, mv_n1_src);
            vload(src_n2 + i, mv_n2_src);

            for (MI_S32 ch = 0; ch < C; ch++)
            {
                MI_S32 *row_tmp          = row + j + ch * AURA_ALIGN(iwidth, QUAR_AURA_HVLEN);
                HVX_VectorPair w_c_src   = Q6_Wuh_vunpack_Vub(Q6_Vb_vshuff_Vb(mv_c_src.val[ch]));
                HVX_VectorPair w_n0_src  = Q6_Wuh_vunpack_Vub(Q6_Vb_vshuff_Vb(mv_n0_src.val[ch]));
                HVX_VectorPair w_n1_src  = Q6_Wuh_vunpack_Vub(Q6_Vb_vshuff_Vb(mv_n1_src.val[ch]));
                HVX_VectorPair w_n2_src  = Q6_Wuh_vunpack_Vub(Q6_Vb_vshuff_Vb(mv_n2_src.val[ch]));
                HVX_VectorPair w_result0 = Q6_Ww_vmpy_VhVh(Q6_V_lo_W(w_c_src), v_beta0);  // lo: 0  1  2  ... 31 hi: 64 65 66  ... 95
                HVX_VectorPair w_result1 = Q6_Ww_vmpy_VhVh(Q6_V_hi_W(w_c_src), v_beta0);  // lo:32 33 34  ... 63 hi: 96 97 98  ... 127
                HVX_VectorPair w_result2 = Q6_Ww_vmpy_VhVh(Q6_V_lo_W(w_n1_src), v_beta2);
                HVX_VectorPair w_result3 = Q6_Ww_vmpy_VhVh(Q6_V_hi_W(w_n1_src), v_beta2);

                w_result0 = Q6_Ww_vmpyacc_WwVhVh(w_result0, Q6_V_lo_W(w_n0_src), v_beta1);
                w_result1 = Q6_Ww_vmpyacc_WwVhVh(w_result1, Q6_V_hi_W(w_n0_src), v_beta1);
                w_result2 = Q6_Ww_vmpyacc_WwVhVh(w_result2, Q6_V_lo_W(w_n2_src), v_beta3);
                w_result3 = Q6_Ww_vmpyacc_WwVhVh(w_result3, Q6_V_hi_W(w_n2_src), v_beta3);
                w_result0 = Q6_Ww_vadd_WwWw(w_result0, w_result2);
                w_result1 = Q6_Ww_vadd_WwWw(w_result1, w_result3);

                vmemu(row_tmp) = Q6_V_lo_W(w_result0);
                vmemu(row_tmp + QUAR_AURA_HVLEN) = Q6_V_lo_W(w_result1);
                vmemu(row_tmp + HALF_AURA_HVLEN) = Q6_V_hi_W(w_result0);
                vmemu(row_tmp + QUAR_AURA_HVLEN + HALF_AURA_HVLEN) = Q6_V_hi_W(w_result1);
            }
        }
    }

    // part2: load vtcm and calculate dst
    {
        MI_S32 *vtcm = reinterpret_cast<MI_S32 *>(vtcm_base);
        MI_S32 owidth_align = ostep / elem_counts * elem_counts;

        HVX_Vector *v_ofs     = (HVX_Vector *)xofs;
        HVX_Vector *v_alpha0  = (HVX_Vector *)alpha0; HVX_Vector *v_alpha1 = (HVX_Vector *)alpha1;
        HVX_Vector *v_alpha2  = (HVX_Vector *)alpha2; HVX_Vector *v_alpha3 = (HVX_Vector *)alpha3;
        HVX_Vector **l_gather = (HVX_Vector **)l_gather_base;
        HVX_Vector **r_gather = (HVX_Vector **)r_gather_base;
        HVX_Vector *v_tmp[4];
        MI_S32 *row_ch[C];

        {
            HVX_Vector v_offset_c  = *v_ofs++;
            HVX_Vector v_offset_r0 = *v_ofs++;
            HVX_VectorPair w_result0 = Q6_Wuw_vunpack_Vuh(v_offset_c);
            HVX_VectorPair w_result1 = Q6_Wuw_vunpack_Vuh(v_offset_r0);
            HVX_Vector v_offset_l1 = Q6_V_lo_W(w_result0);
            HVX_Vector v_offset_l0 = Q6_V_hi_W(w_result0);
            v_offset_c  = Q6_V_lo_W(w_result1);
            v_offset_r0 = Q6_V_hi_W(w_result1);

            HVX_Vector v_offset_add1_l1 = Q6_Vuw_vadd_VuwVuw_sat(v_offset_l1, Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add1_l0 = Q6_Vuw_vadd_VuwVuw_sat(v_offset_l0, Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add1_c  = Q6_Vuw_vadd_VuwVuw_sat(v_offset_c , Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add1_r0 = Q6_Vuw_vadd_VuwVuw_sat(v_offset_r0, Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add2_l1 = Q6_Vuw_vadd_VuwVuw_sat(v_offset_add1_l1, Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add2_l0 = Q6_Vuw_vadd_VuwVuw_sat(v_offset_add1_l0, Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add2_c  = Q6_Vuw_vadd_VuwVuw_sat(v_offset_add1_c , Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add2_r0 = Q6_Vuw_vadd_VuwVuw_sat(v_offset_add1_r0, Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add3_l1 = Q6_Vuw_vadd_VuwVuw_sat(v_offset_add2_l1, Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add3_l0 = Q6_Vuw_vadd_VuwVuw_sat(v_offset_add2_l0, Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add3_c  = Q6_Vuw_vadd_VuwVuw_sat(v_offset_add2_c , Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add3_r0 = Q6_Vuw_vadd_VuwVuw_sat(v_offset_add2_r0, Q6_V_vsplat_R(4));

            for (MI_S32 ch = 0; ch < C; ch++)
            {
                MI_S32 *vtcm_gather = vtcm + ch * (QUAR_AURA_HVLEN << 5);
                row_ch[ch] = row + ch * AURA_ALIGN(iwidth, QUAR_AURA_HVLEN);

                l_gather[ch * 16 + 0]  = (HVX_Vector *)(vtcm_gather);
                l_gather[ch * 16 + 1]  = (HVX_Vector *)(vtcm_gather + QUAR_AURA_HVLEN);
                l_gather[ch * 16 + 2]  = (HVX_Vector *)(vtcm_gather + HALF_AURA_HVLEN);
                l_gather[ch * 16 + 3]  = (HVX_Vector *)(vtcm_gather + QUAR_AURA_HVLEN * 3);
                l_gather[ch * 16 + 4]  = (HVX_Vector *)(vtcm_gather + AURA_HVLEN);
                l_gather[ch * 16 + 5]  = (HVX_Vector *)(vtcm_gather + QUAR_AURA_HVLEN * 5);
                l_gather[ch * 16 + 6]  = (HVX_Vector *)(vtcm_gather + QUAR_AURA_HVLEN * 6);
                l_gather[ch * 16 + 7]  = (HVX_Vector *)(vtcm_gather + QUAR_AURA_HVLEN * 7);
                l_gather[ch * 16 + 8]  = (HVX_Vector *)(vtcm_gather + (AURA_HVLEN << 1));
                l_gather[ch * 16 + 9]  = (HVX_Vector *)(vtcm_gather + QUAR_AURA_HVLEN * 9);
                l_gather[ch * 16 + 10] = (HVX_Vector *)(vtcm_gather + QUAR_AURA_HVLEN * 10);
                l_gather[ch * 16 + 11] = (HVX_Vector *)(vtcm_gather + QUAR_AURA_HVLEN * 11);
                l_gather[ch * 16 + 12] = (HVX_Vector *)(vtcm_gather + QUAR_AURA_HVLEN * 12);
                l_gather[ch * 16 + 13] = (HVX_Vector *)(vtcm_gather + QUAR_AURA_HVLEN * 13);
                l_gather[ch * 16 + 14] = (HVX_Vector *)(vtcm_gather + QUAR_AURA_HVLEN * 14);
                l_gather[ch * 16 + 15] = (HVX_Vector *)(vtcm_gather + QUAR_AURA_HVLEN * 15);
                r_gather[ch * 16 + 0]  = (HVX_Vector *)(vtcm_gather + (AURA_HVLEN << 2));
                r_gather[ch * 16 + 1]  = (HVX_Vector *)(vtcm_gather + QUAR_AURA_HVLEN * 17);
                r_gather[ch * 16 + 2]  = (HVX_Vector *)(vtcm_gather + QUAR_AURA_HVLEN * 18);
                r_gather[ch * 16 + 3]  = (HVX_Vector *)(vtcm_gather + QUAR_AURA_HVLEN * 19);
                r_gather[ch * 16 + 4]  = (HVX_Vector *)(vtcm_gather + QUAR_AURA_HVLEN * 20);
                r_gather[ch * 16 + 5]  = (HVX_Vector *)(vtcm_gather + QUAR_AURA_HVLEN * 21);
                r_gather[ch * 16 + 6]  = (HVX_Vector *)(vtcm_gather + QUAR_AURA_HVLEN * 22);
                r_gather[ch * 16 + 7]  = (HVX_Vector *)(vtcm_gather + QUAR_AURA_HVLEN * 23);
                r_gather[ch * 16 + 8]  = (HVX_Vector *)(vtcm_gather + QUAR_AURA_HVLEN * 24);
                r_gather[ch * 16 + 9]  = (HVX_Vector *)(vtcm_gather + QUAR_AURA_HVLEN * 25);
                r_gather[ch * 16 + 10] = (HVX_Vector *)(vtcm_gather + QUAR_AURA_HVLEN * 26);
                r_gather[ch * 16 + 11] = (HVX_Vector *)(vtcm_gather + QUAR_AURA_HVLEN * 27);
                r_gather[ch * 16 + 12] = (HVX_Vector *)(vtcm_gather + QUAR_AURA_HVLEN * 28);
                r_gather[ch * 16 + 13] = (HVX_Vector *)(vtcm_gather + QUAR_AURA_HVLEN * 29);
                r_gather[ch * 16 + 14] = (HVX_Vector *)(vtcm_gather + QUAR_AURA_HVLEN * 30);
                r_gather[ch * 16 + 15] = (HVX_Vector *)(vtcm_gather + QUAR_AURA_HVLEN * 31);

                Q6_vgather_ARMVw(l_gather[ch * 16 + 0] , (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_l1);
                Q6_vgather_ARMVw(l_gather[ch * 16 + 1] , (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_l0);
                Q6_vgather_ARMVw(l_gather[ch * 16 + 2] , (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_c);
                Q6_vgather_ARMVw(l_gather[ch * 16 + 3] , (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_r0);
                Q6_vgather_ARMVw(l_gather[ch * 16 + 4] , (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add1_l1);
                Q6_vgather_ARMVw(l_gather[ch * 16 + 5] , (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add1_l0);
                Q6_vgather_ARMVw(l_gather[ch * 16 + 6] , (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add1_c);
                Q6_vgather_ARMVw(l_gather[ch * 16 + 7] , (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add1_r0);
                Q6_vgather_ARMVw(l_gather[ch * 16 + 8] , (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add2_l1);
                Q6_vgather_ARMVw(l_gather[ch * 16 + 9] , (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add2_l0);
                Q6_vgather_ARMVw(l_gather[ch * 16 + 10], (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add2_c);
                Q6_vgather_ARMVw(l_gather[ch * 16 + 11], (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add2_r0);
                Q6_vgather_ARMVw(l_gather[ch * 16 + 12], (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add3_l1);
                Q6_vgather_ARMVw(l_gather[ch * 16 + 13], (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add3_l0);
                Q6_vgather_ARMVw(l_gather[ch * 16 + 14], (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add3_c);
                Q6_vgather_ARMVw(l_gather[ch * 16 + 15], (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add3_r0);
            }
        }

        i = 0;
        for (; i < owidth_align - elem_counts; i += elem_counts)
        {
            HVX_Vector v_offset_c  = *v_ofs++;
            HVX_Vector v_offset_r0 = *v_ofs++;
            HVX_Vector v_alpha0_c  = *v_alpha0++;
            HVX_Vector v_alpha0_r0 = *v_alpha0++;
            HVX_Vector v_alpha1_c  = *v_alpha1++;
            HVX_Vector v_alpha1_r0 = *v_alpha1++;
            HVX_Vector v_alpha2_c  = *v_alpha2++;
            HVX_Vector v_alpha2_r0 = *v_alpha2++;
            HVX_Vector v_alpha3_c  = *v_alpha3++;
            HVX_Vector v_alpha3_r0 = *v_alpha3++;
            HVX_VectorPair w_result0 = Q6_Wuw_vunpack_Vuh(v_offset_c);
            HVX_VectorPair w_result1 = Q6_Wuw_vunpack_Vuh(v_offset_r0);
            HVX_Vector v_offset_l1 = Q6_V_lo_W(w_result0);
            HVX_Vector v_offset_l0 = Q6_V_hi_W(w_result0);
            v_offset_c  = Q6_V_lo_W(w_result1);
            v_offset_r0 = Q6_V_hi_W(w_result1);

            HVX_Vector v_offset_add1_l1 = Q6_Vuw_vadd_VuwVuw_sat(v_offset_l1, Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add1_l0 = Q6_Vuw_vadd_VuwVuw_sat(v_offset_l0, Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add1_c  = Q6_Vuw_vadd_VuwVuw_sat(v_offset_c , Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add1_r0 = Q6_Vuw_vadd_VuwVuw_sat(v_offset_r0, Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add2_l1 = Q6_Vuw_vadd_VuwVuw_sat(v_offset_add1_l1, Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add2_l0 = Q6_Vuw_vadd_VuwVuw_sat(v_offset_add1_l0, Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add2_c  = Q6_Vuw_vadd_VuwVuw_sat(v_offset_add1_c , Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add2_r0 = Q6_Vuw_vadd_VuwVuw_sat(v_offset_add1_r0, Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add3_l1 = Q6_Vuw_vadd_VuwVuw_sat(v_offset_add2_l1, Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add3_l0 = Q6_Vuw_vadd_VuwVuw_sat(v_offset_add2_l0, Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add3_c  = Q6_Vuw_vadd_VuwVuw_sat(v_offset_add2_c , Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add3_r0 = Q6_Vuw_vadd_VuwVuw_sat(v_offset_add2_r0, Q6_V_vsplat_R(4));

            w_result0 = Q6_Ww_vunpack_Vh(v_alpha0_c);
            w_result1 = Q6_Ww_vunpack_Vh(v_alpha0_r0);
            HVX_Vector v_alpha0_l1 = Q6_V_lo_W(w_result0);
            HVX_Vector v_alpha0_l0 = Q6_V_hi_W(w_result0);
            v_alpha0_c  = Q6_V_lo_W(w_result1);
            v_alpha0_r0 = Q6_V_hi_W(w_result1);
            w_result0 = Q6_Ww_vunpack_Vh(v_alpha1_c);
            w_result1 = Q6_Ww_vunpack_Vh(v_alpha1_r0);
            HVX_Vector v_alpha1_l1 = Q6_V_lo_W(w_result0);
            HVX_Vector v_alpha1_l0 = Q6_V_hi_W(w_result0);
            v_alpha1_c  = Q6_V_lo_W(w_result1);
            v_alpha1_r0 = Q6_V_hi_W(w_result1);
            w_result0 = Q6_Ww_vunpack_Vh(v_alpha2_c);
            w_result1 = Q6_Ww_vunpack_Vh(v_alpha2_r0);
            HVX_Vector v_alpha2_l1 = Q6_V_lo_W(w_result0);
            HVX_Vector v_alpha2_l0 = Q6_V_hi_W(w_result0);
            v_alpha2_c  = Q6_V_lo_W(w_result1);
            v_alpha2_r0 = Q6_V_hi_W(w_result1);
            w_result0 = Q6_Ww_vunpack_Vh(v_alpha3_c);
            w_result1 = Q6_Ww_vunpack_Vh(v_alpha3_r0);
            HVX_Vector v_alpha3_l1 = Q6_V_lo_W(w_result0);
            HVX_Vector v_alpha3_l0 = Q6_V_hi_W(w_result0);
            v_alpha3_c  = Q6_V_lo_W(w_result1);
            v_alpha3_r0 = Q6_V_hi_W(w_result1);

            MVType mv_result;
            for (MI_S32 ch = 0; ch < C; ch++)
            {
                Q6_vgather_ARMVw(r_gather[ch * 16 + 0] , (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_l1);
                Q6_vgather_ARMVw(r_gather[ch * 16 + 1] , (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_l0);
                Q6_vgather_ARMVw(r_gather[ch * 16 + 2] , (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_c);
                Q6_vgather_ARMVw(r_gather[ch * 16 + 3] , (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_r0);
                Q6_vgather_ARMVw(r_gather[ch * 16 + 4] , (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add1_l1);
                Q6_vgather_ARMVw(r_gather[ch * 16 + 5] , (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add1_l0);
                Q6_vgather_ARMVw(r_gather[ch * 16 + 6] , (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add1_c);
                Q6_vgather_ARMVw(r_gather[ch * 16 + 7] , (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add1_r0);
                Q6_vgather_ARMVw(r_gather[ch * 16 + 8] , (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add2_l1);
                Q6_vgather_ARMVw(r_gather[ch * 16 + 9] , (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add2_l0);
                Q6_vgather_ARMVw(r_gather[ch * 16 + 10], (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add2_c);
                Q6_vgather_ARMVw(r_gather[ch * 16 + 11], (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add2_r0);
                Q6_vgather_ARMVw(r_gather[ch * 16 + 12], (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add3_l1);
                Q6_vgather_ARMVw(r_gather[ch * 16 + 13], (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add3_l0);
                Q6_vgather_ARMVw(r_gather[ch * 16 + 14], (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add3_c);
                Q6_vgather_ARMVw(r_gather[ch * 16 + 15], (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add3_r0);

                w_result0 = Q6_Wd_vmul_VwVw(*l_gather[ch * 16 + 0], v_alpha0_l1);
                w_result1 = Q6_Wd_vmul_VwVw(*l_gather[ch * 16 + 1], v_alpha0_l0);
                w_result0 = Q6_Wd_vmulacc_WdVwVw(w_result0, *l_gather[ch * 16 + 4], v_alpha1_l1);
                w_result1 = Q6_Wd_vmulacc_WdVwVw(w_result1, *l_gather[ch * 16 + 5], v_alpha1_l0);
                w_result0 = Q6_Wd_vmulacc_WdVwVw(w_result0, *l_gather[ch * 16 + 8], v_alpha2_l1);
                w_result1 = Q6_Wd_vmulacc_WdVwVw(w_result1, *l_gather[ch * 16 + 9], v_alpha2_l0);
                w_result0 = Q6_Wd_vmulacc_WdVwVw(w_result0, *l_gather[ch * 16 + 12], v_alpha3_l1);
                w_result1 = Q6_Wd_vmulacc_WdVwVw(w_result1, *l_gather[ch * 16 + 13], v_alpha3_l0);

                HVX_Vector v_result0 = Q6_Vw_vasl_VwR(Q6_V_hi_W(w_result0), 10);
                HVX_Vector v_result1 = Q6_Vw_vasl_VwR(Q6_V_hi_W(w_result1), 10);
                HVX_Vector v_result2 = Q6_Vh_vpacko_VwVw(Q6_V_lo_W(w_result1), Q6_V_lo_W(w_result0));
                HVX_Vector v_result3 = Q6_Vh_vadd_VhVh(v_result2, Q6_Vh_vsplat_R(32));
                v_result0 = Q6_Vh_vpack_VwVw_sat(v_result1, v_result0);
                v_result2 = Q6_Vh_vasr_VhR(v_result3, 6);
                v_result3 = Q6_Vh_vadd_VhVh(v_result0, v_result2);

                w_result0 = Q6_Wd_vmul_VwVw(*l_gather[ch * 16 + 2], v_alpha0_c );
                w_result1 = Q6_Wd_vmul_VwVw(*l_gather[ch * 16 + 3], v_alpha0_r0);
                w_result0 = Q6_Wd_vmulacc_WdVwVw(w_result0, *l_gather[ch * 16 + 6], v_alpha1_c );
                w_result1 = Q6_Wd_vmulacc_WdVwVw(w_result1, *l_gather[ch * 16 + 7], v_alpha1_r0);
                w_result0 = Q6_Wd_vmulacc_WdVwVw(w_result0, *l_gather[ch * 16 + 10], v_alpha2_c );
                w_result1 = Q6_Wd_vmulacc_WdVwVw(w_result1, *l_gather[ch * 16 + 11], v_alpha2_r0);
                w_result0 = Q6_Wd_vmulacc_WdVwVw(w_result0, *l_gather[ch * 16 + 14], v_alpha3_c );
                w_result1 = Q6_Wd_vmulacc_WdVwVw(w_result1, *l_gather[ch * 16 + 15], v_alpha3_r0);

                v_result0 = Q6_Vw_vasl_VwR(Q6_V_hi_W(w_result0), 10);
                v_result1 = Q6_Vw_vasl_VwR(Q6_V_hi_W(w_result1), 10);
                v_result2 = Q6_Vh_vpacko_VwVw(Q6_V_lo_W(w_result1), Q6_V_lo_W(w_result0));
                v_result1 = Q6_Vh_vpack_VwVw_sat(v_result1, v_result0);
                v_result0 = Q6_Vh_vadd_VhVh(v_result2, Q6_Vh_vsplat_R(32));
                v_result2 = Q6_Vh_vasr_VhR(v_result0, 6);
                v_result0 = Q6_Vh_vadd_VhVh(v_result2, v_result1);

                mv_result.val[ch] = Q6_Vub_vpack_VhVh_sat(v_result0, v_result3);

                v_tmp[0] = r_gather[ch * 16 + 0] ; r_gather[ch * 16 + 0]  = l_gather[ch * 16 + 0] ; l_gather[ch * 16 + 0]  = v_tmp[0];
                v_tmp[1] = r_gather[ch * 16 + 1] ; r_gather[ch * 16 + 1]  = l_gather[ch * 16 + 1] ; l_gather[ch * 16 + 1]  = v_tmp[1];
                v_tmp[2] = r_gather[ch * 16 + 2] ; r_gather[ch * 16 + 2]  = l_gather[ch * 16 + 2] ; l_gather[ch * 16 + 2]  = v_tmp[2];
                v_tmp[3] = r_gather[ch * 16 + 3] ; r_gather[ch * 16 + 3]  = l_gather[ch * 16 + 3] ; l_gather[ch * 16 + 3]  = v_tmp[3];
                v_tmp[0] = r_gather[ch * 16 + 4] ; r_gather[ch * 16 + 4]  = l_gather[ch * 16 + 4] ; l_gather[ch * 16 + 4]  = v_tmp[0];
                v_tmp[1] = r_gather[ch * 16 + 5] ; r_gather[ch * 16 + 5]  = l_gather[ch * 16 + 5] ; l_gather[ch * 16 + 5]  = v_tmp[1];
                v_tmp[2] = r_gather[ch * 16 + 6] ; r_gather[ch * 16 + 6]  = l_gather[ch * 16 + 6] ; l_gather[ch * 16 + 6]  = v_tmp[2];
                v_tmp[3] = r_gather[ch * 16 + 7] ; r_gather[ch * 16 + 7]  = l_gather[ch * 16 + 7] ; l_gather[ch * 16 + 7]  = v_tmp[3];
                v_tmp[0] = r_gather[ch * 16 + 8] ; r_gather[ch * 16 + 8]  = l_gather[ch * 16 + 8] ; l_gather[ch * 16 + 8]  = v_tmp[0];
                v_tmp[1] = r_gather[ch * 16 + 9] ; r_gather[ch * 16 + 9]  = l_gather[ch * 16 + 9] ; l_gather[ch * 16 + 9]  = v_tmp[1];
                v_tmp[2] = r_gather[ch * 16 + 10]; r_gather[ch * 16 + 10] = l_gather[ch * 16 + 10]; l_gather[ch * 16 + 10] = v_tmp[2];
                v_tmp[3] = r_gather[ch * 16 + 11]; r_gather[ch * 16 + 11] = l_gather[ch * 16 + 11]; l_gather[ch * 16 + 11] = v_tmp[3];
                v_tmp[0] = r_gather[ch * 16 + 12]; r_gather[ch * 16 + 12] = l_gather[ch * 16 + 12]; l_gather[ch * 16 + 12] = v_tmp[0];
                v_tmp[1] = r_gather[ch * 16 + 13]; r_gather[ch * 16 + 13] = l_gather[ch * 16 + 13]; l_gather[ch * 16 + 13] = v_tmp[1];
                v_tmp[2] = r_gather[ch * 16 + 14]; r_gather[ch * 16 + 14] = l_gather[ch * 16 + 14]; l_gather[ch * 16 + 14] = v_tmp[2];
                v_tmp[3] = r_gather[ch * 16 + 15]; r_gather[ch * 16 + 15] = l_gather[ch * 16 + 15]; l_gather[ch * 16 + 15] = v_tmp[3];
            }

            vstore(dst + i, mv_result);
        }

        {
            HVX_Vector v_alpha0_c  = *v_alpha0++;
            HVX_Vector v_alpha0_r0 = *v_alpha0++;
            HVX_Vector v_alpha1_c  = *v_alpha1++;
            HVX_Vector v_alpha1_r0 = *v_alpha1++;
            HVX_Vector v_alpha2_c  = *v_alpha2++;
            HVX_Vector v_alpha2_r0 = *v_alpha2++;
            HVX_Vector v_alpha3_c  = *v_alpha3++;
            HVX_Vector v_alpha3_r0 = *v_alpha3++;

            HVX_VectorPair w_result0 = Q6_Ww_vunpack_Vh(v_alpha0_c);
            HVX_VectorPair w_result1 = Q6_Ww_vunpack_Vh(v_alpha0_r0);
            HVX_Vector v_alpha0_l1 = Q6_V_lo_W(w_result0);
            HVX_Vector v_alpha0_l0 = Q6_V_hi_W(w_result0);
            v_alpha0_c  = Q6_V_lo_W(w_result1);
            v_alpha0_r0 = Q6_V_hi_W(w_result1);
            w_result0 = Q6_Ww_vunpack_Vh(v_alpha1_c);
            w_result1 = Q6_Ww_vunpack_Vh(v_alpha1_r0);
            HVX_Vector v_alpha1_l1 = Q6_V_lo_W(w_result0);
            HVX_Vector v_alpha1_l0 = Q6_V_hi_W(w_result0);
            v_alpha1_c  = Q6_V_lo_W(w_result1);
            v_alpha1_r0 = Q6_V_hi_W(w_result1);
            w_result0 = Q6_Ww_vunpack_Vh(v_alpha2_c);
            w_result1 = Q6_Ww_vunpack_Vh(v_alpha2_r0);
            HVX_Vector v_alpha2_l1 = Q6_V_lo_W(w_result0);
            HVX_Vector v_alpha2_l0 = Q6_V_hi_W(w_result0);
            v_alpha2_c  = Q6_V_lo_W(w_result1);
            v_alpha2_r0 = Q6_V_hi_W(w_result1);
            w_result0 = Q6_Ww_vunpack_Vh(v_alpha3_c);
            w_result1 = Q6_Ww_vunpack_Vh(v_alpha3_r0);
            HVX_Vector v_alpha3_l1 = Q6_V_lo_W(w_result0);
            HVX_Vector v_alpha3_l0 = Q6_V_hi_W(w_result0);
            v_alpha3_c  = Q6_V_lo_W(w_result1);
            v_alpha3_r0 = Q6_V_hi_W(w_result1);

            MVType mv_result;
            for (MI_S32 ch = 0; ch < C; ch++)
            {
                w_result0 = Q6_Wd_vmul_VwVw(*l_gather[ch * 16 + 0], v_alpha0_l1);
                w_result1 = Q6_Wd_vmul_VwVw(*l_gather[ch * 16 + 1], v_alpha0_l0);
                w_result0 = Q6_Wd_vmulacc_WdVwVw(w_result0, *l_gather[ch * 16 + 4], v_alpha1_l1);
                w_result1 = Q6_Wd_vmulacc_WdVwVw(w_result1, *l_gather[ch * 16 + 5], v_alpha1_l0);
                w_result0 = Q6_Wd_vmulacc_WdVwVw(w_result0, *l_gather[ch * 16 + 8], v_alpha2_l1);
                w_result1 = Q6_Wd_vmulacc_WdVwVw(w_result1, *l_gather[ch * 16 + 9], v_alpha2_l0);
                w_result0 = Q6_Wd_vmulacc_WdVwVw(w_result0, *l_gather[ch * 16 + 12], v_alpha3_l1);
                w_result1 = Q6_Wd_vmulacc_WdVwVw(w_result1, *l_gather[ch * 16 + 13], v_alpha3_l0);

                HVX_Vector v_result0 = Q6_Vw_vasl_VwR(Q6_V_hi_W(w_result0), 10);
                HVX_Vector v_result1 = Q6_Vw_vasl_VwR(Q6_V_hi_W(w_result1), 10);
                HVX_Vector v_result2 = Q6_Vh_vpacko_VwVw(Q6_V_lo_W(w_result1), Q6_V_lo_W(w_result0));
                HVX_Vector v_result3 = Q6_Vh_vadd_VhVh(v_result2, Q6_Vh_vsplat_R(32));
                v_result0 = Q6_Vh_vpack_VwVw_sat(v_result1, v_result0);
                v_result2 = Q6_Vh_vasr_VhR(v_result3, 6);
                v_result3 = Q6_Vh_vadd_VhVh(v_result0, v_result2);

                w_result0 = Q6_Wd_vmul_VwVw(*l_gather[ch * 16 + 2], v_alpha0_c);
                w_result1 = Q6_Wd_vmul_VwVw(*l_gather[ch * 16 + 3], v_alpha0_r0);
                w_result0 = Q6_Wd_vmulacc_WdVwVw(w_result0, *l_gather[ch * 16 + 6], v_alpha1_c);
                w_result1 = Q6_Wd_vmulacc_WdVwVw(w_result1, *l_gather[ch * 16 + 7], v_alpha1_r0);
                w_result0 = Q6_Wd_vmulacc_WdVwVw(w_result0, *l_gather[ch * 16 + 10], v_alpha2_c);
                w_result1 = Q6_Wd_vmulacc_WdVwVw(w_result1, *l_gather[ch * 16 + 11], v_alpha2_r0);
                w_result0 = Q6_Wd_vmulacc_WdVwVw(w_result0, *l_gather[ch * 16 + 14], v_alpha3_c);
                w_result1 = Q6_Wd_vmulacc_WdVwVw(w_result1, *l_gather[ch * 16 + 15], v_alpha3_r0);

                v_result0 = Q6_Vw_vasl_VwR(Q6_V_hi_W(w_result0), 10);
                v_result1 = Q6_Vw_vasl_VwR(Q6_V_hi_W(w_result1), 10);
                v_result2 = Q6_Vh_vpacko_VwVw(Q6_V_lo_W(w_result1), Q6_V_lo_W(w_result0));
                v_result1 = Q6_Vh_vpack_VwVw_sat(v_result1, v_result0);
                v_result0 = Q6_Vh_vadd_VhVh(v_result2, Q6_Vh_vsplat_R(32));
                v_result2 = Q6_Vh_vasr_VhR(v_result0, 6);
                v_result0 = Q6_Vh_vadd_VhVh(v_result2, v_result1);

                mv_result.val[ch] = Q6_Vub_vpack_VhVh_sat(v_result0, v_result3);
            }

            vstore(dst + i, mv_result);
        }

        if (ostep > owidth_align)
        {
            i = ostep - elem_counts;
            j = i / C;
            HVX_Vector v_offset_c   = vmemu(xofs + j);
            HVX_Vector v_offset_r0  = vmemu(xofs + j + HALF_AURA_HVLEN);
            HVX_Vector v_alpha0_c   = vmemu(alpha0 + j);
            HVX_Vector v_alpha0_r0  = vmemu(alpha0 + j + HALF_AURA_HVLEN);
            HVX_Vector v_alpha1_c   = vmemu(alpha1 + j);
            HVX_Vector v_alpha1_r0  = vmemu(alpha1 + j + HALF_AURA_HVLEN);
            HVX_Vector v_alpha2_c   = vmemu(alpha2 + j);
            HVX_Vector v_alpha2_r0  = vmemu(alpha2 + j + HALF_AURA_HVLEN);
            HVX_Vector v_alpha3_c   = vmemu(alpha3 + j);
            HVX_Vector v_alpha3_r0  = vmemu(alpha3 + j + HALF_AURA_HVLEN);
            HVX_VectorPair w_result0 = Q6_Wuw_vunpack_Vuh(v_offset_c);
            HVX_VectorPair w_result1 = Q6_Wuw_vunpack_Vuh(v_offset_r0);
            HVX_Vector v_offset_l1 = Q6_V_lo_W(w_result0);
            HVX_Vector v_offset_l0 = Q6_V_hi_W(w_result0);
            v_offset_c  = Q6_V_lo_W(w_result1);
            v_offset_r0 = Q6_V_hi_W(w_result1);

            HVX_Vector v_offset_add1_l1 = Q6_Vuw_vadd_VuwVuw_sat(v_offset_l1, Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add1_l0 = Q6_Vuw_vadd_VuwVuw_sat(v_offset_l0, Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add1_c  = Q6_Vuw_vadd_VuwVuw_sat(v_offset_c , Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add1_r0 = Q6_Vuw_vadd_VuwVuw_sat(v_offset_r0, Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add2_l1 = Q6_Vuw_vadd_VuwVuw_sat(v_offset_add1_l1, Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add2_l0 = Q6_Vuw_vadd_VuwVuw_sat(v_offset_add1_l0, Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add2_c  = Q6_Vuw_vadd_VuwVuw_sat(v_offset_add1_c , Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add2_r0 = Q6_Vuw_vadd_VuwVuw_sat(v_offset_add1_r0, Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add3_l1 = Q6_Vuw_vadd_VuwVuw_sat(v_offset_add2_l1, Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add3_l0 = Q6_Vuw_vadd_VuwVuw_sat(v_offset_add2_l0, Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add3_c  = Q6_Vuw_vadd_VuwVuw_sat(v_offset_add2_c , Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add3_r0 = Q6_Vuw_vadd_VuwVuw_sat(v_offset_add2_r0, Q6_V_vsplat_R(4));

            w_result0 = Q6_Ww_vunpack_Vh(v_alpha0_c);
            w_result1 = Q6_Ww_vunpack_Vh(v_alpha0_r0);
            HVX_Vector v_alpha0_l1 = Q6_V_lo_W(w_result0);
            HVX_Vector v_alpha0_l0 = Q6_V_hi_W(w_result0);
            v_alpha0_c  = Q6_V_lo_W(w_result1);
            v_alpha0_r0 = Q6_V_hi_W(w_result1);
            w_result0 = Q6_Ww_vunpack_Vh(v_alpha1_c);
            w_result1 = Q6_Ww_vunpack_Vh(v_alpha1_r0);
            HVX_Vector v_alpha1_l1 = Q6_V_lo_W(w_result0);
            HVX_Vector v_alpha1_l0 = Q6_V_hi_W(w_result0);
            v_alpha1_c  = Q6_V_lo_W(w_result1);
            v_alpha1_r0 = Q6_V_hi_W(w_result1);
            w_result0 = Q6_Ww_vunpack_Vh(v_alpha2_c);
            w_result1 = Q6_Ww_vunpack_Vh(v_alpha2_r0);
            HVX_Vector v_alpha2_l1 = Q6_V_lo_W(w_result0);
            HVX_Vector v_alpha2_l0 = Q6_V_hi_W(w_result0);
            v_alpha2_c  = Q6_V_lo_W(w_result1);
            v_alpha2_r0 = Q6_V_hi_W(w_result1);
            w_result0 = Q6_Ww_vunpack_Vh(v_alpha3_c);
            w_result1 = Q6_Ww_vunpack_Vh(v_alpha3_r0);
            HVX_Vector v_alpha3_l1 = Q6_V_lo_W(w_result0);
            HVX_Vector v_alpha3_l0 = Q6_V_hi_W(w_result0);
            v_alpha3_c  = Q6_V_lo_W(w_result1);
            v_alpha3_r0 = Q6_V_hi_W(w_result1);

            MVType mv_result;
            for (MI_S32 ch = 0; ch < C; ch++)
            {
                Q6_vgather_ARMVw(l_gather[ch * 16 + 0], (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_l1);
                Q6_vgather_ARMVw(l_gather[ch * 16 + 1], (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_l0);
                Q6_vgather_ARMVw(l_gather[ch * 16 + 2], (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_c);
                Q6_vgather_ARMVw(l_gather[ch * 16 + 3], (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_r0);
                Q6_vgather_ARMVw(l_gather[ch * 16 + 4], (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add1_l1);
                Q6_vgather_ARMVw(l_gather[ch * 16 + 5], (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add1_l0);
                Q6_vgather_ARMVw(l_gather[ch * 16 + 6], (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add1_c);
                Q6_vgather_ARMVw(l_gather[ch * 16 + 7], (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add1_r0);
                Q6_vgather_ARMVw(l_gather[ch * 16 + 8], (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add2_l1);
                Q6_vgather_ARMVw(l_gather[ch * 16 + 9], (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add2_l0);
                Q6_vgather_ARMVw(l_gather[ch * 16 + 10], (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add2_c);
                Q6_vgather_ARMVw(l_gather[ch * 16 + 11], (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add2_r0);
                Q6_vgather_ARMVw(l_gather[ch * 16 + 12], (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add3_l1);
                Q6_vgather_ARMVw(l_gather[ch * 16 + 13], (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add3_l0);
                Q6_vgather_ARMVw(l_gather[ch * 16 + 14], (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add3_c);
                Q6_vgather_ARMVw(l_gather[ch * 16 + 15], (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add3_r0);

                w_result0 = Q6_Wd_vmul_VwVw(*l_gather[ch * 16 + 0], v_alpha0_l1);
                w_result1 = Q6_Wd_vmul_VwVw(*l_gather[ch * 16 + 1], v_alpha0_l0);
                w_result0 = Q6_Wd_vmulacc_WdVwVw(w_result0, *l_gather[ch * 16 + 4], v_alpha1_l1);
                w_result1 = Q6_Wd_vmulacc_WdVwVw(w_result1, *l_gather[ch * 16 + 5], v_alpha1_l0);
                w_result0 = Q6_Wd_vmulacc_WdVwVw(w_result0, *l_gather[ch * 16 + 8], v_alpha2_l1);
                w_result1 = Q6_Wd_vmulacc_WdVwVw(w_result1, *l_gather[ch * 16 + 9], v_alpha2_l0);
                w_result0 = Q6_Wd_vmulacc_WdVwVw(w_result0, *l_gather[ch * 16 + 12], v_alpha3_l1);
                w_result1 = Q6_Wd_vmulacc_WdVwVw(w_result1, *l_gather[ch * 16 + 13], v_alpha3_l0);

                HVX_Vector v_result0 = Q6_Vw_vasl_VwR(Q6_V_hi_W(w_result0), 10);
                HVX_Vector v_result1 = Q6_Vw_vasl_VwR(Q6_V_hi_W(w_result1), 10);
                HVX_Vector v_result2 = Q6_Vh_vpacko_VwVw(Q6_V_lo_W(w_result1), Q6_V_lo_W(w_result0));
                HVX_Vector v_result3 = Q6_Vh_vadd_VhVh(v_result2, Q6_Vh_vsplat_R(32));
                v_result0 = Q6_Vh_vpack_VwVw_sat(v_result1, v_result0);
                v_result2 = Q6_Vh_vasr_VhR(v_result3, 6);
                v_result3 = Q6_Vh_vadd_VhVh(v_result0, v_result2);

                w_result0 = Q6_Wd_vmul_VwVw(*l_gather[ch * 16 + 2], v_alpha0_c);
                w_result1 = Q6_Wd_vmul_VwVw(*l_gather[ch * 16 + 3], v_alpha0_r0);
                w_result0 = Q6_Wd_vmulacc_WdVwVw(w_result0, *l_gather[ch * 16 + 6], v_alpha1_c);
                w_result1 = Q6_Wd_vmulacc_WdVwVw(w_result1, *l_gather[ch * 16 + 7], v_alpha1_r0);
                w_result0 = Q6_Wd_vmulacc_WdVwVw(w_result0, *l_gather[ch * 16 + 10], v_alpha2_c);
                w_result1 = Q6_Wd_vmulacc_WdVwVw(w_result1, *l_gather[ch * 16 + 11], v_alpha2_r0);
                w_result0 = Q6_Wd_vmulacc_WdVwVw(w_result0, *l_gather[ch * 16 + 14], v_alpha3_c);
                w_result1 = Q6_Wd_vmulacc_WdVwVw(w_result1, *l_gather[ch * 16 + 15], v_alpha3_r0);

                v_result0 = Q6_Vw_vasl_VwR(Q6_V_hi_W(w_result0), 10);
                v_result1 = Q6_Vw_vasl_VwR(Q6_V_hi_W(w_result1), 10);
                v_result2 = Q6_Vh_vpacko_VwVw(Q6_V_lo_W(w_result1), Q6_V_lo_W(w_result0));
                v_result1 = Q6_Vh_vpack_VwVw_sat(v_result1, v_result0);
                v_result0 = Q6_Vh_vadd_VhVh(v_result2, Q6_Vh_vsplat_R(32));
                v_result2 = Q6_Vh_vasr_VhR(v_result0, 6);
                v_result0 = Q6_Vh_vadd_VhVh(v_result2, v_result1);

                mv_result.val[ch] = Q6_Vub_vpack_VhVh_sat(v_result0, v_result3);
            }

            vstore(dst + i, mv_result);
        }
    }
}

// Tp = MI_S8
template<typename Tp, MI_S32 C>
static typename std::enable_if<std::is_same<MI_S8, Tp>::value, AURA_VOID>::type
ResizeCuCommPerRow(const Tp *src_c, Tp *dst, MI_S32 *row_base, MI_S32 *vtcm_base, MI_U16 *xofs, MI_S16 *beta,
                   MI_S16 *alpha0, MI_S16 *alpha1, MI_S16 *alpha2, MI_S16 *alpha3, AURA_VOID *l_gather_base,
                   AURA_VOID *r_gather_base, MI_S32 iwidth, MI_S32 istride, MI_S32 istep, MI_S32 ostep)
{
    using MVType = typename MVHvxVector<C>::Type;
    MI_S32 *row  = reinterpret_cast<MI_S32 *>(row_base);
    constexpr MI_S32 elem_counts = AURA_HVLEN * C / sizeof(Tp);
    constexpr MI_S32 elem_counts_channel = elem_counts / C;
    MI_S32 iwidth_align = istep / elem_counts * elem_counts;

    // part1: calculate the median and save to vtcm
    MI_S32 i = 0, j = 0;
    {
        const Tp *src_n0   = src_c + istride;
        const Tp *src_n1   = src_n0 + istride;
        const Tp *src_n2   = src_n1 + istride;
        HVX_Vector v_beta0 = Q6_Vh_vsplat_R(beta[0]);
        HVX_Vector v_beta1 = Q6_Vh_vsplat_R(beta[1]);
        HVX_Vector v_beta2 = Q6_Vh_vsplat_R(beta[2]);
        HVX_Vector v_beta3 = Q6_Vh_vsplat_R(beta[3]);

        for (; i < iwidth_align; i += elem_counts, j += elem_counts_channel)
        {
            MVType mv_c_src, mv_n0_src, mv_n1_src, mv_n2_src;
            vload(src_c + i,  mv_c_src);
            vload(src_n0 + i, mv_n0_src);
            vload(src_n1 + i, mv_n1_src);
            vload(src_n2 + i, mv_n2_src);

            for (MI_S32 ch = 0; ch < C; ch++)
            {
                HVX_VectorPair w_c_src   = Q6_Wh_vunpack_Vb(Q6_Vb_vshuff_Vb(mv_c_src.val[ch]));
                HVX_VectorPair w_n0_src  = Q6_Wh_vunpack_Vb(Q6_Vb_vshuff_Vb(mv_n0_src.val[ch]));
                HVX_VectorPair w_n1_src  = Q6_Wh_vunpack_Vb(Q6_Vb_vshuff_Vb(mv_n1_src.val[ch]));
                HVX_VectorPair w_n2_src  = Q6_Wh_vunpack_Vb(Q6_Vb_vshuff_Vb(mv_n2_src.val[ch]));
                HVX_VectorPair w_result0 = Q6_Ww_vmpy_VhVh(Q6_V_lo_W(w_c_src), v_beta0);  // lo: 0  1  2  ... 31 hi: 64 65 66  ... 95
                HVX_VectorPair w_result1 = Q6_Ww_vmpy_VhVh(Q6_V_hi_W(w_c_src), v_beta0);  // lo:32 33 34  ... 63 hi: 96 97 98  ... 127
                HVX_VectorPair w_result2 = Q6_Ww_vmpy_VhVh(Q6_V_lo_W(w_n1_src), v_beta2);
                HVX_VectorPair w_result3 = Q6_Ww_vmpy_VhVh(Q6_V_hi_W(w_n1_src), v_beta2);

                w_result0 = Q6_Ww_vmpyacc_WwVhVh(w_result0, Q6_V_lo_W(w_n0_src), v_beta1);
                w_result1 = Q6_Ww_vmpyacc_WwVhVh(w_result1, Q6_V_hi_W(w_n0_src), v_beta1);
                w_result2 = Q6_Ww_vmpyacc_WwVhVh(w_result2, Q6_V_lo_W(w_n2_src), v_beta3);
                w_result3 = Q6_Ww_vmpyacc_WwVhVh(w_result3, Q6_V_hi_W(w_n2_src), v_beta3);
                w_result0 = Q6_Ww_vadd_WwWw(w_result0, w_result2);
                w_result1 = Q6_Ww_vadd_WwWw(w_result1, w_result3);

                HVX_Vector *v_dst = (HVX_Vector *)(row + j + ch * AURA_ALIGN(iwidth, QUAR_AURA_HVLEN));
                *v_dst++ = Q6_V_lo_W(w_result0);
                *v_dst++ = Q6_V_lo_W(w_result1);
                *v_dst++ = Q6_V_hi_W(w_result0);
                *v_dst   = Q6_V_hi_W(w_result1);
            }
        }

        if (istep > iwidth_align)
        {
            MVType mv_c_src, mv_n0_src, mv_n1_src, mv_n2_src;
            i = istep - elem_counts;
            j = i / C;

            vload(src_c + i,  mv_c_src);
            vload(src_n0 + i, mv_n0_src);
            vload(src_n1 + i, mv_n1_src);
            vload(src_n2 + i, mv_n2_src);

            for (MI_S32 ch = 0; ch < C; ch++)
            {
                MI_S32 *row_tmp          = row + j + ch * AURA_ALIGN(iwidth, QUAR_AURA_HVLEN);
                HVX_VectorPair w_c_src   = Q6_Wh_vunpack_Vb(Q6_Vb_vshuff_Vb(mv_c_src.val[ch]));
                HVX_VectorPair w_n0_src  = Q6_Wh_vunpack_Vb(Q6_Vb_vshuff_Vb(mv_n0_src.val[ch]));
                HVX_VectorPair w_n1_src  = Q6_Wh_vunpack_Vb(Q6_Vb_vshuff_Vb(mv_n1_src.val[ch]));
                HVX_VectorPair w_n2_src  = Q6_Wh_vunpack_Vb(Q6_Vb_vshuff_Vb(mv_n2_src.val[ch]));
                HVX_VectorPair w_result0 = Q6_Ww_vmpy_VhVh(Q6_V_lo_W(w_c_src), v_beta0);  // lo: 0  1  2  ... 31 hi: 64 65 66  ... 95
                HVX_VectorPair w_result1 = Q6_Ww_vmpy_VhVh(Q6_V_hi_W(w_c_src), v_beta0);  // lo:32 33 34  ... 63 hi: 96 97 98  ... 127
                HVX_VectorPair w_result2 = Q6_Ww_vmpy_VhVh(Q6_V_lo_W(w_n1_src), v_beta2);
                HVX_VectorPair w_result3 = Q6_Ww_vmpy_VhVh(Q6_V_hi_W(w_n1_src), v_beta2);

                w_result0 = Q6_Ww_vmpyacc_WwVhVh(w_result0, Q6_V_lo_W(w_n0_src), v_beta1);
                w_result1 = Q6_Ww_vmpyacc_WwVhVh(w_result1, Q6_V_hi_W(w_n0_src), v_beta1);
                w_result2 = Q6_Ww_vmpyacc_WwVhVh(w_result2, Q6_V_lo_W(w_n2_src), v_beta3);
                w_result3 = Q6_Ww_vmpyacc_WwVhVh(w_result3, Q6_V_hi_W(w_n2_src), v_beta3);
                w_result0 = Q6_Ww_vadd_WwWw(w_result0, w_result2);
                w_result1 = Q6_Ww_vadd_WwWw(w_result1, w_result3);

                vmemu(row_tmp) = Q6_V_lo_W(w_result0);
                vmemu(row_tmp + QUAR_AURA_HVLEN) = Q6_V_lo_W(w_result1);
                vmemu(row_tmp + HALF_AURA_HVLEN) = Q6_V_hi_W(w_result0);
                vmemu(row_tmp + QUAR_AURA_HVLEN + HALF_AURA_HVLEN) = Q6_V_hi_W(w_result1);
            }
        }
    }

    // part2: load vtcm and calculate dst
    {
        MI_S32 *vtcm = reinterpret_cast<MI_S32 *>(vtcm_base);
        MI_S32 owidth_align = ostep / elem_counts * elem_counts;

        HVX_Vector *v_ofs     = (HVX_Vector *)xofs;
        HVX_Vector *v_alpha0  = (HVX_Vector *)alpha0; HVX_Vector *v_alpha1 = (HVX_Vector *)alpha1;
        HVX_Vector *v_alpha2  = (HVX_Vector *)alpha2; HVX_Vector *v_alpha3 = (HVX_Vector *)alpha3;
        HVX_Vector **l_gather = (HVX_Vector **)l_gather_base;
        HVX_Vector **r_gather = (HVX_Vector **)r_gather_base;
        HVX_Vector *v_tmp[4];
        MI_S32 *row_ch[C];

        {
            HVX_Vector v_offset_c  = *v_ofs++;
            HVX_Vector v_offset_r0 = *v_ofs++;
            HVX_VectorPair w_result0 = Q6_Wuw_vunpack_Vuh(v_offset_c);
            HVX_VectorPair w_result1 = Q6_Wuw_vunpack_Vuh(v_offset_r0);
            HVX_Vector v_offset_l1 = Q6_V_lo_W(w_result0);
            HVX_Vector v_offset_l0 = Q6_V_hi_W(w_result0);
            v_offset_c  = Q6_V_lo_W(w_result1);
            v_offset_r0 = Q6_V_hi_W(w_result1);

            HVX_Vector v_offset_add1_l1 = Q6_Vuw_vadd_VuwVuw_sat(v_offset_l1, Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add1_l0 = Q6_Vuw_vadd_VuwVuw_sat(v_offset_l0, Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add1_c  = Q6_Vuw_vadd_VuwVuw_sat(v_offset_c , Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add1_r0 = Q6_Vuw_vadd_VuwVuw_sat(v_offset_r0, Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add2_l1 = Q6_Vuw_vadd_VuwVuw_sat(v_offset_add1_l1, Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add2_l0 = Q6_Vuw_vadd_VuwVuw_sat(v_offset_add1_l0, Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add2_c  = Q6_Vuw_vadd_VuwVuw_sat(v_offset_add1_c , Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add2_r0 = Q6_Vuw_vadd_VuwVuw_sat(v_offset_add1_r0, Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add3_l1 = Q6_Vuw_vadd_VuwVuw_sat(v_offset_add2_l1, Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add3_l0 = Q6_Vuw_vadd_VuwVuw_sat(v_offset_add2_l0, Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add3_c  = Q6_Vuw_vadd_VuwVuw_sat(v_offset_add2_c , Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add3_r0 = Q6_Vuw_vadd_VuwVuw_sat(v_offset_add2_r0, Q6_V_vsplat_R(4));

            for (MI_S32 ch = 0; ch < C; ch++)
            {
                MI_S32 *vtcm_gather = vtcm + ch * (QUAR_AURA_HVLEN << 5);
                row_ch[ch] = row + ch * AURA_ALIGN(iwidth, QUAR_AURA_HVLEN);

                l_gather[ch * 16 + 0]  = (HVX_Vector *)(vtcm_gather);
                l_gather[ch * 16 + 1]  = (HVX_Vector *)(vtcm_gather + QUAR_AURA_HVLEN);
                l_gather[ch * 16 + 2]  = (HVX_Vector *)(vtcm_gather + HALF_AURA_HVLEN);
                l_gather[ch * 16 + 3]  = (HVX_Vector *)(vtcm_gather + QUAR_AURA_HVLEN * 3);
                l_gather[ch * 16 + 4]  = (HVX_Vector *)(vtcm_gather + AURA_HVLEN);
                l_gather[ch * 16 + 5]  = (HVX_Vector *)(vtcm_gather + QUAR_AURA_HVLEN * 5);
                l_gather[ch * 16 + 6]  = (HVX_Vector *)(vtcm_gather + QUAR_AURA_HVLEN * 6);
                l_gather[ch * 16 + 7]  = (HVX_Vector *)(vtcm_gather + QUAR_AURA_HVLEN * 7);
                l_gather[ch * 16 + 8]  = (HVX_Vector *)(vtcm_gather + (AURA_HVLEN << 1));
                l_gather[ch * 16 + 9]  = (HVX_Vector *)(vtcm_gather + QUAR_AURA_HVLEN * 9);
                l_gather[ch * 16 + 10] = (HVX_Vector *)(vtcm_gather + QUAR_AURA_HVLEN * 10);
                l_gather[ch * 16 + 11] = (HVX_Vector *)(vtcm_gather + QUAR_AURA_HVLEN * 11);
                l_gather[ch * 16 + 12] = (HVX_Vector *)(vtcm_gather + QUAR_AURA_HVLEN * 12);
                l_gather[ch * 16 + 13] = (HVX_Vector *)(vtcm_gather + QUAR_AURA_HVLEN * 13);
                l_gather[ch * 16 + 14] = (HVX_Vector *)(vtcm_gather + QUAR_AURA_HVLEN * 14);
                l_gather[ch * 16 + 15] = (HVX_Vector *)(vtcm_gather + QUAR_AURA_HVLEN * 15);
                r_gather[ch * 16 + 0]  = (HVX_Vector *)(vtcm_gather + (AURA_HVLEN << 2));
                r_gather[ch * 16 + 1]  = (HVX_Vector *)(vtcm_gather + QUAR_AURA_HVLEN * 17);
                r_gather[ch * 16 + 2]  = (HVX_Vector *)(vtcm_gather + QUAR_AURA_HVLEN * 18);
                r_gather[ch * 16 + 3]  = (HVX_Vector *)(vtcm_gather + QUAR_AURA_HVLEN * 19);
                r_gather[ch * 16 + 4]  = (HVX_Vector *)(vtcm_gather + QUAR_AURA_HVLEN * 20);
                r_gather[ch * 16 + 5]  = (HVX_Vector *)(vtcm_gather + QUAR_AURA_HVLEN * 21);
                r_gather[ch * 16 + 6]  = (HVX_Vector *)(vtcm_gather + QUAR_AURA_HVLEN * 22);
                r_gather[ch * 16 + 7]  = (HVX_Vector *)(vtcm_gather + QUAR_AURA_HVLEN * 23);
                r_gather[ch * 16 + 8]  = (HVX_Vector *)(vtcm_gather + QUAR_AURA_HVLEN * 24);
                r_gather[ch * 16 + 9]  = (HVX_Vector *)(vtcm_gather + QUAR_AURA_HVLEN * 25);
                r_gather[ch * 16 + 10] = (HVX_Vector *)(vtcm_gather + QUAR_AURA_HVLEN * 26);
                r_gather[ch * 16 + 11] = (HVX_Vector *)(vtcm_gather + QUAR_AURA_HVLEN * 27);
                r_gather[ch * 16 + 12] = (HVX_Vector *)(vtcm_gather + QUAR_AURA_HVLEN * 28);
                r_gather[ch * 16 + 13] = (HVX_Vector *)(vtcm_gather + QUAR_AURA_HVLEN * 29);
                r_gather[ch * 16 + 14] = (HVX_Vector *)(vtcm_gather + QUAR_AURA_HVLEN * 30);
                r_gather[ch * 16 + 15] = (HVX_Vector *)(vtcm_gather + QUAR_AURA_HVLEN * 31);

                Q6_vgather_ARMVw(l_gather[ch * 16 + 0] , (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_l1);
                Q6_vgather_ARMVw(l_gather[ch * 16 + 1] , (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_l0);
                Q6_vgather_ARMVw(l_gather[ch * 16 + 2] , (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_c);
                Q6_vgather_ARMVw(l_gather[ch * 16 + 3] , (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_r0);
                Q6_vgather_ARMVw(l_gather[ch * 16 + 4] , (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add1_l1);
                Q6_vgather_ARMVw(l_gather[ch * 16 + 5] , (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add1_l0);
                Q6_vgather_ARMVw(l_gather[ch * 16 + 6] , (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add1_c);
                Q6_vgather_ARMVw(l_gather[ch * 16 + 7] , (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add1_r0);
                Q6_vgather_ARMVw(l_gather[ch * 16 + 8] , (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add2_l1);
                Q6_vgather_ARMVw(l_gather[ch * 16 + 9] , (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add2_l0);
                Q6_vgather_ARMVw(l_gather[ch * 16 + 10], (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add2_c);
                Q6_vgather_ARMVw(l_gather[ch * 16 + 11], (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add2_r0);
                Q6_vgather_ARMVw(l_gather[ch * 16 + 12], (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add3_l1);
                Q6_vgather_ARMVw(l_gather[ch * 16 + 13], (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add3_l0);
                Q6_vgather_ARMVw(l_gather[ch * 16 + 14], (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add3_c);
                Q6_vgather_ARMVw(l_gather[ch * 16 + 15], (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add3_r0);
            }
        }

        i = 0;
        for (; i < owidth_align - elem_counts; i += elem_counts)
        {
            HVX_Vector v_offset_c  = *v_ofs++;
            HVX_Vector v_offset_r0 = *v_ofs++;
            HVX_Vector v_alpha0_c  = *v_alpha0++;
            HVX_Vector v_alpha0_r0 = *v_alpha0++;
            HVX_Vector v_alpha1_c  = *v_alpha1++;
            HVX_Vector v_alpha1_r0 = *v_alpha1++;
            HVX_Vector v_alpha2_c  = *v_alpha2++;
            HVX_Vector v_alpha2_r0 = *v_alpha2++;
            HVX_Vector v_alpha3_c  = *v_alpha3++;
            HVX_Vector v_alpha3_r0 = *v_alpha3++;
            HVX_VectorPair w_result0 = Q6_Wuw_vunpack_Vuh(v_offset_c);
            HVX_VectorPair w_result1 = Q6_Wuw_vunpack_Vuh(v_offset_r0);
            HVX_Vector v_offset_l1 = Q6_V_lo_W(w_result0);
            HVX_Vector v_offset_l0 = Q6_V_hi_W(w_result0);
            v_offset_c  = Q6_V_lo_W(w_result1);
            v_offset_r0 = Q6_V_hi_W(w_result1);

            HVX_Vector v_offset_add1_l1 = Q6_Vuw_vadd_VuwVuw_sat(v_offset_l1, Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add1_l0 = Q6_Vuw_vadd_VuwVuw_sat(v_offset_l0, Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add1_c  = Q6_Vuw_vadd_VuwVuw_sat(v_offset_c , Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add1_r0 = Q6_Vuw_vadd_VuwVuw_sat(v_offset_r0, Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add2_l1 = Q6_Vuw_vadd_VuwVuw_sat(v_offset_add1_l1, Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add2_l0 = Q6_Vuw_vadd_VuwVuw_sat(v_offset_add1_l0, Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add2_c  = Q6_Vuw_vadd_VuwVuw_sat(v_offset_add1_c , Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add2_r0 = Q6_Vuw_vadd_VuwVuw_sat(v_offset_add1_r0, Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add3_l1 = Q6_Vuw_vadd_VuwVuw_sat(v_offset_add2_l1, Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add3_l0 = Q6_Vuw_vadd_VuwVuw_sat(v_offset_add2_l0, Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add3_c  = Q6_Vuw_vadd_VuwVuw_sat(v_offset_add2_c , Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add3_r0 = Q6_Vuw_vadd_VuwVuw_sat(v_offset_add2_r0, Q6_V_vsplat_R(4));

            w_result0 = Q6_Ww_vunpack_Vh(v_alpha0_c);
            w_result1 = Q6_Ww_vunpack_Vh(v_alpha0_r0);
            HVX_Vector v_alpha0_l1 = Q6_V_lo_W(w_result0);
            HVX_Vector v_alpha0_l0 = Q6_V_hi_W(w_result0);
            v_alpha0_c  = Q6_V_lo_W(w_result1);
            v_alpha0_r0 = Q6_V_hi_W(w_result1);
            w_result0 = Q6_Ww_vunpack_Vh(v_alpha1_c);
            w_result1 = Q6_Ww_vunpack_Vh(v_alpha1_r0);
            HVX_Vector v_alpha1_l1 = Q6_V_lo_W(w_result0);
            HVX_Vector v_alpha1_l0 = Q6_V_hi_W(w_result0);
            v_alpha1_c  = Q6_V_lo_W(w_result1);
            v_alpha1_r0 = Q6_V_hi_W(w_result1);
            w_result0 = Q6_Ww_vunpack_Vh(v_alpha2_c);
            w_result1 = Q6_Ww_vunpack_Vh(v_alpha2_r0);
            HVX_Vector v_alpha2_l1 = Q6_V_lo_W(w_result0);
            HVX_Vector v_alpha2_l0 = Q6_V_hi_W(w_result0);
            v_alpha2_c  = Q6_V_lo_W(w_result1);
            v_alpha2_r0 = Q6_V_hi_W(w_result1);
            w_result0 = Q6_Ww_vunpack_Vh(v_alpha3_c);
            w_result1 = Q6_Ww_vunpack_Vh(v_alpha3_r0);
            HVX_Vector v_alpha3_l1 = Q6_V_lo_W(w_result0);
            HVX_Vector v_alpha3_l0 = Q6_V_hi_W(w_result0);
            v_alpha3_c  = Q6_V_lo_W(w_result1);
            v_alpha3_r0 = Q6_V_hi_W(w_result1);

            MVType mv_result;
            for (MI_S32 ch = 0; ch < C; ch++)
            {
                Q6_vgather_ARMVw(r_gather[ch * 16 + 0] , (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_l1);
                Q6_vgather_ARMVw(r_gather[ch * 16 + 1] , (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_l0);
                Q6_vgather_ARMVw(r_gather[ch * 16 + 2] , (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_c);
                Q6_vgather_ARMVw(r_gather[ch * 16 + 3] , (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_r0);
                Q6_vgather_ARMVw(r_gather[ch * 16 + 4] , (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add1_l1);
                Q6_vgather_ARMVw(r_gather[ch * 16 + 5] , (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add1_l0);
                Q6_vgather_ARMVw(r_gather[ch * 16 + 6] , (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add1_c);
                Q6_vgather_ARMVw(r_gather[ch * 16 + 7] , (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add1_r0);
                Q6_vgather_ARMVw(r_gather[ch * 16 + 8] , (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add2_l1);
                Q6_vgather_ARMVw(r_gather[ch * 16 + 9] , (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add2_l0);
                Q6_vgather_ARMVw(r_gather[ch * 16 + 10], (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add2_c);
                Q6_vgather_ARMVw(r_gather[ch * 16 + 11], (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add2_r0);
                Q6_vgather_ARMVw(r_gather[ch * 16 + 12], (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add3_l1);
                Q6_vgather_ARMVw(r_gather[ch * 16 + 13], (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add3_l0);
                Q6_vgather_ARMVw(r_gather[ch * 16 + 14], (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add3_c);
                Q6_vgather_ARMVw(r_gather[ch * 16 + 15], (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add3_r0);

                // q0, q1 to avoid rebundant addition from high 32-bits when the high-32bits are sign bits
                // (high 32-bits equal to -1, low 32-bits less than 0)
                w_result0 = Q6_Wd_vmul_VwVw(*l_gather[ch * 16 + 0], v_alpha0_l1);
                w_result1 = Q6_Wd_vmul_VwVw(*l_gather[ch * 16 + 1], v_alpha0_l0);
                w_result0 = Q6_Wd_vmulacc_WdVwVw(w_result0, *l_gather[ch * 16 + 4], v_alpha1_l1);
                w_result1 = Q6_Wd_vmulacc_WdVwVw(w_result1, *l_gather[ch * 16 + 5], v_alpha1_l0);
                w_result0 = Q6_Wd_vmulacc_WdVwVw(w_result0, *l_gather[ch * 16 + 8], v_alpha2_l1);
                w_result1 = Q6_Wd_vmulacc_WdVwVw(w_result1, *l_gather[ch * 16 + 9], v_alpha2_l0);
                w_result0 = Q6_Wd_vmulacc_WdVwVw(w_result0, *l_gather[ch * 16 + 12], v_alpha3_l1);
                w_result1 = Q6_Wd_vmulacc_WdVwVw(w_result1, *l_gather[ch * 16 + 13], v_alpha3_l0);

                HVX_Vector v_result0 = Q6_Vw_vasl_VwR(Q6_V_hi_W(w_result0), 10);
                HVX_Vector v_result1 = Q6_Vw_vasl_VwR(Q6_V_hi_W(w_result1), 10);
                HVX_Vector v_result2 = Q6_Vh_vpacko_VwVw(Q6_V_lo_W(w_result1), Q6_V_lo_W(w_result0));
                HVX_Vector v_result3 = Q6_Vh_vadd_VhVh(v_result2, Q6_Vh_vsplat_R(32));
                HVX_VectorPred q0    = Q6_Q_vcmp_gt_VhVh(Q6_Vh_vsplat_R(0), v_result2);
                v_result0            = Q6_Vh_vpack_VwVw_sat(v_result1, v_result0);
                v_result2            = Q6_Vh_vasr_VhR(v_result3, 6);
                HVX_VectorPred q1    = Q6_Q_vcmp_eqand_QVhVh(q0, v_result0, Q6_Vh_vsplat_R(-1024));
                v_result1            = Q6_V_vmux_QVV(q1, Q6_Vh_vsplat_R(0), v_result0);
                v_result3            = Q6_Vh_vadd_VhVh(v_result1, v_result2);

                w_result0 = Q6_Wd_vmul_VwVw(*l_gather[ch * 16 + 2], v_alpha0_c );
                w_result1 = Q6_Wd_vmul_VwVw(*l_gather[ch * 16 + 3], v_alpha0_r0);
                w_result0 = Q6_Wd_vmulacc_WdVwVw(w_result0, *l_gather[ch * 16 + 6], v_alpha1_c );
                w_result1 = Q6_Wd_vmulacc_WdVwVw(w_result1, *l_gather[ch * 16 + 7], v_alpha1_r0);
                w_result0 = Q6_Wd_vmulacc_WdVwVw(w_result0, *l_gather[ch * 16 + 10], v_alpha2_c );
                w_result1 = Q6_Wd_vmulacc_WdVwVw(w_result1, *l_gather[ch * 16 + 11], v_alpha2_r0);
                w_result0 = Q6_Wd_vmulacc_WdVwVw(w_result0, *l_gather[ch * 16 + 14], v_alpha3_c );
                w_result1 = Q6_Wd_vmulacc_WdVwVw(w_result1, *l_gather[ch * 16 + 15], v_alpha3_r0);

                v_result0 = Q6_Vw_vasl_VwR(Q6_V_hi_W(w_result0), 10);
                v_result1 = Q6_Vw_vasl_VwR(Q6_V_hi_W(w_result1), 10);
                v_result2 = Q6_Vh_vpacko_VwVw(Q6_V_lo_W(w_result1), Q6_V_lo_W(w_result0));
                v_result1 = Q6_Vh_vpack_VwVw_sat(v_result1, v_result0);
                q0        = Q6_Q_vcmp_gt_VhVh(Q6_Vh_vsplat_R(0), v_result2);
                v_result0 = Q6_Vh_vadd_VhVh(v_result2, Q6_Vh_vsplat_R(32));
                q1        = Q6_Q_vcmp_eqand_QVhVh(q0, v_result1, Q6_Vh_vsplat_R(-1024));
                v_result2 = Q6_Vh_vasr_VhR(v_result0, 6);
                v_result0 = Q6_V_vmux_QVV(q1, Q6_Vh_vsplat_R(0), v_result1);
                v_result1 = Q6_Vh_vadd_VhVh(v_result0, v_result2);

                mv_result.val[ch] = Q6_Vb_vpack_VhVh_sat(v_result1, v_result3);

                v_tmp[0] = r_gather[ch * 16 + 0] ; r_gather[ch * 16 + 0]  = l_gather[ch * 16 + 0] ; l_gather[ch * 16 + 0]  = v_tmp[0];
                v_tmp[1] = r_gather[ch * 16 + 1] ; r_gather[ch * 16 + 1]  = l_gather[ch * 16 + 1] ; l_gather[ch * 16 + 1]  = v_tmp[1];
                v_tmp[2] = r_gather[ch * 16 + 2] ; r_gather[ch * 16 + 2]  = l_gather[ch * 16 + 2] ; l_gather[ch * 16 + 2]  = v_tmp[2];
                v_tmp[3] = r_gather[ch * 16 + 3] ; r_gather[ch * 16 + 3]  = l_gather[ch * 16 + 3] ; l_gather[ch * 16 + 3]  = v_tmp[3];
                v_tmp[0] = r_gather[ch * 16 + 4] ; r_gather[ch * 16 + 4]  = l_gather[ch * 16 + 4] ; l_gather[ch * 16 + 4]  = v_tmp[0];
                v_tmp[1] = r_gather[ch * 16 + 5] ; r_gather[ch * 16 + 5]  = l_gather[ch * 16 + 5] ; l_gather[ch * 16 + 5]  = v_tmp[1];
                v_tmp[2] = r_gather[ch * 16 + 6] ; r_gather[ch * 16 + 6]  = l_gather[ch * 16 + 6] ; l_gather[ch * 16 + 6]  = v_tmp[2];
                v_tmp[3] = r_gather[ch * 16 + 7] ; r_gather[ch * 16 + 7]  = l_gather[ch * 16 + 7] ; l_gather[ch * 16 + 7]  = v_tmp[3];
                v_tmp[0] = r_gather[ch * 16 + 8] ; r_gather[ch * 16 + 8]  = l_gather[ch * 16 + 8] ; l_gather[ch * 16 + 8]  = v_tmp[0];
                v_tmp[1] = r_gather[ch * 16 + 9] ; r_gather[ch * 16 + 9]  = l_gather[ch * 16 + 9] ; l_gather[ch * 16 + 9]  = v_tmp[1];
                v_tmp[2] = r_gather[ch * 16 + 10]; r_gather[ch * 16 + 10] = l_gather[ch * 16 + 10]; l_gather[ch * 16 + 10] = v_tmp[2];
                v_tmp[3] = r_gather[ch * 16 + 11]; r_gather[ch * 16 + 11] = l_gather[ch * 16 + 11]; l_gather[ch * 16 + 11] = v_tmp[3];
                v_tmp[0] = r_gather[ch * 16 + 12]; r_gather[ch * 16 + 12] = l_gather[ch * 16 + 12]; l_gather[ch * 16 + 12] = v_tmp[0];
                v_tmp[1] = r_gather[ch * 16 + 13]; r_gather[ch * 16 + 13] = l_gather[ch * 16 + 13]; l_gather[ch * 16 + 13] = v_tmp[1];
                v_tmp[2] = r_gather[ch * 16 + 14]; r_gather[ch * 16 + 14] = l_gather[ch * 16 + 14]; l_gather[ch * 16 + 14] = v_tmp[2];
                v_tmp[3] = r_gather[ch * 16 + 15]; r_gather[ch * 16 + 15] = l_gather[ch * 16 + 15]; l_gather[ch * 16 + 15] = v_tmp[3];
            }

            vstore(dst + i, mv_result);
        }

        {
            HVX_Vector v_alpha0_c  = *v_alpha0++;
            HVX_Vector v_alpha0_r0 = *v_alpha0++;
            HVX_Vector v_alpha1_c  = *v_alpha1++;
            HVX_Vector v_alpha1_r0 = *v_alpha1++;
            HVX_Vector v_alpha2_c  = *v_alpha2++;
            HVX_Vector v_alpha2_r0 = *v_alpha2++;
            HVX_Vector v_alpha3_c  = *v_alpha3++;
            HVX_Vector v_alpha3_r0 = *v_alpha3++;

            HVX_VectorPair w_result0 = Q6_Ww_vunpack_Vh(v_alpha0_c);
            HVX_VectorPair w_result1 = Q6_Ww_vunpack_Vh(v_alpha0_r0);
            HVX_Vector v_alpha0_l1 = Q6_V_lo_W(w_result0);
            HVX_Vector v_alpha0_l0 = Q6_V_hi_W(w_result0);
            v_alpha0_c  = Q6_V_lo_W(w_result1);
            v_alpha0_r0 = Q6_V_hi_W(w_result1);
            w_result0 = Q6_Ww_vunpack_Vh(v_alpha1_c);
            w_result1 = Q6_Ww_vunpack_Vh(v_alpha1_r0);
            HVX_Vector v_alpha1_l1 = Q6_V_lo_W(w_result0);
            HVX_Vector v_alpha1_l0 = Q6_V_hi_W(w_result0);
            v_alpha1_c  = Q6_V_lo_W(w_result1);
            v_alpha1_r0 = Q6_V_hi_W(w_result1);
            w_result0 = Q6_Ww_vunpack_Vh(v_alpha2_c);
            w_result1 = Q6_Ww_vunpack_Vh(v_alpha2_r0);
            HVX_Vector v_alpha2_l1 = Q6_V_lo_W(w_result0);
            HVX_Vector v_alpha2_l0 = Q6_V_hi_W(w_result0);
            v_alpha2_c  = Q6_V_lo_W(w_result1);
            v_alpha2_r0 = Q6_V_hi_W(w_result1);
            w_result0 = Q6_Ww_vunpack_Vh(v_alpha3_c);
            w_result1 = Q6_Ww_vunpack_Vh(v_alpha3_r0);
            HVX_Vector v_alpha3_l1 = Q6_V_lo_W(w_result0);
            HVX_Vector v_alpha3_l0 = Q6_V_hi_W(w_result0);
            v_alpha3_c  = Q6_V_lo_W(w_result1);
            v_alpha3_r0 = Q6_V_hi_W(w_result1);

            MVType mv_result;
            for (MI_S32 ch = 0; ch < C; ch++)
            {
                w_result0 = Q6_Wd_vmul_VwVw(*l_gather[ch * 16 + 0], v_alpha0_l1);
                w_result1 = Q6_Wd_vmul_VwVw(*l_gather[ch * 16 + 1], v_alpha0_l0);
                w_result0 = Q6_Wd_vmulacc_WdVwVw(w_result0, *l_gather[ch * 16 + 4], v_alpha1_l1);
                w_result1 = Q6_Wd_vmulacc_WdVwVw(w_result1, *l_gather[ch * 16 + 5], v_alpha1_l0);
                w_result0 = Q6_Wd_vmulacc_WdVwVw(w_result0, *l_gather[ch * 16 + 8], v_alpha2_l1);
                w_result1 = Q6_Wd_vmulacc_WdVwVw(w_result1, *l_gather[ch * 16 + 9], v_alpha2_l0);
                w_result0 = Q6_Wd_vmulacc_WdVwVw(w_result0, *l_gather[ch * 16 + 12], v_alpha3_l1);
                w_result1 = Q6_Wd_vmulacc_WdVwVw(w_result1, *l_gather[ch * 16 + 13], v_alpha3_l0);

                HVX_Vector v_result0 = Q6_Vw_vasl_VwR(Q6_V_hi_W(w_result0), 10);
                HVX_Vector v_result1 = Q6_Vw_vasl_VwR(Q6_V_hi_W(w_result1), 10);
                HVX_Vector v_result2 = Q6_Vh_vpacko_VwVw(Q6_V_lo_W(w_result1), Q6_V_lo_W(w_result0));
                HVX_Vector v_result3 = Q6_Vh_vadd_VhVh(v_result2, Q6_Vh_vsplat_R(32));
                HVX_VectorPred q0    = Q6_Q_vcmp_gt_VhVh(Q6_Vh_vsplat_R(0), v_result2);
                v_result0            = Q6_Vh_vpack_VwVw_sat(v_result1, v_result0);
                v_result2            = Q6_Vh_vasr_VhR(v_result3, 6);
                HVX_VectorPred q1    = Q6_Q_vcmp_eqand_QVhVh(q0, v_result0, Q6_Vh_vsplat_R(-1024));
                v_result1            = Q6_V_vmux_QVV(q1, Q6_Vh_vsplat_R(0), v_result0);
                v_result3            = Q6_Vh_vadd_VhVh(v_result1, v_result2);

                w_result0 = Q6_Wd_vmul_VwVw(*l_gather[ch * 16 + 2], v_alpha0_c);
                w_result1 = Q6_Wd_vmul_VwVw(*l_gather[ch * 16 + 3], v_alpha0_r0);
                w_result0 = Q6_Wd_vmulacc_WdVwVw(w_result0, *l_gather[ch * 16 + 6], v_alpha1_c);
                w_result1 = Q6_Wd_vmulacc_WdVwVw(w_result1, *l_gather[ch * 16 + 7], v_alpha1_r0);
                w_result0 = Q6_Wd_vmulacc_WdVwVw(w_result0, *l_gather[ch * 16 + 10], v_alpha2_c);
                w_result1 = Q6_Wd_vmulacc_WdVwVw(w_result1, *l_gather[ch * 16 + 11], v_alpha2_r0);
                w_result0 = Q6_Wd_vmulacc_WdVwVw(w_result0, *l_gather[ch * 16 + 14], v_alpha3_c);
                w_result1 = Q6_Wd_vmulacc_WdVwVw(w_result1, *l_gather[ch * 16 + 15], v_alpha3_r0);

                v_result0 = Q6_Vw_vasl_VwR(Q6_V_hi_W(w_result0), 10);
                v_result1 = Q6_Vw_vasl_VwR(Q6_V_hi_W(w_result1), 10);
                v_result2 = Q6_Vh_vpacko_VwVw(Q6_V_lo_W(w_result1), Q6_V_lo_W(w_result0));
                v_result1 = Q6_Vh_vpack_VwVw_sat(v_result1, v_result0);
                q0        = Q6_Q_vcmp_gt_VhVh(Q6_Vh_vsplat_R(0), v_result2);
                v_result0 = Q6_Vh_vadd_VhVh(v_result2, Q6_Vh_vsplat_R(32));
                q1        = Q6_Q_vcmp_eqand_QVhVh(q0, v_result1, Q6_Vh_vsplat_R(-1024));
                v_result2 = Q6_Vh_vasr_VhR(v_result0, 6);
                v_result0 = Q6_V_vmux_QVV(q1, Q6_Vh_vsplat_R(0), v_result1);
                v_result1 = Q6_Vh_vadd_VhVh(v_result0, v_result2);

                mv_result.val[ch] = Q6_Vb_vpack_VhVh_sat(v_result1, v_result3);
            }

            vstore(dst + i, mv_result);
        }

        if (ostep > owidth_align)
        {
            i = ostep - elem_counts;
            j = i / C;
            HVX_Vector v_offset_c   = vmemu(xofs + j);
            HVX_Vector v_offset_r0  = vmemu(xofs + j + HALF_AURA_HVLEN);
            HVX_Vector v_alpha0_c   = vmemu(alpha0 + j);
            HVX_Vector v_alpha0_r0  = vmemu(alpha0 + j + HALF_AURA_HVLEN);
            HVX_Vector v_alpha1_c   = vmemu(alpha1 + j);
            HVX_Vector v_alpha1_r0  = vmemu(alpha1 + j + HALF_AURA_HVLEN);
            HVX_Vector v_alpha2_c   = vmemu(alpha2 + j);
            HVX_Vector v_alpha2_r0  = vmemu(alpha2 + j + HALF_AURA_HVLEN);
            HVX_Vector v_alpha3_c   = vmemu(alpha3 + j);
            HVX_Vector v_alpha3_r0  = vmemu(alpha3 + j + HALF_AURA_HVLEN);
            HVX_VectorPair w_result0 = Q6_Wuw_vunpack_Vuh(v_offset_c);
            HVX_VectorPair w_result1 = Q6_Wuw_vunpack_Vuh(v_offset_r0);
            HVX_Vector v_offset_l1 = Q6_V_lo_W(w_result0);
            HVX_Vector v_offset_l0 = Q6_V_hi_W(w_result0);
            v_offset_c  = Q6_V_lo_W(w_result1);
            v_offset_r0 = Q6_V_hi_W(w_result1);

            HVX_Vector v_offset_add1_l1 = Q6_Vuw_vadd_VuwVuw_sat(v_offset_l1, Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add1_l0 = Q6_Vuw_vadd_VuwVuw_sat(v_offset_l0, Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add1_c  = Q6_Vuw_vadd_VuwVuw_sat(v_offset_c , Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add1_r0 = Q6_Vuw_vadd_VuwVuw_sat(v_offset_r0, Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add2_l1 = Q6_Vuw_vadd_VuwVuw_sat(v_offset_add1_l1, Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add2_l0 = Q6_Vuw_vadd_VuwVuw_sat(v_offset_add1_l0, Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add2_c  = Q6_Vuw_vadd_VuwVuw_sat(v_offset_add1_c , Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add2_r0 = Q6_Vuw_vadd_VuwVuw_sat(v_offset_add1_r0, Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add3_l1 = Q6_Vuw_vadd_VuwVuw_sat(v_offset_add2_l1, Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add3_l0 = Q6_Vuw_vadd_VuwVuw_sat(v_offset_add2_l0, Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add3_c  = Q6_Vuw_vadd_VuwVuw_sat(v_offset_add2_c , Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add3_r0 = Q6_Vuw_vadd_VuwVuw_sat(v_offset_add2_r0, Q6_V_vsplat_R(4));

            w_result0 = Q6_Ww_vunpack_Vh(v_alpha0_c);
            w_result1 = Q6_Ww_vunpack_Vh(v_alpha0_r0);
            HVX_Vector v_alpha0_l1 = Q6_V_lo_W(w_result0);
            HVX_Vector v_alpha0_l0 = Q6_V_hi_W(w_result0);
            v_alpha0_c  = Q6_V_lo_W(w_result1);
            v_alpha0_r0 = Q6_V_hi_W(w_result1);
            w_result0 = Q6_Ww_vunpack_Vh(v_alpha1_c);
            w_result1 = Q6_Ww_vunpack_Vh(v_alpha1_r0);
            HVX_Vector v_alpha1_l1 = Q6_V_lo_W(w_result0);
            HVX_Vector v_alpha1_l0 = Q6_V_hi_W(w_result0);
            v_alpha1_c  = Q6_V_lo_W(w_result1);
            v_alpha1_r0 = Q6_V_hi_W(w_result1);
            w_result0 = Q6_Ww_vunpack_Vh(v_alpha2_c);
            w_result1 = Q6_Ww_vunpack_Vh(v_alpha2_r0);
            HVX_Vector v_alpha2_l1 = Q6_V_lo_W(w_result0);
            HVX_Vector v_alpha2_l0 = Q6_V_hi_W(w_result0);
            v_alpha2_c  = Q6_V_lo_W(w_result1);
            v_alpha2_r0 = Q6_V_hi_W(w_result1);
            w_result0 = Q6_Ww_vunpack_Vh(v_alpha3_c);
            w_result1 = Q6_Ww_vunpack_Vh(v_alpha3_r0);
            HVX_Vector v_alpha3_l1 = Q6_V_lo_W(w_result0);
            HVX_Vector v_alpha3_l0 = Q6_V_hi_W(w_result0);
            v_alpha3_c  = Q6_V_lo_W(w_result1);
            v_alpha3_r0 = Q6_V_hi_W(w_result1);

            MVType mv_result;
            for (MI_S32 ch = 0; ch < C; ch++)
            {
                Q6_vgather_ARMVw(l_gather[ch * 16 +  0], (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_l1);
                Q6_vgather_ARMVw(l_gather[ch * 16 +  1], (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_l0);
                Q6_vgather_ARMVw(l_gather[ch * 16 +  2], (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_c);
                Q6_vgather_ARMVw(l_gather[ch * 16 +  3], (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_r0);
                Q6_vgather_ARMVw(l_gather[ch * 16 +  4], (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add1_l1);
                Q6_vgather_ARMVw(l_gather[ch * 16 +  5], (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add1_l0);
                Q6_vgather_ARMVw(l_gather[ch * 16 +  6], (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add1_c);
                Q6_vgather_ARMVw(l_gather[ch * 16 +  7], (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add1_r0);
                Q6_vgather_ARMVw(l_gather[ch * 16 +  8], (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add2_l1);
                Q6_vgather_ARMVw(l_gather[ch * 16 +  9], (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add2_l0);
                Q6_vgather_ARMVw(l_gather[ch * 16 + 10], (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add2_c);
                Q6_vgather_ARMVw(l_gather[ch * 16 + 11], (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add2_r0);
                Q6_vgather_ARMVw(l_gather[ch * 16 + 12], (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add3_l1);
                Q6_vgather_ARMVw(l_gather[ch * 16 + 13], (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add3_l0);
                Q6_vgather_ARMVw(l_gather[ch * 16 + 14], (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add3_c);
                Q6_vgather_ARMVw(l_gather[ch * 16 + 15], (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add3_r0);

                w_result0 = Q6_Wd_vmul_VwVw(*l_gather[ch * 16 + 0], v_alpha0_l1);
                w_result1 = Q6_Wd_vmul_VwVw(*l_gather[ch * 16 + 1], v_alpha0_l0);
                w_result0 = Q6_Wd_vmulacc_WdVwVw(w_result0, *l_gather[ch * 16 + 4], v_alpha1_l1);
                w_result1 = Q6_Wd_vmulacc_WdVwVw(w_result1, *l_gather[ch * 16 + 5], v_alpha1_l0);
                w_result0 = Q6_Wd_vmulacc_WdVwVw(w_result0, *l_gather[ch * 16 + 8], v_alpha2_l1);
                w_result1 = Q6_Wd_vmulacc_WdVwVw(w_result1, *l_gather[ch * 16 + 9], v_alpha2_l0);
                w_result0 = Q6_Wd_vmulacc_WdVwVw(w_result0, *l_gather[ch * 16 + 12], v_alpha3_l1);
                w_result1 = Q6_Wd_vmulacc_WdVwVw(w_result1, *l_gather[ch * 16 + 13], v_alpha3_l0);

                HVX_Vector v_result0 = Q6_Vw_vasl_VwR(Q6_V_hi_W(w_result0), 10);
                HVX_Vector v_result1 = Q6_Vw_vasl_VwR(Q6_V_hi_W(w_result1), 10);
                HVX_Vector v_result2 = Q6_Vh_vpacko_VwVw(Q6_V_lo_W(w_result1), Q6_V_lo_W(w_result0));
                HVX_Vector v_result3 = Q6_Vh_vadd_VhVh(v_result2, Q6_Vh_vsplat_R(32));
                HVX_VectorPred q0    = Q6_Q_vcmp_gt_VhVh(Q6_Vh_vsplat_R(0), v_result2);
                v_result0            = Q6_Vh_vpack_VwVw_sat(v_result1, v_result0);
                v_result2            = Q6_Vh_vasr_VhR(v_result3, 6);
                HVX_VectorPred q1    = Q6_Q_vcmp_eqand_QVhVh(q0, v_result0, Q6_Vh_vsplat_R(-1024));
                v_result1            = Q6_V_vmux_QVV(q1, Q6_Vh_vsplat_R(0), v_result0);
                v_result3            = Q6_Vh_vadd_VhVh(v_result1, v_result2);

                w_result0 = Q6_Wd_vmul_VwVw(*l_gather[ch * 16 + 2], v_alpha0_c);
                w_result1 = Q6_Wd_vmul_VwVw(*l_gather[ch * 16 + 3], v_alpha0_r0);
                w_result0 = Q6_Wd_vmulacc_WdVwVw(w_result0, *l_gather[ch * 16 + 6], v_alpha1_c);
                w_result1 = Q6_Wd_vmulacc_WdVwVw(w_result1, *l_gather[ch * 16 + 7], v_alpha1_r0);
                w_result0 = Q6_Wd_vmulacc_WdVwVw(w_result0, *l_gather[ch * 16 + 10], v_alpha2_c);
                w_result1 = Q6_Wd_vmulacc_WdVwVw(w_result1, *l_gather[ch * 16 + 11], v_alpha2_r0);
                w_result0 = Q6_Wd_vmulacc_WdVwVw(w_result0, *l_gather[ch * 16 + 14], v_alpha3_c);
                w_result1 = Q6_Wd_vmulacc_WdVwVw(w_result1, *l_gather[ch * 16 + 15], v_alpha3_r0);

                v_result0 = Q6_Vw_vasl_VwR(Q6_V_hi_W(w_result0), 10);
                v_result1 = Q6_Vw_vasl_VwR(Q6_V_hi_W(w_result1), 10);
                v_result2 = Q6_Vh_vpacko_VwVw(Q6_V_lo_W(w_result1), Q6_V_lo_W(w_result0));
                v_result1 = Q6_Vh_vpack_VwVw_sat(v_result1, v_result0);
                q0        = Q6_Q_vcmp_gt_VhVh(Q6_Vh_vsplat_R(0), v_result2);
                v_result0 = Q6_Vh_vadd_VhVh(v_result2, Q6_Vh_vsplat_R(32));
                q1        = Q6_Q_vcmp_eqand_QVhVh(q0, v_result1, Q6_Vh_vsplat_R(-1024));
                v_result2 = Q6_Vh_vasr_VhR(v_result0, 6);
                v_result0 = Q6_V_vmux_QVV(q1, Q6_Vh_vsplat_R(0), v_result1);
                v_result1 = Q6_Vh_vadd_VhVh(v_result0, v_result2);

                mv_result.val[ch] = Q6_Vb_vpack_VhVh_sat(v_result1, v_result3);
            }

            vstore(dst + i, mv_result);
        }
    }
}

// Tp = MI_U16
template<typename Tp, MI_S32 C>
static typename std::enable_if<std::is_same<MI_U16, Tp>::value, AURA_VOID>::type
ResizeCuCommPerRow(const Tp *src_c, Tp *dst, MI_S32 *row_base, MI_S32 *vtcm_base, MI_U16 *xofs,
                   MI_S16 *beta_base, MI_S16 *alpha0_base, MI_S16 *alpha1_base, MI_S16 *alpha2_base,
                   MI_S16 *alpha3_base, AURA_VOID *l_gather_base, AURA_VOID *r_gather_base, MI_S32 iwidth,
                   MI_S32 istride, MI_S32 istep, MI_S32 ostep)
{
    AURA_UNUSED(l_gather_base);
    AURA_UNUSED(r_gather_base);
    using MVType = typename MVHvxVector<C>::Type;
    constexpr MI_S32 elem_counts = AURA_HVLEN * C / sizeof(Tp);
    constexpr MI_S32 elem_counts_channel = elem_counts / C;
    MI_S32 iwidth_align = istep / elem_counts * elem_counts;

    MI_F32 *row    = reinterpret_cast<MI_F32 *>(row_base);
    MI_S32 *beta   = reinterpret_cast<MI_S32 *>(beta_base);
    MI_F32 *alpha0 = reinterpret_cast<MI_F32 *>(alpha0_base);
    MI_F32 *alpha1 = reinterpret_cast<MI_F32 *>(alpha1_base);
    MI_F32 *alpha2 = reinterpret_cast<MI_F32 *>(alpha2_base);
    MI_F32 *alpha3 = reinterpret_cast<MI_F32 *>(alpha3_base);

    // part1: calculate the median and save to vtcm
    MI_S32 i = 0, j = 0;
    {
        const Tp *src_n0 = src_c +  istride;
        const Tp *src_n1 = src_n0 + istride;
        const Tp *src_n2 = src_n1 + istride;

        HVX_Vector v_beta0 = Q6_V_vsplat_R(beta[0]);
        HVX_Vector v_beta1 = Q6_V_vsplat_R(beta[1]);
        HVX_Vector v_beta2 = Q6_V_vsplat_R(beta[2]);
        HVX_Vector v_beta3 = Q6_V_vsplat_R(beta[3]);

        for (; i < iwidth_align; i += elem_counts, j += elem_counts_channel)
        {
            MVType mv_c_src, mv_n0_src, mv_n1_src, mv_n2_src;
            vload(src_c + i,  mv_c_src);
            vload(src_n0 + i, mv_n0_src);
            vload(src_n1 + i, mv_n1_src);
            vload(src_n2 + i, mv_n2_src);

            for (MI_S32 ch = 0; ch < C; ch++)
            {
                HVX_Vector *v_dst = (HVX_Vector *)(row + j + ch * AURA_ALIGN(iwidth, QUAR_AURA_HVLEN));
                HVX_VectorPair w_result0 = Q6_Wuw_vunpack_Vuh(mv_c_src.val[ch]);
                HVX_VectorPair w_result1 = Q6_Wuw_vunpack_Vuh(mv_n0_src.val[ch]);

                HVX_Vector v_src0 = Q6_Vsf_vcvt_Vw(Q6_V_lo_W(w_result0));
                HVX_Vector v_src1 = Q6_Vsf_vcvt_Vw(Q6_V_hi_W(w_result0));
                HVX_Vector v_src2 = Q6_Vsf_vcvt_Vw(Q6_V_lo_W(w_result1));
                HVX_Vector v_src3 = Q6_Vsf_vcvt_Vw(Q6_V_hi_W(w_result1));
                v_src0 = Q6_Vqf32_vmpy_VsfVsf(v_src0, v_beta0);
                v_src1 = Q6_Vqf32_vmpy_VsfVsf(v_src1, v_beta0);
                v_src2 = Q6_Vqf32_vmpy_VsfVsf(v_src2, v_beta1);
                v_src3 = Q6_Vqf32_vmpy_VsfVsf(v_src3, v_beta1);

                w_result0 = Q6_Wuw_vunpack_Vuh(mv_n1_src.val[ch]);
                w_result1 = Q6_Wuw_vunpack_Vuh(mv_n2_src.val[ch]);
                v_src0 = Q6_Vqf32_vadd_Vqf32Vqf32(v_src0, Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_vcvt_Vw(Q6_V_lo_W(w_result0)), v_beta2));
                v_src1 = Q6_Vqf32_vadd_Vqf32Vqf32(v_src1, Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_vcvt_Vw(Q6_V_hi_W(w_result0)), v_beta2));
                v_src2 = Q6_Vqf32_vadd_Vqf32Vqf32(v_src2, Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_vcvt_Vw(Q6_V_lo_W(w_result1)), v_beta3));
                v_src3 = Q6_Vqf32_vadd_Vqf32Vqf32(v_src3, Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_vcvt_Vw(Q6_V_hi_W(w_result1)), v_beta3));
                v_src0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(v_src0, v_src2));
                v_src1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(v_src1, v_src3));

                *v_dst++ = v_src0;
                *v_dst = v_src1;
            }
        }

        if (istep > iwidth_align)
        {
            MVType mv_c_src, mv_n0_src, mv_n1_src, mv_n2_src;
            i = istep - elem_counts;
            j = i / C;

            vload(src_c + i,  mv_c_src);
            vload(src_n0 + i, mv_n0_src);
            vload(src_n1 + i, mv_n1_src);
            vload(src_n2 + i, mv_n2_src);

            for (MI_S32 ch = 0; ch < C; ch++)
            {
                MI_F32 *row_tmp = row + j + ch * AURA_ALIGN(iwidth, QUAR_AURA_HVLEN);
                HVX_VectorPair w_result0 = Q6_Wuw_vunpack_Vuh(mv_c_src.val[ch]);
                HVX_VectorPair w_result1 = Q6_Wuw_vunpack_Vuh(mv_n0_src.val[ch]);

                HVX_Vector v_src0 = Q6_Vsf_vcvt_Vw(Q6_V_lo_W(w_result0));
                HVX_Vector v_src1 = Q6_Vsf_vcvt_Vw(Q6_V_hi_W(w_result0));
                HVX_Vector v_src2 = Q6_Vsf_vcvt_Vw(Q6_V_lo_W(w_result1));
                HVX_Vector v_src3 = Q6_Vsf_vcvt_Vw(Q6_V_hi_W(w_result1));
                v_src0 = Q6_Vqf32_vmpy_VsfVsf(v_src0, v_beta0);
                v_src1 = Q6_Vqf32_vmpy_VsfVsf(v_src1, v_beta0);
                v_src2 = Q6_Vqf32_vmpy_VsfVsf(v_src2, v_beta1);
                v_src3 = Q6_Vqf32_vmpy_VsfVsf(v_src3, v_beta1);

                w_result0 = Q6_Wuw_vunpack_Vuh(mv_n1_src.val[ch]);
                w_result1 = Q6_Wuw_vunpack_Vuh(mv_n2_src.val[ch]);

                v_src0 = Q6_Vqf32_vadd_Vqf32Vqf32(v_src0, Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_vcvt_Vw(Q6_V_lo_W(w_result0)), v_beta2));
                v_src1 = Q6_Vqf32_vadd_Vqf32Vqf32(v_src1, Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_vcvt_Vw(Q6_V_hi_W(w_result0)), v_beta2));
                v_src2 = Q6_Vqf32_vadd_Vqf32Vqf32(v_src2, Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_vcvt_Vw(Q6_V_lo_W(w_result1)), v_beta3));
                v_src3 = Q6_Vqf32_vadd_Vqf32Vqf32(v_src3, Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_vcvt_Vw(Q6_V_hi_W(w_result1)), v_beta3));
                v_src0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(v_src0, v_src2));
                v_src1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(v_src1, v_src3));

                vmemu(row_tmp) = v_src0;
                vmemu(row_tmp + QUAR_AURA_HVLEN) = v_src1;
            }
        }
    }

    // part2: load vtcm and calculate dst
    {
        MI_S32 *vtcm = reinterpret_cast<MI_S32 *>(vtcm_base);
        MI_S32 owidth_align = ostep / elem_counts * elem_counts;

        HVX_Vector *v_ofs    = (HVX_Vector *)xofs;
        HVX_Vector *v_alpha0 = (HVX_Vector *)alpha0; HVX_Vector *v_alpha1 = (HVX_Vector *)alpha1;
        HVX_Vector *v_alpha2 = (HVX_Vector *)alpha2; HVX_Vector *v_alpha3 = (HVX_Vector *)alpha3;
        HVX_Vector *l_gather[C][8];
        HVX_Vector *r_gather[C][8];
        HVX_Vector *v_tmp[4];
        MI_F32 *row_ch[C];

        {
            HVX_Vector v_offset_c  = *v_ofs++;
            HVX_VectorPair w_result0 = Q6_Wuw_vunpack_Vuh(v_offset_c);
            v_offset_c = Q6_V_lo_W(w_result0);
            HVX_Vector v_offset_r0   = Q6_V_hi_W(w_result0);

            HVX_Vector v_offset_add1_c  = Q6_Vuw_vadd_VuwVuw_sat(v_offset_c , Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add1_r0 = Q6_Vuw_vadd_VuwVuw_sat(v_offset_r0, Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add2_c  = Q6_Vuw_vadd_VuwVuw_sat(v_offset_add1_c , Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add2_r0 = Q6_Vuw_vadd_VuwVuw_sat(v_offset_add1_r0, Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add3_c  = Q6_Vuw_vadd_VuwVuw_sat(v_offset_add2_c , Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add3_r0 = Q6_Vuw_vadd_VuwVuw_sat(v_offset_add2_r0, Q6_V_vsplat_R(4));

            for (MI_S32 ch = 0; ch < C; ch++)
            {
                MI_S32 *vtcm_gather = vtcm + ch * (QUAR_AURA_HVLEN << 4);
                row_ch[ch] = row + ch * AURA_ALIGN(iwidth, QUAR_AURA_HVLEN);

                l_gather[ch][0] = (HVX_Vector *)(vtcm_gather);
                l_gather[ch][1] = (HVX_Vector *)(vtcm_gather + QUAR_AURA_HVLEN);
                l_gather[ch][2] = (HVX_Vector *)(vtcm_gather + HALF_AURA_HVLEN);
                l_gather[ch][3] = (HVX_Vector *)(vtcm_gather + QUAR_AURA_HVLEN * 3);
                l_gather[ch][4] = (HVX_Vector *)(vtcm_gather + AURA_HVLEN);
                l_gather[ch][5] = (HVX_Vector *)(vtcm_gather + QUAR_AURA_HVLEN * 5);
                l_gather[ch][6] = (HVX_Vector *)(vtcm_gather + QUAR_AURA_HVLEN * 6);
                l_gather[ch][7] = (HVX_Vector *)(vtcm_gather + QUAR_AURA_HVLEN * 7);
                r_gather[ch][0] = (HVX_Vector *)(vtcm_gather + (AURA_HVLEN << 1));
                r_gather[ch][1] = (HVX_Vector *)(vtcm_gather + QUAR_AURA_HVLEN * 9);
                r_gather[ch][2] = (HVX_Vector *)(vtcm_gather + QUAR_AURA_HVLEN * 10);
                r_gather[ch][3] = (HVX_Vector *)(vtcm_gather + QUAR_AURA_HVLEN * 11);
                r_gather[ch][4] = (HVX_Vector *)(vtcm_gather + QUAR_AURA_HVLEN * 12);
                r_gather[ch][5] = (HVX_Vector *)(vtcm_gather + QUAR_AURA_HVLEN * 13);
                r_gather[ch][6] = (HVX_Vector *)(vtcm_gather + QUAR_AURA_HVLEN * 14);
                r_gather[ch][7] = (HVX_Vector *)(vtcm_gather + QUAR_AURA_HVLEN * 15);

                Q6_vgather_ARMVw(l_gather[ch][0] , (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_c );
                Q6_vgather_ARMVw(l_gather[ch][1] , (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_r0);
                Q6_vgather_ARMVw(l_gather[ch][2] , (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add1_c );
                Q6_vgather_ARMVw(l_gather[ch][3] , (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add1_r0);
                Q6_vgather_ARMVw(l_gather[ch][4] , (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add2_c );
                Q6_vgather_ARMVw(l_gather[ch][5] , (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add2_r0);
                Q6_vgather_ARMVw(l_gather[ch][6] , (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add3_c );
                Q6_vgather_ARMVw(l_gather[ch][7] , (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add3_r0);
            }
        }

        i = 0;
        for (; i < owidth_align - elem_counts; i += elem_counts)
        {
            HVX_Vector v_offset_c  = *v_ofs++;
            HVX_Vector v_alpha0_c  = *v_alpha0++;
            HVX_Vector v_alpha0_r0 = *v_alpha0++;
            HVX_Vector v_alpha1_c  = *v_alpha1++;
            HVX_Vector v_alpha1_r0 = *v_alpha1++;
            HVX_Vector v_alpha2_c  = *v_alpha2++;
            HVX_Vector v_alpha2_r0 = *v_alpha2++;
            HVX_Vector v_alpha3_c  = *v_alpha3++;
            HVX_Vector v_alpha3_r0 = *v_alpha3++;

            HVX_VectorPair w_result0 = Q6_Wuw_vunpack_Vuh(v_offset_c);
            HVX_Vector v_offset_r0 = Q6_V_hi_W(w_result0);
            v_offset_c = Q6_V_lo_W(w_result0);

            HVX_Vector v_offset_add1_c  = Q6_Vuw_vadd_VuwVuw_sat(v_offset_c , Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add1_r0 = Q6_Vuw_vadd_VuwVuw_sat(v_offset_r0, Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add2_c  = Q6_Vuw_vadd_VuwVuw_sat(v_offset_add1_c , Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add2_r0 = Q6_Vuw_vadd_VuwVuw_sat(v_offset_add1_r0, Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add3_c  = Q6_Vuw_vadd_VuwVuw_sat(v_offset_add2_c , Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add3_r0 = Q6_Vuw_vadd_VuwVuw_sat(v_offset_add2_r0, Q6_V_vsplat_R(4));

            MVType mv_result;
            for (MI_S32 ch = 0; ch < C; ch++)
            {
                Q6_vgather_ARMVw(r_gather[ch][0], (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_c );
                Q6_vgather_ARMVw(r_gather[ch][1], (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_r0);
                Q6_vgather_ARMVw(r_gather[ch][2], (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add1_c );
                Q6_vgather_ARMVw(r_gather[ch][3], (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add1_r0);
                Q6_vgather_ARMVw(r_gather[ch][4], (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add2_c );
                Q6_vgather_ARMVw(r_gather[ch][5], (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add2_r0);
                Q6_vgather_ARMVw(r_gather[ch][6], (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add3_c );
                Q6_vgather_ARMVw(r_gather[ch][7], (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add3_r0);

                HVX_Vector v_res0_c  = Q6_Vqf32_vmpy_VsfVsf(*l_gather[ch][0], v_alpha0_c);
                HVX_Vector v_res0_r0 = Q6_Vqf32_vmpy_VsfVsf(*l_gather[ch][1], v_alpha0_r0);
                HVX_Vector v_res1_c  = Q6_Vqf32_vmpy_VsfVsf(*l_gather[ch][2], v_alpha1_c);
                HVX_Vector v_res1_r0 = Q6_Vqf32_vmpy_VsfVsf(*l_gather[ch][3], v_alpha1_r0);
                HVX_Vector v_res2_c  = Q6_Vqf32_vmpy_VsfVsf(*l_gather[ch][4], v_alpha2_c);
                HVX_Vector v_res2_r0 = Q6_Vqf32_vmpy_VsfVsf(*l_gather[ch][5], v_alpha2_r0);
                HVX_Vector v_res3_c  = Q6_Vqf32_vmpy_VsfVsf(*l_gather[ch][6], v_alpha3_c);
                HVX_Vector v_res3_r0 = Q6_Vqf32_vmpy_VsfVsf(*l_gather[ch][7], v_alpha3_r0);

                v_res1_c  = Q6_Vqf32_vadd_Vqf32Vqf32(v_res0_c,  v_res1_c);
                v_res1_r0 = Q6_Vqf32_vadd_Vqf32Vqf32(v_res0_r0, v_res1_r0);
                v_res3_c  = Q6_Vqf32_vadd_Vqf32Vqf32(v_res2_c,  v_res3_c);
                v_res3_r0 = Q6_Vqf32_vadd_Vqf32Vqf32(v_res2_r0, v_res3_r0);
                v_res1_c  = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(v_res3_c,  v_res1_c));
                v_res1_r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(v_res3_r0, v_res1_r0));
                v_res0_c  = Q6_Vw_vcvt_Vsf_rnd(v_res1_c);
                v_res0_r0 = Q6_Vw_vcvt_Vsf_rnd(v_res1_r0);
                mv_result.val[ch] = Q6_Vuh_vpack_VwVw_sat(v_res0_r0, v_res0_c);

                v_tmp[0] = r_gather[ch][0]; r_gather[ch][0] = l_gather[ch][0]; l_gather[ch][0] = v_tmp[0];
                v_tmp[1] = r_gather[ch][1]; r_gather[ch][1] = l_gather[ch][1]; l_gather[ch][1] = v_tmp[1];
                v_tmp[2] = r_gather[ch][2]; r_gather[ch][2] = l_gather[ch][2]; l_gather[ch][2] = v_tmp[2];
                v_tmp[3] = r_gather[ch][3]; r_gather[ch][3] = l_gather[ch][3]; l_gather[ch][3] = v_tmp[3];
                v_tmp[0] = r_gather[ch][4]; r_gather[ch][4] = l_gather[ch][4]; l_gather[ch][4] = v_tmp[0];
                v_tmp[1] = r_gather[ch][5]; r_gather[ch][5] = l_gather[ch][5]; l_gather[ch][5] = v_tmp[1];
                v_tmp[2] = r_gather[ch][6]; r_gather[ch][6] = l_gather[ch][6]; l_gather[ch][6] = v_tmp[2];
                v_tmp[3] = r_gather[ch][7]; r_gather[ch][7] = l_gather[ch][7]; l_gather[ch][7] = v_tmp[3];
            }

            vstore(dst + i, mv_result);
        }

        {
            HVX_Vector v_alpha0_c  = *v_alpha0++;
            HVX_Vector v_alpha0_r0 = *v_alpha0++;
            HVX_Vector v_alpha1_c  = *v_alpha1++;
            HVX_Vector v_alpha1_r0 = *v_alpha1++;
            HVX_Vector v_alpha2_c  = *v_alpha2++;
            HVX_Vector v_alpha2_r0 = *v_alpha2++;
            HVX_Vector v_alpha3_c  = *v_alpha3++;
            HVX_Vector v_alpha3_r0 = *v_alpha3++;

            MVType mv_result;
            for (MI_S32 ch = 0; ch < C; ch++)
            {
                HVX_Vector v_res0_c  = Q6_Vqf32_vmpy_VsfVsf(*l_gather[ch][0], v_alpha0_c);
                HVX_Vector v_res0_r0 = Q6_Vqf32_vmpy_VsfVsf(*l_gather[ch][1], v_alpha0_r0);
                HVX_Vector v_res1_c  = Q6_Vqf32_vmpy_VsfVsf(*l_gather[ch][2], v_alpha1_c);
                HVX_Vector v_res1_r0 = Q6_Vqf32_vmpy_VsfVsf(*l_gather[ch][3], v_alpha1_r0);
                HVX_Vector v_res2_c  = Q6_Vqf32_vmpy_VsfVsf(*l_gather[ch][4], v_alpha2_c);
                HVX_Vector v_res2_r0 = Q6_Vqf32_vmpy_VsfVsf(*l_gather[ch][5], v_alpha2_r0);
                HVX_Vector v_res3_c  = Q6_Vqf32_vmpy_VsfVsf(*l_gather[ch][6], v_alpha3_c);
                HVX_Vector v_res3_r0 = Q6_Vqf32_vmpy_VsfVsf(*l_gather[ch][7], v_alpha3_r0);

                v_res1_c  = Q6_Vqf32_vadd_Vqf32Vqf32(v_res0_c,  v_res1_c);
                v_res1_r0 = Q6_Vqf32_vadd_Vqf32Vqf32(v_res0_r0, v_res1_r0);
                v_res3_c  = Q6_Vqf32_vadd_Vqf32Vqf32(v_res2_c,  v_res3_c);
                v_res3_r0 = Q6_Vqf32_vadd_Vqf32Vqf32(v_res2_r0, v_res3_r0);
                v_res1_c  = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(v_res3_c,  v_res1_c));
                v_res1_r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(v_res3_r0, v_res1_r0));
                v_res0_c  = Q6_Vw_vcvt_Vsf_rnd(v_res1_c);
                v_res0_r0 = Q6_Vw_vcvt_Vsf_rnd(v_res1_r0);

                mv_result.val[ch] = Q6_Vuh_vpack_VwVw_sat(v_res0_r0, v_res0_c);
            }

            vstore(dst + i, mv_result);
        }

        if (ostep > owidth_align)
        {
            i = ostep - elem_counts;
            j = i / C;

            HVX_Vector v_offset_c    = vmemu(xofs + j);
            HVX_Vector v_alpha0_c    = vmemu(alpha0 + j);
            HVX_Vector v_alpha0_r0   = vmemu(alpha0 + j + QUAR_AURA_HVLEN);
            HVX_Vector v_alpha1_c    = vmemu(alpha1 + j);
            HVX_Vector v_alpha1_r0   = vmemu(alpha1 + j + QUAR_AURA_HVLEN);
            HVX_Vector v_alpha2_c    = vmemu(alpha2 + j);
            HVX_Vector v_alpha2_r0   = vmemu(alpha2 + j + QUAR_AURA_HVLEN);
            HVX_Vector v_alpha3_c    = vmemu(alpha3 + j);
            HVX_Vector v_alpha3_r0   = vmemu(alpha3 + j + QUAR_AURA_HVLEN);
            HVX_VectorPair w_result0 = Q6_Wuw_vunpack_Vuh(v_offset_c);
            HVX_Vector v_offset_r0   = Q6_V_hi_W(w_result0);
            v_offset_c               = Q6_V_lo_W(w_result0);

            HVX_Vector v_offset_add1_c  = Q6_Vuw_vadd_VuwVuw_sat(v_offset_c , Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add1_r0 = Q6_Vuw_vadd_VuwVuw_sat(v_offset_r0, Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add2_c  = Q6_Vuw_vadd_VuwVuw_sat(v_offset_add1_c , Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add2_r0 = Q6_Vuw_vadd_VuwVuw_sat(v_offset_add1_r0, Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add3_c  = Q6_Vuw_vadd_VuwVuw_sat(v_offset_add2_c , Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add3_r0 = Q6_Vuw_vadd_VuwVuw_sat(v_offset_add2_r0, Q6_V_vsplat_R(4));

            MVType mv_result;
            for (MI_S32 ch = 0; ch < C; ch++)
            {
                Q6_vgather_ARMVw(l_gather[ch][0], (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_c );
                Q6_vgather_ARMVw(l_gather[ch][1], (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_r0);
                Q6_vgather_ARMVw(l_gather[ch][2], (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add1_c );
                Q6_vgather_ARMVw(l_gather[ch][3], (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add1_r0);
                Q6_vgather_ARMVw(l_gather[ch][4], (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add2_c );
                Q6_vgather_ARMVw(l_gather[ch][5], (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add2_r0);
                Q6_vgather_ARMVw(l_gather[ch][6], (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add3_c );
                Q6_vgather_ARMVw(l_gather[ch][7], (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add3_r0);

                HVX_Vector v_res0_c  = Q6_Vqf32_vmpy_VsfVsf(*l_gather[ch][0], v_alpha0_c);
                HVX_Vector v_res0_r0 = Q6_Vqf32_vmpy_VsfVsf(*l_gather[ch][1], v_alpha0_r0);
                HVX_Vector v_res1_c  = Q6_Vqf32_vmpy_VsfVsf(*l_gather[ch][2], v_alpha1_c);
                HVX_Vector v_res1_r0 = Q6_Vqf32_vmpy_VsfVsf(*l_gather[ch][3], v_alpha1_r0);
                HVX_Vector v_res2_c  = Q6_Vqf32_vmpy_VsfVsf(*l_gather[ch][4], v_alpha2_c);
                HVX_Vector v_res2_r0 = Q6_Vqf32_vmpy_VsfVsf(*l_gather[ch][5], v_alpha2_r0);
                HVX_Vector v_res3_c  = Q6_Vqf32_vmpy_VsfVsf(*l_gather[ch][6], v_alpha3_c);
                HVX_Vector v_res3_r0 = Q6_Vqf32_vmpy_VsfVsf(*l_gather[ch][7], v_alpha3_r0);

                v_res1_c  = Q6_Vqf32_vadd_Vqf32Vqf32(v_res0_c,  v_res1_c);
                v_res1_r0 = Q6_Vqf32_vadd_Vqf32Vqf32(v_res0_r0, v_res1_r0);
                v_res3_c  = Q6_Vqf32_vadd_Vqf32Vqf32(v_res2_c,  v_res3_c);
                v_res3_r0 = Q6_Vqf32_vadd_Vqf32Vqf32(v_res2_r0, v_res3_r0);
                v_res1_c  = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(v_res3_c,  v_res1_c));
                v_res1_r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(v_res3_r0, v_res1_r0));
                v_res0_c  = Q6_Vw_vcvt_Vsf_rnd(v_res1_c);
                v_res0_r0 = Q6_Vw_vcvt_Vsf_rnd(v_res1_r0);

                mv_result.val[ch] = Q6_Vuh_vpack_VwVw_sat(v_res0_r0, v_res0_c);
            }

            vstore(dst + i, mv_result);
        }
    }
}

// Tp = MI_S16
template<typename Tp, MI_S32 C>
static typename std::enable_if<std::is_same<MI_S16, Tp>::value, AURA_VOID>::type
ResizeCuCommPerRow(const Tp *src_c, Tp *dst, MI_S32 *row_base, MI_S32 *vtcm_base, MI_U16 *xofs,
                   MI_S16 *beta_base, MI_S16 *alpha0_base, MI_S16 *alpha1_base, MI_S16 *alpha2_base,
                   MI_S16 *alpha3_base, AURA_VOID *l_gather_base, AURA_VOID *r_gather_base, MI_S32 iwidth,
                   MI_S32 istride, MI_S32 istep, MI_S32 ostep)
{
    AURA_UNUSED(l_gather_base);
    AURA_UNUSED(r_gather_base);
    using MVType = typename MVHvxVector<C>::Type;
    constexpr MI_S32 elem_counts = AURA_HVLEN * C / sizeof(Tp);
    constexpr MI_S32 elem_counts_channel = elem_counts / C;
    MI_S32 iwidth_align = istep / elem_counts * elem_counts;

    MI_F32 *row    = reinterpret_cast<MI_F32 *>(row_base);
    MI_S32 *beta   = reinterpret_cast<MI_S32 *>(beta_base);
    MI_F32 *alpha0 = reinterpret_cast<MI_F32 *>(alpha0_base);
    MI_F32 *alpha1 = reinterpret_cast<MI_F32 *>(alpha1_base);
    MI_F32 *alpha2 = reinterpret_cast<MI_F32 *>(alpha2_base);
    MI_F32 *alpha3 = reinterpret_cast<MI_F32 *>(alpha3_base);

    // part1: calculate the median and save to vtcm
    MI_S32 i = 0, j = 0;
    {
        const Tp *src_n0 = src_c +  istride;
        const Tp *src_n1 = src_n0 + istride;
        const Tp *src_n2 = src_n1 + istride;

        HVX_Vector v_beta0 = Q6_V_vsplat_R(beta[0]);
        HVX_Vector v_beta1 = Q6_V_vsplat_R(beta[1]);
        HVX_Vector v_beta2 = Q6_V_vsplat_R(beta[2]);
        HVX_Vector v_beta3 = Q6_V_vsplat_R(beta[3]);

        for (; i < iwidth_align; i += elem_counts, j += elem_counts_channel)
        {
            MVType mv_c_src, mv_n0_src, mv_n1_src, mv_n2_src;
            vload(src_c + i,  mv_c_src);
            vload(src_n0 + i, mv_n0_src);
            vload(src_n1 + i, mv_n1_src);
            vload(src_n2 + i, mv_n2_src);

            for (MI_S32 ch = 0; ch < C; ch++)
            {
                HVX_Vector *v_dst = (HVX_Vector *)(row + j + ch * AURA_ALIGN(iwidth, QUAR_AURA_HVLEN));
                HVX_VectorPair w_result0 = Q6_Ww_vunpack_Vh(mv_c_src.val[ch]);
                HVX_VectorPair w_result1 = Q6_Ww_vunpack_Vh(mv_n0_src.val[ch]);

                HVX_Vector v_src0 = Q6_Vsf_vcvt_Vw(Q6_V_lo_W(w_result0));
                HVX_Vector v_src1 = Q6_Vsf_vcvt_Vw(Q6_V_hi_W(w_result0));
                HVX_Vector v_src2 = Q6_Vsf_vcvt_Vw(Q6_V_lo_W(w_result1));
                HVX_Vector v_src3 = Q6_Vsf_vcvt_Vw(Q6_V_hi_W(w_result1));
                v_src0 = Q6_Vqf32_vmpy_VsfVsf(v_src0, v_beta0);
                v_src1 = Q6_Vqf32_vmpy_VsfVsf(v_src1, v_beta0);
                v_src2 = Q6_Vqf32_vmpy_VsfVsf(v_src2, v_beta1);
                v_src3 = Q6_Vqf32_vmpy_VsfVsf(v_src3, v_beta1);

                w_result0 = Q6_Ww_vunpack_Vh(mv_n1_src.val[ch]);
                w_result1 = Q6_Ww_vunpack_Vh(mv_n2_src.val[ch]);
                v_src0 = Q6_Vqf32_vadd_Vqf32Vqf32(v_src0, Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_vcvt_Vw(Q6_V_lo_W(w_result0)), v_beta2));
                v_src1 = Q6_Vqf32_vadd_Vqf32Vqf32(v_src1, Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_vcvt_Vw(Q6_V_hi_W(w_result0)), v_beta2));
                v_src2 = Q6_Vqf32_vadd_Vqf32Vqf32(v_src2, Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_vcvt_Vw(Q6_V_lo_W(w_result1)), v_beta3));
                v_src3 = Q6_Vqf32_vadd_Vqf32Vqf32(v_src3, Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_vcvt_Vw(Q6_V_hi_W(w_result1)), v_beta3));
                v_src0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(v_src0, v_src2));
                v_src1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(v_src1, v_src3));

                *v_dst++ = v_src0;
                *v_dst = v_src1;
            }
        }

        if (istep > iwidth_align)
        {
            MVType mv_c_src, mv_n0_src, mv_n1_src, mv_n2_src;
            i = istep - elem_counts;
            j = i / C;

            vload(src_c + i,  mv_c_src);
            vload(src_n0 + i, mv_n0_src);
            vload(src_n1 + i, mv_n1_src);
            vload(src_n2 + i, mv_n2_src);

            for (MI_S32 ch = 0; ch < C; ch++)
            {
                MI_F32 *row_tmp = row + j + ch * AURA_ALIGN(iwidth, QUAR_AURA_HVLEN);
                HVX_VectorPair w_result0 = Q6_Ww_vunpack_Vh(mv_c_src.val[ch]);
                HVX_VectorPair w_result1 = Q6_Ww_vunpack_Vh(mv_n0_src.val[ch]);

                HVX_Vector v_src0 = Q6_Vsf_vcvt_Vw(Q6_V_lo_W(w_result0));
                HVX_Vector v_src1 = Q6_Vsf_vcvt_Vw(Q6_V_hi_W(w_result0));
                HVX_Vector v_src2 = Q6_Vsf_vcvt_Vw(Q6_V_lo_W(w_result1));
                HVX_Vector v_src3 = Q6_Vsf_vcvt_Vw(Q6_V_hi_W(w_result1));
                v_src0 = Q6_Vqf32_vmpy_VsfVsf(v_src0, v_beta0);
                v_src1 = Q6_Vqf32_vmpy_VsfVsf(v_src1, v_beta0);
                v_src2 = Q6_Vqf32_vmpy_VsfVsf(v_src2, v_beta1);
                v_src3 = Q6_Vqf32_vmpy_VsfVsf(v_src3, v_beta1);

                w_result0 = Q6_Ww_vunpack_Vh(mv_n1_src.val[ch]);
                w_result1 = Q6_Ww_vunpack_Vh(mv_n2_src.val[ch]);

                v_src0 = Q6_Vqf32_vadd_Vqf32Vqf32(v_src0, Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_vcvt_Vw(Q6_V_lo_W(w_result0)), v_beta2));
                v_src1 = Q6_Vqf32_vadd_Vqf32Vqf32(v_src1, Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_vcvt_Vw(Q6_V_hi_W(w_result0)), v_beta2));
                v_src2 = Q6_Vqf32_vadd_Vqf32Vqf32(v_src2, Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_vcvt_Vw(Q6_V_lo_W(w_result1)), v_beta3));
                v_src3 = Q6_Vqf32_vadd_Vqf32Vqf32(v_src3, Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_vcvt_Vw(Q6_V_hi_W(w_result1)), v_beta3));
                v_src0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(v_src0, v_src2));
                v_src1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(v_src1, v_src3));

                vmemu(row_tmp) = v_src0;
                vmemu(row_tmp + QUAR_AURA_HVLEN) = v_src1;
            }
        }
    }

    // part2: load vtcm and calculate dst
    {
        MI_S32 *vtcm = reinterpret_cast<MI_S32 *>(vtcm_base);
        MI_S32 owidth_align = ostep / elem_counts * elem_counts;

        HVX_Vector *v_ofs     = (HVX_Vector *)xofs;
        HVX_Vector *v_alpha0  = (HVX_Vector *)alpha0; HVX_Vector *v_alpha1 = (HVX_Vector *)alpha1;
        HVX_Vector *v_alpha2  = (HVX_Vector *)alpha2; HVX_Vector *v_alpha3 = (HVX_Vector *)alpha3;
        HVX_Vector *l_gather[C][8];
        HVX_Vector *r_gather[C][8];
        HVX_Vector *v_tmp[4];
        MI_F32 *row_ch[C];

        {
            HVX_Vector v_offset_c  = *v_ofs++;
            HVX_VectorPair w_result0 = Q6_Wuw_vunpack_Vuh(v_offset_c);
            v_offset_c = Q6_V_lo_W(w_result0);
            HVX_Vector v_offset_r0   = Q6_V_hi_W(w_result0);

            HVX_Vector v_offset_add1_c  = Q6_Vuw_vadd_VuwVuw_sat(v_offset_c , Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add1_r0 = Q6_Vuw_vadd_VuwVuw_sat(v_offset_r0, Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add2_c  = Q6_Vuw_vadd_VuwVuw_sat(v_offset_add1_c , Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add2_r0 = Q6_Vuw_vadd_VuwVuw_sat(v_offset_add1_r0, Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add3_c  = Q6_Vuw_vadd_VuwVuw_sat(v_offset_add2_c , Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add3_r0 = Q6_Vuw_vadd_VuwVuw_sat(v_offset_add2_r0, Q6_V_vsplat_R(4));

            for (MI_S32 ch = 0; ch < C; ch++)
            {
                MI_S32 *vtcm_gather = vtcm + ch * (QUAR_AURA_HVLEN << 4);
                row_ch[ch] = row + ch * AURA_ALIGN(iwidth, QUAR_AURA_HVLEN);

                l_gather[ch][0] = (HVX_Vector *)(vtcm_gather);
                l_gather[ch][1] = (HVX_Vector *)(vtcm_gather + QUAR_AURA_HVLEN);
                l_gather[ch][2] = (HVX_Vector *)(vtcm_gather + HALF_AURA_HVLEN);
                l_gather[ch][3] = (HVX_Vector *)(vtcm_gather + QUAR_AURA_HVLEN * 3);
                l_gather[ch][4] = (HVX_Vector *)(vtcm_gather + AURA_HVLEN);
                l_gather[ch][5] = (HVX_Vector *)(vtcm_gather + QUAR_AURA_HVLEN * 5);
                l_gather[ch][6] = (HVX_Vector *)(vtcm_gather + QUAR_AURA_HVLEN * 6);
                l_gather[ch][7] = (HVX_Vector *)(vtcm_gather + QUAR_AURA_HVLEN * 7);
                r_gather[ch][0] = (HVX_Vector *)(vtcm_gather + (AURA_HVLEN << 1));
                r_gather[ch][1] = (HVX_Vector *)(vtcm_gather + QUAR_AURA_HVLEN * 9);
                r_gather[ch][2] = (HVX_Vector *)(vtcm_gather + QUAR_AURA_HVLEN * 10);
                r_gather[ch][3] = (HVX_Vector *)(vtcm_gather + QUAR_AURA_HVLEN * 11);
                r_gather[ch][4] = (HVX_Vector *)(vtcm_gather + QUAR_AURA_HVLEN * 12);
                r_gather[ch][5] = (HVX_Vector *)(vtcm_gather + QUAR_AURA_HVLEN * 13);
                r_gather[ch][6] = (HVX_Vector *)(vtcm_gather + QUAR_AURA_HVLEN * 14);
                r_gather[ch][7] = (HVX_Vector *)(vtcm_gather + QUAR_AURA_HVLEN * 15);

                Q6_vgather_ARMVw(l_gather[ch][0] , (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_c );
                Q6_vgather_ARMVw(l_gather[ch][1] , (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_r0);
                Q6_vgather_ARMVw(l_gather[ch][2] , (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add1_c );
                Q6_vgather_ARMVw(l_gather[ch][3] , (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add1_r0);
                Q6_vgather_ARMVw(l_gather[ch][4] , (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add2_c );
                Q6_vgather_ARMVw(l_gather[ch][5] , (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add2_r0);
                Q6_vgather_ARMVw(l_gather[ch][6] , (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add3_c );
                Q6_vgather_ARMVw(l_gather[ch][7] , (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add3_r0);
            }
        }

        i = 0;
        for (; i < owidth_align - elem_counts; i += elem_counts)
        {
            HVX_Vector v_offset_c  = *v_ofs++;
            HVX_Vector v_alpha0_c  = *v_alpha0++;
            HVX_Vector v_alpha0_r0 = *v_alpha0++;
            HVX_Vector v_alpha1_c  = *v_alpha1++;
            HVX_Vector v_alpha1_r0 = *v_alpha1++;
            HVX_Vector v_alpha2_c  = *v_alpha2++;
            HVX_Vector v_alpha2_r0 = *v_alpha2++;
            HVX_Vector v_alpha3_c  = *v_alpha3++;
            HVX_Vector v_alpha3_r0 = *v_alpha3++;

            HVX_VectorPair w_result0 = Q6_Wuw_vunpack_Vuh(v_offset_c);
            HVX_Vector v_offset_r0 = Q6_V_hi_W(w_result0);
            v_offset_c = Q6_V_lo_W(w_result0);

            HVX_Vector v_offset_add1_c  = Q6_Vuw_vadd_VuwVuw_sat(v_offset_c , Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add1_r0 = Q6_Vuw_vadd_VuwVuw_sat(v_offset_r0, Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add2_c  = Q6_Vuw_vadd_VuwVuw_sat(v_offset_add1_c , Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add2_r0 = Q6_Vuw_vadd_VuwVuw_sat(v_offset_add1_r0, Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add3_c  = Q6_Vuw_vadd_VuwVuw_sat(v_offset_add2_c , Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add3_r0 = Q6_Vuw_vadd_VuwVuw_sat(v_offset_add2_r0, Q6_V_vsplat_R(4));

            MVType mv_result;
            for (MI_S32 ch = 0; ch < C; ch++)
            {
                Q6_vgather_ARMVw(r_gather[ch][0], (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_c );
                Q6_vgather_ARMVw(r_gather[ch][1], (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_r0);
                Q6_vgather_ARMVw(r_gather[ch][2], (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add1_c );
                Q6_vgather_ARMVw(r_gather[ch][3], (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add1_r0);
                Q6_vgather_ARMVw(r_gather[ch][4], (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add2_c );
                Q6_vgather_ARMVw(r_gather[ch][5], (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add2_r0);
                Q6_vgather_ARMVw(r_gather[ch][6], (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add3_c );
                Q6_vgather_ARMVw(r_gather[ch][7], (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add3_r0);

                HVX_Vector v_res0_c  = Q6_Vqf32_vmpy_VsfVsf(*l_gather[ch][0], v_alpha0_c);
                HVX_Vector v_res0_r0 = Q6_Vqf32_vmpy_VsfVsf(*l_gather[ch][1], v_alpha0_r0);
                HVX_Vector v_res1_c  = Q6_Vqf32_vmpy_VsfVsf(*l_gather[ch][2], v_alpha1_c);
                HVX_Vector v_res1_r0 = Q6_Vqf32_vmpy_VsfVsf(*l_gather[ch][3], v_alpha1_r0);
                HVX_Vector v_res2_c  = Q6_Vqf32_vmpy_VsfVsf(*l_gather[ch][4], v_alpha2_c);
                HVX_Vector v_res2_r0 = Q6_Vqf32_vmpy_VsfVsf(*l_gather[ch][5], v_alpha2_r0);
                HVX_Vector v_res3_c  = Q6_Vqf32_vmpy_VsfVsf(*l_gather[ch][6], v_alpha3_c);
                HVX_Vector v_res3_r0 = Q6_Vqf32_vmpy_VsfVsf(*l_gather[ch][7], v_alpha3_r0);

                v_res1_c  = Q6_Vqf32_vadd_Vqf32Vqf32(v_res0_c,  v_res1_c);
                v_res1_r0 = Q6_Vqf32_vadd_Vqf32Vqf32(v_res0_r0, v_res1_r0);
                v_res3_c  = Q6_Vqf32_vadd_Vqf32Vqf32(v_res2_c,  v_res3_c);
                v_res3_r0 = Q6_Vqf32_vadd_Vqf32Vqf32(v_res2_r0, v_res3_r0);
                v_res1_c  = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(v_res3_c,  v_res1_c));
                v_res1_r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(v_res3_r0, v_res1_r0));
                v_res0_c  = Q6_Vw_vcvt_Vsf_rnd(v_res1_c);
                v_res0_r0 = Q6_Vw_vcvt_Vsf_rnd(v_res1_r0);
                mv_result.val[ch] = Q6_Vh_vpack_VwVw_sat(v_res0_r0, v_res0_c);

                v_tmp[0] = r_gather[ch][0]; r_gather[ch][0] = l_gather[ch][0]; l_gather[ch][0] = v_tmp[0];
                v_tmp[1] = r_gather[ch][1]; r_gather[ch][1] = l_gather[ch][1]; l_gather[ch][1] = v_tmp[1];
                v_tmp[2] = r_gather[ch][2]; r_gather[ch][2] = l_gather[ch][2]; l_gather[ch][2] = v_tmp[2];
                v_tmp[3] = r_gather[ch][3]; r_gather[ch][3] = l_gather[ch][3]; l_gather[ch][3] = v_tmp[3];
                v_tmp[0] = r_gather[ch][4]; r_gather[ch][4] = l_gather[ch][4]; l_gather[ch][4] = v_tmp[0];
                v_tmp[1] = r_gather[ch][5]; r_gather[ch][5] = l_gather[ch][5]; l_gather[ch][5] = v_tmp[1];
                v_tmp[2] = r_gather[ch][6]; r_gather[ch][6] = l_gather[ch][6]; l_gather[ch][6] = v_tmp[2];
                v_tmp[3] = r_gather[ch][7]; r_gather[ch][7] = l_gather[ch][7]; l_gather[ch][7] = v_tmp[3];
            }

            vstore(dst + i, mv_result);
        }

        {
            HVX_Vector v_alpha0_c  = *v_alpha0++;
            HVX_Vector v_alpha0_r0 = *v_alpha0++;
            HVX_Vector v_alpha1_c  = *v_alpha1++;
            HVX_Vector v_alpha1_r0 = *v_alpha1++;
            HVX_Vector v_alpha2_c  = *v_alpha2++;
            HVX_Vector v_alpha2_r0 = *v_alpha2++;
            HVX_Vector v_alpha3_c  = *v_alpha3++;
            HVX_Vector v_alpha3_r0 = *v_alpha3++;

            MVType mv_result;
            for (MI_S32 ch = 0; ch < C; ch++)
            {
                HVX_Vector v_res0_c  = Q6_Vqf32_vmpy_VsfVsf(*l_gather[ch][0], v_alpha0_c);
                HVX_Vector v_res0_r0 = Q6_Vqf32_vmpy_VsfVsf(*l_gather[ch][1], v_alpha0_r0);
                HVX_Vector v_res1_c  = Q6_Vqf32_vmpy_VsfVsf(*l_gather[ch][2], v_alpha1_c);
                HVX_Vector v_res1_r0 = Q6_Vqf32_vmpy_VsfVsf(*l_gather[ch][3], v_alpha1_r0);
                HVX_Vector v_res2_c  = Q6_Vqf32_vmpy_VsfVsf(*l_gather[ch][4], v_alpha2_c);
                HVX_Vector v_res2_r0 = Q6_Vqf32_vmpy_VsfVsf(*l_gather[ch][5], v_alpha2_r0);
                HVX_Vector v_res3_c  = Q6_Vqf32_vmpy_VsfVsf(*l_gather[ch][6], v_alpha3_c);
                HVX_Vector v_res3_r0 = Q6_Vqf32_vmpy_VsfVsf(*l_gather[ch][7], v_alpha3_r0);

                v_res1_c  = Q6_Vqf32_vadd_Vqf32Vqf32(v_res0_c,  v_res1_c);
                v_res1_r0 = Q6_Vqf32_vadd_Vqf32Vqf32(v_res0_r0, v_res1_r0);
                v_res3_c  = Q6_Vqf32_vadd_Vqf32Vqf32(v_res2_c,  v_res3_c);
                v_res3_r0 = Q6_Vqf32_vadd_Vqf32Vqf32(v_res2_r0, v_res3_r0);
                v_res1_c  = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(v_res3_c,  v_res1_c));
                v_res1_r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(v_res3_r0, v_res1_r0));
                v_res0_c  = Q6_Vw_vcvt_Vsf_rnd(v_res1_c);
                v_res0_r0 = Q6_Vw_vcvt_Vsf_rnd(v_res1_r0);

                mv_result.val[ch] = Q6_Vh_vpack_VwVw_sat(v_res0_r0, v_res0_c);
            }

            vstore(dst + i, mv_result);
        }

        if (ostep > owidth_align)
        {
            i = ostep - elem_counts;
            j = i / C;

            HVX_Vector v_offset_c    = vmemu(xofs + j);
            HVX_Vector v_alpha0_c    = vmemu(alpha0 + j);
            HVX_Vector v_alpha0_r0   = vmemu(alpha0 + j + QUAR_AURA_HVLEN);
            HVX_Vector v_alpha1_c    = vmemu(alpha1 + j);
            HVX_Vector v_alpha1_r0   = vmemu(alpha1 + j + QUAR_AURA_HVLEN);
            HVX_Vector v_alpha2_c    = vmemu(alpha2 + j);
            HVX_Vector v_alpha2_r0   = vmemu(alpha2 + j + QUAR_AURA_HVLEN);
            HVX_Vector v_alpha3_c    = vmemu(alpha3 + j);
            HVX_Vector v_alpha3_r0   = vmemu(alpha3 + j + QUAR_AURA_HVLEN);
            HVX_VectorPair w_result0 = Q6_Wuw_vunpack_Vuh(v_offset_c);
            HVX_Vector v_offset_r0   = Q6_V_hi_W(w_result0);
            v_offset_c               = Q6_V_lo_W(w_result0);

            HVX_Vector v_offset_add1_c  = Q6_Vuw_vadd_VuwVuw_sat(v_offset_c , Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add1_r0 = Q6_Vuw_vadd_VuwVuw_sat(v_offset_r0, Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add2_c  = Q6_Vuw_vadd_VuwVuw_sat(v_offset_add1_c , Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add2_r0 = Q6_Vuw_vadd_VuwVuw_sat(v_offset_add1_r0, Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add3_c  = Q6_Vuw_vadd_VuwVuw_sat(v_offset_add2_c , Q6_V_vsplat_R(4));
            HVX_Vector v_offset_add3_r0 = Q6_Vuw_vadd_VuwVuw_sat(v_offset_add2_r0, Q6_V_vsplat_R(4));

            MVType mv_result;
            for (MI_S32 ch = 0; ch < C; ch++)
            {
                Q6_vgather_ARMVw(l_gather[ch][0], (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_c );
                Q6_vgather_ARMVw(l_gather[ch][1], (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_r0);
                Q6_vgather_ARMVw(l_gather[ch][2], (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add1_c );
                Q6_vgather_ARMVw(l_gather[ch][3], (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add1_r0);
                Q6_vgather_ARMVw(l_gather[ch][4], (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add2_c );
                Q6_vgather_ARMVw(l_gather[ch][5], (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add2_r0);
                Q6_vgather_ARMVw(l_gather[ch][6], (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add3_c );
                Q6_vgather_ARMVw(l_gather[ch][7], (MI_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add3_r0);

                HVX_Vector v_res0_c  = Q6_Vqf32_vmpy_VsfVsf(*l_gather[ch][0], v_alpha0_c);
                HVX_Vector v_res0_r0 = Q6_Vqf32_vmpy_VsfVsf(*l_gather[ch][1], v_alpha0_r0);
                HVX_Vector v_res1_c  = Q6_Vqf32_vmpy_VsfVsf(*l_gather[ch][2], v_alpha1_c);
                HVX_Vector v_res1_r0 = Q6_Vqf32_vmpy_VsfVsf(*l_gather[ch][3], v_alpha1_r0);
                HVX_Vector v_res2_c  = Q6_Vqf32_vmpy_VsfVsf(*l_gather[ch][4], v_alpha2_c);
                HVX_Vector v_res2_r0 = Q6_Vqf32_vmpy_VsfVsf(*l_gather[ch][5], v_alpha2_r0);
                HVX_Vector v_res3_c  = Q6_Vqf32_vmpy_VsfVsf(*l_gather[ch][6], v_alpha3_c);
                HVX_Vector v_res3_r0 = Q6_Vqf32_vmpy_VsfVsf(*l_gather[ch][7], v_alpha3_r0);

                v_res1_c  = Q6_Vqf32_vadd_Vqf32Vqf32(v_res0_c,  v_res1_c);
                v_res1_r0 = Q6_Vqf32_vadd_Vqf32Vqf32(v_res0_r0, v_res1_r0);
                v_res3_c  = Q6_Vqf32_vadd_Vqf32Vqf32(v_res2_c,  v_res3_c);
                v_res3_r0 = Q6_Vqf32_vadd_Vqf32Vqf32(v_res2_r0, v_res3_r0);
                v_res1_c  = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(v_res3_c,  v_res1_c));
                v_res1_r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(v_res3_r0, v_res1_r0));
                v_res0_c  = Q6_Vw_vcvt_Vsf_rnd(v_res1_c);
                v_res0_r0 = Q6_Vw_vcvt_Vsf_rnd(v_res1_r0);
                mv_result.val[ch] = Q6_Vh_vpack_VwVw_sat(v_res0_r0, v_res0_c);
            }

            vstore(dst + i, mv_result);
        }
    }
}

template<typename Tp, MI_S32 C>
static typename std::enable_if<std::is_same<MI_U8, Tp>::value || std::is_same<MI_S8, Tp>::value, AURA_VOID>::type
ResizeCuCommCore(const Mat &src, Mat &dst, CubicBufStruct *cubic_buf_struct, MI_S32 thread_num, MI_S32 start_height, MI_S32 end_height)
{
    MI_S32 iwidth    = src.GetSizes().m_width;
    MI_S32 istride   = src.GetStrides().m_width;
    MI_S32 owidth    = dst.GetSizes().m_width;
    MI_S32 oheight   = dst.GetSizes().m_height;
    MI_S32 istep     = iwidth * C;
    MI_S32 ostep     = owidth * C;
    MI_S32 thread_id = SaturateCast<MI_S32>(static_cast<MI_F32>(start_height) * thread_num / oheight);

    MI_U16 *xofs         = reinterpret_cast<MI_U16 *>(cubic_buf_struct->xofs);
    MI_U16 *yofs         = reinterpret_cast<MI_U16 *>(cubic_buf_struct->yofs);
    MI_S16 *alpha0       = reinterpret_cast<MI_S16 *>(cubic_buf_struct->alpha0);
    MI_S16 *alpha1       = reinterpret_cast<MI_S16 *>(cubic_buf_struct->alpha1);
    MI_S16 *alpha2       = reinterpret_cast<MI_S16 *>(cubic_buf_struct->alpha2);
    MI_S16 *alpha3       = reinterpret_cast<MI_S16 *>(cubic_buf_struct->alpha3);
    MI_S16 *beta         = reinterpret_cast<MI_S16 *>(cubic_buf_struct->beta);
    MI_S32 *row_base     = reinterpret_cast<MI_S32 *>(cubic_buf_struct->row + AURA_ALIGN(iwidth * sizeof(MI_S32), AURA_HVLEN) * C * thread_id);
    MI_S32 *vtcm_base    = reinterpret_cast<MI_S32 *>(cubic_buf_struct->vtcm + AURA_HVLEN * 32 * C * thread_id);
    MI_U8 *l_gather_base = cubic_buf_struct->l_gather_base + thread_id * 16 * C * sizeof(HVX_Vector *);
    MI_U8 *r_gather_base = cubic_buf_struct->r_gather_base + thread_id * 16 * C * sizeof(HVX_Vector *);
    MI_U64 l2fetch_param = L2PfParam(istride, iwidth * sizeof(Tp) * C, 4, 0);

    for (MI_S32 y = start_height; y < end_height; y++)
    {
        if (y + 1 < end_height)
        {
            L2Fetch(reinterpret_cast<MI_U64>(src.Ptr<Tp>(yofs[y + 1])), l2fetch_param);
        }

        const Tp *src_c = src.Ptr<Tp>(yofs[y]);
        Tp *dst_c       = dst.Ptr<Tp>(y);
        ResizeCuCommPerRow<Tp, C>(src_c, dst_c, row_base, vtcm_base, xofs, beta + (y << 2), alpha0, alpha1,
                                  alpha2, alpha3, l_gather_base, r_gather_base, iwidth, istride, istep, ostep);
    }
}

template<typename Tp, MI_S32 C>
static typename std::enable_if<std::is_same<MI_U16, Tp>::value || std::is_same<MI_S16, Tp>::value, AURA_VOID>::type
ResizeCuCommCore(const Mat &src, Mat &dst, CubicBufStruct *cubic_buf_struct, MI_S32 thread_num, MI_S32 start_height, MI_S32 end_height)
{
    MI_S32 iwidth    = src.GetSizes().m_width;
    MI_S32 istride   = src.GetStrides().m_width;
    MI_S32 owidth    = dst.GetSizes().m_width;
    MI_S32 oheight   = dst.GetSizes().m_height;
    MI_S32 istep     = iwidth * C;
    MI_S32 ostep     = owidth * C;
    MI_S32 thread_id = SaturateCast<MI_S32>(static_cast<MI_F32>(start_height) * thread_num / oheight);

    MI_U16 *xofs         = reinterpret_cast<MI_U16 *>(cubic_buf_struct->xofs);
    MI_U16 *yofs         = reinterpret_cast<MI_U16 *>(cubic_buf_struct->yofs);
    MI_S16 *alpha0       = reinterpret_cast<MI_S16 *>(cubic_buf_struct->alpha0);
    MI_S16 *alpha1       = reinterpret_cast<MI_S16 *>(cubic_buf_struct->alpha1);
    MI_S16 *alpha2       = reinterpret_cast<MI_S16 *>(cubic_buf_struct->alpha2);
    MI_S16 *alpha3       = reinterpret_cast<MI_S16 *>(cubic_buf_struct->alpha3);
    MI_S16 *beta         = reinterpret_cast<MI_S16 *>(cubic_buf_struct->beta);
    MI_S32 *row_base     = reinterpret_cast<MI_S32 *>(cubic_buf_struct->row + AURA_ALIGN(iwidth * sizeof(MI_S32), AURA_HVLEN) * C * thread_id);
    MI_S32 *vtcm_base    = reinterpret_cast<MI_S32 *>(cubic_buf_struct->vtcm + (AURA_HVLEN * 32 / sizeof(Tp)) * C * thread_id);
    MI_U64 l2fetch_param = L2PfParam(istride, iwidth * sizeof(Tp) * C, 4, 0);

    for (MI_S32 y = start_height; y < end_height; y++)
    {
        if (y + 1 < end_height)
        {
            L2Fetch(reinterpret_cast<MI_U64>(src.Ptr<Tp>(yofs[y + 1])), l2fetch_param);
        }

        const Tp *src_c = src.Ptr<Tp>(yofs[y]);
        Tp *dst_c       = dst.Ptr<Tp>(y);
        ResizeCuCommPerRow<Tp, C>(src_c, dst_c, row_base, vtcm_base, xofs, beta + (y << 2) * sizeof(Tp), alpha0, alpha1,
                                  alpha2, alpha3, NULL, NULL, iwidth, istride / sizeof(Tp), istep, ostep);
    }
}

template<typename Tp>
static Status ResizeCuCommHvxImpl(Context *ctx, const Mat &src, Mat &dst, CubicBufStruct *cubic_buf_struct,
                                  MI_S32 thread_num, MI_S32 start_height, MI_S32 end_height)
{
    MI_S32 channel = src.GetSizes().m_channel;
    Status ret = Status::OK;

    switch (channel)
    {
        case 1:
            ResizeCuCommCore<Tp, 1>(src, dst, cubic_buf_struct, thread_num, start_height, end_height);
            break;

        case 2:
            ResizeCuCommCore<Tp, 2>(src, dst, cubic_buf_struct, thread_num, start_height, end_height);
            break;

        case 3:
            ResizeCuCommCore<Tp, 3>(src, dst, cubic_buf_struct, thread_num, start_height, end_height);
            break;

        default:
            AURA_ADD_ERROR_STRING(ctx, "only support channel 1,2,3.");
            ret = Status::ERROR;
    }
    return ret;
}

template<typename Tp>
static Status ResizeCuCommHvxHelper(Context *ctx, const Mat &src, Mat &dst)
{
    Status ret = Status::ERROR;

    WorkerPool *wp = ctx->GetWorkerPool();
    if (MI_NULL == wp)
    {
        return Status::ERROR;
    }

    MI_S32 iwidth  = src.GetSizes().m_width;
    MI_S32 iheight = src.GetSizes().m_height;
    MI_S32 channel = src.GetSizes().m_channel;
    MI_S32 owidth  = dst.GetSizes().m_width;
    MI_S32 oheight = dst.GetSizes().m_height;
    MI_S32 thread_num = wp->GetComputeThreadNum();

    MI_S32 xofs_size   = AURA_ALIGN(owidth * sizeof(MI_U16), AURA_HVLEN);
    MI_S32 yofs_size   = AURA_ALIGN(oheight * sizeof(MI_U16), AURA_HVLEN);
    MI_S32 alpha_size  = AURA_ALIGN(owidth * sizeof(MI_U16) * sizeof(Tp), AURA_HVLEN);
    MI_S32 beta_size   = AURA_ALIGN(oheight * 4 * sizeof(MI_U16) * sizeof(Tp), AURA_HVLEN);
    MI_S32 row_size    = AURA_ALIGN(iwidth * sizeof(MI_S32), AURA_HVLEN) * channel * thread_num;
    MI_S32 vtcm_size   = (AURA_HVLEN * 32 / sizeof(Tp)) * channel * thread_num;// U8/S8 need 8*AURA_HVLEN*S32 vtcm, S16/U16 only need 4*AURA_HVLEN*S32 vtcm
    MI_S32 buffer_size = xofs_size + yofs_size + alpha_size * 4 + beta_size + row_size + vtcm_size;

    MI_U8 *vtcm_mem = static_cast<MI_U8*>(AURA_ALLOC_PARAM(ctx, AURA_MEM_VTCM, buffer_size, AURA_HVLEN));
    if (MI_NULL == vtcm_mem)
    {
        AURA_ADD_ERROR_STRING(ctx, "alloc vtcm memory failed");
        AURA_FREE(ctx, vtcm_mem);
        return Status::ABORT;
    }

    CubicBufStruct cubic_buf_struct;
    cubic_buf_struct.xofs   = vtcm_mem;
    cubic_buf_struct.yofs   = cubic_buf_struct.xofs + xofs_size;
    cubic_buf_struct.alpha0 = cubic_buf_struct.yofs + yofs_size;
    cubic_buf_struct.alpha1 = cubic_buf_struct.alpha0 + alpha_size;
    cubic_buf_struct.alpha2 = cubic_buf_struct.alpha1 + alpha_size;
    cubic_buf_struct.alpha3 = cubic_buf_struct.alpha2 + alpha_size;
    cubic_buf_struct.beta   = cubic_buf_struct.alpha3 + alpha_size;
    cubic_buf_struct.row    = cubic_buf_struct.beta + beta_size;
    cubic_buf_struct.vtcm   = cubic_buf_struct.row + row_size;

    MI_U8 *gather_base = NULL;
    if (sizeof(Tp) == 1)
    {
        MI_S32 gather_size = 32 * channel * sizeof(HVX_Vector *) * thread_num;
        gather_base = (MI_U8 *)AURA_ALLOC(ctx, gather_size);

        if (MI_NULL == gather_base)
        {
            AURA_ADD_ERROR_STRING(ctx, "AURA_ALLOC failed.");
            AURA_FREE(ctx, vtcm_mem);
            AURA_FREE(ctx, gather_base);
            return Status::ERROR;
        }
        cubic_buf_struct.l_gather_base = gather_base;
        cubic_buf_struct.r_gather_base = gather_base + (gather_size >> 1);
    }

    GetCuOffset<Tp>(&cubic_buf_struct, iwidth, owidth, iheight, oheight);

    ret = wp->ParallelFor((MI_S32)0, oheight, ResizeCuCommHvxImpl<Tp>, ctx, std::cref(src), std::ref(dst),
                          &cubic_buf_struct, thread_num);

    AURA_FREE(ctx, vtcm_mem);
    AURA_FREE(ctx, gather_base);
    AURA_RETURN(ctx, ret);
}

Status ResizeCuCommHvx(Context *ctx, const Mat &src, Mat &dst)
{
    Status ret = Status::ERROR;

    switch (src.GetElemType())
    {
        case ElemType::U8:
        {
            ret = ResizeCuCommHvxHelper<MI_U8>(ctx, src, dst);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeCuCommHvxHelper run failed, type: MI_U8");
            }
            break;
        }

        case ElemType::S8:
        {
            ret = ResizeCuCommHvxHelper<MI_S8>(ctx, src, dst);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeCuCommHvxHelper run failed, type: MI_S8");
            }
            break;
        }

        case ElemType::U16:
        {
            ret = ResizeCuCommHvxHelper<MI_U16>(ctx, src, dst);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeCuCommHvxHelper run failed, type: MI_U16");
            }
            break;
        }

        case ElemType::S16:
        {
            ret = ResizeCuCommHvxHelper<MI_S16>(ctx, src, dst);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeCuCommHvxHelper run failed, type: MI_S16");
            }
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "elem type is not supported.");
            ret = Status::ERROR;
        }
    }

    AURA_RETURN(ctx, ret);
}

}