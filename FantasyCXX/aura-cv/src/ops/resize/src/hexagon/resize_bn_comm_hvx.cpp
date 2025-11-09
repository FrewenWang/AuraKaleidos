#include "aura/runtime/logger.h"
#include "aura/runtime/worker_pool.h"
#include "resize_impl.hpp"

namespace aura
{

#define  HALF_AURA_HVLEN    64
#define  QUAR_AURA_HVLEN    32

struct ResizeBnVtcmBuffer
{
    DT_U8 *xofs;
    DT_U8 *yofs;
    DT_U8 *alpha;
    DT_U8 *beta;
    DT_U8 *src_buffer;
    DT_U8 *gather_buffer;
};

template<typename Tp>
static Status GetResizeBnCommOffset(ResizeBnVtcmBuffer *vtcm_buffer, DT_S32 iwidth, DT_S32 owidth, DT_S32 iheight, DT_S32 oheight, DT_BOOL is_area)
{
    if (DT_NULL == vtcm_buffer)
    {
        return Status::ERROR;
    }

    DT_F64 scale_x = static_cast<DT_F64>(iwidth) / owidth;
    DT_F64 scale_y = static_cast<DT_F64>(iheight) / oheight;
    DT_S32 fixed_coef = (1 == sizeof(Tp)) ? 2048 : 32768;

    DT_U16 *xofs  = reinterpret_cast<DT_U16 *>(vtcm_buffer->xofs);
    DT_U16 *yofs  = reinterpret_cast<DT_U16 *>(vtcm_buffer->yofs);
    DT_U16 *alpha = reinterpret_cast<DT_U16 *>(vtcm_buffer->alpha);
    DT_U16 *beta  = reinterpret_cast<DT_U16 *>(vtcm_buffer->beta);

    for (DT_S32 x = 0; x < owidth; x++)
    {
        DT_F32 fx;
        DT_S32 sx;
        if (!is_area)
        {
            fx = static_cast<DT_F32>((x + 0.5) * scale_x - 0.5);
            sx = static_cast<DT_S32>(Floor(fx));
            fx -= sx;
        }
        else
        {
            sx = static_cast<DT_S32>(Floor(x * scale_x));
            fx = static_cast<DT_F32>((x + 1) - (sx + 1) / scale_x);
            fx = fx <= 0 ? 0.f : fx - Floor(fx);
        }

        DT_U32 tx = SaturateCast<DT_U32>(fx * fixed_coef);

        if (sx < 0)
        {
            sx = 0;
            tx = 0;
        }

        if (sx > iwidth - 2)
        {
            sx = iwidth - 2;
            tx = fixed_coef;
        }

        xofs[x]  = (sx << 1) * sizeof(Tp);
        alpha[x] = fixed_coef - tx;
    }

    for (DT_S32 y = 0; y < oheight; y++)
    {
        DT_F32 fy;
        DT_S32 sy;
        if (!is_area)
        {
            fy = static_cast<DT_F32>((y + 0.5) * scale_y - 0.5);
            sy = static_cast<DT_S32>(Floor(fy));
            fy -= sy;
        }
        else
        {
            sy = static_cast<DT_S32>(Floor(y * scale_y));
            fy = static_cast<DT_F32>((y + 1) - (sy + 1) / scale_y);
            fy = fy <= 0 ? 0.f : fy - Floor(fy);
        }

        DT_U32 ty = SaturateCast<DT_U32>(fy * fixed_coef);

        if (sy < 0)
        {
            sy = 0;
            ty = 0;
        }

        if (sy > iheight - 2)
        {
            sy = iheight - 2;
            ty = fixed_coef;
        }

        yofs[y] = sy;
        beta[y] = fixed_coef - ty;
    }

    return Status::OK;
}

// Tp = DT_U8
template<typename Tp, typename Vt, DT_S32 C>
static typename std::enable_if<std::is_same<DT_U8, Tp>::value, DT_VOID>::type
ResizeBnCommRow(const Tp *src_c, const Tp *src_n0, Vt *row_base, Vt *vtcm_base, DT_U16 *xofs, Tp *dst,
                DT_U16 beta, DT_U16 *alpha, DT_S32 iwidth, DT_S32 istep, DT_S32 ostep)
{
    using MVType = typename MVHvxVector<C>::Type;
    DT_U16 *row  = reinterpret_cast<DT_U16 *>(row_base);
    constexpr DT_S32 elem_counts = AURA_HVLEN * C / sizeof(Tp);
    constexpr DT_S32 elem_counts_channel = elem_counts / C;
    DT_S32 iwidth_align = istep / elem_counts * elem_counts;

    HVX_Vector v_beta0 = Q6_Vh_vsplat_R(beta);
    HVX_Vector v_beta1 = Q6_Vh_vsplat_R(2048 - beta);
    DT_S32 i = 0, j = 0;

    // part1: calculate the median and save to vtcm
    for (; i < iwidth_align; i += elem_counts, j += elem_counts_channel)
    {
        MVType mv_c_src, mv_n0_src;
        vload(src_c + i, mv_c_src);
        vload(src_n0 + i, mv_n0_src);

        for (DT_S32 ch = 0; ch < C; ch++)
        {
            HVX_VectorPair w_c_src   = Q6_Wuh_vzxt_Vub(mv_c_src.val[ch]);
            HVX_VectorPair w_n0_src  = Q6_Wuh_vzxt_Vub(mv_n0_src.val[ch]);
            HVX_VectorPair w_result0 = Q6_Ww_vmpy_VhVh(Q6_V_lo_W(w_c_src), v_beta0);
            HVX_VectorPair w_result1 = Q6_Ww_vmpy_VhVh(Q6_V_hi_W(w_c_src), v_beta0);

            w_result0 = Q6_Ww_vmpyacc_WwVhVh(w_result0, Q6_V_lo_W(w_n0_src), v_beta1);
            w_result1 = Q6_Ww_vmpyacc_WwVhVh(w_result1, Q6_V_hi_W(w_n0_src), v_beta1);

            mv_c_src.val[ch]  = Q6_Vh_vasr_VwVwR(Q6_V_hi_W(w_result0), Q6_V_lo_W(w_result0), 5);
            mv_n0_src.val[ch] = Q6_Vh_vasr_VwVwR(Q6_V_hi_W(w_result1), Q6_V_lo_W(w_result1), 5);
            w_c_src = Q6_W_vshuff_VVR(mv_n0_src.val[ch], mv_c_src.val[ch], -2);

            HVX_Vector *v_dst = (HVX_Vector *)(row + j + ch * AURA_ALIGN(iwidth, HALF_AURA_HVLEN));
            *v_dst++ = Q6_V_lo_W(w_c_src);
            *v_dst   = Q6_V_hi_W(w_c_src);
        }
    }

    if (istep > iwidth_align)
    {
        MVType mv_c_src, mv_n0_src;
        i = istep - elem_counts;
        j = i / C;

        vload(src_c + i, mv_c_src);
        vload(src_n0 + i, mv_n0_src);

        for (DT_S32 ch = 0; ch < C; ch++)
        {
            DT_U16 *row_tmp         = row + j + ch * AURA_ALIGN(iwidth, HALF_AURA_HVLEN);
            HVX_VectorPair w_c_src  = Q6_Wuh_vzxt_Vub(mv_c_src.val[ch]);
            HVX_VectorPair w_n0_src = Q6_Wuh_vzxt_Vub(mv_n0_src.val[ch]);

            HVX_VectorPair w_result0 = Q6_Ww_vmpy_VhVh(Q6_V_lo_W(w_c_src), v_beta0);
            HVX_VectorPair w_result1 = Q6_Ww_vmpy_VhVh(Q6_V_hi_W(w_c_src), v_beta0);

            w_result0 = Q6_Ww_vmpyacc_WwVhVh(w_result0, Q6_V_lo_W(w_n0_src), v_beta1);
            w_result1 = Q6_Ww_vmpyacc_WwVhVh(w_result1, Q6_V_hi_W(w_n0_src), v_beta1);

            mv_c_src.val[ch]  = Q6_Vh_vasr_VwVwR(Q6_V_hi_W(w_result0), Q6_V_lo_W(w_result0), 5);
            mv_n0_src.val[ch] = Q6_Vh_vasr_VwVwR(Q6_V_hi_W(w_result1), Q6_V_lo_W(w_result1), 5);
            w_c_src = Q6_W_vshuff_VVR(mv_n0_src.val[ch], mv_c_src.val[ch], -2);

            vmemu(row_tmp) = Q6_V_lo_W(w_c_src);
            vmemu(row_tmp + HALF_AURA_HVLEN) = Q6_V_hi_W(w_c_src);
        }
    }

    // part2: load vtcm and calculate dst
    DT_U16 *vtcm = reinterpret_cast<DT_U16 *>(vtcm_base);
    DT_S32 owidth_align = ostep / elem_counts * elem_counts;

    HVX_Vector *v_ofs   = (HVX_Vector *)xofs;
    HVX_Vector *v_alpha = (HVX_Vector *)alpha;

    HVX_Vector *l1_gather_lo[C];
    HVX_Vector *l1_gather_hi[C];
    HVX_Vector *l0_gather_lo[C];
    HVX_Vector *l0_gather_hi[C];
    HVX_Vector *r0_gather_lo[C];
    HVX_Vector *r0_gather_hi[C];
    HVX_Vector *r1_gather_lo[C];
    HVX_Vector *r1_gather_hi[C];
    DT_U16 *row_ch[C];

    {
        HVX_Vector v_offset_c       = *v_ofs++;
        HVX_Vector v_offset_r0      = *v_ofs++;
        HVX_Vector v_offset_add1_c  = Q6_Vuh_vadd_VuhVuh_sat(v_offset_c, Q6_Vh_vsplat_R(2));
        HVX_Vector v_offset_add1_r0 = Q6_Vuh_vadd_VuhVuh_sat(v_offset_r0, Q6_Vh_vsplat_R(2));

        for (DT_S32 ch = 0; ch < C; ch++)
        {
            DT_U16 *vtcm_gather = vtcm + ch * HALF_AURA_HVLEN * 8;

            row_ch[ch]       = row + ch * AURA_ALIGN(iwidth, HALF_AURA_HVLEN);
            l1_gather_lo[ch] = (HVX_Vector *)(vtcm_gather);
            l1_gather_hi[ch] = (HVX_Vector *)(vtcm_gather + HALF_AURA_HVLEN);
            l0_gather_lo[ch] = (HVX_Vector *)(vtcm_gather + AURA_HVLEN);
            l0_gather_hi[ch] = (HVX_Vector *)(vtcm_gather + HALF_AURA_HVLEN * 3);
            r0_gather_lo[ch] = (HVX_Vector *)(vtcm_gather + (AURA_HVLEN << 1));
            r0_gather_hi[ch] = (HVX_Vector *)(vtcm_gather + HALF_AURA_HVLEN * 5);
            r1_gather_lo[ch] = (HVX_Vector *)(vtcm_gather + HALF_AURA_HVLEN * 6);
            r1_gather_hi[ch] = (HVX_Vector *)(vtcm_gather + HALF_AURA_HVLEN * 7);

            Q6_vgather_ARMVh(l1_gather_lo[ch], (DT_U32)(row_ch[ch]), (iwidth << 1) - 1, v_offset_c);
            Q6_vgather_ARMVh(l0_gather_lo[ch], (DT_U32)(row_ch[ch]), (iwidth << 1) - 1, v_offset_add1_c);
            Q6_vgather_ARMVh(l1_gather_hi[ch], (DT_U32)(row_ch[ch]), (iwidth << 1) - 1, v_offset_r0);
            Q6_vgather_ARMVh(l0_gather_hi[ch], (DT_U32)(row_ch[ch]), (iwidth << 1) - 1, v_offset_add1_r0);
        }
    }

    i = 0;
    for (; i < owidth_align - elem_counts; i += elem_counts)
    {
        HVX_Vector v_offset_c       = *v_ofs++;
        HVX_Vector v_offset_r0      = *v_ofs++;
        HVX_Vector v_alpha_c        = *v_alpha++;
        HVX_Vector v_alpha_r0       = *v_alpha++;
        HVX_Vector v_offset_add1_c  = Q6_Vuh_vadd_VuhVuh_sat(v_offset_c , Q6_Vh_vsplat_R(2));
        HVX_Vector v_offset_add1_r0 = Q6_Vuh_vadd_VuhVuh_sat(v_offset_r0, Q6_Vh_vsplat_R(2));
        HVX_Vector v_alpha1_c       = Q6_Vuh_vsub_VuhVuh_sat(Q6_Vh_vsplat_R(2048), v_alpha_c);
        HVX_Vector v_alpha1_r0      = Q6_Vuh_vsub_VuhVuh_sat(Q6_Vh_vsplat_R(2048), v_alpha_r0);

        MVType mv_result;
        for (DT_S32 ch = 0; ch < C; ch++)
        {
            HVX_Vector *v_tmp[4];
            Q6_vgather_ARMVh(r0_gather_lo[ch], (DT_U32)(row_ch[ch]), (iwidth << 1) - 1, v_offset_c);
            Q6_vgather_ARMVh(r1_gather_lo[ch], (DT_U32)(row_ch[ch]), (iwidth << 1) - 1, v_offset_add1_c);
            Q6_vgather_ARMVh(r0_gather_hi[ch], (DT_U32)(row_ch[ch]), (iwidth << 1) - 1, v_offset_r0);
            Q6_vgather_ARMVh(r1_gather_hi[ch], (DT_U32)(row_ch[ch]), (iwidth << 1) - 1, v_offset_add1_r0);

            HVX_VectorPair w_result0 = Q6_Wuw_vmpy_VuhVuh(*l1_gather_lo[ch], v_alpha_c);
            HVX_VectorPair w_result1 = Q6_Wuw_vmpy_VuhVuh(*l1_gather_hi[ch], v_alpha_r0);

            w_result0 = Q6_Ww_vmpyacc_WwVhVh(w_result0, *l0_gather_lo[ch], v_alpha1_c);
            w_result1 = Q6_Ww_vmpyacc_WwVhVh(w_result1, *l0_gather_hi[ch], v_alpha1_r0);

            HVX_Vector v_result0 = Q6_Vh_vpacko_VwVw(Q6_V_hi_W(w_result0), Q6_V_lo_W(w_result0));
            HVX_Vector v_result1 = Q6_Vh_vpacko_VwVw(Q6_V_hi_W(w_result1), Q6_V_lo_W(w_result1));

            v_result0 = Q6_Vh_vshuff_Vh(v_result0);
            v_result1 = Q6_Vh_vshuff_Vh(v_result1);
            v_result1 = Q6_Vub_vasr_VuhVuhR_rnd_sat(v_result1, v_result0, 1);
            mv_result.val[ch] = Q6_Vb_vdeal_Vb(v_result1);

            v_tmp[0]         = r0_gather_lo[ch];
            r0_gather_lo[ch] = l1_gather_lo[ch];
            l1_gather_lo[ch] = v_tmp[0];

            v_tmp[1]         = r0_gather_hi[ch];
            r0_gather_hi[ch] = l1_gather_hi[ch];
            l1_gather_hi[ch] = v_tmp[1];

            v_tmp[2]         = r1_gather_lo[ch];
            r1_gather_lo[ch] = l0_gather_lo[ch];
            l0_gather_lo[ch] = v_tmp[2];

            v_tmp[3]         = r1_gather_hi[ch];
            r1_gather_hi[ch] = l0_gather_hi[ch];
            l0_gather_hi[ch] = v_tmp[3];
        }

        vstore(dst + i, mv_result);
    }

    {
        HVX_Vector v_alpha_c   = *v_alpha++;
        HVX_Vector v_alpha_r0  = *v_alpha;
        HVX_Vector v_alpha1_c  = Q6_Vuh_vsub_VuhVuh_sat(Q6_Vh_vsplat_R(2048), v_alpha_c);
        HVX_Vector v_alpha1_r0 = Q6_Vuh_vsub_VuhVuh_sat(Q6_Vh_vsplat_R(2048), v_alpha_r0);
        MVType mv_result;
        for (DT_S32 ch = 0; ch < C; ch++)
        {
            HVX_VectorPair w_result0 = Q6_Wuw_vmpy_VuhVuh(*l1_gather_lo[ch], v_alpha_c);
            HVX_VectorPair w_result1 = Q6_Wuw_vmpy_VuhVuh(*l1_gather_hi[ch], v_alpha_r0);

            w_result0 = Q6_Ww_vmpyacc_WwVhVh(w_result0, *l0_gather_lo[ch], v_alpha1_c);
            w_result1 = Q6_Ww_vmpyacc_WwVhVh(w_result1, *l0_gather_hi[ch], v_alpha1_r0);

            HVX_Vector v_result0 = Q6_Vh_vpacko_VwVw(Q6_V_hi_W(w_result0), Q6_V_lo_W(w_result0));
            HVX_Vector v_result1 = Q6_Vh_vpacko_VwVw(Q6_V_hi_W(w_result1), Q6_V_lo_W(w_result1));

            v_result0         = Q6_Vh_vshuff_Vh(v_result0);
            v_result1         = Q6_Vh_vshuff_Vh(v_result1);
            v_result1         = Q6_Vub_vasr_VuhVuhR_rnd_sat(v_result1, v_result0, 1);
            mv_result.val[ch] = Q6_Vb_vdeal_Vb(v_result1);
        }

        vstore(dst + i, mv_result);
    }

    if (ostep > owidth_align)
    {
        i = ostep - elem_counts;
        j = i / C;

        HVX_Vector v_offset_c       = vmemu(xofs + j);
        HVX_Vector v_offset_r0      = vmemu(xofs + j + HALF_AURA_HVLEN);
        HVX_Vector v_alpha_c        = vmemu(alpha + j);
        HVX_Vector v_alpha_r0       = vmemu(alpha + j + HALF_AURA_HVLEN);
        HVX_Vector v_offset_add1_c  = Q6_Vuh_vadd_VuhVuh_sat(v_offset_c , Q6_Vh_vsplat_R(2));
        HVX_Vector v_offset_add1_r0 = Q6_Vuh_vadd_VuhVuh_sat(v_offset_r0, Q6_Vh_vsplat_R(2));
        HVX_Vector v_alpha1_c       = Q6_Vuh_vsub_VuhVuh_sat(Q6_Vh_vsplat_R(2048), v_alpha_c);
        HVX_Vector v_alpha1_r0      = Q6_Vuh_vsub_VuhVuh_sat(Q6_Vh_vsplat_R(2048), v_alpha_r0);

        MVType mv_result;
        for (DT_S32 ch = 0; ch < C; ch++)
        {
            Q6_vgather_ARMVh(l1_gather_lo[ch], (DT_U32)(row_ch[ch]), (iwidth << 1) - 1, v_offset_c);
            Q6_vgather_ARMVh(l0_gather_lo[ch], (DT_U32)(row_ch[ch]), (iwidth << 1) - 1, v_offset_add1_c);
            Q6_vgather_ARMVh(l1_gather_hi[ch], (DT_U32)(row_ch[ch]), (iwidth << 1) - 1, v_offset_r0);
            Q6_vgather_ARMVh(l0_gather_hi[ch], (DT_U32)(row_ch[ch]), (iwidth << 1) - 1, v_offset_add1_r0);

            HVX_VectorPair w_result0 = Q6_Wuw_vmpy_VuhVuh(*l1_gather_lo[ch], v_alpha_c);
            HVX_VectorPair w_result1 = Q6_Wuw_vmpy_VuhVuh(*l1_gather_hi[ch], v_alpha_r0);

            w_result0 = Q6_Ww_vmpyacc_WwVhVh(w_result0, *l0_gather_lo[ch], v_alpha1_c);
            w_result1 = Q6_Ww_vmpyacc_WwVhVh(w_result1, *l0_gather_hi[ch], v_alpha1_r0);

            HVX_Vector v_result0 = Q6_Vh_vpacko_VwVw(Q6_V_hi_W(w_result0), Q6_V_lo_W(w_result0));
            HVX_Vector v_result1 = Q6_Vh_vpacko_VwVw(Q6_V_hi_W(w_result1), Q6_V_lo_W(w_result1));

            v_result0         = Q6_Vh_vshuff_Vh(v_result0);
            v_result1         = Q6_Vh_vshuff_Vh(v_result1);
            v_result1         = Q6_Vub_vasr_VuhVuhR_rnd_sat(v_result1, v_result0, 1);
            mv_result.val[ch] = Q6_Vb_vdeal_Vb(v_result1);
        }

        vstore(dst + i, mv_result);
    }
}

// Tp = DT_S8
template<typename Tp, typename Vt, DT_S32 C>
static typename std::enable_if<std::is_same<DT_S8, Tp>::value, DT_VOID>::type
ResizeBnCommRow(const Tp *src_c, const Tp *src_n0, Vt *row_base, Vt *vtcm_base, DT_U16 *xofs, Tp *dst,
                DT_U16 beta, DT_U16 *alpha, DT_S32 iwidth, DT_S32 istep, DT_S32 ostep)
{
    using MVType = typename MVHvxVector<C>::Type;
    DT_U16 *row  = reinterpret_cast<DT_U16 *>(row_base);
    constexpr DT_S32 elem_counts = AURA_HVLEN * C / sizeof(Tp);
    constexpr DT_S32 elem_counts_channel = elem_counts / C;
    DT_S32 iwidth_align = istep / elem_counts * elem_counts;

    HVX_Vector v_beta0 = Q6_Vh_vsplat_R(beta);
    HVX_Vector v_beta1 = Q6_Vh_vsplat_R(2048 - beta);
    DT_S32 i = 0, j = 0;

    // part1: calculate the median and save to vtcm
    for (; i < iwidth_align; i += elem_counts, j += elem_counts_channel)
    {
        MVType mv_c_src, mv_n0_src;
        vload(src_c + i, mv_c_src);
        vload(src_n0 + i, mv_n0_src);

        for (DT_S32 ch = 0; ch < C; ch++)
        {
            HVX_VectorPair w_c_src   = Q6_Wh_vsxt_Vb(mv_c_src.val[ch]);
            HVX_VectorPair w_n0_src  = Q6_Wh_vsxt_Vb(mv_n0_src.val[ch]);
            HVX_VectorPair w_result0 = Q6_Ww_vmpy_VhVh(Q6_V_lo_W(w_c_src), v_beta0);
            HVX_VectorPair w_result1 = Q6_Ww_vmpy_VhVh(Q6_V_hi_W(w_c_src), v_beta0);

            w_result0 = Q6_Ww_vmpyacc_WwVhVh(w_result0, Q6_V_lo_W(w_n0_src), v_beta1);
            w_result1 = Q6_Ww_vmpyacc_WwVhVh(w_result1, Q6_V_hi_W(w_n0_src), v_beta1);

            mv_c_src.val[ch]  = Q6_Vh_vasr_VwVwR(Q6_V_hi_W(w_result0), Q6_V_lo_W(w_result0), 5);
            mv_n0_src.val[ch] = Q6_Vh_vasr_VwVwR(Q6_V_hi_W(w_result1), Q6_V_lo_W(w_result1), 5);
            w_c_src = Q6_W_vshuff_VVR(mv_n0_src.val[ch], mv_c_src.val[ch], -2);

            HVX_Vector *v_dst = (HVX_Vector *)(row + j + ch * AURA_ALIGN(iwidth, HALF_AURA_HVLEN));

            *v_dst++ = Q6_V_lo_W(w_c_src);
            *v_dst   = Q6_V_hi_W(w_c_src);
        }
    }

    if (istep > iwidth_align)
    {
        MVType mv_c_src, mv_n0_src;
        i = istep - elem_counts;
        j = i / C;

        vload(src_c + i, mv_c_src);
        vload(src_n0 + i, mv_n0_src);

        for (DT_S32 ch = 0; ch < C; ch++)
        {
            DT_U16 *row_tmp = row + j + ch * AURA_ALIGN(iwidth, HALF_AURA_HVLEN);
            HVX_VectorPair w_c_src  = Q6_Wh_vsxt_Vb(mv_c_src.val[ch]);
            HVX_VectorPair w_n0_src = Q6_Wh_vsxt_Vb(mv_n0_src.val[ch]);

            HVX_VectorPair w_result0 = Q6_Ww_vmpy_VhVh(Q6_V_lo_W(w_c_src), v_beta0);
            HVX_VectorPair w_result1 = Q6_Ww_vmpy_VhVh(Q6_V_hi_W(w_c_src), v_beta0);

            w_result0 = Q6_Ww_vmpyacc_WwVhVh(w_result0, Q6_V_lo_W(w_n0_src), v_beta1);
            w_result1 = Q6_Ww_vmpyacc_WwVhVh(w_result1, Q6_V_hi_W(w_n0_src), v_beta1);

            mv_c_src.val[ch]  = Q6_Vh_vasr_VwVwR(Q6_V_hi_W(w_result0), Q6_V_lo_W(w_result0), 5);
            mv_n0_src.val[ch] = Q6_Vh_vasr_VwVwR(Q6_V_hi_W(w_result1), Q6_V_lo_W(w_result1), 5);
            w_c_src = Q6_W_vshuff_VVR(mv_n0_src.val[ch], mv_c_src.val[ch], -2);

            vmemu(row_tmp) = Q6_V_lo_W(w_c_src);
            vmemu(row_tmp + HALF_AURA_HVLEN) = Q6_V_hi_W(w_c_src);
        }
    }

    // part2: load vtcm and calculate dst
    DT_U16 *vtcm = reinterpret_cast<DT_U16 *>(vtcm_base);
    DT_S32 owidth_align = ostep / elem_counts * elem_counts;

    HVX_Vector *v_ofs   = (HVX_Vector *)xofs;
    HVX_Vector *v_alpha = (HVX_Vector *)alpha;

    HVX_Vector *l1_gather_lo[C];
    HVX_Vector *l1_gather_hi[C];
    HVX_Vector *l0_gather_lo[C];
    HVX_Vector *l0_gather_hi[C];
    HVX_Vector *r0_gather_lo[C];
    HVX_Vector *r0_gather_hi[C];
    HVX_Vector *r1_gather_lo[C];
    HVX_Vector *r1_gather_hi[C];
    DT_U16 *row_ch[C];

    {
        HVX_Vector v_offset_c       = *v_ofs++;
        HVX_Vector v_offset_r0      = *v_ofs++;
        HVX_Vector v_offset_add1_c  = Q6_Vuh_vadd_VuhVuh_sat(v_offset_c, Q6_Vh_vsplat_R(2));
        HVX_Vector v_offset_add1_r0 = Q6_Vuh_vadd_VuhVuh_sat(v_offset_r0, Q6_Vh_vsplat_R(2));

        for (DT_S32 ch = 0; ch < C; ch++)
        {
            DT_U16 *vtcm_gather = vtcm + ch * HALF_AURA_HVLEN * 8;
            row_ch[ch]       = row + ch * AURA_ALIGN(iwidth, HALF_AURA_HVLEN);
            l1_gather_lo[ch] = (HVX_Vector *)vtcm_gather;
            l1_gather_hi[ch] = (HVX_Vector *)(vtcm_gather + HALF_AURA_HVLEN);
            l0_gather_lo[ch] = (HVX_Vector *)(vtcm_gather + AURA_HVLEN);
            l0_gather_hi[ch] = (HVX_Vector *)(vtcm_gather + HALF_AURA_HVLEN * 3);
            r0_gather_lo[ch] = (HVX_Vector *)(vtcm_gather + (AURA_HVLEN << 1));
            r0_gather_hi[ch] = (HVX_Vector *)(vtcm_gather + HALF_AURA_HVLEN * 5);
            r1_gather_lo[ch] = (HVX_Vector *)(vtcm_gather + HALF_AURA_HVLEN * 6);
            r1_gather_hi[ch] = (HVX_Vector *)(vtcm_gather + HALF_AURA_HVLEN * 7);

            Q6_vgather_ARMVh(l1_gather_lo[ch], (DT_U32)(row_ch[ch]), (iwidth << 1) - 1, v_offset_c);
            Q6_vgather_ARMVh(l0_gather_lo[ch], (DT_U32)(row_ch[ch]), (iwidth << 1) - 1, v_offset_add1_c);
            Q6_vgather_ARMVh(l1_gather_hi[ch], (DT_U32)(row_ch[ch]), (iwidth << 1) - 1, v_offset_r0);
            Q6_vgather_ARMVh(l0_gather_hi[ch], (DT_U32)(row_ch[ch]), (iwidth << 1) - 1, v_offset_add1_r0);
        }
    }

    i = 0;
    for (; i < owidth_align - elem_counts; i += elem_counts)
    {
        HVX_Vector v_offset_c       = *v_ofs++;
        HVX_Vector v_offset_r0      = *v_ofs++;
        HVX_Vector v_alpha_c        = *v_alpha++;
        HVX_Vector v_alpha_r0       = *v_alpha++;
        HVX_Vector v_offset_add1_c  = Q6_Vuh_vadd_VuhVuh_sat(v_offset_c , Q6_Vh_vsplat_R(2));
        HVX_Vector v_offset_add1_r0 = Q6_Vuh_vadd_VuhVuh_sat(v_offset_r0, Q6_Vh_vsplat_R(2));
        HVX_Vector v_alpha1_c       = Q6_Vuh_vsub_VuhVuh_sat(Q6_Vh_vsplat_R(2048), v_alpha_c);
        HVX_Vector v_alpha1_r0      = Q6_Vuh_vsub_VuhVuh_sat(Q6_Vh_vsplat_R(2048), v_alpha_r0);

        MVType mv_result;
        for (DT_S32 ch = 0; ch < C; ch++)
        {
            HVX_Vector *v_tmp[4];
            Q6_vgather_ARMVh(r0_gather_lo[ch], (DT_U32)(row_ch[ch]), (iwidth << 1) - 1, v_offset_c);
            Q6_vgather_ARMVh(r1_gather_lo[ch], (DT_U32)(row_ch[ch]), (iwidth << 1) - 1, v_offset_add1_c);
            Q6_vgather_ARMVh(r0_gather_hi[ch], (DT_U32)(row_ch[ch]), (iwidth << 1) - 1, v_offset_r0);
            Q6_vgather_ARMVh(r1_gather_hi[ch], (DT_U32)(row_ch[ch]), (iwidth << 1) - 1, v_offset_add1_r0);

            HVX_VectorPair w_result0 = Q6_Ww_vmpy_VhVh(*l1_gather_lo[ch], v_alpha_c);
            HVX_VectorPair w_result1 = Q6_Ww_vmpy_VhVh(*l1_gather_hi[ch], v_alpha_r0);

            w_result0 = Q6_Ww_vmpyacc_WwVhVh(w_result0, *l0_gather_lo[ch], v_alpha1_c);
            w_result1 = Q6_Ww_vmpyacc_WwVhVh(w_result1, *l0_gather_hi[ch], v_alpha1_r0);

            HVX_Vector v_result0 = Q6_Vh_vpacko_VwVw(Q6_V_hi_W(w_result0), Q6_V_lo_W(w_result0));
            HVX_Vector v_result1 = Q6_Vh_vpacko_VwVw(Q6_V_hi_W(w_result1), Q6_V_lo_W(w_result1));

            v_result0 = Q6_Vh_vshuff_Vh(v_result0);
            v_result1 = Q6_Vh_vshuff_Vh(v_result1);
            v_result1 = Q6_Vb_vasr_VhVhR_rnd_sat(v_result1, v_result0, 1);
            mv_result.val[ch] = Q6_Vb_vdeal_Vb(v_result1);

            v_tmp[0]         = r0_gather_lo[ch];
            r0_gather_lo[ch] = l1_gather_lo[ch];
            l1_gather_lo[ch] = v_tmp[0];

            v_tmp[1]         = r0_gather_hi[ch];
            r0_gather_hi[ch] = l1_gather_hi[ch];
            l1_gather_hi[ch] = v_tmp[1];

            v_tmp[2]         = r1_gather_lo[ch];
            r1_gather_lo[ch] = l0_gather_lo[ch];
            l0_gather_lo[ch] = v_tmp[2];

            v_tmp[3]         = r1_gather_hi[ch];
            r1_gather_hi[ch] = l0_gather_hi[ch];
            l0_gather_hi[ch] = v_tmp[3];
        }

        vstore(dst + i, mv_result);
    }

    {
        HVX_Vector v_alpha_c   = *v_alpha++;
        HVX_Vector v_alpha_r0  = *v_alpha;
        HVX_Vector v_alpha1_c  = Q6_Vuh_vsub_VuhVuh_sat(Q6_Vh_vsplat_R(2048), v_alpha_c);
        HVX_Vector v_alpha1_r0 = Q6_Vuh_vsub_VuhVuh_sat(Q6_Vh_vsplat_R(2048), v_alpha_r0);
        MVType mv_result;
        for (DT_S32 ch = 0; ch < C; ch++)
        {
            HVX_VectorPair w_result0 = Q6_Ww_vmpy_VhVh(*l1_gather_lo[ch], v_alpha_c);
            HVX_VectorPair w_result1 = Q6_Ww_vmpy_VhVh(*l1_gather_hi[ch], v_alpha_r0);

            w_result0 = Q6_Ww_vmpyacc_WwVhVh(w_result0, *l0_gather_lo[ch], v_alpha1_c);
            w_result1 = Q6_Ww_vmpyacc_WwVhVh(w_result1, *l0_gather_hi[ch], v_alpha1_r0);

            HVX_Vector v_result0 = Q6_Vh_vpacko_VwVw(Q6_V_hi_W(w_result0), Q6_V_lo_W(w_result0));
            HVX_Vector v_result1 = Q6_Vh_vpacko_VwVw(Q6_V_hi_W(w_result1), Q6_V_lo_W(w_result1));

            v_result0         = Q6_Vh_vshuff_Vh(v_result0);
            v_result1         = Q6_Vh_vshuff_Vh(v_result1);
            v_result1         = Q6_Vb_vasr_VhVhR_rnd_sat(v_result1, v_result0, 1);
            mv_result.val[ch] = Q6_Vb_vdeal_Vb(v_result1);
        }

        vstore(dst + i, mv_result);
    }

    if (ostep > owidth_align)
    {
        i = ostep - elem_counts;
        j = i / C;

        HVX_Vector v_offset_c       = vmemu(xofs + j);
        HVX_Vector v_offset_r0      = vmemu(xofs + j + HALF_AURA_HVLEN);
        HVX_Vector v_alpha_c        = vmemu(alpha + j);
        HVX_Vector v_alpha_r0       = vmemu(alpha + j + HALF_AURA_HVLEN);
        HVX_Vector v_offset_add1_c  = Q6_Vuh_vadd_VuhVuh_sat(v_offset_c , Q6_Vh_vsplat_R(2));
        HVX_Vector v_offset_add1_r0 = Q6_Vuh_vadd_VuhVuh_sat(v_offset_r0, Q6_Vh_vsplat_R(2));
        HVX_Vector v_alpha1_c       = Q6_Vuh_vsub_VuhVuh_sat(Q6_Vh_vsplat_R(2048), v_alpha_c );
        HVX_Vector v_alpha1_r0      = Q6_Vuh_vsub_VuhVuh_sat(Q6_Vh_vsplat_R(2048), v_alpha_r0);

        MVType mv_result;
        for (DT_S32 ch = 0; ch < C; ch++)
        {
            Q6_vgather_ARMVh(l1_gather_lo[ch], (DT_U32)(row_ch[ch]), (iwidth << 1) - 1, v_offset_c);
            Q6_vgather_ARMVh(l0_gather_lo[ch], (DT_U32)(row_ch[ch]), (iwidth << 1) - 1, v_offset_add1_c);
            Q6_vgather_ARMVh(l1_gather_hi[ch], (DT_U32)(row_ch[ch]), (iwidth << 1) - 1, v_offset_r0);
            Q6_vgather_ARMVh(l0_gather_hi[ch], (DT_U32)(row_ch[ch]), (iwidth << 1) - 1, v_offset_add1_r0);

            HVX_VectorPair w_result0 = Q6_Ww_vmpy_VhVh(*l1_gather_lo[ch], v_alpha_c);
            HVX_VectorPair w_result1 = Q6_Ww_vmpy_VhVh(*l1_gather_hi[ch], v_alpha_r0);

            w_result0 = Q6_Ww_vmpyacc_WwVhVh(w_result0, *l0_gather_lo[ch], v_alpha1_c);
            w_result1 = Q6_Ww_vmpyacc_WwVhVh(w_result1, *l0_gather_hi[ch], v_alpha1_r0);

            HVX_Vector v_result0 = Q6_Vh_vpacko_VwVw(Q6_V_hi_W(w_result0), Q6_V_lo_W(w_result0));
            HVX_Vector v_result1 = Q6_Vh_vpacko_VwVw(Q6_V_hi_W(w_result1), Q6_V_lo_W(w_result1));

            v_result0         = Q6_Vh_vshuff_Vh(v_result0);
            v_result1         = Q6_Vh_vshuff_Vh(v_result1);
            v_result1         = Q6_Vb_vasr_VhVhR_rnd_sat(v_result1, v_result0, 1);
            mv_result.val[ch] = Q6_Vb_vdeal_Vb(v_result1);
        }
        vstore(dst + i, mv_result);
    }
}

// Tp = DT_U16
template<typename Tp, typename Vt, DT_S32 C>
static typename std::enable_if<std::is_same<DT_U16, Tp>::value, DT_VOID>::type
ResizeBnCommRow(const Tp *src_c, const Tp *src_n0, Vt *row_base, Vt *vtcm_base, DT_U16 *xofs, Tp *dst,
                DT_U16 beta, DT_U16 *alpha, DT_S32 iwidth, DT_S32 istep, DT_S32 ostep)
{
    using MVType = typename MVHvxVector<C>::Type;
    DT_S32 *row  = reinterpret_cast<DT_S32 *>(row_base);
    constexpr DT_S32 elem_counts         = AURA_HVLEN * C / sizeof(Tp);
    constexpr DT_S32 elem_counts_channel = elem_counts / C;
    DT_S32 iwidth_align = istep / elem_counts * elem_counts;

    HVX_Vector v_beta0 = Q6_Vh_vsplat_R(beta);
    HVX_Vector v_beta1 = Q6_Vh_vsplat_R(32768 - beta);
    DT_S32 i = 0, j = 0;

    // part1: calculate the median and save to vtcm
    for (; i < iwidth_align; i += elem_counts, j += elem_counts_channel)
    {
        MVType mv_c_src, mv_n0_src;
        vload(src_c + i, mv_c_src);
        vload(src_n0 + i, mv_n0_src);

        for (DT_S32 ch = 0; ch < C; ch++)
        {
            mv_c_src.val[ch]  = Q6_Vh_vshuff_Vh(mv_c_src.val[ch]);
            mv_n0_src.val[ch] = Q6_Vh_vshuff_Vh(mv_n0_src.val[ch]);
            HVX_VectorPair w_result = Q6_Wuw_vmpy_VuhVuh(mv_c_src.val[ch], v_beta0);
            w_result = Q6_Wuw_vmpyacc_WuwVuhVuh(w_result, mv_n0_src.val[ch], v_beta1);

            HVX_Vector *v_dst = (HVX_Vector *)(row + j + ch * AURA_ALIGN(iwidth, QUAR_AURA_HVLEN));
            *v_dst++ = Q6_V_lo_W(w_result);
            *v_dst   = Q6_V_hi_W(w_result);
        }
    }

    if (istep > iwidth_align)
    {
        MVType mv_c_src, mv_n0_src;
        i = istep - elem_counts;
        j = i / C;

        vload(src_c + i, mv_c_src);
        vload(src_n0 + i, mv_n0_src);

        for (DT_S32 ch = 0; ch < C; ch++)
        {
            DT_S32 *row_tmp   = row + j + ch * AURA_ALIGN(iwidth, QUAR_AURA_HVLEN);
            mv_c_src.val[ch]  = Q6_Vh_vshuff_Vh(mv_c_src.val[ch]);
            mv_n0_src.val[ch] = Q6_Vh_vshuff_Vh(mv_n0_src.val[ch]);
            HVX_VectorPair w_result = Q6_Wuw_vmpy_VuhVuh(mv_c_src.val[ch], v_beta0);
            w_result = Q6_Wuw_vmpyacc_WuwVuhVuh(w_result, mv_n0_src.val[ch], v_beta1);

            vmemu(row_tmp) = Q6_V_lo_W(w_result);
            vmemu(row_tmp + QUAR_AURA_HVLEN) = Q6_V_hi_W(w_result);
        }
    }

    // part2: load vtcm and calculate dst
    DT_S32 *vtcm = reinterpret_cast<DT_S32 *>(vtcm_base);
    DT_S32 owidth_align = ostep / elem_counts * elem_counts;

    HVX_Vector *v_ofs   = (HVX_Vector *)xofs;
    HVX_Vector *v_alpha = (HVX_Vector *)alpha;

    HVX_Vector *l1_gather_lo[C];
    HVX_Vector *l1_gather_hi[C];
    HVX_Vector *l0_gather_lo[C];
    HVX_Vector *l0_gather_hi[C];
    HVX_Vector *r0_gather_lo[C];
    HVX_Vector *r0_gather_hi[C];
    HVX_Vector *r1_gather_lo[C];
    HVX_Vector *r1_gather_hi[C];
    DT_S32 *row_ch[C];

    {
        HVX_Vector v_offset_c        = *v_ofs++;
        HVX_Vector v_offset_add1_c   = Q6_Vuh_vadd_VuhVuh_sat(v_offset_c, Q6_Vh_vsplat_R(4));
        HVX_VectorPair w_offset      = Q6_Wuw_vunpack_Vuh(v_offset_c);
        HVX_VectorPair w_offset_add1 = Q6_Wuw_vunpack_Vuh(v_offset_add1_c);
        HVX_Vector v_offset_r0       = Q6_V_hi_W(w_offset);
        HVX_Vector v_offset_add1_r0  = Q6_V_hi_W(w_offset_add1);
        v_offset_c                   = Q6_V_lo_W(w_offset);
        v_offset_add1_c              = Q6_V_lo_W(w_offset_add1);

        for (DT_S32 ch = 0; ch < C; ch++)
        {
            DT_S32 *vtcm_gather = vtcm + ch * QUAR_AURA_HVLEN * 8;
            row_ch[ch]          = row + ch * AURA_ALIGN(iwidth, QUAR_AURA_HVLEN);
            l1_gather_lo[ch]    = (HVX_Vector *)(vtcm_gather);
            l1_gather_hi[ch]    = (HVX_Vector *)(vtcm_gather + QUAR_AURA_HVLEN);
            l0_gather_lo[ch]    = (HVX_Vector *)(vtcm_gather + HALF_AURA_HVLEN);
            l0_gather_hi[ch]    = (HVX_Vector *)(vtcm_gather + QUAR_AURA_HVLEN * 3);
            r0_gather_lo[ch]    = (HVX_Vector *)(vtcm_gather + AURA_HVLEN);
            r0_gather_hi[ch]    = (HVX_Vector *)(vtcm_gather + QUAR_AURA_HVLEN * 5);
            r1_gather_lo[ch]    = (HVX_Vector *)(vtcm_gather + QUAR_AURA_HVLEN * 6);
            r1_gather_hi[ch]    = (HVX_Vector *)(vtcm_gather + QUAR_AURA_HVLEN * 7);

            Q6_vgather_ARMVw(l1_gather_lo[ch], (DT_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_c);
            Q6_vgather_ARMVw(l0_gather_lo[ch], (DT_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add1_c);
            Q6_vgather_ARMVw(l1_gather_hi[ch], (DT_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_r0);
            Q6_vgather_ARMVw(l0_gather_hi[ch], (DT_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add1_r0);
        }
    }

    i = 0;
    for (; i < owidth_align - elem_counts; i += elem_counts)
    {
        HVX_Vector v_offset_c       = *v_ofs++;
        HVX_Vector v_offset_add1_c  = Q6_Vuh_vadd_VuhVuh_sat(v_offset_c, Q6_Vh_vsplat_R(4));
        HVX_VectorPair w_tmp0       = Q6_Wuw_vunpack_Vuh(v_offset_c);
        HVX_VectorPair w_tmp1       = Q6_Wuw_vunpack_Vuh(v_offset_add1_c);
        HVX_Vector v_offset_r0      = Q6_V_hi_W(w_tmp0);
        HVX_Vector v_offset_add1_r0 = Q6_V_hi_W(w_tmp1);
        v_offset_c                  = Q6_V_lo_W(w_tmp0);
        v_offset_add1_c             = Q6_V_lo_W(w_tmp1);

        HVX_Vector v_alpha_c   = *v_alpha++;
        HVX_Vector v_alpha1_c  = Q6_Vuh_vsub_VuhVuh_sat(Q6_Vh_vsplat_R(32768), v_alpha_c);
        w_tmp0                 = Q6_Wuw_vunpack_Vuh(v_alpha_c);
        w_tmp1                 = Q6_Wuw_vunpack_Vuh(v_alpha1_c);
        HVX_Vector v_alpha_r0  = Q6_V_hi_W(w_tmp0);
        HVX_Vector v_alpha1_r0 = Q6_V_hi_W(w_tmp1);
        v_alpha_c              = Q6_V_lo_W(w_tmp0);
        v_alpha1_c             = Q6_V_lo_W(w_tmp1);

        MVType mv_result;
        for (DT_S32 ch = 0; ch < C; ch++)
        {
            HVX_Vector *v_tmp[4];
            Q6_vgather_ARMVw(r0_gather_lo[ch], (DT_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_c);
            Q6_vgather_ARMVw(r1_gather_lo[ch], (DT_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add1_c);
            Q6_vgather_ARMVw(r0_gather_hi[ch], (DT_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_r0);
            Q6_vgather_ARMVw(r1_gather_hi[ch], (DT_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add1_r0);

            HVX_Vector v_result0 = Q6_Vuw_vmul32xlo16lsr16_VuwVuw(*l1_gather_lo[ch], v_alpha_c);
            HVX_Vector v_result1 = Q6_Vuw_vmul32xlo16lsr16_VuwVuw(*l1_gather_hi[ch], v_alpha_r0);
            HVX_Vector v_result2 = Q6_Vuw_vmul32xlo16lsr16_VuwVuw(*l0_gather_lo[ch], v_alpha1_c);
            HVX_Vector v_result3 = Q6_Vuw_vmul32xlo16lsr16_VuwVuw(*l0_gather_hi[ch], v_alpha1_r0);

            v_result0 = Q6_Vuw_vadd_VuwVuw_sat(v_result0, v_result2);
            v_result1 = Q6_Vuw_vadd_VuwVuw_sat(v_result1, v_result3);
            v_result2 = Q6_Vuh_vasr_VuwVuwR_rnd_sat(v_result1, v_result0, 14);
            mv_result.val[ch] = Q6_Vh_vdeal_Vh(v_result2);

            v_tmp[0]         = r0_gather_lo[ch];
            r0_gather_lo[ch] = l1_gather_lo[ch];
            l1_gather_lo[ch] = v_tmp[0];

            v_tmp[1]         = r0_gather_hi[ch];
            r0_gather_hi[ch] = l1_gather_hi[ch];
            l1_gather_hi[ch] = v_tmp[1];

            v_tmp[2]         = r1_gather_lo[ch];
            r1_gather_lo[ch] = l0_gather_lo[ch];
            l0_gather_lo[ch] = v_tmp[2];

            v_tmp[3]         = r1_gather_hi[ch];
            r1_gather_hi[ch] = l0_gather_hi[ch];
            l0_gather_hi[ch] = v_tmp[3];
        }

        vstore(dst + i, mv_result);
    }

    {
        HVX_Vector v_alpha_c   = *v_alpha;
        HVX_Vector v_alpha1_c  = Q6_Vuh_vsub_VuhVuh_sat(Q6_Vh_vsplat_R(32768), v_alpha_c);
        HVX_VectorPair w_tmp0  = Q6_Wuw_vunpack_Vuh(v_alpha_c);
        HVX_VectorPair w_tmp1  = Q6_Wuw_vunpack_Vuh(v_alpha1_c);
        HVX_Vector v_alpha_r0  = Q6_V_hi_W(w_tmp0);
        HVX_Vector v_alpha1_r0 = Q6_V_hi_W(w_tmp1);
        v_alpha_c              = Q6_V_lo_W(w_tmp0);
        v_alpha1_c             = Q6_V_lo_W(w_tmp1);

        MVType mv_result;
        for (DT_S32 ch = 0; ch < C; ch++)
        {
            HVX_Vector v_result0 = Q6_Vuw_vmul32xlo16lsr16_VuwVuw(*l1_gather_lo[ch], v_alpha_c);
            HVX_Vector v_result1 = Q6_Vuw_vmul32xlo16lsr16_VuwVuw(*l1_gather_hi[ch], v_alpha_r0);
            HVX_Vector v_result2 = Q6_Vuw_vmul32xlo16lsr16_VuwVuw(*l0_gather_lo[ch], v_alpha1_c);
            HVX_Vector v_result3 = Q6_Vuw_vmul32xlo16lsr16_VuwVuw(*l0_gather_hi[ch], v_alpha1_r0);

            v_result0 = Q6_Vuw_vadd_VuwVuw_sat(v_result0, v_result2);
            v_result1 = Q6_Vuw_vadd_VuwVuw_sat(v_result1, v_result3);
            v_result2 = Q6_Vuh_vasr_VuwVuwR_rnd_sat(v_result1, v_result0, 14);

            mv_result.val[ch] = Q6_Vh_vdeal_Vh(v_result2);
        }

        vstore(dst + i, mv_result);
    }

    if (ostep > owidth_align)
    {
        i = ostep - elem_counts;
        j = i / C;

        HVX_Vector v_offset_c      = vmemu(xofs + j);
        HVX_Vector v_alpha_c       = vmemu(alpha + j);
        HVX_Vector v_offset_add1_c = Q6_Vuh_vadd_VuhVuh_sat(v_offset_c, Q6_Vh_vsplat_R(4));
        HVX_Vector v_alpha1_c      = Q6_Vuh_vsub_VuhVuh_sat(Q6_Vh_vsplat_R(32768), v_alpha_c);

        HVX_VectorPair w_tmp0       = Q6_Wuw_vunpack_Vuh(v_offset_c);
        HVX_VectorPair w_tmp1       = Q6_Wuw_vunpack_Vuh(v_offset_add1_c);
        HVX_Vector v_offset_r0      = Q6_V_hi_W(w_tmp0);
        HVX_Vector v_offset_add1_r0 = Q6_V_hi_W(w_tmp1);
        v_offset_c                  = Q6_V_lo_W(w_tmp0);
        v_offset_add1_c             = Q6_V_lo_W(w_tmp1);

        w_tmp0                 = Q6_Wuw_vunpack_Vuh(v_alpha_c);
        w_tmp1                 = Q6_Wuw_vunpack_Vuh(v_alpha1_c);
        HVX_Vector v_alpha_r0  = Q6_V_hi_W(w_tmp0);
        HVX_Vector v_alpha1_r0 = Q6_V_hi_W(w_tmp1);
        v_alpha_c              = Q6_V_lo_W(w_tmp0);
        v_alpha1_c             = Q6_V_lo_W(w_tmp1);

        MVType mv_result;
        for (DT_S32 ch = 0; ch < C; ch++)
        {
            Q6_vgather_ARMVw(l1_gather_lo[ch], (DT_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_c);
            Q6_vgather_ARMVw(l0_gather_lo[ch], (DT_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add1_c);
            Q6_vgather_ARMVw(l1_gather_hi[ch], (DT_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_r0);
            Q6_vgather_ARMVw(l0_gather_hi[ch], (DT_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add1_r0);

            HVX_Vector v_result0 = Q6_Vuw_vmul32xlo16lsr16_VuwVuw(*l1_gather_lo[ch], v_alpha_c);
            HVX_Vector v_result1 = Q6_Vuw_vmul32xlo16lsr16_VuwVuw(*l1_gather_hi[ch], v_alpha_r0);
            HVX_Vector v_result2 = Q6_Vuw_vmul32xlo16lsr16_VuwVuw(*l0_gather_lo[ch], v_alpha1_c);
            HVX_Vector v_result3 = Q6_Vuw_vmul32xlo16lsr16_VuwVuw(*l0_gather_hi[ch], v_alpha1_r0);

            v_result0 = Q6_Vuw_vadd_VuwVuw_sat(v_result0, v_result2);
            v_result1 = Q6_Vuw_vadd_VuwVuw_sat(v_result1, v_result3);
            v_result2 = Q6_Vuh_vasr_VuwVuwR_rnd_sat(v_result1, v_result0, 14);

            mv_result.val[ch] = Q6_Vh_vdeal_Vh(v_result2);
        }

        vstore(dst + i, mv_result);
    }
}

// Tp = DT_S16
template<typename Tp, typename Vt, DT_S32 C>
static typename std::enable_if<std::is_same<DT_S16, Tp>::value, DT_VOID>::type
ResizeBnCommRow(const Tp *src_c, const Tp *src_n0, Vt *row_base, Vt *vtcm_base, DT_U16 *xofs, Tp *dst,
                DT_U16 beta, DT_U16 *alpha, DT_S32 iwidth, DT_S32 istep, DT_S32 ostep)
{
    using MVType = typename MVHvxVector<C>::Type;
    DT_S32 *row  = reinterpret_cast<DT_S32 *>(row_base);
    constexpr DT_S32 elem_counts = AURA_HVLEN * C / sizeof(Tp);
    constexpr DT_S32 elem_counts_channel = elem_counts / C;
    DT_S32 iwidth_align = istep / elem_counts * elem_counts;

    HVX_Vector v_beta0 = Q6_Vh_vsplat_R(beta);
    HVX_Vector v_beta1 = Q6_Vh_vsplat_R(32768 - beta);
    DT_S32 i = 0, j = 0;

    // part1: calculate the median and save to vtcm
    for (; i < iwidth_align; i += elem_counts, j += elem_counts_channel)
    {
        MVType mv_c_src, mv_n0_src;
        vload(src_c + i, mv_c_src);
        vload(src_n0 + i, mv_n0_src);

        for (DT_S32 ch = 0; ch < C; ch++)
        {
            mv_c_src.val[ch] = Q6_Vh_vshuff_Vh(mv_c_src.val[ch]);
            mv_n0_src.val[ch] = Q6_Vh_vshuff_Vh(mv_n0_src.val[ch]);
            HVX_VectorPair w_result = Q6_Ww_vmpy_VhVuh(mv_c_src.val[ch], v_beta0);
            w_result = Q6_Ww_vmpyacc_WwVhVuh(w_result, mv_n0_src.val[ch], v_beta1);

            HVX_Vector *v_dst = (HVX_Vector *)(row + j + ch * AURA_ALIGN(iwidth, QUAR_AURA_HVLEN));
            *v_dst++ = Q6_V_lo_W(w_result);
            *v_dst   = Q6_V_hi_W(w_result);
        }
    }

    if (istep > iwidth_align)
    {
        MVType mv_c_src, mv_n0_src;
        i = istep - elem_counts;
        j = i / C;

        vload(src_c + i, mv_c_src);
        vload(src_n0 + i, mv_n0_src);

        for (DT_S32 ch = 0; ch < C; ch++)
        {
            DT_S32 *row_tmp = row + j + ch * AURA_ALIGN(iwidth, QUAR_AURA_HVLEN);
            mv_c_src.val[ch] = Q6_Vh_vshuff_Vh(mv_c_src.val[ch]);
            mv_n0_src.val[ch] = Q6_Vh_vshuff_Vh(mv_n0_src.val[ch]);
            HVX_VectorPair w_result = Q6_Ww_vmpy_VhVuh(mv_c_src.val[ch], v_beta0);
            w_result = Q6_Ww_vmpyacc_WwVhVuh(w_result, mv_n0_src.val[ch], v_beta1);

            vmemu(row_tmp) = Q6_V_lo_W(w_result);
            vmemu(row_tmp + QUAR_AURA_HVLEN) = Q6_V_hi_W(w_result);
        }
    }

    // part2: load vtcm and calculate dst
    DT_S32 *vtcm = reinterpret_cast<DT_S32 *>(vtcm_base);
    DT_S32 owidth_align = ostep / elem_counts * elem_counts;

    HVX_Vector *v_ofs   = (HVX_Vector *)xofs;
    HVX_Vector *v_alpha = (HVX_Vector *)alpha;

    HVX_Vector *l1_gather_lo[C];
    HVX_Vector *l1_gather_hi[C];
    HVX_Vector *l0_gather_lo[C];
    HVX_Vector *l0_gather_hi[C];
    HVX_Vector *r0_gather_lo[C];
    HVX_Vector *r0_gather_hi[C];
    HVX_Vector *r1_gather_lo[C];
    HVX_Vector *r1_gather_hi[C];
    DT_S32 *row_ch[C];

    {
        HVX_Vector v_offset_c        = *v_ofs++;
        HVX_Vector v_offset_add1_c   = Q6_Vuh_vadd_VuhVuh_sat(v_offset_c, Q6_Vh_vsplat_R(4));
        HVX_VectorPair w_offset      = Q6_Wuw_vunpack_Vuh(v_offset_c);
        HVX_VectorPair w_offset_add1 = Q6_Wuw_vunpack_Vuh(v_offset_add1_c);
        HVX_Vector v_offset_r0       = Q6_V_hi_W(w_offset);
        HVX_Vector v_offset_add1_r0  = Q6_V_hi_W(w_offset_add1);
        v_offset_c                   = Q6_V_lo_W(w_offset);
        v_offset_add1_c              = Q6_V_lo_W(w_offset_add1);

        for (DT_S32 ch = 0; ch < C; ch++)
        {
            DT_S32 *vtcm_gather = vtcm + ch * QUAR_AURA_HVLEN * 8;
            row_ch[ch]          = row + ch * AURA_ALIGN(iwidth, QUAR_AURA_HVLEN);
            
            l1_gather_lo[ch] = (HVX_Vector *)(vtcm_gather);
            l1_gather_hi[ch] = (HVX_Vector *)(vtcm_gather + QUAR_AURA_HVLEN);
            l0_gather_lo[ch] = (HVX_Vector *)(vtcm_gather + HALF_AURA_HVLEN);
            l0_gather_hi[ch] = (HVX_Vector *)(vtcm_gather + QUAR_AURA_HVLEN * 3);
            r0_gather_lo[ch] = (HVX_Vector *)(vtcm_gather + AURA_HVLEN);
            r0_gather_hi[ch] = (HVX_Vector *)(vtcm_gather + QUAR_AURA_HVLEN * 5);
            r1_gather_lo[ch] = (HVX_Vector *)(vtcm_gather + QUAR_AURA_HVLEN * 6);
            r1_gather_hi[ch] = (HVX_Vector *)(vtcm_gather + QUAR_AURA_HVLEN * 7);

            Q6_vgather_ARMVw(l1_gather_lo[ch], (DT_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_c);
            Q6_vgather_ARMVw(l0_gather_lo[ch], (DT_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add1_c);
            Q6_vgather_ARMVw(l1_gather_hi[ch], (DT_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_r0);
            Q6_vgather_ARMVw(l0_gather_hi[ch], (DT_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add1_r0);
        }
    }

    i = 0;
    for (; i < owidth_align - elem_counts; i += elem_counts)
    {
        HVX_Vector v_offset_c       = *v_ofs++;
        HVX_Vector v_offset_add1_c  = Q6_Vuh_vadd_VuhVuh_sat(v_offset_c, Q6_Vh_vsplat_R(4));
        HVX_VectorPair w_tmp0       = Q6_Wuw_vunpack_Vuh(v_offset_c);
        HVX_VectorPair w_tmp1       = Q6_Wuw_vunpack_Vuh(v_offset_add1_c);
        HVX_Vector v_offset_r0      = Q6_V_hi_W(w_tmp0);
        HVX_Vector v_offset_add1_r0 = Q6_V_hi_W(w_tmp1);
        v_offset_c                  = Q6_V_lo_W(w_tmp0);
        v_offset_add1_c             = Q6_V_lo_W(w_tmp1);

        HVX_Vector v_alpha_c   = *v_alpha++;
        HVX_Vector v_alpha1_c  = Q6_Vuh_vsub_VuhVuh_sat(Q6_Vh_vsplat_R(32768), v_alpha_c);
        w_tmp0                 = Q6_Wuw_vunpack_Vuh(v_alpha_c);
        w_tmp1                 = Q6_Wuw_vunpack_Vuh(v_alpha1_c);
        HVX_Vector v_alpha_r0  = Q6_V_hi_W(w_tmp0);
        HVX_Vector v_alpha1_r0 = Q6_V_hi_W(w_tmp1);
        v_alpha_c              = Q6_V_lo_W(w_tmp0);
        v_alpha1_c             = Q6_V_lo_W(w_tmp1);

        MVType mv_result;
        for (DT_S32 ch = 0; ch < C; ch++)
        {
            HVX_Vector *v_tmp[4];
            Q6_vgather_ARMVw(r0_gather_lo[ch], (DT_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_c);
            Q6_vgather_ARMVw(r1_gather_lo[ch], (DT_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add1_c);
            Q6_vgather_ARMVw(r0_gather_hi[ch], (DT_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_r0);
            Q6_vgather_ARMVw(r1_gather_hi[ch], (DT_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add1_r0);

            HVX_Vector v_result0 = Q6_Vw_vmul32xlo16lsr16_VwVuw(*l1_gather_lo[ch], v_alpha_c);
            HVX_Vector v_result1 = Q6_Vw_vmul32xlo16lsr16_VwVuw(*l1_gather_hi[ch], v_alpha_r0);
            HVX_Vector v_result2 = Q6_Vw_vmul32xlo16lsr16_VwVuw(*l0_gather_lo[ch], v_alpha1_c);
            HVX_Vector v_result3 = Q6_Vw_vmul32xlo16lsr16_VwVuw(*l0_gather_hi[ch], v_alpha1_r0);

            v_result0 = Q6_Vw_vadd_VwVw_sat(v_result0, v_result2);
            v_result1 = Q6_Vw_vadd_VwVw_sat(v_result1, v_result3);
            v_result2 = Q6_Vh_vasr_VwVwR_rnd_sat(v_result1, v_result0, 14);
            mv_result.val[ch] = Q6_Vh_vdeal_Vh(v_result2);

            v_tmp[0]         = r0_gather_lo[ch];
            r0_gather_lo[ch] = l1_gather_lo[ch];
            l1_gather_lo[ch] = v_tmp[0];

            v_tmp[1]         = r0_gather_hi[ch];
            r0_gather_hi[ch] = l1_gather_hi[ch];
            l1_gather_hi[ch] = v_tmp[1];

            v_tmp[2]         = r1_gather_lo[ch];
            r1_gather_lo[ch] = l0_gather_lo[ch];
            l0_gather_lo[ch] = v_tmp[2];

            v_tmp[3]         = r1_gather_hi[ch];
            r1_gather_hi[ch] = l0_gather_hi[ch];
            l0_gather_hi[ch] = v_tmp[3];
        }

        vstore(dst + i, mv_result);
    }

    {
        HVX_Vector v_alpha_c   = *v_alpha;
        HVX_Vector v_alpha1_c  = Q6_Vuh_vsub_VuhVuh_sat(Q6_Vh_vsplat_R(32768), v_alpha_c);
        HVX_VectorPair w_tmp0  = Q6_Wuw_vunpack_Vuh(v_alpha_c);
        HVX_VectorPair w_tmp1  = Q6_Wuw_vunpack_Vuh(v_alpha1_c);
        HVX_Vector v_alpha_r0  = Q6_V_hi_W(w_tmp0);
        HVX_Vector v_alpha1_r0 = Q6_V_hi_W(w_tmp1);
        v_alpha_c              = Q6_V_lo_W(w_tmp0);
        v_alpha1_c             = Q6_V_lo_W(w_tmp1);

        MVType mv_result;
        for (DT_S32 ch = 0; ch < C; ch++)
        {
            HVX_Vector v_result0 = Q6_Vw_vmul32xlo16lsr16_VwVuw(*l1_gather_lo[ch], v_alpha_c);
            HVX_Vector v_result1 = Q6_Vw_vmul32xlo16lsr16_VwVuw(*l1_gather_hi[ch], v_alpha_r0);
            HVX_Vector v_result2 = Q6_Vw_vmul32xlo16lsr16_VwVuw(*l0_gather_lo[ch], v_alpha1_c);
            HVX_Vector v_result3 = Q6_Vw_vmul32xlo16lsr16_VwVuw(*l0_gather_hi[ch], v_alpha1_r0);

            v_result0 = Q6_Vw_vadd_VwVw_sat(v_result0, v_result2);
            v_result1 = Q6_Vw_vadd_VwVw_sat(v_result1, v_result3);
            v_result2 = Q6_Vh_vasr_VwVwR_rnd_sat(v_result1, v_result0, 14);
            mv_result.val[ch] = Q6_Vh_vdeal_Vh(v_result2);
        }

        vstore(dst + i, mv_result);
    }

    if (ostep > owidth_align)
    {
        i = ostep - elem_counts;
        j = i / C;

        HVX_Vector v_offset_c      = vmemu(xofs + j);
        HVX_Vector v_alpha_c       = vmemu(alpha + j);
        HVX_Vector v_offset_add1_c = Q6_Vuh_vadd_VuhVuh_sat(v_offset_c, Q6_Vh_vsplat_R(4));
        HVX_Vector v_alpha1_c      = Q6_Vuh_vsub_VuhVuh_sat(Q6_Vh_vsplat_R(32768), v_alpha_c);

        HVX_VectorPair w_tmp0       = Q6_Wuw_vunpack_Vuh(v_offset_c);
        HVX_VectorPair w_tmp1       = Q6_Wuw_vunpack_Vuh(v_offset_add1_c);
        HVX_Vector v_offset_r0      = Q6_V_hi_W(w_tmp0);
        HVX_Vector v_offset_add1_r0 = Q6_V_hi_W(w_tmp1);
        v_offset_c                  = Q6_V_lo_W(w_tmp0);
        v_offset_add1_c             = Q6_V_lo_W(w_tmp1);

        w_tmp0                 = Q6_Wuw_vunpack_Vuh(v_alpha_c);
        w_tmp1                 = Q6_Wuw_vunpack_Vuh(v_alpha1_c);
        HVX_Vector v_alpha_r0  = Q6_V_hi_W(w_tmp0);
        HVX_Vector v_alpha1_r0 = Q6_V_hi_W(w_tmp1);
        v_alpha_c              = Q6_V_lo_W(w_tmp0);
        v_alpha1_c             = Q6_V_lo_W(w_tmp1);

        MVType mv_result;
        for (DT_S32 ch = 0; ch < C; ch++)
        {
            Q6_vgather_ARMVw(l1_gather_lo[ch], (DT_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_c);
            Q6_vgather_ARMVw(l0_gather_lo[ch], (DT_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add1_c);
            Q6_vgather_ARMVw(l1_gather_hi[ch], (DT_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_r0);
            Q6_vgather_ARMVw(l0_gather_hi[ch], (DT_U32)(row_ch[ch]), (iwidth << 2) - 1, v_offset_add1_r0);

            HVX_Vector v_result0 = Q6_Vw_vmul32xlo16lsr16_VwVuw(*l1_gather_lo[ch], v_alpha_c);
            HVX_Vector v_result1 = Q6_Vw_vmul32xlo16lsr16_VwVuw(*l1_gather_hi[ch], v_alpha_r0);
            HVX_Vector v_result2 = Q6_Vw_vmul32xlo16lsr16_VwVuw(*l0_gather_lo[ch], v_alpha1_c);
            HVX_Vector v_result3 = Q6_Vw_vmul32xlo16lsr16_VwVuw(*l0_gather_hi[ch], v_alpha1_r0);

            v_result0 = Q6_Vw_vadd_VwVw_sat(v_result0, v_result2);
            v_result1 = Q6_Vw_vadd_VwVw_sat(v_result1, v_result3);
            v_result2 = Q6_Vh_vasr_VwVwR_rnd_sat(v_result1, v_result0, 14);

            mv_result.val[ch] = Q6_Vh_vdeal_Vh(v_result2);
        }

        vstore(dst + i, mv_result);
    }
}

template<typename Tp, typename Vt, DT_S32 C>
static Status ResizeBnCommHvxImpl(const Mat &src, Mat &dst, ResizeBnVtcmBuffer *vtcm_buffer, DT_S32 thread_num, DT_S32 start_height, DT_S32 end_height)
{
    DT_S32 iwidth    = src.GetSizes().m_width;
    DT_S32 istride   = src.GetStrides().m_width;
    DT_S32 owidth    = dst.GetSizes().m_width;
    DT_S32 oheight   = dst.GetSizes().m_height;
    DT_S32 istep     = iwidth * C;
    DT_S32 ostep     = owidth * C;
    DT_S32 thread_id = SaturateCast<DT_S32>(static_cast<DT_F32>(start_height) * thread_num / oheight);

    DT_U16 *xofs  = reinterpret_cast<DT_U16 *>(vtcm_buffer->xofs);
    DT_U16 *yofs  = reinterpret_cast<DT_U16 *>(vtcm_buffer->yofs);
    DT_U16 *alpha = reinterpret_cast<DT_U16 *>(vtcm_buffer->alpha);
    DT_U16 *beta  = reinterpret_cast<DT_U16 *>(vtcm_buffer->beta);
    Vt *row_base  = reinterpret_cast<Vt *>(vtcm_buffer->src_buffer + AURA_ALIGN(iwidth * sizeof(Vt), AURA_HVLEN) * C * thread_id);
    Vt *vtcm_base = reinterpret_cast<Vt *>(vtcm_buffer->gather_buffer + (AURA_HVLEN * 8) * C * thread_id);
    DT_U64 l2fetch_param = L2PfParam(istride, iwidth * sizeof(Tp) * C, 2, 0);

    for (DT_S32 y = start_height; y < end_height; y++)
    {
        if (y + 1 < end_height)
        {
            L2Fetch(reinterpret_cast<DT_U64>(src.Ptr<Tp>(yofs[y + 1])), l2fetch_param);
        }

        const Tp *src_c  = src.Ptr<Tp>(yofs[y]);
        const Tp *src_n0 = src_c + istride / sizeof(Tp);
        Tp *dst_c        = dst.Ptr<Tp>(y);
        ResizeBnCommRow<Tp, Vt, C>(src_c, src_n0, row_base, vtcm_base, xofs, dst_c, beta[y], alpha, iwidth, istep, ostep);
    }

    return Status::OK;
}

template<typename Tp, typename Vt>
static Status ResizeBnCommHvxHelper(Context *ctx, const Mat &src, Mat &dst, DT_BOOL is_area)
{
    WorkerPool *wp = ctx->GetWorkerPool();
    if (DT_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerPool fail");
        return Status::ERROR;
    }

    Status ret        = Status::ERROR;
    DT_S32 iwidth     = src.GetSizes().m_width;
    DT_S32 iheight    = src.GetSizes().m_height;
    DT_S32 channel    = src.GetSizes().m_channel;
    DT_S32 owidth     = dst.GetSizes().m_width;
    DT_S32 oheight    = dst.GetSizes().m_height;
    DT_S32 thread_num = wp->GetComputeThreadNum();

    DT_S32 xofs_size          = AURA_ALIGN(owidth  * sizeof(DT_U16), AURA_HVLEN);
    DT_S32 yofs_size          = AURA_ALIGN(oheight * sizeof(DT_U16), AURA_HVLEN);
    DT_S32 alpha_size         = AURA_ALIGN(owidth  * sizeof(DT_U16), AURA_HVLEN);
    DT_S32 beta_size          = AURA_ALIGN(oheight * sizeof(DT_U16), AURA_HVLEN);
    DT_S32 src_buffer_size    = AURA_ALIGN(iwidth  * sizeof(Vt), AURA_HVLEN) * channel * thread_num;
    DT_S32 gather_buffer_size = (AURA_HVLEN * 8) * channel * thread_num;
    DT_S32 total_buffer_size  = xofs_size + yofs_size + alpha_size + beta_size + src_buffer_size + gather_buffer_size;

    DT_U8 *vtcm_mem = static_cast<DT_U8*>(AURA_ALLOC_PARAM(ctx, AURA_MEM_VTCM, total_buffer_size, AURA_HVLEN));
    if (DT_NULL == vtcm_mem)
    {
        AURA_ADD_ERROR_STRING(ctx, "alloc vtcm memory failed");
        AURA_FREE(ctx, vtcm_mem);
        return Status::ABORT;
    }

    ResizeBnVtcmBuffer vtcm_buffer;
    vtcm_buffer.xofs          = vtcm_mem;
    vtcm_buffer.yofs          = vtcm_buffer.xofs  + xofs_size;
    vtcm_buffer.alpha         = vtcm_buffer.yofs  + yofs_size;
    vtcm_buffer.beta          = vtcm_buffer.alpha + alpha_size;
    vtcm_buffer.src_buffer    = vtcm_buffer.beta  + beta_size;
    vtcm_buffer.gather_buffer = vtcm_buffer.src_buffer + src_buffer_size;

    ret = GetResizeBnCommOffset<Tp>(&vtcm_buffer, iwidth, owidth, iheight, oheight, is_area);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetResizeBnCommOffset failed");
        AURA_FREE(ctx, vtcm_mem);
        return ret;
    }

    switch (channel)
    {
        case 1:
        {
            ret = wp->ParallelFor((DT_S32)0, oheight, ResizeBnCommHvxImpl<Tp, Vt, 1>, std::cref(src), std::ref(dst), &vtcm_buffer, thread_num);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ParallelFor for ResizeBnCommHvxImpl of c1 failed");
            }
            break;
        }

        case 2:
        {
            ret = wp->ParallelFor((DT_S32)0, oheight, ResizeBnCommHvxImpl<Tp, Vt, 2>, std::cref(src), std::ref(dst), &vtcm_buffer, thread_num);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ParallelFor for ResizeBnCommHvxImpl of c2 failed");
            }
            break;
        }

        case 3:
        {
            ret = wp->ParallelFor((DT_S32)0, oheight, ResizeBnCommHvxImpl<Tp, Vt, 3>, std::cref(src), std::ref(dst), &vtcm_buffer, thread_num);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ParallelFor for ResizeBnCommHvxImpl of c3 failed");
            }
            break;
        }

        default:
        {
            ret = Status::ERROR;
            AURA_ADD_ERROR_STRING(ctx, "only support channel 1,2,3");
        }
    }

    AURA_FREE(ctx, vtcm_mem);
    AURA_RETURN(ctx, ret);
}

Status ResizeBnCommHvx(Context *ctx, const Mat &src, Mat &dst, DT_BOOL is_area)
{
    Status ret = Status::ERROR;

    switch (src.GetElemType())
    {
        case ElemType::U8:
        {
            ret = ResizeBnCommHvxHelper<DT_U8, DT_U16>(ctx, src, dst, is_area);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeBnCommHvxHelper run failed, type: DT_U8");
            }
            break;
        }

        case ElemType::S8:
        {
            ret = ResizeBnCommHvxHelper<DT_S8, DT_U16>(ctx, src, dst, is_area);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeBnCommHvxHelper run failed, type: DT_S8");
            }
            break;
        }

        case ElemType::U16:
        {
            ret = ResizeBnCommHvxHelper<DT_U16, DT_S32>(ctx, src, dst, is_area);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeBnCommHvxHelper run failed, type: DT_U16");
            }
            break;
        }

        case ElemType::S16:
        {
            ret = ResizeBnCommHvxHelper<DT_S16, DT_S32>(ctx, src, dst, is_area);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeBnCommHvxHelper run failed, type: DT_S16");
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

} // namespace aura