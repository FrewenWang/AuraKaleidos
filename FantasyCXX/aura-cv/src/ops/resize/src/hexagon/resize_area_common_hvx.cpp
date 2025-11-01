#include "resize_impl.hpp"
#include "aura/runtime/worker_pool.h"
#include "aura/runtime/logger.h"

namespace aura
{

struct ResizeAreaDnVtcmBuffer
{
    MI_U8 *xtab;
    MI_U8 *alpha;
    MI_U8 *ytab;
    MI_U8 *beta;
    MI_U8 *src_buffer;
    MI_S32 src_buffer_pitch;
    MI_U8 *gather_buffer;
};

template<typename Tp>
static Status GetResizeAreaCommDnOffset(MI_S32 iwidth, MI_S32 owidth, MI_S32 iheight, MI_S32 oheight, ResizeAreaDnVtcmBuffer *vtcm_buffer)
{
    if (MI_NULL == vtcm_buffer)
    {
        return Status::ERROR;
    }

    MI_F32 scale_x = static_cast<MI_F32>(iwidth) / owidth;
    MI_F32 scale_y = static_cast<MI_F32>(iheight) / oheight;
    MI_S32 fixed_coef = (1 == sizeof(Tp)) ? 2048 : 32768;

    MI_U16 *xtab  = reinterpret_cast<MI_U16*>(vtcm_buffer->xtab);
    MI_U16 *ytab  = reinterpret_cast<MI_U16*>(vtcm_buffer->ytab);
    MI_U16 *alpha = reinterpret_cast<MI_U16*>(vtcm_buffer->alpha);
    MI_U16 *beta  = reinterpret_cast<MI_U16*>(vtcm_buffer->beta);

    MI_S32 kx = 0;
    for (MI_S32 x = 0; x < owidth; x++)
    {
        MI_F32 fsx1 = x * scale_x;
        MI_F32 fsx2 = fsx1 + scale_x;
        MI_F32 cell_width = Min(scale_x, iwidth - fsx1);

        MI_S32 sx1 = Ceil(fsx1);
        MI_S32 sx2 = Floor(fsx2);
        sx2 = Min(sx2, iwidth - 1);
        sx1 = Min(sx1, sx2);

        MI_S32 end_id = kx + Floor(scale_x) + 2;
        MI_U16 *cell_alpha = alpha + x;
        MI_U16 *cell_xtab  = xtab + x;

        if (sx1 - fsx1 > RESIZE_AREA_MIN_FLOAT)
        {
            cell_xtab[0]  = sx1 - 1;
            cell_alpha[0] = SaturateCast<MI_U16>(((sx1 - fsx1) / cell_width) * fixed_coef);

            cell_xtab  += owidth;
            cell_alpha += owidth;
            kx++;
        }

        for (MI_S32 sx = sx1; sx < sx2; sx++)
        {
            cell_xtab[0]  = sx;
            cell_alpha[0] = SaturateCast<MI_U16>((1.0f / cell_width) * fixed_coef);

            cell_xtab  += owidth;
            cell_alpha += owidth;
            kx++;
        }

        if (fsx2 - sx2 > RESIZE_AREA_MIN_FLOAT)
        {
            cell_xtab[0]  = sx2;
            cell_alpha[0] = SaturateCast<MI_U16>((Min(Min(fsx2 - sx2, 1.0f), cell_width) / cell_width) * fixed_coef);

            cell_xtab  += owidth;
            cell_alpha += owidth;
            kx++;
        }

        for (; kx < end_id; kx++)
        {
            cell_xtab[0]  = sx2;
            cell_alpha[0] = 0;

            cell_xtab  += owidth;
            cell_alpha += owidth;
        }
    }

    MI_S32 ky = 0;
    for (MI_S32 y = 0; y < oheight; y++)
    {
        MI_F32 fsy1 = y * scale_y;
        MI_F32 fsy2 = fsy1 + scale_y;
        MI_F32 cell_width = Min(scale_y, iheight - fsy1);

        MI_S32 sy1 = Ceil(fsy1);
        MI_S32 sy2 = Floor(fsy2);
        sy2 = Min(sy2, iheight - 1);
        sy1 = Min(sy1, sy2);

        MI_S32 end_id = ky + Floor(scale_y) + 2;

        if (sy1 - fsy1 > RESIZE_AREA_MIN_FLOAT)
        {
            ytab[ky]   = sy1 - 1;
            beta[ky++] = SaturateCast<MI_U16>(((sy1 - fsy1) / cell_width) * fixed_coef);
        }

        for (MI_S32 sy = sy1; sy < sy2; sy++)
        {
            ytab[ky]   = sy;
            beta[ky++] = SaturateCast<MI_U16>((1.0f / cell_width) * fixed_coef);
        }

        if (fsy2 - sy2 > RESIZE_AREA_MIN_FLOAT)
        {
            ytab[ky]   = sy2;
            beta[ky++] = SaturateCast<MI_U16>((Min(Min(fsy2 - sy2, 1.0f), cell_width) / cell_width) * fixed_coef);
        }

        for (; ky < end_id; ky++)
        {
            ytab[ky] = sy2;
            beta[ky] = 0;
        }
    }

    return Status::OK;
}

template <typename Tp, typename std::enable_if<std::is_same<MI_U8, Tp>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID ResizeAreaDnCommVCore(HVX_VectorPair &wu32_src_row_sum0, HVX_VectorPair &wu32_src_row_sum1, HVX_Vector &vu8_x_src, MI_U16 beta_row)
{
    HVX_Vector vu16_beta = Q6_Vh_vsplat_R(beta_row);
    HVX_VectorPair wu16_x_src = Q6_Wuh_vunpack_Vub(vu8_x_src);
    wu32_src_row_sum0 = Q6_Wuw_vmpyacc_WuwVuhVuh(wu32_src_row_sum0, Q6_V_lo_W(wu16_x_src), vu16_beta);
    wu32_src_row_sum1 = Q6_Wuw_vmpyacc_WuwVuhVuh(wu32_src_row_sum1, Q6_V_hi_W(wu16_x_src), vu16_beta);
}

template <typename Tp, typename std::enable_if<std::is_same<MI_S8, Tp>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID ResizeAreaDnCommVCore(HVX_VectorPair &ws32_src_row_sum0, HVX_VectorPair &ws32_src_row_sum1, HVX_Vector &vs8_x_src, MI_U16 beta_row)
{
    HVX_Vector vu16_beta = Q6_Vh_vsplat_R(beta_row);
    HVX_VectorPair ws16_x_src = Q6_Wh_vunpack_Vb(vs8_x_src);
    ws32_src_row_sum0 = Q6_Ww_vmpyacc_WwVhVuh(ws32_src_row_sum0, Q6_V_lo_W(ws16_x_src), vu16_beta);
    ws32_src_row_sum1 = Q6_Ww_vmpyacc_WwVhVuh(ws32_src_row_sum1, Q6_V_hi_W(ws16_x_src), vu16_beta);
}

template <typename Tp, typename std::enable_if<std::is_same<MI_U16, Tp>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID ResizeAreaDnCommVCore(HVX_VectorPair &wu32_src_row_sum, HVX_Vector &vu16_x_src, MI_U16 beta_row)
{
    HVX_Vector vu16_beta = Q6_Vh_vsplat_R(beta_row);
    wu32_src_row_sum = Q6_Wuw_vmpyacc_WuwVuhVuh(wu32_src_row_sum, vu16_x_src, vu16_beta);
}

template <typename Tp, typename std::enable_if<std::is_same<MI_S16, Tp>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID ResizeAreaDnCommVCore(HVX_VectorPair &ws32_src_row_sum, HVX_Vector &vs16_x_src, MI_U16 beta_row)
{
    HVX_Vector vu16_beta = Q6_Vh_vsplat_R(beta_row);
    ws32_src_row_sum = Q6_Ww_vmpyacc_WwVhVuh(ws32_src_row_sum, vs16_x_src, vu16_beta);
}

template <typename Tp, typename std::enable_if<std::is_same<MI_U8, Tp>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID ResizeAreaDnCommVCore(HVX_VectorPair &wu32_src_row_sum0, HVX_VectorPair &wu32_src_row_sum1, HVX_Vector &vu16_row0, HVX_Vector &vu16_row1)
{
    vu16_row0 = Q6_Vuh_vasr_VuwVuwR_rnd_sat(Q6_V_hi_W(wu32_src_row_sum0), Q6_V_lo_W(wu32_src_row_sum0), 11);
    vu16_row1 = Q6_Vuh_vasr_VuwVuwR_rnd_sat(Q6_V_hi_W(wu32_src_row_sum1), Q6_V_lo_W(wu32_src_row_sum1), 11);
}

template <typename Tp, typename std::enable_if<std::is_same<MI_S8, Tp>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID ResizeAreaDnCommVCore(HVX_VectorPair &ws32_src_row_sum0, HVX_VectorPair &ws32_src_row_sum1, HVX_Vector &vs16_row0, HVX_Vector &vs16_row1)
{
    vs16_row0 = Q6_Vh_vasr_VwVwR_rnd_sat(Q6_V_hi_W(ws32_src_row_sum0), Q6_V_lo_W(ws32_src_row_sum0), 11);
    vs16_row1 = Q6_Vh_vasr_VwVwR_rnd_sat(Q6_V_hi_W(ws32_src_row_sum1), Q6_V_lo_W(ws32_src_row_sum1), 11);
}

template <typename Tp, typename std::enable_if<std::is_same<MI_U16, Tp>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID ResizeAreaDnCommVCore(HVX_VectorPair &wu32_src_row_sum, HVX_Vector &vu16_row)
{
    vu16_row = Q6_Vuh_vasr_VuwVuwR_rnd_sat(Q6_V_hi_W(wu32_src_row_sum), Q6_V_lo_W(wu32_src_row_sum), 15);
}

template <typename Tp, typename std::enable_if<std::is_same<MI_S16, Tp>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID ResizeAreaDnCommVCore(HVX_VectorPair &ws32_src_row_sum, HVX_Vector &vs16_row)
{
    vs16_row = Q6_Vh_vasr_VwVwR_rnd_sat(Q6_V_hi_W(ws32_src_row_sum), Q6_V_lo_W(ws32_src_row_sum), 15);
}

template <typename Tp, typename std::enable_if<std::is_same<MI_U8, Tp>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID ResizeAreaDnCommHCore(HVX_Vector &vu8_result, MI_U16 *xtab_vec, MI_U16 *alpha_vec, MI_U16 *src_buffer, HVX_Vector *vu16_gather0,
                                                 HVX_Vector *vu16_gather1, MI_S32 int_scale_x, MI_S32 ch, MI_S32 iwidth, MI_S32 owidth)
{
    HVX_Vector vu16_x0_idx, vu16_x1_idx, vu16_x0_alpha, vu16_x1_alpha, vu16_x0_gather, vu16_x1_gather;
    HVX_VectorPair wu32_src_sum0 = Q6_W_vzero();
    HVX_VectorPair wu32_src_sum1 = Q6_W_vzero();

    for (MI_S32 sx = 0; sx < int_scale_x; sx++)
    {
        HVX_Vector *vu16_xtab  = (HVX_Vector *)xtab_vec;
        HVX_Vector *vu16_alpha = (HVX_Vector *)alpha_vec;

        vu16_x0_idx   = vmemu(vu16_xtab++);
        vu16_x1_idx   = vmemu(vu16_xtab);
        vu16_x0_alpha = vmemu(vu16_alpha++);
        vu16_x1_alpha = vmemu(vu16_alpha);

        vu16_x0_idx = Q6_Vuh_vadd_VuhVuh_sat(vu16_x0_idx, vu16_x0_idx);
        vu16_x1_idx = Q6_Vuh_vadd_VuhVuh_sat(vu16_x1_idx, vu16_x1_idx);

        Q6_vgather_ARMVh(vu16_gather0, (MI_U32)(src_buffer + ch * iwidth), (iwidth << 1) - 1, vu16_x0_idx);
        Q6_vgather_ARMVh(vu16_gather1, (MI_U32)(src_buffer + ch * iwidth), (iwidth << 1) - 1, vu16_x1_idx);

        vu16_x0_gather = *vu16_gather0;
        vu16_x1_gather = *vu16_gather1;

        wu32_src_sum0 = Q6_Wuw_vmpyacc_WuwVuhVuh(wu32_src_sum0, vu16_x0_gather, vu16_x0_alpha);
        wu32_src_sum1 = Q6_Wuw_vmpyacc_WuwVuhVuh(wu32_src_sum1, vu16_x1_gather, vu16_x1_alpha);

        xtab_vec  += owidth;
        alpha_vec += owidth;
    }

    HVX_Vector vu16_row0 = Q6_Vuh_vasr_VuwVuwR_rnd_sat(Q6_V_hi_W(wu32_src_sum0), Q6_V_lo_W(wu32_src_sum0), 11);
    HVX_Vector vu16_row1 = Q6_Vuh_vasr_VuwVuwR_rnd_sat(Q6_V_hi_W(wu32_src_sum1), Q6_V_lo_W(wu32_src_sum1), 11);
    vu8_result = Q6_Vub_vpack_VhVh_sat(vu16_row1, vu16_row0);
}

template <typename Tp, typename std::enable_if<std::is_same<MI_S8, Tp>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID ResizeAreaDnCommHCore(HVX_Vector &vs8_result, MI_U16 *xtab_vec, MI_U16 *alpha_vec, MI_U16 *src_buffer, HVX_Vector *vs16_gather0,
                                                 HVX_Vector *vs16_gather1, MI_S32 int_scale_x, MI_S32 ch, MI_S32 iwidth, MI_S32 owidth)
{
    HVX_Vector vu16_x0_idx, vu16_x1_idx, vu16_x0_alpha, vu16_x1_alpha, vs16_x0_gather, vs16_x1_gather;
    HVX_VectorPair ws32_src_sum0 = Q6_W_vzero();
    HVX_VectorPair ws32_src_sum1 = Q6_W_vzero();

    MI_S32 sx = 0;
    for (; sx < int_scale_x; sx++)
    {
        HVX_Vector *vu16_xtab  = (HVX_Vector *)xtab_vec;
        HVX_Vector *vu16_alpha = (HVX_Vector *)alpha_vec;

        vu16_x0_idx   = vmemu(vu16_xtab++);
        vu16_x1_idx   = vmemu(vu16_xtab);
        vu16_x0_alpha = vmemu(vu16_alpha++);
        vu16_x1_alpha = vmemu(vu16_alpha);

        vu16_x0_idx = Q6_Vuh_vadd_VuhVuh_sat(vu16_x0_idx, vu16_x0_idx);
        vu16_x1_idx = Q6_Vuh_vadd_VuhVuh_sat(vu16_x1_idx, vu16_x1_idx);

        Q6_vgather_ARMVh(vs16_gather0, (MI_U32)(src_buffer + ch * iwidth), (iwidth << 1) - 1, vu16_x0_idx);
        Q6_vgather_ARMVh(vs16_gather1, (MI_U32)(src_buffer + ch * iwidth), (iwidth << 1) - 1, vu16_x1_idx);

        vs16_x0_gather = *vs16_gather0;
        vs16_x1_gather = *vs16_gather1;

        ws32_src_sum0 = Q6_Ww_vmpyacc_WwVhVuh(ws32_src_sum0, vs16_x0_gather, vu16_x0_alpha);
        ws32_src_sum1 = Q6_Ww_vmpyacc_WwVhVuh(ws32_src_sum1, vs16_x1_gather, vu16_x1_alpha);

        xtab_vec  += owidth;
        alpha_vec += owidth;
    }

    HVX_Vector vs16_row0 = Q6_Vh_vasr_VwVwR_rnd_sat(Q6_V_hi_W(ws32_src_sum0), Q6_V_lo_W(ws32_src_sum0), 11);
    HVX_Vector vs16_row1 = Q6_Vh_vasr_VwVwR_rnd_sat(Q6_V_hi_W(ws32_src_sum1), Q6_V_lo_W(ws32_src_sum1), 11);
    vs8_result = Q6_Vb_vpack_VhVh_sat(vs16_row1, vs16_row0);
}

template <typename Tp, typename std::enable_if<std::is_same<MI_U16, Tp>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID ResizeAreaDnCommHCore(HVX_Vector &vu16_result, MI_U16 *xtab_vec, MI_U16 *alpha_vec, MI_U16 *src_buffer, HVX_Vector *vu16_gather,
                                                 MI_S32 int_scale_x, MI_S32 ch, MI_S32 iwidth, MI_S32 owidth)
{
    HVX_Vector vu16_x_idx, vu16_x_alpha, vu16_x_gather;
    HVX_VectorPair wu32_src_sum = Q6_W_vzero();

    for (MI_S32 sx = 0; sx < int_scale_x; sx++)
    {
        HVX_Vector *vu16_xtab  = (HVX_Vector *)xtab_vec;
        HVX_Vector *vu16_alpha = (HVX_Vector *)alpha_vec;

        vu16_x_idx   = vmemu(vu16_xtab);
        vu16_x_alpha = vmemu(vu16_alpha);

        vu16_x_idx = Q6_Vuh_vadd_VuhVuh_sat(vu16_x_idx, vu16_x_idx);

        Q6_vgather_ARMVh(vu16_gather, (MI_U32)(src_buffer + ch * iwidth), (iwidth << 1) - 1, vu16_x_idx);

        vu16_x_gather = *vu16_gather;

        wu32_src_sum = Q6_Wuw_vmpyacc_WuwVuhVuh(wu32_src_sum, vu16_x_gather, vu16_x_alpha);

        xtab_vec  += owidth;
        alpha_vec += owidth;
    }

    vu16_result = Q6_Vuh_vasr_VuwVuwR_rnd_sat(Q6_V_hi_W(wu32_src_sum), Q6_V_lo_W(wu32_src_sum), 15);
}

template <typename Tp, typename std::enable_if<std::is_same<MI_S16, Tp>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID ResizeAreaDnCommHCore(HVX_Vector &vs16_result, MI_U16 *xtab_vec, MI_U16 *alpha_vec, MI_U16 *src_buffer, HVX_Vector *vs16_gather,
                                                 MI_S32 int_scale_x, MI_S32 ch, MI_S32 iwidth, MI_S32 owidth)
{
    HVX_Vector vu16_x_idx, vu16_x_alpha, vs16_x_gather;
    HVX_VectorPair ws32_src_sum = Q6_W_vzero();

    MI_S32 sx = 0;
    for (; sx < int_scale_x; sx++)
    {
        HVX_Vector *vu16_xtab  = (HVX_Vector *)xtab_vec;
        HVX_Vector *vu16_alpha = (HVX_Vector *)alpha_vec;

        vu16_x_idx   = vmemu(vu16_xtab);
        vu16_x_alpha = vmemu(vu16_alpha);

        vu16_x_idx = Q6_Vuh_vadd_VuhVuh_sat(vu16_x_idx, vu16_x_idx);

        Q6_vgather_ARMVh(vs16_gather, (MI_U32)(src_buffer + ch * iwidth), (iwidth << 1) - 1, vu16_x_idx);

        vs16_x_gather = *vs16_gather;

        ws32_src_sum = Q6_Ww_vmpyacc_WwVhVuh(ws32_src_sum, vs16_x_gather, vu16_x_alpha);

        xtab_vec  += owidth;
        alpha_vec += owidth;
    }

    vs16_result = Q6_Vh_vasr_VwVwR_rnd_sat(Q6_V_hi_W(ws32_src_sum), Q6_V_lo_W(ws32_src_sum), 15);
}

template <typename Tp, MI_S32 C>
static typename std::enable_if<std::is_same<MI_U8, Tp>::value || std::is_same<MI_S8, Tp>::value, Status>::type
ResizeAreaDnCommRow(Tp *src, MI_U16 *xtab, MI_U16 *alpha, MI_U16 *ytab_row, MI_U16 *beta_row, MI_U16 *src_buffer, MI_U16 *gather_buffer,
                    Tp *dst_row, MI_S32 int_scale_y, MI_S32 iwidth, MI_S32 istride, MI_S32 owidth)
{
    using MVType = typename MVHvxVector<C>::Type;
    using MWType = typename MWHvxVector<C>::Type;

    MI_S32 elem_counts  = AURA_HVLEN / sizeof(Tp);
    MI_S32 iwidth_align = iwidth & (-elem_counts);
    MI_S32 owidth_align = owidth & (-elem_counts);
    MI_S32 istep        = istride / sizeof(Tp);
    MI_S32 half_elem    = elem_counts >> 1;
    MI_S32 int_scale_x  = Floor((MI_F32)(iwidth / owidth)) + 2;

    MVType mv_src, mv_row0, mv_row1, mv_result;
    MWType mw_src_row_sum0, mw_src_row_sum1;

    for (MI_S32 ch = 0; ch < C; ch++)
    {
        mw_src_row_sum0.val[ch] = Q6_W_vzero();
        mw_src_row_sum1.val[ch] = Q6_W_vzero();
    }

    MI_S32 i = 0;
    for (; i < iwidth_align; i += elem_counts)
    {
        MI_U16 *src_buffer_ptr = src_buffer + i;

        for(MI_S32 sy = 0; sy < int_scale_y; sy++)
        {
            vload(src + i * C + ytab_row[sy] * istep, mv_src);
            #pragma unroll(C)
            for (MI_S32 ch = 0; ch < C; ch++)
            {
                ResizeAreaDnCommVCore<Tp>(mw_src_row_sum0.val[ch], mw_src_row_sum1.val[ch], mv_src.val[ch], beta_row[sy]);
            }
        }

        #pragma unroll(C)
        for (MI_S32 ch = 0; ch < C; ch++)
        {
            ResizeAreaDnCommVCore<Tp>(mw_src_row_sum0.val[ch], mw_src_row_sum1.val[ch], mv_row0.val[ch], mv_row1.val[ch]);

            vmemu(src_buffer_ptr + ch * iwidth)             = mv_row0.val[ch];
            vmemu(src_buffer_ptr + ch * iwidth + half_elem) = mv_row1.val[ch];

            mw_src_row_sum0.val[ch] = Q6_W_vzero();
            mw_src_row_sum1.val[ch] = Q6_W_vzero();
        }
    }

    if (iwidth_align < iwidth)
    {
        i = iwidth - elem_counts;
        MI_U16 *src_buffer_ptr = src_buffer + i;

        for(MI_S32 sy = 0; sy < int_scale_y; sy++)
        {
            vload(src + i * C + ytab_row[sy] * istep, mv_src);
            #pragma unroll(C)
            for (MI_S32 ch = 0; ch < C; ch++)
            {
                ResizeAreaDnCommVCore<Tp>(mw_src_row_sum0.val[ch], mw_src_row_sum1.val[ch], mv_src.val[ch], beta_row[sy]);
            }
        }

        #pragma unroll(C)
        for (MI_S32 ch = 0; ch < C; ch++)
        {
            ResizeAreaDnCommVCore<Tp>(mw_src_row_sum0.val[ch], mw_src_row_sum1.val[ch], mv_row0.val[ch], mv_row1.val[ch]);

            vmemu(src_buffer_ptr + ch * iwidth)             = mv_row0.val[ch];
            vmemu(src_buffer_ptr + ch * iwidth + half_elem) = mv_row1.val[ch];

            mw_src_row_sum0.val[ch] = Q6_W_vzero();
            mw_src_row_sum1.val[ch] = Q6_W_vzero();
        }
    }

    HVX_Vector *gather_buffer_ptr0 = (HVX_Vector *)gather_buffer;
    HVX_Vector *gather_buffer_ptr1 = (HVX_Vector *)(gather_buffer + half_elem);

    MI_S32 j = 0;
    for (; j < owidth_align; j += elem_counts)
    {
        MI_U16 *xtab_vec  = xtab + j;
        MI_U16 *alpha_vec = alpha + j;

        #pragma unroll(C)
        for (MI_S32 ch = 0; ch < C; ch++)
        {
            ResizeAreaDnCommHCore<Tp>(mv_result.val[ch], xtab_vec, alpha_vec, src_buffer, gather_buffer_ptr0, gather_buffer_ptr1,
                                         int_scale_x, ch, iwidth, owidth);
        }
        vstore((dst_row + C * j), mv_result);
    }

    if (owidth_align < owidth)
    {
        j = owidth - elem_counts;
        MI_U16 *xtab_vec  = xtab + j;
        MI_U16 *alpha_vec = alpha + j;

        #pragma unroll(C)
        for (MI_S32 ch = 0; ch < C; ch++)
        {
            ResizeAreaDnCommHCore<Tp>(mv_result.val[ch], xtab_vec, alpha_vec, src_buffer, gather_buffer_ptr0, gather_buffer_ptr1,
                                         int_scale_x, ch, iwidth, owidth);
        }
        vstore((dst_row + C * j), mv_result);
    }

    return Status::OK;
}

template <typename Tp, MI_S32 C>
static typename std::enable_if<std::is_same<MI_U16, Tp>::value || std::is_same<MI_S16, Tp>::value, Status>::type
ResizeAreaDnCommRow(Tp *src, MI_U16 *xtab, MI_U16 *alpha, MI_U16 *ytab_row, MI_U16 *beta_row, MI_U16 *src_buffer, MI_U16 *gather_buffer,
                    Tp *dst_row, MI_S32 int_scale_y, MI_S32 iwidth, MI_S32 istride, MI_S32 owidth)
{
    using MVType = typename MVHvxVector<C>::Type;
    using MWType = typename MWHvxVector<C>::Type;

    MI_S32 elem_counts  = AURA_HVLEN / sizeof(Tp);
    MI_S32 iwidth_align = iwidth & (-elem_counts);
    MI_S32 owidth_align = owidth & (-elem_counts);
    MI_S32 istep        = istride / sizeof(Tp);
    MI_S32 int_scale_x  = Floor((MI_F32)(iwidth / owidth)) + 2;

    MVType mv_src, mv_row, mv_result;
    MWType mw_src_row_sum;

    for (MI_S32 ch = 0; ch < C; ch++)
    {
        mw_src_row_sum.val[ch] = Q6_W_vzero();
    }

    MI_S32 i = 0;
    for (; i < iwidth_align; i += elem_counts)
    {
        MI_U16 *src_buffer_ptr = src_buffer + i;

        for(MI_S32 sy = 0; sy < int_scale_y; sy++)
        {
            vload(src + i * C + ytab_row[sy] * istep, mv_src);
            #pragma unroll(C)
            for (MI_S32 ch = 0; ch < C; ch++)
            {
                ResizeAreaDnCommVCore<Tp>(mw_src_row_sum.val[ch], mv_src.val[ch], beta_row[sy]);
            }
        }

        #pragma unroll(C)
        for (MI_S32 ch = 0; ch < C; ch++)
        {
            ResizeAreaDnCommVCore<Tp>(mw_src_row_sum.val[ch], mv_row.val[ch]);

            vmemu(src_buffer_ptr + ch * iwidth) = mv_row.val[ch];

            mw_src_row_sum.val[ch] = Q6_W_vzero();
        }
    }

    if (iwidth_align < iwidth)
    {
        i = iwidth - elem_counts;
        MI_U16 *src_buffer_ptr = (MI_U16 *)src_buffer + i;

        for(MI_S32 sy = 0; sy < int_scale_y; sy++)
        {
            vload(src + i * C + ytab_row[sy] * istep, mv_src);
            #pragma unroll(C)
            for (MI_S32 ch = 0; ch < C; ch++)
            {
                ResizeAreaDnCommVCore<Tp>(mw_src_row_sum.val[ch], mv_src.val[ch], beta_row[sy]);
            }
        }

        #pragma unroll(C)
        for (MI_S32 ch = 0; ch < C; ch++)
        {
            ResizeAreaDnCommVCore<Tp>(mw_src_row_sum.val[ch], mv_row.val[ch]);

            vmemu(src_buffer_ptr + ch * iwidth) = mv_row.val[ch];

            mw_src_row_sum.val[ch] = Q6_W_vzero();
        }
    }

    HVX_Vector *gather_buffer_ptr = (HVX_Vector *)gather_buffer;

    MI_S32 j = 0;
    for (; j < owidth_align; j += elem_counts)
    {
        MI_U16 *xtab_vec  = xtab + j;
        MI_U16 *alpha_vec = alpha + j;

        #pragma unroll(C)
        for (MI_S32 ch = 0; ch < C; ch++)
        {
            ResizeAreaDnCommHCore<Tp>(mv_result.val[ch], xtab_vec, alpha_vec, src_buffer, gather_buffer_ptr,
                                         int_scale_x, ch, iwidth, owidth);
        }
        vstore((dst_row + C * j), mv_result);
    }

    if (owidth_align < owidth)
    {
        j = owidth - elem_counts;
        MI_U16 *xtab_vec  = xtab + j;
        MI_U16 *alpha_vec = alpha + j;

        #pragma unroll(C)
        for (MI_S32 ch = 0; ch < C; ch++)
        {
            ResizeAreaDnCommHCore<Tp>(mv_result.val[ch], xtab_vec, alpha_vec, src_buffer, gather_buffer_ptr,
                                         int_scale_x, ch, iwidth, owidth);
        }
        vstore((dst_row + C * j), mv_result);
    }

    return Status::OK;
}

template <typename Tp, MI_S32 C>
static Status ResizeAreaDnCommHvxImpl(const Mat &src, Mat &dst, ResizeAreaDnVtcmBuffer *vtcm_buffer, MI_S32 thread_num, MI_S32 start_row, MI_S32 end_row)
{
    MI_S32 iwidth    = src.GetSizes().m_width;
    MI_S32 iheight   = src.GetSizes().m_height;
    MI_S32 istride   = src.GetStrides().m_width;
    MI_S32 owidth    = dst.GetSizes().m_width;
    MI_S32 oheight   = dst.GetSizes().m_height;
    MI_S32 thread_id = SaturateCast<MI_S32>(static_cast<MI_F32>(start_row) * thread_num / oheight);

    MI_U16 *xtab          = reinterpret_cast<MI_U16*>(vtcm_buffer->xtab);
    MI_U16 *alpha         = reinterpret_cast<MI_U16*>(vtcm_buffer->alpha);
    MI_U16 *ytab          = reinterpret_cast<MI_U16*>(vtcm_buffer->ytab);
    MI_U16 *beta          = reinterpret_cast<MI_U16*>(vtcm_buffer->beta);
    MI_U16 *src_buffer    = reinterpret_cast<MI_U16*>(vtcm_buffer->src_buffer + C * vtcm_buffer->src_buffer_pitch * thread_id);
    MI_U16 *gather_buffer = reinterpret_cast<MI_U16*>(vtcm_buffer->gather_buffer + C * (AURA_HVLEN << 2) * thread_id);

    MI_S32 int_scale_y   = Floor((MI_F32)(iheight / oheight)) + 2;
    MI_U64 l2fetch_param = L2PfParam(istride, iwidth * sizeof(Tp) * C, int_scale_y - 1, 0);

    for (MI_S32 y = start_row; y < end_row; y++)
    {
        if (y + 1 < end_row)
        {
            L2Fetch(reinterpret_cast<MI_U32>(src.Ptr<Tp>(ytab[(y + 1) * int_scale_y])), l2fetch_param);
        }

        Tp *src_s   = (Tp *)src.Ptr<Tp>(0);
        Tp *dst_row = (Tp *)dst.Ptr<Tp>(y);

        MI_U16 *ytab_row = ytab + y * int_scale_y;
        MI_U16 *beta_row = beta + y * int_scale_y;

        ResizeAreaDnCommRow<Tp, C>(src_s, xtab, alpha, ytab_row, beta_row, src_buffer, gather_buffer, dst_row, int_scale_y, iwidth, istride, owidth);
    }

    return Status::OK;
}

template <typename Tp>
static Status ResizeAreaCommHvxHelper(Context *ctx, const Mat &src, Mat &dst)
{
    WorkerPool *wp = ctx->GetWorkerPool();
    if (MI_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkPool fail");
        return Status::ERROR;
    }

    Status ret     = Status::ERROR;
    MI_S32 iwidth  = src.GetSizes().m_width;
    MI_S32 iheight = src.GetSizes().m_height;
    MI_S32 owidth  = dst.GetSizes().m_width;
    MI_S32 oheight = dst.GetSizes().m_height;
    MI_S32 channel = dst.GetSizes().m_channel;
    MI_S32 thread_num = wp->GetComputeThreadNum();

    MI_F32 scale_x = static_cast<MI_F32>(iwidth) / owidth;
    MI_F32 scale_y = static_cast<MI_F32>(iheight) / oheight;

    if (scale_x < 1.0f || scale_y < 1.0f)
    {
        ret = ResizeBnCommHvx(ctx, src, dst, MI_TRUE);
    }
    else
    {
        MI_S32 owidth_xscale  = owidth * (Floor(scale_x) + 2);
        MI_S32 oheight_xscale = oheight * (Floor(scale_y) + 2);

        MI_S32 xtab_size          = AURA_ALIGN(owidth_xscale  * sizeof(MI_U16), AURA_HVLEN);
        MI_S32 alpha_size         = AURA_ALIGN(owidth_xscale  * sizeof(MI_U16), AURA_HVLEN);
        MI_S32 ytab_size          = AURA_ALIGN(oheight_xscale * sizeof(MI_U16), AURA_HVLEN);
        MI_S32 beta_size          = AURA_ALIGN(oheight_xscale * sizeof(MI_U16), AURA_HVLEN);
        MI_S32 src_buffer_size    = AURA_ALIGN(iwidth * sizeof(MI_U16), AURA_HVLEN) * thread_num * channel;
        MI_S32 gather_buffer_size = (AURA_HVLEN << 2) * thread_num * channel;
        MI_S32 total_buffer_size  = xtab_size + alpha_size + ytab_size + beta_size + src_buffer_size + gather_buffer_size;

        MI_U8 *vtcm_mem = static_cast<MI_U8*>(AURA_ALLOC_PARAM(ctx, AURA_MEM_VTCM, total_buffer_size, AURA_HVLEN));
        if (MI_NULL == vtcm_mem)
        {
            AURA_ADD_ERROR_STRING(ctx, "alloc vtcm memory failed");
            AURA_FREE(ctx, vtcm_mem);
            return Status::ABORT;
        }

        struct ResizeAreaDnVtcmBuffer vtcm_buffer;
        vtcm_buffer.xtab             = vtcm_mem;
        vtcm_buffer.alpha            = vtcm_buffer.xtab  + xtab_size;
        vtcm_buffer.ytab             = vtcm_buffer.alpha + alpha_size;
        vtcm_buffer.beta             = vtcm_buffer.ytab + ytab_size;
        vtcm_buffer.src_buffer       = vtcm_buffer.beta + beta_size;
        vtcm_buffer.src_buffer_pitch = src_buffer_size / (thread_num * channel);
        vtcm_buffer.gather_buffer    = vtcm_buffer.src_buffer + src_buffer_size;

        ret  = GetResizeAreaCommDnOffset<Tp>(iwidth, owidth, iheight, oheight, &vtcm_buffer);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(ctx, "GetResizeAreaCommDnOffset failed");
            AURA_FREE(ctx, vtcm_mem);
            return ret;
        }

        switch (channel)
        {
            case 1:
            {
                ret = wp->ParallelFor((MI_S32)0, oheight, ResizeAreaDnCommHvxImpl<Tp, 1>, std::cref(src), std::ref(dst), &vtcm_buffer, thread_num);
                if (ret != Status::OK)
                {
                    AURA_ADD_ERROR_STRING(ctx, "ParallelFor for ResizeAreaDnCommHvxImpl of c1 failed");
                }
                break;
            }

            case 2:
            {
                ret = wp->ParallelFor((MI_S32)0, oheight, ResizeAreaDnCommHvxImpl<Tp, 2>, std::cref(src), std::ref(dst), &vtcm_buffer, thread_num);
                if (ret != Status::OK)
                {
                    AURA_ADD_ERROR_STRING(ctx, "ParallelFor for ResizeAreaDnCommHvxImpl of c2 failed");
                }
                break;
            }

            case 3:
            {
                ret = wp->ParallelFor((MI_S32)0, oheight, ResizeAreaDnCommHvxImpl<Tp, 3>, std::cref(src), std::ref(dst), &vtcm_buffer, thread_num);
                if (ret != Status::OK)
                {
                    AURA_ADD_ERROR_STRING(ctx, "ParallelFor for ResizeAreaDnCommHvxImpl of c3 failed");
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

    return Status::OK;
}

Status ResizeAreaCommonHvx(Context *ctx, const Mat &src, Mat &dst)
{
    Status ret = Status::ERROR;

    switch (src.GetElemType())
    {
        case ElemType::U8:
        {
            ret = ResizeAreaCommHvxHelper<MI_U8>(ctx, src, dst);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeAreaCommHvxHelper run failed, type: MI_U8");
            }
            break;
        }

        case ElemType::S8:
        {
            ret = ResizeAreaCommHvxHelper<MI_S8>(ctx, src, dst);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeAreaCommHvxHelper run failed, type: MI_S8");
            }
            break;
        }

        case ElemType::U16:
        {
            ret = ResizeAreaCommHvxHelper<MI_U16>(ctx, src, dst);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeAreaCommHvxHelper run failed, type: MI_U16");
            }
            break;
        }

        case ElemType::S16:
        {
            ret = ResizeAreaCommHvxHelper<MI_S16>(ctx, src, dst);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeAreaCommHvxHelper run failed, type: MI_S16");
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