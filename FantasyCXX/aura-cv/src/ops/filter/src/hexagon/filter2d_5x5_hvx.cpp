#include "filter2d_impl.hpp"
#include "aura/runtime/worker_pool.h"
#include "aura/runtime/logger.h"

namespace aura
{

// using Tp = MI_U8
template <typename Tp, typename std::enable_if<std::is_same<Tp, MI_U8>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID Filter2d5x5Core(HVX_Vector &vu8_src_x0, HVX_Vector &vu8_src_x1, HVX_Vector &vu8_src_x2,
                                           HVX_VectorPair &wu32_sum_lo, HVX_VectorPair &wu32_sum_hi,
                                           const MI_S16 *kernel, MI_S32 line)
{
    const MI_S32 idx = 5 * line;
    MI_U32 k0k0 = (kernel[idx + 0] << 16) | kernel[idx + 0];
    MI_U32 k1k1 = (kernel[idx + 1] << 16) | kernel[idx + 1];
    MI_U32 k2k2 = (kernel[idx + 2] << 16) | kernel[idx + 2];
    MI_U32 k3k3 = (kernel[idx + 3] << 16) | kernel[idx + 3];
    MI_U32 k4k4 = (kernel[idx + 4] << 16) | kernel[idx + 4];

    HVX_VectorPair wu16_src_l1 = Q6_Wuh_vzxt_Vub(Q6_V_vlalign_VVR(vu8_src_x1, vu8_src_x0, 2));
    HVX_VectorPair wu16_src_l0 = Q6_Wuh_vzxt_Vub(Q6_V_vlalign_VVR(vu8_src_x1, vu8_src_x0, 1));
    HVX_VectorPair wu16_src_c  = Q6_Wuh_vzxt_Vub(vu8_src_x1);
    HVX_VectorPair wu16_src_r0 = Q6_Wuh_vzxt_Vub(Q6_V_valign_VVR(vu8_src_x2, vu8_src_x1, 1));
    HVX_VectorPair wu16_src_r1 = Q6_Wuh_vzxt_Vub(Q6_V_valign_VVR(vu8_src_x2, vu8_src_x1, 2));

    wu32_sum_lo = Q6_Wuw_vmpyacc_WuwVuhRuh(wu32_sum_lo, Q6_V_lo_W(wu16_src_l1), k0k0);
    wu32_sum_hi = Q6_Wuw_vmpyacc_WuwVuhRuh(wu32_sum_hi, Q6_V_hi_W(wu16_src_l1), k0k0);
    wu32_sum_lo = Q6_Wuw_vmpyacc_WuwVuhRuh(wu32_sum_lo, Q6_V_lo_W(wu16_src_l0), k1k1);
    wu32_sum_hi = Q6_Wuw_vmpyacc_WuwVuhRuh(wu32_sum_hi, Q6_V_hi_W(wu16_src_l0), k1k1);
    wu32_sum_lo = Q6_Wuw_vmpyacc_WuwVuhRuh(wu32_sum_lo, Q6_V_lo_W(wu16_src_c),  k2k2);
    wu32_sum_hi = Q6_Wuw_vmpyacc_WuwVuhRuh(wu32_sum_hi, Q6_V_hi_W(wu16_src_c),  k2k2);
    wu32_sum_lo = Q6_Wuw_vmpyacc_WuwVuhRuh(wu32_sum_lo, Q6_V_lo_W(wu16_src_r0), k3k3);
    wu32_sum_hi = Q6_Wuw_vmpyacc_WuwVuhRuh(wu32_sum_hi, Q6_V_hi_W(wu16_src_r0), k3k3);
    wu32_sum_lo = Q6_Wuw_vmpyacc_WuwVuhRuh(wu32_sum_lo, Q6_V_lo_W(wu16_src_r1), k4k4);
    wu32_sum_hi = Q6_Wuw_vmpyacc_WuwVuhRuh(wu32_sum_hi, Q6_V_hi_W(wu16_src_r1), k4k4);
}

template <typename Tp, typename std::enable_if<std::is_same<Tp, MI_U8>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE HVX_Vector Filter2d5x5Vector(HVX_Vector &vu8_src_p1x0, HVX_Vector &vu8_src_p1x1, HVX_Vector &vu8_src_p1x2,
                                                HVX_Vector &vu8_src_p0x0, HVX_Vector &vu8_src_p0x1, HVX_Vector &vu8_src_p0x2,
                                                HVX_Vector &vu8_src_cx0,  HVX_Vector &vu8_src_cx1,  HVX_Vector &vu8_src_cx2,
                                                HVX_Vector &vu8_src_n0x0, HVX_Vector &vu8_src_n0x1, HVX_Vector &vu8_src_n0x2,
                                                HVX_Vector &vu8_src_n1x0, HVX_Vector &vu8_src_n1x1, HVX_Vector &vu8_src_n1x2,
                                                const MI_S16 *kernel)
{
    HVX_VectorPair wu32_sum_lo = Q6_W_vcombine_VV(Q6_V_vzero(), Q6_V_vzero());
    HVX_VectorPair wu32_sum_hi = wu32_sum_lo;

    Filter2d5x5Core<Tp>(vu8_src_p1x0, vu8_src_p1x1, vu8_src_p1x2, wu32_sum_lo, wu32_sum_hi, kernel, 0);
    Filter2d5x5Core<Tp>(vu8_src_p0x0, vu8_src_p0x1, vu8_src_p0x2, wu32_sum_lo, wu32_sum_hi, kernel, 1);
    Filter2d5x5Core<Tp>(vu8_src_cx0,  vu8_src_cx1,  vu8_src_cx2,  wu32_sum_lo, wu32_sum_hi, kernel, 2);
    Filter2d5x5Core<Tp>(vu8_src_n0x0, vu8_src_n0x1, vu8_src_n0x2, wu32_sum_lo, wu32_sum_hi, kernel, 3);
    Filter2d5x5Core<Tp>(vu8_src_n1x0, vu8_src_n1x1, vu8_src_n1x2, wu32_sum_lo, wu32_sum_hi, kernel, 4);

    HVX_Vector vu16_result_lo = Q6_Vuh_vasr_VuwVuwR_rnd_sat(Q6_V_hi_W(wu32_sum_lo), Q6_V_lo_W(wu32_sum_lo), 12);
    HVX_Vector vu16_result_hi = Q6_Vuh_vasr_VuwVuwR_rnd_sat(Q6_V_hi_W(wu32_sum_hi), Q6_V_lo_W(wu32_sum_hi), 12);

    return Q6_Vub_vsat_VhVh(vu16_result_hi, vu16_result_lo);
}

// using Tp = MI_U16
template <typename Tp, typename std::enable_if<std::is_same<Tp, MI_U16>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID Filter2d5x5Core(HVX_Vector &vu16_src_x0, HVX_Vector &vu16_src_x1, HVX_Vector &vu16_src_x2,
                                           HVX_VectorPair &wu32_sum, const MI_S16 *kernel, MI_S32 line)
{
    const MI_S32 idx = 5 * line;
    MI_U32 k0k0 = (kernel[idx + 0] << 16) | kernel[idx + 0];
    MI_U32 k1k1 = (kernel[idx + 1] << 16) | kernel[idx + 1];
    MI_U32 k2k2 = (kernel[idx + 2] << 16) | kernel[idx + 2];
    MI_U32 k3k3 = (kernel[idx + 3] << 16) | kernel[idx + 3];
    MI_U32 k4k4 = (kernel[idx + 4] << 16) | kernel[idx + 4];

    HVX_Vector vu16_src_l1 = Q6_V_vlalign_VVR(vu16_src_x1, vu16_src_x0, 4);
    HVX_Vector vu16_src_l0 = Q6_V_vlalign_VVR(vu16_src_x1, vu16_src_x0, 2);
    HVX_Vector vu16_src_r0 = Q6_V_valign_VVR(vu16_src_x2, vu16_src_x1, 2);
    HVX_Vector vu16_src_r1 = Q6_V_valign_VVR(vu16_src_x2, vu16_src_x1, 4);

    wu32_sum = Q6_Wuw_vmpyacc_WuwVuhRuh(wu32_sum, vu16_src_l1, k0k0);
    wu32_sum = Q6_Wuw_vmpyacc_WuwVuhRuh(wu32_sum, vu16_src_l0, k1k1);
    wu32_sum = Q6_Wuw_vmpyacc_WuwVuhRuh(wu32_sum, vu16_src_x1, k2k2);
    wu32_sum = Q6_Wuw_vmpyacc_WuwVuhRuh(wu32_sum, vu16_src_r0, k3k3);
    wu32_sum = Q6_Wuw_vmpyacc_WuwVuhRuh(wu32_sum, vu16_src_r1, k4k4);
}

template <typename Tp, typename std::enable_if<std::is_same<Tp, MI_U16>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE HVX_Vector Filter2d5x5Vector(HVX_Vector &vu16_src_p1x0, HVX_Vector &vu16_src_p1x1, HVX_Vector &vu16_src_p1x2,
                                                HVX_Vector &vu16_src_p0x0, HVX_Vector &vu16_src_p0x1, HVX_Vector &vu16_src_p0x2,
                                                HVX_Vector &vu16_src_cx0,  HVX_Vector &vu16_src_cx1,  HVX_Vector &vu16_src_cx2,
                                                HVX_Vector &vu16_src_n0x0, HVX_Vector &vu16_src_n0x1, HVX_Vector &vu16_src_n0x2,
                                                HVX_Vector &vu16_src_n1x0, HVX_Vector &vu16_src_n1x1, HVX_Vector &vu16_src_n1x2,
                                                const MI_S16 *kernel)
{
    HVX_VectorPair wu32_sum = Q6_W_vcombine_VV(Q6_V_vzero(), Q6_V_vzero());

    Filter2d5x5Core<Tp>(vu16_src_p1x0, vu16_src_p1x1, vu16_src_p1x2, wu32_sum, kernel, 0);
    Filter2d5x5Core<Tp>(vu16_src_p0x0, vu16_src_p0x1, vu16_src_p0x2, wu32_sum, kernel, 1);
    Filter2d5x5Core<Tp>(vu16_src_cx0,  vu16_src_cx1,  vu16_src_cx2,  wu32_sum, kernel, 2);
    Filter2d5x5Core<Tp>(vu16_src_n0x0, vu16_src_n0x1, vu16_src_n0x2, wu32_sum, kernel, 3);
    Filter2d5x5Core<Tp>(vu16_src_n1x0, vu16_src_n1x1, vu16_src_n1x2, wu32_sum, kernel, 4);

    return Q6_Vuh_vasr_VuwVuwR_rnd_sat(Q6_V_hi_W(wu32_sum), Q6_V_lo_W(wu32_sum), 12);
}

// using Tp = MI_S16
template <typename Tp, typename std::enable_if<std::is_same<Tp, MI_S16>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID Filter2d5x5Core(HVX_Vector &vs16_src_x0, HVX_Vector &vs16_src_x1, HVX_Vector &vs16_src_x2,
                                           HVX_VectorPair &ws32_sum, const MI_S16 *kernel, MI_S32 line)
{
    const MI_S32 idx = 5 * line;
    MI_U32 k0k0 = (kernel[idx + 0] << 16) | kernel[idx + 0];
    MI_U32 k1k1 = (kernel[idx + 1] << 16) | kernel[idx + 1];
    MI_U32 k2k2 = (kernel[idx + 2] << 16) | kernel[idx + 2];
    MI_U32 k3k3 = (kernel[idx + 3] << 16) | kernel[idx + 3];
    MI_U32 k4k4 = (kernel[idx + 4] << 16) | kernel[idx + 4];

    HVX_Vector vs16_src_l1 = Q6_V_vlalign_VVR(vs16_src_x1, vs16_src_x0, 4);
    HVX_Vector vs16_src_l0 = Q6_V_vlalign_VVR(vs16_src_x1, vs16_src_x0, 2);
    HVX_Vector vs16_src_r0 = Q6_V_valign_VVR(vs16_src_x2, vs16_src_x1, 2);
    HVX_Vector vs16_src_r1 = Q6_V_valign_VVR(vs16_src_x2, vs16_src_x1, 4);

    ws32_sum = Q6_Ww_vmpyacc_WwVhRh(ws32_sum, vs16_src_l1, k0k0);
    ws32_sum = Q6_Ww_vmpyacc_WwVhRh(ws32_sum, vs16_src_l0, k1k1);
    ws32_sum = Q6_Ww_vmpyacc_WwVhRh(ws32_sum, vs16_src_x1, k2k2);
    ws32_sum = Q6_Ww_vmpyacc_WwVhRh(ws32_sum, vs16_src_r0, k3k3);
    ws32_sum = Q6_Ww_vmpyacc_WwVhRh(ws32_sum, vs16_src_r1, k4k4);
}

template <typename Tp, typename std::enable_if<std::is_same<Tp, MI_S16>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE HVX_Vector Filter2d5x5Vector(HVX_Vector &vs16_src_p1x0, HVX_Vector &vs16_src_p1x1, HVX_Vector &vs16_src_p1x2,
                                                HVX_Vector &vs16_src_p0x0, HVX_Vector &vs16_src_p0x1, HVX_Vector &vs16_src_p0x2,
                                                HVX_Vector &vs16_src_cx0,  HVX_Vector &vs16_src_cx1,  HVX_Vector &vs16_src_cx2,
                                                HVX_Vector &vs16_src_n0x0, HVX_Vector &vs16_src_n0x1, HVX_Vector &vs16_src_n0x2,
                                                HVX_Vector &vs16_src_n1x0, HVX_Vector &vs16_src_n1x1, HVX_Vector &vs16_src_n1x2,
                                                const MI_S16 *kernel)
{
    HVX_VectorPair ws32_sum = Q6_W_vcombine_VV(Q6_V_vzero(), Q6_V_vzero());

    Filter2d5x5Core<Tp>(vs16_src_p1x0, vs16_src_p1x1, vs16_src_p1x2, ws32_sum, kernel, 0);
    Filter2d5x5Core<Tp>(vs16_src_p0x0, vs16_src_p0x1, vs16_src_p0x2, ws32_sum, kernel, 1);
    Filter2d5x5Core<Tp>(vs16_src_cx0,  vs16_src_cx1,  vs16_src_cx2,  ws32_sum, kernel, 2);
    Filter2d5x5Core<Tp>(vs16_src_n0x0, vs16_src_n0x1, vs16_src_n0x2, ws32_sum, kernel, 3);
    Filter2d5x5Core<Tp>(vs16_src_n1x0, vs16_src_n1x1, vs16_src_n1x2, ws32_sum, kernel, 4);

    return Q6_Vh_vasr_VwVwR_rnd_sat(Q6_V_hi_W(ws32_sum), Q6_V_lo_W(ws32_sum), 12);
}

template <typename Tp>
AURA_ALWAYS_INLINE AURA_VOID Filter2d5x5Vector(HVX_Vector &v_src_p1x0, HVX_Vector &v_src_p1x1, HVX_Vector &v_src_p1x2, HVX_Vector &v_border_p1,
                                             HVX_Vector &v_src_p0x0, HVX_Vector &v_src_p0x1, HVX_Vector &v_src_p0x2, HVX_Vector &v_border_p0,
                                             HVX_Vector &v_src_cx0,  HVX_Vector &v_src_cx1,  HVX_Vector &v_src_cx2,  HVX_Vector &v_border_c,
                                             HVX_Vector &v_src_n0x0, HVX_Vector &v_src_n0x1, HVX_Vector &v_src_n0x2, HVX_Vector &v_border_n0,
                                             HVX_Vector &v_src_n1x0, HVX_Vector &v_src_n1x1, HVX_Vector &v_src_n1x2, HVX_Vector &v_border_n1,
                                             HVX_Vector &v_result_x0, HVX_Vector &v_result_x1,
                                             const MI_S16 *kernel, MI_S32 align_size)
{
    HVX_Vector v_src_p1r0 = Q6_V_vlalign_VVR(v_border_p1, v_src_p1x2, align_size);
    HVX_Vector v_src_p0r0 = Q6_V_vlalign_VVR(v_border_p0, v_src_p0x2, align_size);
    HVX_Vector v_src_cr0  = Q6_V_vlalign_VVR(v_border_c,  v_src_cx2,  align_size);
    HVX_Vector v_src_n0r0 = Q6_V_vlalign_VVR(v_border_n0, v_src_n0x2, align_size);
    HVX_Vector v_src_n1r0 = Q6_V_vlalign_VVR(v_border_n1, v_src_n1x2, align_size);
    v_result_x0 = Filter2d5x5Vector<Tp>(v_src_p1x0, v_src_p1x1, v_src_p1r0,
                                        v_src_p0x0, v_src_p0x1, v_src_p0r0,
                                        v_src_cx0,  v_src_cx1,  v_src_cr0,
                                        v_src_n0x0, v_src_n0x1, v_src_n0r0,
                                        v_src_n1x0, v_src_n1x1, v_src_n1r0,
                                        kernel);

    HVX_Vector v_src_p1l0 = Q6_V_valign_VVR(v_src_p1x1, v_src_p1x0, align_size);
    HVX_Vector v_src_p0l0 = Q6_V_valign_VVR(v_src_p0x1, v_src_p0x0, align_size);
    HVX_Vector v_src_cl0  = Q6_V_valign_VVR(v_src_cx1,  v_src_cx0,  align_size);
    HVX_Vector v_src_n0l0 = Q6_V_valign_VVR(v_src_n0x1, v_src_n0x0, align_size);
    HVX_Vector v_src_n1l0 = Q6_V_valign_VVR(v_src_n1x1, v_src_n1x0, align_size);
    v_result_x1 = Filter2d5x5Vector<Tp>(v_src_p1l0, v_src_p1x2, v_border_p1,
                                        v_src_p0l0, v_src_p0x2, v_border_p0,
                                        v_src_cl0,  v_src_cx2,  v_border_c,
                                        v_src_n0l0, v_src_n0x2, v_border_n0,
                                        v_src_n1l0, v_src_n1x2, v_border_n1,
                                        kernel);
}

template <typename Tp, BorderType BORDER_TYPE, MI_S32 C>
static AURA_VOID Filter2d5x5Row(const Tp *src_p1, const Tp *src_p0, const Tp *src_c, const Tp *src_n0, const Tp *src_n1,
                              Tp *dst, MI_S32 width, const MI_S16 *kernel, const std::vector<Tp> &border_value)
{
    using MVType = typename MVHvxVector<C>::Type;

    constexpr MI_S32 ELEM_COUNTS = AURA_HVLEN / sizeof(Tp);
    MI_S32 back_offset = width - ELEM_COUNTS;

    MVType mv_src_p1[3], mv_src_p0[3], mv_src_c[3], mv_src_n0[3], mv_src_n1[3];
    MVType mv_result;

    // left border
    {
        vload(src_p1, mv_src_p1[1]);
        vload(src_p0, mv_src_p0[1]);
        vload(src_c,  mv_src_c[1]);
        vload(src_n0, mv_src_n0[1]);
        vload(src_n1, mv_src_n1[1]);

        #pragma unroll(C)
        for (MI_S32 ch = 0; ch < C; ch++)
        {
            mv_src_p1[0].val[ch] = GetBorderVector<Tp, BORDER_TYPE, BorderArea::LEFT>(mv_src_p1[1].val[ch], src_p1[ch], border_value[ch]);
            mv_src_p0[0].val[ch] = GetBorderVector<Tp, BORDER_TYPE, BorderArea::LEFT>(mv_src_p0[1].val[ch], src_p0[ch], border_value[ch]);
            mv_src_c[0].val[ch]  = GetBorderVector<Tp, BORDER_TYPE, BorderArea::LEFT>(mv_src_c[1].val[ch],  src_c[ch],  border_value[ch]);
            mv_src_n0[0].val[ch] = GetBorderVector<Tp, BORDER_TYPE, BorderArea::LEFT>(mv_src_n0[1].val[ch], src_n0[ch], border_value[ch]);
            mv_src_n1[0].val[ch] = GetBorderVector<Tp, BORDER_TYPE, BorderArea::LEFT>(mv_src_n1[1].val[ch], src_n1[ch], border_value[ch]);
        }
    }

    // main(0 ~ n-2)
    {
        for (MI_S32 x = ELEM_COUNTS; x <= back_offset; x += ELEM_COUNTS)
        {
            vload(src_p1 + C * x, mv_src_p1[2]);
            vload(src_p0 + C * x, mv_src_p0[2]);
            vload(src_c  + C * x, mv_src_c[2]);
            vload(src_n0 + C * x, mv_src_n0[2]);
            vload(src_n1 + C * x, mv_src_n1[2]);

            #pragma unroll(C)
            for (MI_S32 ch = 0; ch < C; ch++)
            {
                mv_result.val[ch] = Filter2d5x5Vector<Tp>(mv_src_p1[0].val[ch], mv_src_p1[1].val[ch], mv_src_p1[2].val[ch],
                                                          mv_src_p0[0].val[ch], mv_src_p0[1].val[ch], mv_src_p0[2].val[ch],
                                                          mv_src_c[0].val[ch],  mv_src_c[1].val[ch],  mv_src_c[2].val[ch],
                                                          mv_src_n0[0].val[ch], mv_src_n0[1].val[ch], mv_src_n0[2].val[ch],
                                                          mv_src_n1[0].val[ch], mv_src_n1[1].val[ch], mv_src_n1[2].val[ch],
                                                          kernel);
            }
            vstore(dst + C * (x - ELEM_COUNTS), mv_result);

            mv_src_p1[0] = mv_src_p1[1];
            mv_src_p0[0] = mv_src_p0[1];
            mv_src_c[0]  = mv_src_c[1];
            mv_src_n0[0] = mv_src_n0[1];
            mv_src_n1[0] = mv_src_n1[1];

            mv_src_p1[1] = mv_src_p1[2];
            mv_src_p0[1] = mv_src_p0[2];
            mv_src_c[1]  = mv_src_c[2];
            mv_src_n0[1] = mv_src_n0[2];
            mv_src_n1[1] = mv_src_n1[2];
        }
    }

    // remain
    {
        MI_S32 last = C * (width - 1);
        MI_S32 rest = width % ELEM_COUNTS;
        MVType mv_last;

        vload(src_p1 + C * back_offset, mv_src_p1[2]);
        vload(src_p0 + C * back_offset, mv_src_p0[2]);
        vload(src_c  + C * back_offset, mv_src_c[2]);
        vload(src_n0 + C * back_offset, mv_src_n0[2]);
        vload(src_n1 + C * back_offset, mv_src_n1[2]);

        #pragma unroll(C)
        for (MI_S32 ch = 0; ch < C; ch++)
        {
            HVX_Vector v_border_p1 = GetBorderVector<Tp, BORDER_TYPE, BorderArea::RIGHT>(mv_src_p1[2].val[ch], src_p1[ch + last], border_value[ch]);
            HVX_Vector v_border_p0 = GetBorderVector<Tp, BORDER_TYPE, BorderArea::RIGHT>(mv_src_p0[2].val[ch], src_p0[ch + last], border_value[ch]);
            HVX_Vector v_border_c  = GetBorderVector<Tp, BORDER_TYPE, BorderArea::RIGHT>(mv_src_c[2].val[ch],  src_c[ch + last],  border_value[ch]);
            HVX_Vector v_border_n0 = GetBorderVector<Tp, BORDER_TYPE, BorderArea::RIGHT>(mv_src_n0[2].val[ch], src_n0[ch + last], border_value[ch]);
            HVX_Vector v_border_n1 = GetBorderVector<Tp, BORDER_TYPE, BorderArea::RIGHT>(mv_src_n1[2].val[ch], src_n1[ch + last], border_value[ch]);

            Filter2d5x5Vector<Tp>(mv_src_p1[0].val[ch], mv_src_p1[1].val[ch], mv_src_p1[2].val[ch], v_border_p1,
                                  mv_src_p0[0].val[ch], mv_src_p0[1].val[ch], mv_src_p0[2].val[ch], v_border_p0,
                                  mv_src_c[0].val[ch],  mv_src_c[1].val[ch],  mv_src_c[2].val[ch],  v_border_c,
                                  mv_src_n0[0].val[ch], mv_src_n0[1].val[ch], mv_src_n0[2].val[ch], v_border_n0,
                                  mv_src_n1[0].val[ch], mv_src_n1[1].val[ch], mv_src_n1[2].val[ch], v_border_n1,
                                  mv_result.val[ch], mv_last.val[ch],
                                  kernel, rest * sizeof(Tp));
        }

        vstore(dst + C * (back_offset - rest), mv_result);
        vstore(dst + C * back_offset, mv_last);
    }
}

template <typename Tp, BorderType BORDER_TYPE, MI_S32 C>
static Status Filter2d5x5HvxImpl(const Mat &src, Mat &dst, const std::vector<MI_S16> &kdata, const std::vector<Tp> &border_value,
                                 const Tp *border_buffer, MI_S32 start_row, MI_S32 end_row)
{
    MI_S32 width   = src.GetSizes().m_width;
    MI_S32 height  = src.GetSizes().m_height;
    MI_S32 istride = src.GetStrides().m_width;

    const MI_S16 *kernel = kdata.data();

    const Tp *src_p1 = src.Ptr<Tp, BORDER_TYPE>(start_row - 2, border_buffer);
    const Tp *src_p0 = src.Ptr<Tp, BORDER_TYPE>(start_row - 1, border_buffer);
    const Tp *src_c  = src.Ptr<Tp>(start_row);
    const Tp *src_n0 = src.Ptr<Tp, BORDER_TYPE>(start_row + 1, border_buffer);
    const Tp *src_n1 = src.Ptr<Tp, BORDER_TYPE>(start_row + 2, border_buffer);

    MI_U64 L2fetch_param = L2PfParam(istride, width * C * ElemTypeSize(src.GetElemType()), 1, 0);
    for (MI_S32 y = start_row; y < end_row; y++)
    {
        if (y + 3 < height)
        {
            L2Fetch(reinterpret_cast<MI_U32>(src.Ptr<Tp>(y + 3)), L2fetch_param);
        }

        Tp *dst_row = dst.Ptr<Tp>(y);
        Filter2d5x5Row<Tp, BORDER_TYPE, C>(src_p1, src_p0, src_c, src_n0, src_n1, dst_row, width, kernel, border_value);

        src_p1 = src_p0;
        src_p0 = src_c;
        src_c  = src_n0;
        src_n0 = src_n1;
        src_n1 = src.Ptr<Tp, BORDER_TYPE>(y + 3, border_buffer);
    }

    return Status::OK;
}

template<typename Tp, BorderType BORDER_TYPE>
static Status Filter2d5x5HvxHelper(Context *ctx, const Mat &src, Mat &dst, const std::vector<MI_S16> &kdata,
                                   const std::vector<Tp> &border_value, const Tp *border_buffer)
{
    Status ret = Status::ERROR;

    WorkerPool *wp = ctx->GetWorkerPool();
    if (MI_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerpool failed");
        return ret;
    }

    MI_S32 height  = src.GetSizes().m_height;
    MI_S32 channel = src.GetSizes().m_channel;

    switch (channel)
    {
        case 1:
        {
            ret = wp->ParallelFor((MI_S32)0, height, Filter2d5x5HvxImpl<Tp, BORDER_TYPE, 1>,
                                  std::cref(src), std::ref(dst), std::cref(kdata), std::cref(border_value), border_buffer);
            break;
        }

        case 2:
        {
            ret = wp->ParallelFor((MI_S32)0, height, Filter2d5x5HvxImpl<Tp, BORDER_TYPE, 2>,
                                  std::cref(src), std::ref(dst), std::cref(kdata), std::cref(border_value), border_buffer);
            break;
        }

        case 3:
        {
            ret = wp->ParallelFor((MI_S32)0, height, Filter2d5x5HvxImpl<Tp, BORDER_TYPE, 3>,
                                  std::cref(src), std::ref(dst), std::cref(kdata), std::cref(border_value), border_buffer);
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "unsupported channel");
            break;
        }
    }

    AURA_RETURN(ctx, ret);
}

template <typename Tp>
static Status Filter2d5x5HvxHelper(Context *ctx, const Mat &src, Mat &dst, const std::vector<MI_S16> &kdata,
                                   BorderType &border_type, const Scalar &border_value)
{
    Status ret = Status::ERROR;

    Tp *border_buffer = MI_NULL;
    std::vector<Tp> vec_border_value = border_value.ToVector<Tp>();

    MI_S32 width   = dst.GetSizes().m_width;
    MI_S32 channel = dst.GetSizes().m_channel;

    switch (border_type)
    {
        case BorderType::CONSTANT:
        {
            border_buffer = CreateBorderBuffer(ctx, width, channel, vec_border_value);
            if (MI_NULL == border_buffer)
            {
                AURA_ADD_ERROR_STRING(ctx, "CreateBorderBuffer failed");
                return Status::ERROR;
            }

            ret = Filter2d5x5HvxHelper<Tp, BorderType::CONSTANT>(ctx, src, dst, kdata, vec_border_value, border_buffer);

            AURA_FREE(ctx, border_buffer);
            break;
        }

        case BorderType::REPLICATE:
        {
            ret = Filter2d5x5HvxHelper<Tp, BorderType::REPLICATE>(ctx, src, dst, kdata, vec_border_value, border_buffer);
            break;
        }

        case BorderType::REFLECT_101:
        {
            ret = Filter2d5x5HvxHelper<Tp, BorderType::REFLECT_101>(ctx, src, dst, kdata, vec_border_value, border_buffer);
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "unsupported border_type");
            return Status::ERROR;
        }
    }

    AURA_RETURN(ctx, ret);
}

Status Filter2d5x5Hvx(Context *ctx, const Mat &src, Mat &dst, const std::vector<MI_S16> &kdata,
                      BorderType border_type, const Scalar &border_value)
{
    Status ret = Status::ERROR;
    MI_S32 pattern = AURA_MAKE_PATTERN(src.GetElemType(), dst.GetElemType());

    switch (pattern)
    {
        case AURA_MAKE_PATTERN(ElemType::U8, ElemType::U8):
        {
            ret = Filter2d5x5HvxHelper<MI_U8>(ctx, src, dst, kdata, border_type, border_value);
            break;
        }

        case AURA_MAKE_PATTERN(ElemType::U16, ElemType::U16):
        {
            ret = Filter2d5x5HvxHelper<MI_U16>(ctx, src, dst, kdata, border_type, border_value);
            break;
        }

        case AURA_MAKE_PATTERN(ElemType::S16, ElemType::S16):
        {
            ret = Filter2d5x5HvxHelper<MI_S16>(ctx, src, dst, kdata, border_type, border_value);
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "unsupported source format");
            return Status::ERROR;
        }
    }

    AURA_RETURN(ctx, ret);
}

} // namespace aura