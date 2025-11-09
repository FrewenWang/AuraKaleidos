#include "pyrdown_impl.hpp"
#include "aura/runtime/logger.h"
#include "aura/runtime/worker_pool.h"

namespace aura
{

// using Tp = DT_U8
template <typename Tp, typename std::enable_if<std::is_same<Tp, DT_U8>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID Pyrdown5x5VCore(const HVX_Vector &vu8_src_p1, const HVX_Vector &vu8_src_p0,
                                           const HVX_Vector &vu8_src_c,  const HVX_Vector &vu8_src_n0,
                                           const HVX_Vector &vu8_src_n1, HVX_VectorPair &wu16_sum, const DT_U16 *kernel)
{
    DT_U32 k0k0k0k0 = Q6_R_vsplatb_R(kernel[0]);
    DT_U32 k1k1k1k1 = Q6_R_vsplatb_R(kernel[1]);
    DT_U32 k2k2k2k2 = Q6_R_vsplatb_R(kernel[2]);

    wu16_sum = Q6_Wh_vmpa_WubRub(Q6_W_vcombine_VV(vu8_src_p1, vu8_src_n1), k0k0k0k0);
    wu16_sum = Q6_Wh_vmpaacc_WhWubRub(wu16_sum, Q6_W_vcombine_VV(vu8_src_p0, vu8_src_n0), k1k1k1k1);
    wu16_sum = Q6_Wuh_vmpyacc_WuhVubRub(wu16_sum, vu8_src_c, k2k2k2k2);
}

// using Tp = DT_U8
template <typename Tp, typename std::enable_if<std::is_same<Tp, DT_U8>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID Pyrdown5x5HCore(const HVX_Vector &vu16_sum_l1, const HVX_Vector &vu16_sum_l0,
                                           const HVX_Vector &vu16_sum_c,  const HVX_Vector &vu16_sum_r0,
                                           const HVX_Vector &vu16_sum_r1, HVX_Vector &vu16_result, const DT_U16 *kernel)
{
    DT_U32 k0k0k0k0 = Q6_R_vsplatb_R(kernel[0]);
    DT_U32 k1k1k1k1 = Q6_R_vsplatb_R(kernel[1]);
    DT_U32 k2k2     = (kernel[2] << 16) | kernel[2];

    HVX_VectorPair wu32_result;

    wu32_result = Q6_Ww_vmpa_WuhRb(Q6_W_vcombine_VV(vu16_sum_l1, vu16_sum_r1), k0k0k0k0);
    wu32_result = Q6_Ww_vmpaacc_WwWuhRb(wu32_result, Q6_W_vcombine_VV(vu16_sum_l0, vu16_sum_r0), k1k1k1k1);
    wu32_result = Q6_Wuw_vmpyacc_WuwVuhRuh(wu32_result, vu16_sum_c, k2k2);
    vu16_result = Q6_Vuh_vround_VuwVuw_sat(Q6_V_hi_W(wu32_result), Q6_V_lo_W(wu32_result));
}

// using Tp = DT_U8
template <typename Tp, typename std::enable_if<std::is_same<Tp, DT_U8>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE HVX_Vector Pyrdown5x5Pack(const HVX_Vector &vu16_result_hi, const HVX_Vector &vu16_result_lo)
{
    return Q6_Vub_vpack_VhVh_sat(vu16_result_hi, vu16_result_lo);
}

// using Tp = DT_U16
template <typename Tp, typename std::enable_if<std::is_same<Tp, DT_U16>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID Pyrdown5x5VCore(const HVX_Vector &vu16_src_p1, const HVX_Vector &vu16_src_p0,
                                           const HVX_Vector &vu16_src_c,  const HVX_Vector &vu16_src_n0,
                                           const HVX_Vector &vu16_src_n1, HVX_VectorPair &wu32_sum, const DT_U32 *kernel)
{
    DT_U32 k0k0 = (kernel[0] << 16) | kernel[0];
    DT_U32 k1k1 = (kernel[1] << 16) | kernel[1];
    DT_U32 k2k2 = (kernel[2] << 16) | kernel[2];

    wu32_sum = Q6_Wuw_vmpy_VuhRuh(vu16_src_p1, k0k0);
    wu32_sum = Q6_Wuw_vmpyacc_WuwVuhRuh(wu32_sum, vu16_src_p0, k1k1);
    wu32_sum = Q6_Wuw_vmpyacc_WuwVuhRuh(wu32_sum, vu16_src_c,  k2k2);
    wu32_sum = Q6_Wuw_vmpyacc_WuwVuhRuh(wu32_sum, vu16_src_n0, k1k1);
    wu32_sum = Q6_Wuw_vmpyacc_WuwVuhRuh(wu32_sum, vu16_src_n1, k0k0);
}

// using Tp = DT_U16
template <typename Tp, typename std::enable_if<std::is_same<Tp, DT_U16>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID Pyrdown5x5HCore(const HVX_Vector &vu32_sum_l1, const HVX_Vector &vu32_sum_l0,
                                           const HVX_Vector &vu32_sum_c,  const HVX_Vector &vu32_sum_r0,
                                           const HVX_Vector &vu32_sum_r1, HVX_Vector &vu32_result, const DT_U32 *kernel)
{
    HVX_Vector vu32_k0 = Q6_V_vsplat_R(kernel[0]);
    HVX_Vector vu32_k1 = Q6_V_vsplat_R(kernel[1]);
    HVX_Vector vu32_k2 = Q6_V_vsplat_R(kernel[2]);

    HVX_VectorPair vu64_sum;

    vu64_sum = Q6_Wd_vmul_VwVw(Q6_Vw_vadd_VwVw(vu32_sum_l1, vu32_sum_r1), vu32_k0);
    vu64_sum = Q6_Wd_vmulacc_WdVwVw(vu64_sum, Q6_Vw_vadd_VwVw(vu32_sum_l0, vu32_sum_r0), vu32_k1);
    vu64_sum = Q6_Wd_vmulacc_WdVwVw(vu64_sum, vu32_sum_c, vu32_k2);
    vu64_sum = Q6_Wud_vadd_WudWud(vu64_sum, Q6_W_vcombine_VV(Q6_V_vzero(), Q6_V_vsplat_R(1 << 27)));

    // rshift 28
    vu32_result = Q6_V_vor_VV(Q6_Vw_vasl_VwR(Q6_V_hi_W(vu64_sum), 4), Q6_Vuw_vlsr_VuwR(Q6_V_lo_W(vu64_sum), 28));
}

// using Tp = DT_U16
template <typename Tp, typename std::enable_if<std::is_same<Tp, DT_U16>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE HVX_Vector Pyrdown5x5Pack(const HVX_Vector &vu32_result_hi,
                                             const HVX_Vector &vu32_result_lo)
{
    return Q6_Vuh_vpack_VwVw_sat(vu32_result_hi, vu32_result_lo);
}

// using Tp = DT_S16
template <typename Tp, typename std::enable_if<std::is_same<Tp, DT_S16>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID Pyrdown5x5VCore(const HVX_Vector &vs16_src_p1, const HVX_Vector &vs16_src_p0,
                                           const HVX_Vector &vs16_src_c,  const HVX_Vector &vs16_src_n0,
                                           const HVX_Vector &vs16_src_n1, HVX_VectorPair &ws32_sum, const DT_S32 *kernel)
{
    DT_S32 k0k0 = (kernel[0] << 16) | kernel[0];
    DT_S32 k1k1 = (kernel[1] << 16) | kernel[1];
    DT_S32 k2k2 = (kernel[2] << 16) | kernel[2];

    ws32_sum = Q6_Ww_vmpy_VhRh(vs16_src_p1, k0k0);
    ws32_sum = Q6_Ww_vmpyacc_WwVhRh(ws32_sum, vs16_src_p0, k1k1);
    ws32_sum = Q6_Ww_vmpyacc_WwVhRh(ws32_sum, vs16_src_c,  k2k2);
    ws32_sum = Q6_Ww_vmpyacc_WwVhRh(ws32_sum, vs16_src_n0, k1k1);
    ws32_sum = Q6_Ww_vmpyacc_WwVhRh(ws32_sum, vs16_src_n1, k0k0);
}

// using Tp = DT_S16
template <typename Tp, typename std::enable_if<std::is_same<Tp, DT_S16>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID Pyrdown5x5HCore(const HVX_Vector &vs32_sum_l1, const HVX_Vector &vs32_sum_l0,
                                           const HVX_Vector &vs32_sum_c,  const HVX_Vector &vs32_sum_r0,
                                           const HVX_Vector &vs32_sum_r1, HVX_Vector &vs32_result, const DT_S32 *kernel)
{
    HVX_Vector vs32_k0 = Q6_V_vsplat_R(kernel[0]);
    HVX_Vector vs32_k1 = Q6_V_vsplat_R(kernel[1]);
    HVX_Vector vs32_k2 = Q6_V_vsplat_R(kernel[2]);

    HVX_VectorPair vs64_sum;

    vs64_sum = Q6_Wd_vmul_VwVw(Q6_Vw_vadd_VwVw(vs32_sum_l1, vs32_sum_r1), vs32_k0);
    vs64_sum = Q6_Wd_vmulacc_WdVwVw(vs64_sum, Q6_Vw_vadd_VwVw(vs32_sum_l0, vs32_sum_r0), vs32_k1);
    vs64_sum = Q6_Wd_vmulacc_WdVwVw(vs64_sum, vs32_sum_c, vs32_k2);
    vs64_sum = Q6_Wd_vadd_WdWd(vs64_sum, Q6_W_vcombine_VV(Q6_V_vzero(), Q6_V_vsplat_R(1 << 27)));

    // rshift 28
    vs32_result = Q6_V_vor_VV(Q6_Vw_vasl_VwR(Q6_V_hi_W(vs64_sum), 4), Q6_Vuw_vlsr_VuwR(Q6_V_lo_W(vs64_sum), 28));
}

// using Tp = DT_S16
template <typename Tp, typename std::enable_if<std::is_same<Tp, DT_S16>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE HVX_Vector Pyrdown5x5Pack(const HVX_Vector &vs32_result_hi, const HVX_Vector &vs32_result_lo)
{
    return Q6_Vh_vpack_VwVw_sat(vs32_result_hi, vs32_result_lo);
}

// High w_sum_x0         ...  -1
// Low w_sum_x0          ...  -2
// High w_sum_x1   1   3 ... 127
// Low w_sum_x1    0   2 ... 126
// High w_sum_x2 129 131 ... 255
// Low w_sum_x2  128 130 ... 254
// High w_sum_x3 257 259 ... 383
// Low w_sum_x3  256 258 ... 382
template <typename Tp, typename Kt>
AURA_ALWAYS_INLINE DT_VOID Pyrdown5x5HCore(HVX_VectorPair &w_sum_x0, HVX_VectorPair &w_sum_x1, HVX_VectorPair &w_sum_x2,
                                           HVX_VectorPair &w_sum_x3, HVX_Vector &v_result, const Kt *kernel)
{
    DT_S32 align_size = sizeof(Kt);
    // -2  0 ...
    HVX_Vector v_l1_sum = Q6_V_vlalign_VVR(Q6_V_lo_W(w_sum_x1), Q6_V_lo_W(w_sum_x0), align_size);
    // -1  1 ...
    HVX_Vector v_l0_sum = Q6_V_vlalign_VVR(Q6_V_hi_W(w_sum_x1), Q6_V_hi_W(w_sum_x0), align_size);
    //  0  2 ...
    HVX_Vector v_c_sum  = Q6_V_lo_W(w_sum_x1);
    //  1  3 ...
    HVX_Vector v_r0_sum = Q6_V_hi_W(w_sum_x1);
    //  2  4 ...
    HVX_Vector v_r1_sum = Q6_V_valign_VVR(Q6_V_lo_W(w_sum_x2), Q6_V_lo_W(w_sum_x1), align_size);

    // 0 2 ... 126
    HVX_Vector v_result_lo;
    Pyrdown5x5HCore<Tp>(v_l1_sum, v_l0_sum, v_c_sum, v_r0_sum, v_r1_sum, v_result_lo, kernel);

    // 126 128 ...
    v_l1_sum = Q6_V_vlalign_VVR(Q6_V_lo_W(w_sum_x2), Q6_V_lo_W(w_sum_x1), align_size);
    // 127 129 ...
    v_l0_sum = Q6_V_vlalign_VVR(Q6_V_hi_W(w_sum_x2), Q6_V_hi_W(w_sum_x1), align_size);
    // 128 130 ...
    v_c_sum  = Q6_V_lo_W(w_sum_x2);
    // 129 131 ...
    v_r0_sum = Q6_V_hi_W(w_sum_x2);
    // 130 132 ...
    v_r1_sum = Q6_V_valign_VVR(Q6_V_lo_W(w_sum_x3), Q6_V_lo_W(w_sum_x2), align_size);

    HVX_Vector v_result_hi;
    Pyrdown5x5HCore<Tp>(v_l1_sum, v_l0_sum, v_c_sum, v_r0_sum, v_r1_sum, v_result_hi, kernel);

    w_sum_x0 = w_sum_x2;
    w_sum_x1 = w_sum_x3;
    v_result = Pyrdown5x5Pack<Tp>(v_result_hi, v_result_lo);
}

// High w_sum_x0                             ...  w-rest-2*ELEM_COUNTS-1
// Low w_sum_x0                              ...  w-rest-2*ELEM_COUNTS-2
// High w_sum_x1     w-rest-2*ELEM_COUNTS+1  ...  w-rest  -ELEM_COUNTS-1
// Low w_sum_x1      w-rest-2*ELEM_COUNTS    ...  w-rest  -ELEM_COUNTS-2
// High w_sum_x2     w-rest  -ELEM_COUNTS+1  ...  w-rest              -1
// Low w_sum_x2      w-rest  -ELEM_COUNTS    ...  w-rest              -2
// High w_sum_x3     w     -2*ELEM_COUNTS+1  ...  w       -ELEM_COUNTS-1
// Low w_sum_x3      w     -2*ELEM_COUNTS    ...  w       -ELEM_COUNTS-2
// High w_sum_x4     w       -ELEM_COUNTS+1  ...  w                   -1
// Low w_sum_x4      w       -ELEM_COUNTS    ...  w                   -2
// High w_sum_x5     w                  +1   ...
// Low w_sum_x5      w                       ...
template <typename Tp, typename Kt>
AURA_ALWAYS_INLINE DT_VOID Pyrdown5x5HCore(HVX_VectorPair &w_sum_x0, HVX_VectorPair &w_sum_x1,
                                           HVX_VectorPair &w_sum_x2, HVX_VectorPair &w_sum_x3,
                                           HVX_VectorPair &w_sum_x4, HVX_VectorPair &w_sum_x5,
                                           HVX_Vector &v_result, HVX_Vector &v_last,
                                           const Kt *kernel, DT_S32 rest)
{
    constexpr DT_S32 ELEM_COUNTS = AURA_HVLEN / sizeof(Tp);
    HVX_Vector v_sums_r0[2];
    HVX_Vector v_sums_l0[2];
    HVX_VectorPair w_sum_r0, w_sum_l0;

    if (rest < ELEM_COUNTS)
    {
        DT_S32 align_size = (rest) / 2 * sizeof(Kt);
        // align between w4 and w5
        v_sums_r0[0] = Q6_V_vlalign_safe_VVR(Q6_V_lo_W(w_sum_x5), Q6_V_lo_W(w_sum_x4), align_size);
        v_sums_r0[1] = Q6_V_vlalign_safe_VVR(Q6_V_hi_W(w_sum_x5), Q6_V_hi_W(w_sum_x4), align_size);

        // align between w0 and w1
        v_sums_l0[0] = Q6_V_valign_safe_VVR(Q6_V_lo_W(w_sum_x1), Q6_V_lo_W(w_sum_x0), align_size);
        v_sums_l0[1] = Q6_V_valign_safe_VVR(Q6_V_hi_W(w_sum_x1), Q6_V_hi_W(w_sum_x0), align_size);
    }
    else
    {
        DT_S32 align_size = (rest - ELEM_COUNTS) / 2 * sizeof(Kt);
        // align between w3 and w4
        v_sums_r0[0] = Q6_V_vlalign_safe_VVR(Q6_V_lo_W(w_sum_x4), Q6_V_lo_W(w_sum_x3), align_size);
        v_sums_r0[1] = Q6_V_vlalign_safe_VVR(Q6_V_hi_W(w_sum_x4), Q6_V_hi_W(w_sum_x3), align_size);

        // align between w1 and w2
        v_sums_l0[0] = Q6_V_valign_safe_VVR(Q6_V_lo_W(w_sum_x2), Q6_V_lo_W(w_sum_x1), align_size);
        v_sums_l0[1] = Q6_V_valign_safe_VVR(Q6_V_hi_W(w_sum_x2), Q6_V_hi_W(w_sum_x1), align_size);
    }

    w_sum_r0 = Q6_W_vcombine_VV(v_sums_r0[1], v_sums_r0[0]);
    w_sum_l0 = Q6_W_vcombine_VV(v_sums_l0[1], v_sums_l0[0]);

    Pyrdown5x5HCore<Tp, Kt>(w_sum_x0, w_sum_x1, w_sum_x2, w_sum_r0, v_result, kernel);
    Pyrdown5x5HCore<Tp, Kt>(w_sum_l0, w_sum_x3, w_sum_x4, w_sum_x5, v_last, kernel);
}

template <typename Tp, BorderType BORDER_TYPE, typename Kt>
static DT_VOID PyrDown5x5Row(const Tp *src_p1, const Tp *src_p0, const Tp *src_c, const Tp *src_n0,
                             const Tp *src_n1, DT_S32 iwidth, Tp *dst, DT_S32 owidth, const Kt *kernel)
{
    constexpr DT_S32 ELEM_COUNTS = AURA_HVLEN / sizeof(Tp);
    // Ensure remain can process up to 2 vectors
    DT_S32 dst_back_offset = owidth - 2 * ELEM_COUNTS;
    DT_S32 ox              = 0;
    DT_S32 ix              = 0;

    HVX_Vector v_src_p1x0, v_src_p1x1;
    HVX_Vector v_src_p0x0, v_src_p0x1;
    HVX_Vector v_src_cx0,  v_src_cx1;
    HVX_Vector v_src_n0x0, v_src_n0x1;
    HVX_Vector v_src_n1x0, v_src_n1x1;
    HVX_Vector v_result;
    HVX_VectorPair w_sum_x0, w_sum_x1, w_sum_x2, w_sum_x3;

    // left border
    {
        vload(src_p1 + ix, v_src_p1x0);
        vload(src_p0 + ix, v_src_p0x0);
        vload(src_c  + ix, v_src_cx0);
        vload(src_n0 + ix, v_src_n0x0);
        vload(src_n1 + ix, v_src_n1x0);

        HVX_Vector v_border_p1 = GetBorderVector<Tp, BORDER_TYPE, BorderArea::LEFT>(v_src_p1x0, src_p1[0], 0);
        HVX_Vector v_border_p0 = GetBorderVector<Tp, BORDER_TYPE, BorderArea::LEFT>(v_src_p0x0, src_p0[0], 0);
        HVX_Vector v_border_c  = GetBorderVector<Tp, BORDER_TYPE, BorderArea::LEFT>(v_src_cx0,  src_c[0],  0);
        HVX_Vector v_border_n0 = GetBorderVector<Tp, BORDER_TYPE, BorderArea::LEFT>(v_src_n0x0, src_n0[0], 0);
        HVX_Vector v_border_n1 = GetBorderVector<Tp, BORDER_TYPE, BorderArea::LEFT>(v_src_n1x0, src_n1[0], 0);

        Pyrdown5x5VCore<Tp>(v_border_p1, v_border_p0, v_border_c, v_border_n0, v_border_n1, w_sum_x0, kernel);
        Pyrdown5x5VCore<Tp>(v_src_p1x0, v_src_p0x0, v_src_cx0, v_src_n0x0, v_src_n1x0, w_sum_x1, kernel);
    }

    // main (0~n-3)
    for (ox = 0; ox < dst_back_offset; ox += ELEM_COUNTS)
    {
        ix = ox * 2 + ELEM_COUNTS;

        vload(src_p1 + ix, v_src_p1x0);
        vload(src_p0 + ix, v_src_p0x0);
        vload(src_c  + ix, v_src_cx0);
        vload(src_n0 + ix, v_src_n0x0);
        vload(src_n1 + ix, v_src_n1x0);

        vload(src_p1 + ix + ELEM_COUNTS, v_src_p1x1);
        vload(src_p0 + ix + ELEM_COUNTS, v_src_p0x1);
        vload(src_c  + ix + ELEM_COUNTS, v_src_cx1);
        vload(src_n0 + ix + ELEM_COUNTS, v_src_n0x1);
        vload(src_n1 + ix + ELEM_COUNTS, v_src_n1x1);

        Pyrdown5x5VCore<Tp>(v_src_p1x0, v_src_p0x0, v_src_cx0, v_src_n0x0, v_src_n1x0, w_sum_x2, kernel);
        Pyrdown5x5VCore<Tp>(v_src_p1x1, v_src_p0x1, v_src_cx1, v_src_n0x1, v_src_n1x1, w_sum_x3, kernel);

        Pyrdown5x5HCore<Tp, Kt>(w_sum_x0, w_sum_x1, w_sum_x2, w_sum_x3, v_result, kernel);

        vstore(dst + ox, v_result);
    }
    // remain
    {
        ix                     = ox * 2 + ELEM_COUNTS;
        DT_S32 ix_last         = iwidth - 1;
        DT_S32 ox_last         = owidth - ELEM_COUNTS;
        DT_S32 is_odd          = iwidth & 1;
        DT_S32 src_rest        = iwidth + is_odd - ix - ELEM_COUNTS;
        DT_S32 src_back_offset = iwidth - 2 * ELEM_COUNTS;

        HVX_VectorPair w_sum_x4, w_sum_x5;
        HVX_Vector v_last;

        vload(src_p1 + ix, v_src_p1x0);
        vload(src_p0 + ix, v_src_p0x0);
        vload(src_c  + ix, v_src_cx0);
        vload(src_n0 + ix, v_src_n0x0);
        vload(src_n1 + ix, v_src_n1x0);

        Pyrdown5x5VCore<Tp>(v_src_p1x0, v_src_p0x0, v_src_cx0, v_src_n0x0, v_src_n1x0, w_sum_x2, kernel);

        vload(src_p1 + src_back_offset + is_odd, v_src_p1x0);
        vload(src_p0 + src_back_offset + is_odd, v_src_p0x0);
        vload(src_c  + src_back_offset + is_odd, v_src_cx0);
        vload(src_n0 + src_back_offset + is_odd, v_src_n0x0);
        vload(src_n1 + src_back_offset + is_odd, v_src_n1x0);

        vload(src_p1 + src_back_offset + ELEM_COUNTS, v_src_p1x1);
        vload(src_p0 + src_back_offset + ELEM_COUNTS, v_src_p0x1);
        vload(src_c  + src_back_offset + ELEM_COUNTS, v_src_cx1);
        vload(src_n0 + src_back_offset + ELEM_COUNTS, v_src_n0x1);
        vload(src_n1 + src_back_offset + ELEM_COUNTS, v_src_n1x1);

        HVX_Vector v_border_p1 = GetBorderVector<Tp, BORDER_TYPE, BorderArea::RIGHT>(v_src_p1x1, src_p1[ix_last], 0);
        HVX_Vector v_border_p0 = GetBorderVector<Tp, BORDER_TYPE, BorderArea::RIGHT>(v_src_p0x1, src_p0[ix_last], 0);
        HVX_Vector v_border_c  = GetBorderVector<Tp, BORDER_TYPE, BorderArea::RIGHT>(v_src_cx1,  src_c[ix_last],  0);
        HVX_Vector v_border_n0 = GetBorderVector<Tp, BORDER_TYPE, BorderArea::RIGHT>(v_src_n0x1, src_n0[ix_last], 0);
        HVX_Vector v_border_n1 = GetBorderVector<Tp, BORDER_TYPE, BorderArea::RIGHT>(v_src_n1x1, src_n1[ix_last], 0);

        if (is_odd)
        {
            const DT_S32 align_size = sizeof(Tp);

            v_src_p1x1 = Q6_V_valign_VVR(v_border_p1, v_src_p1x1, align_size);
            v_src_p0x1 = Q6_V_valign_VVR(v_border_p0, v_src_p0x1, align_size);
            v_src_cx1  = Q6_V_valign_VVR(v_border_c,  v_src_cx1,  align_size);
            v_src_n0x1 = Q6_V_valign_VVR(v_border_n0, v_src_n0x1, align_size);
            v_src_n1x1 = Q6_V_valign_VVR(v_border_n1, v_src_n1x1, align_size);

            v_border_p1 = Q6_V_vror_VR(v_border_p1, align_size);
            v_border_p0 = Q6_V_vror_VR(v_border_p0, align_size);
            v_border_c  = Q6_V_vror_VR(v_border_c,  align_size);
            v_border_n0 = Q6_V_vror_VR(v_border_n0, align_size);
            v_border_n1 = Q6_V_vror_VR(v_border_n1, align_size);
        }

        Pyrdown5x5VCore<Tp>(v_src_p1x0, v_src_p0x0, v_src_cx0, v_src_n0x0, v_src_n1x0, w_sum_x3, kernel);
        Pyrdown5x5VCore<Tp>(v_src_p1x1, v_src_p0x1, v_src_cx1, v_src_n0x1, v_src_n1x1, w_sum_x4, kernel);
        Pyrdown5x5VCore<Tp>(v_border_p1, v_border_p0, v_border_c, v_border_n0, v_border_n1, w_sum_x5, kernel);

        Pyrdown5x5HCore<Tp, Kt>(w_sum_x0, w_sum_x1, w_sum_x2, w_sum_x3, w_sum_x4, w_sum_x5, v_result, v_last, kernel, src_rest);

        vstore(dst + ox,      v_result);
        vstore(dst + ox_last, v_last);
    }

    return;
}

template <typename Tp, BorderType BORDER_TYPE>
static Status PyrDown5x5HvxImpl(const Mat &src, Mat &dst, const Mat &kmat, DT_S32 start_row, DT_S32 end_row)
{
    using Kt = typename PyrDownTraits<Tp>::KernelType;

    DT_S32 iwidth  = src.GetSizes().m_width;
    DT_S32 iheight = src.GetSizes().m_height;
    DT_S32 istride = src.GetStrides().m_width;
    DT_S32 owidth  = dst.GetSizes().m_width;

    const Kt *kernel = kmat.Ptr<Kt>(0);

    DT_S32 sy = start_row << 1;

    const Tp *src_p1 = src.Ptr<Tp, BORDER_TYPE>(sy - 2, DT_NULL);
    const Tp *src_p0 = src.Ptr<Tp, BORDER_TYPE>(sy - 1, DT_NULL);
    const Tp *src_c  = src.Ptr<Tp>(sy);
    const Tp *src_n0 = src.Ptr<Tp, BORDER_TYPE>(sy + 1, DT_NULL);
    const Tp *src_n1 = src.Ptr<Tp, BORDER_TYPE>(sy + 2, DT_NULL);

    DT_U64 L2fetch_param = L2PfParam(istride, iwidth * ElemTypeSize(src.GetElemType()), 2, 0);
    for (DT_S32 dy = start_row; dy < end_row; dy++)
    {
        sy = dy << 1;
        if (sy + 4 < iheight)
        {
            L2Fetch(reinterpret_cast<DT_U32>(src.Ptr<Tp>(sy + 3)), L2fetch_param);
        }

        Tp *dst_row = dst.Ptr<Tp>(dy);

        PyrDown5x5Row<Tp, BORDER_TYPE, Kt>(src_p1, src_p0, src_c, src_n0, src_n1, iwidth, dst_row, owidth, kernel);

        src_p1 = src_c;
        src_p0 = src_n0;
        src_c  = src_n1;
        src_n0 = src.Ptr<Tp, BORDER_TYPE>(sy + 3, DT_NULL);
        src_n1 = src.Ptr<Tp, BORDER_TYPE>(sy + 4, DT_NULL);
    }

    return Status::OK;
}

template <typename Tp>
static Status PyrDown5x5HvxHelper(Context *ctx, const Mat &src, Mat &dst, const Mat &kmat,
                                  BorderType &border_type)
{
    Status ret = Status::ERROR;

    WorkerPool *wp = ctx->GetWorkerPool();
    if (DT_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerpool failed");
        return ret;
    }

    DT_S32 height = dst.GetSizes().m_height;

    switch (border_type)
    {
        case BorderType::REPLICATE:
        {
            ret = wp->ParallelFor((DT_S32)0, height, PyrDown5x5HvxImpl<Tp, BorderType::REPLICATE>,
                                  std::cref(src), std::ref(dst), std::cref(kmat));
            break;
        }

        case BorderType::REFLECT_101:
        {
            ret = wp->ParallelFor((DT_S32)0, height, PyrDown5x5HvxImpl<Tp, BorderType::REFLECT_101>,
                                  std::cref(src), std::ref(dst), std::cref(kmat));
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

Status PyrDown5x5Hvx(Context *ctx, const Mat &src, Mat &dst, const Mat &kmat,
                     BorderType border_type)
{
    Status ret = Status::ERROR;

    switch (src.GetElemType())
    {
        case ElemType::U8:
        {
            ret = PyrDown5x5HvxHelper<DT_U8>(ctx, src, dst, kmat, border_type);
            break;
        }

        case ElemType::U16:
        {
            ret = PyrDown5x5HvxHelper<DT_U16>(ctx, src, dst, kmat, border_type);
            break;
        }

        case ElemType::S16:
        {
            ret = PyrDown5x5HvxHelper<DT_S16>(ctx, src, dst, kmat, border_type);
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