#include "pyrup_impl.hpp"
#include "aura/runtime/logger.h"
#include "aura/runtime/worker_pool.h"

namespace aura
{

// using Tp = MI_U8
template <typename Tp, typename std::enable_if<std::is_same<Tp, MI_U8>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID Pyrup5x5VCore(const HVX_Vector &vu8_src_p, const HVX_Vector &vu8_src_c,
                                         const HVX_Vector &vu8_src_n, HVX_VectorPair &wu32_sum_c_lo,
                                         HVX_VectorPair &wu32_sum_c_hi, HVX_VectorPair &wu32_sum_n_lo,
                                         HVX_VectorPair &wu32_sum_n_hi, const MI_U16 *kernel)
{
    MI_U32 k0k0 = (kernel[0] << 16) | kernel[0];
    MI_U32 k1k1 = (kernel[1] << 16) | kernel[1];
    MI_U32 k2k2 = (kernel[2] << 16) | kernel[2];

    HVX_VectorPair wu16_sum_c  = Q6_Wh_vadd_VubVub(vu8_src_p, vu8_src_n);
    wu32_sum_c_lo              = Q6_Ww_vmpy_VhRh(Q6_V_lo_W(wu16_sum_c), k0k0);
    wu32_sum_c_hi              = Q6_Ww_vmpy_VhRh(Q6_V_hi_W(wu16_sum_c), k0k0);

    HVX_VectorPair wu16_src_c  = Q6_Wuh_vzxt_Vub(vu8_src_c);

    wu32_sum_c_lo              = Q6_Ww_vmpyacc_WwVhRh(wu32_sum_c_lo, Q6_V_lo_W(wu16_src_c), k2k2);
    wu32_sum_c_hi              = Q6_Ww_vmpyacc_WwVhRh(wu32_sum_c_hi, Q6_V_hi_W(wu16_src_c), k2k2);

    HVX_VectorPair wu16_sum_n  = Q6_Wh_vadd_VubVub(vu8_src_c, vu8_src_n);
    wu32_sum_n_lo              = Q6_Ww_vmpy_VhRh(Q6_V_lo_W(wu16_sum_n), k1k1);
    wu32_sum_n_hi              = Q6_Ww_vmpy_VhRh(Q6_V_hi_W(wu16_sum_n), k1k1);
}

template <typename Tp, typename std::enable_if<std::is_same<Tp, MI_U8>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID Pyrup5x5HCore(const HVX_Vector &vu32_sum_l, const HVX_Vector &vu32_sum_c,
                                         const HVX_Vector &vu32_sum_r, HVX_Vector &vu16_result, const MI_U32 *kk)
{
    HVX_Vector vu32_sum0 = Q6_Vw_vmpyi_VwRh(Q6_Vuw_vadd_VuwVuw_sat(vu32_sum_l, vu32_sum_r), kk[0]);
    vu32_sum0            = Q6_Vw_vmpyiacc_VwVwRh(vu32_sum0, vu32_sum_c, kk[2]);

    HVX_Vector vu32_sum1 = Q6_Vw_vmpyi_VwRh(Q6_Vuw_vadd_VuwVuw_sat(vu32_sum_c, vu32_sum_r), kk[1]);

    vu16_result          = Q6_Vh_vshuffo_VhVh(vu32_sum1, vu32_sum0);
    vu16_result          = Q6_Vh_vadd_VhVh_sat(vu16_result, Q6_Vh_vsplat_R(2));
    vu16_result          = Q6_Vh_vasr_VhR(vu16_result, 2);
}

// high wu32_sum_x0_lo       X   X ...  -2
// low  wu32_sum_x0_lo       X   X ...  -4
// high wu32_sum_x0_hi       X   X ...  -1
// low  wu32_sum_x0_hi       X   X ...  -3

// high wu32_sum_x1_lo       2   6 ... 126
// low  wu32_sum_x1_lo       0   4 ... 124
// high wu32_sum_x1_hi       3   7 ... 127
// low  wu32_sum_x1_hi       1   5 ... 125

// high wu32_sum_x2_lo     130 134 ... 254
// low  wu32_sum_x2_lo     128 132 ... 252
// high wu32_sum_x2_hi     131 135 ... 255
// low  wu32_sum_x2_hi     129 133 ... 253
template <typename Tp, typename std::enable_if<std::is_same<Tp, MI_U8>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID Pyrup5x5HCore(HVX_VectorPair &wu32_sum_x0_lo, HVX_VectorPair &wu32_sum_x0_hi,
                                         HVX_VectorPair &wu32_sum_x1_lo, HVX_VectorPair &wu32_sum_x1_hi,
                                         HVX_VectorPair &wu32_sum_x2_lo, HVX_VectorPair &wu32_sum_x2_hi,
                                         HVX_Vector &vu8_result_x0, HVX_Vector &vu8_result_x1,const MI_U16 *kernel)
{
    const MI_S32 align_size = sizeof(MI_U32);
    MI_U32 kk[3];
    kk[0] = (kernel[0] << 16) | kernel[0];
    kk[1] = (kernel[1] << 16) | kernel[1];
    kk[2] = (kernel[2] << 16) | kernel[2];
    HVX_Vector vu32_sum_minus_1, vu32_sum0, vu32_sum1;
    HVX_Vector vu32_sum2, vu32_sum3, vu32_sum4;
    HVX_Vector vu16_result0, vu16_result1, vu16_result2, vu16_result3;
    HVX_VectorPair wu16_result0, wu16_result1;

    // -1 3 ... 123
    vu32_sum_minus_1 = Q6_V_vlalign_VVR(Q6_V_hi_W(wu32_sum_x1_hi), Q6_V_hi_W(wu32_sum_x0_hi), align_size);
    // 0 4 ... 124
    vu32_sum0 = Q6_V_lo_W(wu32_sum_x1_lo);
    // 1 5 ... 125
    vu32_sum1 = Q6_V_lo_W(wu32_sum_x1_hi);
    // 2 6 ... 126
    vu32_sum2 = Q6_V_hi_W(wu32_sum_x1_lo);
    // 3 7 ... 127
    vu32_sum3 = Q6_V_hi_W(wu32_sum_x1_hi);
    // 4 8 ... 128
    vu32_sum4 = Q6_V_valign_VVR(Q6_V_lo_W(wu32_sum_x2_lo), Q6_V_lo_W(wu32_sum_x1_lo), align_size);

    Pyrup5x5HCore<Tp>(vu32_sum_minus_1, vu32_sum0, vu32_sum1, vu16_result0, kk);
    Pyrup5x5HCore<Tp>(vu32_sum1, vu32_sum2, vu32_sum3, vu16_result2, kk);

    wu16_result0 = Q6_W_vshuff_VVR(vu16_result2, vu16_result0, -align_size);
    vu16_result0 = Q6_Vub_vpack_VhVh_sat(Q6_V_hi_W(wu16_result0), Q6_V_lo_W(wu16_result0));

    Pyrup5x5HCore<Tp>(vu32_sum0, vu32_sum1, vu32_sum2, vu16_result1, kk);
    Pyrup5x5HCore<Tp>(vu32_sum2, vu32_sum3, vu32_sum4, vu16_result3, kk);

    wu16_result1 = Q6_W_vshuff_VVR(vu16_result3, vu16_result1, -align_size);
    vu16_result1 = Q6_Vub_vpack_VhVh_sat(Q6_V_hi_W(wu16_result1), Q6_V_lo_W(wu16_result1));
    // Shuff uh
    HVX_VectorPair wu8_result = Q6_W_vshuff_VVR(vu16_result1, vu16_result0, -2 * sizeof(MI_U8));

    vu8_result_x0 = Q6_V_lo_W(wu8_result);
    vu8_result_x1 = Q6_V_hi_W(wu8_result);

    wu32_sum_x0_lo = wu32_sum_x1_lo;
    wu32_sum_x0_hi = wu32_sum_x1_hi;
    wu32_sum_x1_lo = wu32_sum_x2_lo;
    wu32_sum_x1_hi = wu32_sum_x2_hi;

    return;
}

// high wu32_sum_x0_lo     w-254-rest  ...            w-130-rest
// low  wu32_sum_x0_lo     w-256-rest  ...            w-132-rest
// high wu32_sum_x0_hi     w-253-rest  ...            w-129-rest
// low  wu32_sum_x0_hi     w-255-rest  ...            w-131-rest

// high wu32_sum_x1_lo     w-126-rest  w-122-rest ...   w-2-rest
// low  wu32_sum_x1_lo     w-128-rest  w-124-rest ...   w-4-rest
// high wu32_sum_x1_hi     w-125-rest  w-121-rest ...   w-1-rest
// low  wu32_sum_x1_hi     w-127-rest  w-123-rest ...   w-3-rest

// high wu32_sum_x2_lo          w-126       w-122 ...        w-2
// low  wu32_sum_x2_lo          w-128       w-124 ...        w-4
// high wu32_sum_x2_hi          w-125       w-121 ...        w-1
// low  wu32_sum_x2_hi          w-127       w-123 ...        w-3

// high wu32_sum_x3_lo            w+2           X ...          X
// low  wu32_sum_x3_lo            w+0           X ...          X
// high wu32_sum_x3_hi            w+3           X ...          X
// low  wu32_sum_x3_hi            w+1           X ...          X
template <typename Tp, typename std::enable_if<std::is_same<Tp, MI_U8>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID Pyrup5x5HCore(HVX_VectorPair &wu32_sum_x0_lo, HVX_VectorPair &wu32_sum_x0_hi,
                                         HVX_VectorPair &wu32_sum_x1_lo, HVX_VectorPair &wu32_sum_x1_hi,
                                         HVX_VectorPair &wu32_sum_x2_lo, HVX_VectorPair &wu32_sum_x2_hi,
                                         HVX_VectorPair &wu32_sum_x3_lo, HVX_VectorPair &wu32_sum_x3_hi,
                                         HVX_Vector &vu8_result_x0, HVX_Vector &vu8_result_x1,
                                         HVX_Vector &vu8_last_x0, HVX_Vector &vu8_last_x1,
                                         const MI_U16 *kernel, MI_S32 rest)
{
    HVX_Vector vu32_sums_l[4];
    HVX_Vector vu32_sums_r[4];
    HVX_Vector vu32_sum_l0, vu32_sum_r0;
    HVX_VectorPair wu32_sum_r0, wu32_sum_l0;
    MI_S32 idx;

    vu32_sums_l[0] = Q6_V_lo_W(wu32_sum_x2_lo);
    vu32_sums_l[1] = Q6_V_lo_W(wu32_sum_x2_hi);
    vu32_sums_l[2] = Q6_V_hi_W(wu32_sum_x2_lo);
    vu32_sums_l[3] = Q6_V_hi_W(wu32_sum_x2_hi);

    vu32_sums_r[0] = Q6_V_lo_W(wu32_sum_x3_lo);
    vu32_sums_r[1] = Q6_V_lo_W(wu32_sum_x3_hi);
    vu32_sums_r[2] = Q6_V_hi_W(wu32_sum_x3_lo);
    vu32_sums_r[3] = Q6_V_hi_W(wu32_sum_x3_hi);

    idx         = (AURA_HVLEN - rest) & (4 - 1);
    vu32_sum_r0 = Q6_V_valign_safe_VVR(vu32_sums_r[idx], vu32_sums_l[idx], AURA_HVLEN - idx - rest);

    vu32_sums_l[0] = Q6_V_lo_W(wu32_sum_x0_lo);
    vu32_sums_l[1] = Q6_V_lo_W(wu32_sum_x0_hi);
    vu32_sums_l[2] = Q6_V_hi_W(wu32_sum_x0_lo);
    vu32_sums_l[3] = Q6_V_hi_W(wu32_sum_x0_hi);

    vu32_sums_r[0] = Q6_V_lo_W(wu32_sum_x1_lo);
    vu32_sums_r[1] = Q6_V_lo_W(wu32_sum_x1_hi);
    vu32_sums_r[2] = Q6_V_hi_W(wu32_sum_x1_lo);
    vu32_sums_r[3] = Q6_V_hi_W(wu32_sum_x1_hi);

    idx         = (rest + 3) & (4 - 1);
    vu32_sum_l0 = Q6_V_valign_safe_VVR(vu32_sums_r[idx], vu32_sums_l[idx], rest + 3 - idx);

    wu32_sum_r0 = Q6_W_vcombine_VV(vu32_sum_r0, vu32_sum_r0);
    wu32_sum_l0 = Q6_W_vcombine_VV(vu32_sum_l0, vu32_sum_l0);

    Pyrup5x5HCore<Tp>(wu32_sum_x0_lo, wu32_sum_x0_hi, wu32_sum_x1_lo, wu32_sum_x1_hi, wu32_sum_r0, wu32_sum_r0, vu8_result_x0, vu8_result_x1, kernel);
    Pyrup5x5HCore<Tp>(wu32_sum_l0, wu32_sum_l0, wu32_sum_x2_lo, wu32_sum_x2_hi, wu32_sum_x3_lo, wu32_sum_x3_hi, vu8_last_x0, vu8_last_x1, kernel);

    return;
}

template <typename Tp, BorderType BORDER_TYPE, typename Kt,
          typename std::enable_if<std::is_same<Tp, MI_U8>::value>::type* = MI_NULL>
static AURA_VOID PyrUp5x5TwoRow(const Tp *src_p, const Tp *src_c, const Tp *src_n, MI_S32 iwidth,
                              Tp *dst_c0, Tp *dst_c1, const Kt *kernel)
{
    constexpr MI_S32 ELEM_COUNTS = AURA_HVLEN / sizeof(Tp);

    MI_S32 src_back_offset = iwidth - ELEM_COUNTS;
    MI_S32 ox = 0;
    MI_S32 ix = 0;

    HVX_Vector     vu8_src_p;
    HVX_Vector     vu8_src_c;
    HVX_Vector     vu8_src_n;
    HVX_Vector     vu8_result_c0x0, vu8_result_c0x1;
    HVX_Vector     vu8_result_c1x0, vu8_result_c1x1;
    HVX_VectorPair wu32_sum_c0x0_lo, wu32_sum_c0x0_hi, wu32_sum_c0x1_lo, wu32_sum_c0x1_hi, wu32_sum_c0x2_lo, wu32_sum_c0x2_hi;
    HVX_VectorPair wu32_sum_c1x0_lo, wu32_sum_c1x0_hi, wu32_sum_c1x1_lo, wu32_sum_c1x1_hi, wu32_sum_c1x2_lo, wu32_sum_c1x2_hi;

    // left border
    {
        ix = 0;
        vload(src_p + ix, vu8_src_p);
        vload(src_c + ix, vu8_src_c);
        vload(src_n + ix, vu8_src_n);

        HVX_Vector vu8_border_p = GetBorderVector<Tp, BORDER_TYPE, BorderArea::LEFT>(vu8_src_p, src_p[0], 0);
        HVX_Vector vu8_border_c = GetBorderVector<Tp, BORDER_TYPE, BorderArea::LEFT>(vu8_src_c, src_c[0], 0);
        HVX_Vector vu8_border_n = GetBorderVector<Tp, BORDER_TYPE, BorderArea::LEFT>(vu8_src_n, src_n[0], 0);

        Pyrup5x5VCore<Tp>(vu8_border_p, vu8_border_c, vu8_border_n, wu32_sum_c0x0_lo,
                          wu32_sum_c0x0_hi, wu32_sum_c1x0_lo, wu32_sum_c1x0_hi, kernel);
        Pyrup5x5VCore<Tp>(vu8_src_p, vu8_src_c, vu8_src_n, wu32_sum_c0x1_lo,
                          wu32_sum_c0x1_hi, wu32_sum_c1x1_lo, wu32_sum_c1x1_hi, kernel);
    }

    // main(0~n-2)
    for (ix = ELEM_COUNTS; ix <= src_back_offset; ix += ELEM_COUNTS)
    {
        ox = (ix - ELEM_COUNTS) * 2;
        vload(src_p + ix, vu8_src_p);
        vload(src_c + ix, vu8_src_c);
        vload(src_n + ix, vu8_src_n);

        Pyrup5x5VCore<Tp>(vu8_src_p, vu8_src_c, vu8_src_n, wu32_sum_c0x2_lo,
                          wu32_sum_c0x2_hi, wu32_sum_c1x2_lo, wu32_sum_c1x2_hi, kernel);

        Pyrup5x5HCore<Tp>(wu32_sum_c0x0_lo, wu32_sum_c0x0_hi, wu32_sum_c0x1_lo, wu32_sum_c0x1_hi,
                          wu32_sum_c0x2_lo, wu32_sum_c0x2_hi, vu8_result_c0x0,  vu8_result_c0x1, kernel);
        Pyrup5x5HCore<Tp>(wu32_sum_c1x0_lo, wu32_sum_c1x0_hi, wu32_sum_c1x1_lo, wu32_sum_c1x1_hi,
                          wu32_sum_c1x2_lo, wu32_sum_c1x2_hi, vu8_result_c1x0,  vu8_result_c1x1, kernel);

        vstore(dst_c0 + ox,               vu8_result_c0x0);
        vstore(dst_c0 + ox + ELEM_COUNTS, vu8_result_c0x1);
        vstore(dst_c1 + ox,               vu8_result_c1x0);
        vstore(dst_c1 + ox + ELEM_COUNTS, vu8_result_c1x1);
    }

    // remain
    {
        ox              = (ix - ELEM_COUNTS) * 2;
        MI_S32 ix_last  = iwidth - 1;
        MI_S32 ox_last  = src_back_offset * 2;
        MI_S32 src_rest = iwidth & (ELEM_COUNTS - 1);

        HVX_Vector     vu8_last_c0x0, vu8_last_c0x1;
        HVX_Vector     vu8_last_c1x0, vu8_last_c1x1;
        HVX_VectorPair wu32_esum_c0x3, wu32_osum_c0x3;
        HVX_VectorPair wu32_esum_c1x3, wu32_osum_c1x3;

        vload(src_p + src_back_offset, vu8_src_p);
        vload(src_c + src_back_offset, vu8_src_c);
        vload(src_n + src_back_offset, vu8_src_n);

        HVX_Vector vu8_border_p = GetBorderVector<Tp, BorderType::REPLICATE, BorderArea::RIGHT>(vu8_src_p, src_p[ix_last], 0);
        HVX_Vector vu8_border_c = GetBorderVector<Tp, BorderType::REPLICATE, BorderArea::RIGHT>(vu8_src_c, src_c[ix_last], 0);
        HVX_Vector vu8_border_n = GetBorderVector<Tp, BorderType::REPLICATE, BorderArea::RIGHT>(vu8_src_n, src_n[ix_last], 0);

        Pyrup5x5VCore<Tp>(vu8_src_p, vu8_src_c, vu8_src_n, wu32_sum_c0x2_lo,
                          wu32_sum_c0x2_hi, wu32_sum_c1x2_lo, wu32_sum_c1x2_hi, kernel);
        Pyrup5x5VCore<Tp>(vu8_border_p, vu8_border_c, vu8_border_n, wu32_esum_c0x3,
                          wu32_osum_c0x3, wu32_esum_c1x3, wu32_osum_c1x3, kernel);

        Pyrup5x5HCore<Tp>(wu32_sum_c0x0_lo, wu32_sum_c0x0_hi, wu32_sum_c0x1_lo, wu32_sum_c0x1_hi,
                          wu32_sum_c0x2_lo, wu32_sum_c0x2_hi, wu32_esum_c0x3, wu32_osum_c0x3,
                          vu8_result_c0x0, vu8_result_c0x1, vu8_last_c0x0, vu8_last_c0x1,
                          kernel, src_rest);
        Pyrup5x5HCore<Tp>(wu32_sum_c1x0_lo, wu32_sum_c1x0_hi, wu32_sum_c1x1_lo, wu32_sum_c1x1_hi,
                          wu32_sum_c1x2_lo, wu32_sum_c1x2_hi, wu32_esum_c1x3, wu32_osum_c1x3,
                          vu8_result_c1x0, vu8_result_c1x1, vu8_last_c1x0, vu8_last_c1x1,
                          kernel, src_rest);

        vstore(dst_c0 + ox,               vu8_result_c0x0);
        vstore(dst_c0 + ox + ELEM_COUNTS, vu8_result_c0x1);
        vstore(dst_c1 + ox,               vu8_result_c1x0);
        vstore(dst_c1 + ox + ELEM_COUNTS, vu8_result_c1x1);

        vstore(dst_c0 + ox_last,               vu8_last_c0x0);
        vstore(dst_c0 + ox_last + ELEM_COUNTS, vu8_last_c0x1);
        vstore(dst_c1 + ox_last,               vu8_last_c1x0);
        vstore(dst_c1 + ox_last + ELEM_COUNTS, vu8_last_c1x1);
    }
}

// using Tp = MI_U16
template <typename Tp, typename std::enable_if<std::is_same<Tp, MI_U16>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID Pyrup5x5VCore(const HVX_Vector &vu16_src_p, const HVX_Vector &vu16_src_c, const HVX_Vector &vu16_src_n,
                                        HVX_VectorPair &wu32_sum_c, HVX_VectorPair &wu32_sum_n, const MI_U32 *kernel)
{
    MI_U32 k0k0 = (kernel[0] << 16) | kernel[0];
    MI_U32 k1k1 = (kernel[1] << 16) | kernel[1];
    MI_U32 k2k2 = (kernel[2] << 16) | kernel[2];

    wu32_sum_c = Q6_Wuw_vmpy_VuhRuh(vu16_src_p, k0k0);
    wu32_sum_c = Q6_Wuw_vmpyacc_WuwVuhRuh(wu32_sum_c, vu16_src_n, k0k0);
    wu32_sum_c = Q6_Wuw_vmpyacc_WuwVuhRuh(wu32_sum_c, vu16_src_c, k2k2);

    wu32_sum_n = Q6_Wuw_vmpy_VuhRuh(vu16_src_c, k1k1);
    wu32_sum_n = Q6_Wuw_vmpyacc_WuwVuhRuh(wu32_sum_n, vu16_src_n, k1k1);
}

template <typename Tp, typename std::enable_if<std::is_same<Tp, MI_U16>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID Pyrup5x5HCore(const HVX_Vector &vu32_sum_l, const HVX_Vector &vu32_sum_c,
                                         const HVX_Vector &vu32_sum_r, HVX_Vector &vu16_result, const MI_U32 *kernel)
{
    HVX_Vector vu32_k0 = Q6_V_vsplat_R(kernel[0]);
    HVX_Vector vu32_k1 = Q6_V_vsplat_R(kernel[1]);
    HVX_Vector vu32_k2 = Q6_V_vsplat_R(kernel[2]);

    HVX_VectorPair vu64_sum0, vu64_sum1;
    HVX_Vector     vu32_sum0, vu32_sum1;

    vu64_sum0 = Q6_Wd_vmul_VwVw(Q6_Vw_vadd_VwVw(vu32_sum_l, vu32_sum_r), vu32_k0);
    vu64_sum0 = Q6_Wd_vmulacc_WdVwVw(vu64_sum0, vu32_sum_c, vu32_k2);
    vu64_sum0 = Q6_Wud_vadd_WudWud(vu64_sum0, Q6_W_vcombine_VV(Q6_V_vzero(), Q6_V_vsplat_R(1 << 25)));

    // rshift 26
    vu32_sum0 = Q6_V_vor_VV(Q6_Vw_vasl_VwR(Q6_V_hi_W(vu64_sum0), 6), Q6_Vuw_vlsr_VuwR(Q6_V_lo_W(vu64_sum0), 26));

    vu64_sum1 = Q6_Wd_vmul_VwVw(Q6_Vw_vadd_VwVw(vu32_sum_c, vu32_sum_r), vu32_k1);
    vu64_sum1 = Q6_Wud_vadd_WudWud(vu64_sum1, Q6_W_vcombine_VV(Q6_V_vzero(), Q6_V_vsplat_R(1 << 25)));
    // rshift 26
    vu32_sum1 = Q6_V_vor_VV(Q6_Vw_vasl_VwR(Q6_V_hi_W(vu64_sum1), 6), Q6_Vuw_vlsr_VuwR(Q6_V_lo_W(vu64_sum1), 26));

    vu16_result = Q6_Vh_vshuffe_VhVh(vu32_sum1, vu32_sum0);
}

// using Tp = MI_S16
template <typename Tp, typename std::enable_if<std::is_same<Tp, MI_S16>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID Pyrup5x5VCore(const HVX_Vector &vs16_src_p, const HVX_Vector &vs16_src_c, const HVX_Vector &vs16_src_n,
                                         HVX_VectorPair &ws32_sum_c, HVX_VectorPair &ws32_sum_n, const MI_S32 *kernel)
{
    MI_S32 k0k0 = (kernel[0] << 16) | kernel[0];
    MI_S32 k1k1 = (kernel[1] << 16) | kernel[1];
    MI_S32 k2k2 = (kernel[2] << 16) | kernel[2];

    ws32_sum_c = Q6_Ww_vmpy_VhRh(vs16_src_p, k0k0);
    ws32_sum_c = Q6_Ww_vmpyacc_WwVhRh(ws32_sum_c, vs16_src_n, k0k0);
    ws32_sum_c = Q6_Ww_vmpyacc_WwVhRh(ws32_sum_c, vs16_src_c, k2k2);

    ws32_sum_n = Q6_Ww_vmpy_VhRh(vs16_src_c, k1k1);
    ws32_sum_n = Q6_Ww_vmpyacc_WwVhRh(ws32_sum_n, vs16_src_n, k1k1);
}

template <typename Tp, typename std::enable_if<std::is_same<Tp, MI_S16>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID Pyrup5x5HCore(const HVX_Vector &vs32_sum_l, const HVX_Vector &vs32_sum_c,
                                         const HVX_Vector &vs32_sum_r, HVX_Vector &vs16_result, const MI_S32 *kernel)
{
    HVX_Vector vs32_k0 = Q6_V_vsplat_R(kernel[0]);
    HVX_Vector vs32_k1 = Q6_V_vsplat_R(kernel[1]);
    HVX_Vector vs32_k2 = Q6_V_vsplat_R(kernel[2]);

    HVX_VectorPair vs64_sum0, vs64_sum1;
    HVX_Vector vs32_sum0, vs32_sum1;
    vs64_sum0 = Q6_Wd_vmul_VwVw(Q6_Vw_vadd_VwVw(vs32_sum_l, vs32_sum_r), vs32_k0);
    vs64_sum0 = Q6_Wd_vmulacc_WdVwVw(vs64_sum0, vs32_sum_c, vs32_k2);
    vs64_sum0 = Q6_Wd_vadd_WdWd(vs64_sum0, Q6_W_vcombine_VV(Q6_V_vzero(), Q6_V_vsplat_R(1 << 25)));

    // rshift 26
    vs32_sum0 = Q6_V_vor_VV(Q6_Vw_vasl_VwR(Q6_V_hi_W(vs64_sum0), 6), Q6_Vuw_vlsr_VuwR(Q6_V_lo_W(vs64_sum0), 26));

    vs64_sum1 = Q6_Wd_vmul_VwVw(Q6_Vw_vadd_VwVw(vs32_sum_c, vs32_sum_r), vs32_k1);
    vs64_sum1 = Q6_Wd_vadd_WdWd(vs64_sum1, Q6_W_vcombine_VV(Q6_V_vzero(), Q6_V_vsplat_R(1 << 25)));
    // rshift 26
    vs32_sum1 = Q6_V_vor_VV(Q6_Vw_vasl_VwR(Q6_V_hi_W(vs64_sum1), 6), Q6_Vuw_vlsr_VuwR(Q6_V_lo_W(vs64_sum1), 26));

    vs16_result = Q6_Vh_vshuffe_VhVh(vs32_sum1, vs32_sum0);
}


// high wd32_sum_x0  -63 -61 ...  -1
// low  wd32_sum_x0  -64 -62 ...  -2
// high wd32_sum_x1    1   3 ...  63
// low  wd32_sum_x1    0   2 ...  62
// high wd32_sum_x2    65 67 ... 127
// low  wd32_sum_x2    64 66 ... 126
template <typename Tp, typename Kt,
          typename std::enable_if<std::is_same<Tp, MI_S16>::value || std::is_same<Tp, MI_U16>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID Pyrup5x5HCore(HVX_VectorPair &wd32_sum_x0, HVX_VectorPair &wd32_sum_x1, HVX_VectorPair &wd32_sum_x2,
                                         HVX_Vector &vd16_result_x0, HVX_Vector &vd16_result_x1, const Kt *kernel)
{
    const MI_S32 align_size  = sizeof(Kt);
    HVX_Vector vd32_sum_minus_1 = Q6_V_vlalign_VVR(Q6_V_hi_W(wd32_sum_x1), Q6_V_hi_W(wd32_sum_x0), align_size);
    HVX_Vector vd32_sum0        = Q6_V_lo_W(wd32_sum_x1);
    HVX_Vector vd32_sum1        = Q6_V_hi_W(wd32_sum_x1);
    HVX_Vector vd32_sum2        = Q6_V_valign_VVR(Q6_V_lo_W(wd32_sum_x2), Q6_V_lo_W(wd32_sum_x1), align_size);

    HVX_Vector vd16_result0, vd16_result1;

    Pyrup5x5HCore<Tp>(vd32_sum_minus_1, vd32_sum0, vd32_sum1, vd16_result0, kernel);
    Pyrup5x5HCore<Tp>(vd32_sum0, vd32_sum1, vd32_sum2, vd16_result1, kernel);

    // Shuff uh
    HVX_VectorPair wd16_result = Q6_W_vshuff_VVR(vd16_result1, vd16_result0, -(sizeof(Tp) << 1));

    vd16_result_x0 = Q6_V_lo_W(wd16_result);
    vd16_result_x1 = Q6_V_hi_W(wd16_result);

    wd32_sum_x0 = wd32_sum_x1;
    wd32_sum_x1 = wd32_sum_x2;
}

// high wd32_sum_x0  w-127-rest ... w-65-rest
// low  wd32_sum_x0  w-128-rest ... w-66-rest
// high wd32_sum_x1   w-63-rest ...  w-1-rest
// low  wd32_sum_x1   w-64-rest ...  w-2-rest
// high wd32_sum_x2        w-63 ...       w-1
// low  wd32_sum_x2        w-64 ...       w-2
// high vd32_sum3           w+1 ...      w+63
// low  vd32_sum3           w+0 ...      w+62
template <typename Tp, typename Kt,
          typename std::enable_if<std::is_same<Tp, MI_S16>::value || std::is_same<Tp, MI_U16>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID Pyrup5x5HCore(HVX_VectorPair &wd32_sum_x0, HVX_VectorPair &wd32_sum_x1,
                                         HVX_VectorPair &wd32_sum_x2, HVX_VectorPair &vd32_sum3,
                                         HVX_Vector &vd16_result_x0, HVX_Vector &vd16_result_x1, HVX_Vector &vd16_last_x0,
                                         HVX_Vector &vd16_last_x1, const Kt *kernel, MI_S32 rest)
{
    HVX_Vector vd32_sum_r0;
    HVX_Vector vd32_sum_l0;
    HVX_VectorPair wd32_sum_r0, wd32_sum_l0;

    if (rest & 1)
    {
        // odd
        // i=0 idx=1
        vd32_sum_r0 = Q6_V_valign_safe_VVR(Q6_V_hi_W(vd32_sum3), Q6_V_hi_W(wd32_sum_x2), (AURA_HVLEN / 2 - 1 - rest) * 2);
        // i=1 idx=0
        vd32_sum_l0 = Q6_V_valign_safe_VVR(Q6_V_lo_W(wd32_sum_x1), Q6_V_lo_W(wd32_sum_x0), (rest + 1) * 2);
    }
    else
    {
        // even
        // i=0 idx=0
        vd32_sum_r0 = Q6_V_valign_safe_VVR(Q6_V_lo_W(vd32_sum3), Q6_V_lo_W(wd32_sum_x2), (AURA_HVLEN / 2 - rest) * 2);
        // i=1 idx=1
        vd32_sum_l0 = Q6_V_valign_safe_VVR(Q6_V_hi_W(wd32_sum_x1), Q6_V_hi_W(wd32_sum_x0), rest * 2);
    }

    wd32_sum_r0 = Q6_W_vcombine_VV(vd32_sum_r0, vd32_sum_r0);
    wd32_sum_l0 = Q6_W_vcombine_VV(vd32_sum_l0, vd32_sum_l0);

    Pyrup5x5HCore<Tp>(wd32_sum_x0, wd32_sum_x1, wd32_sum_r0, vd16_result_x0, vd16_result_x1, kernel);
    Pyrup5x5HCore<Tp>(wd32_sum_l0, wd32_sum_x2, vd32_sum3, vd16_last_x0, vd16_last_x1, kernel);

    return;
}

template <typename Tp, BorderType BORDER_TYPE, typename Kt,
          typename std::enable_if<std::is_same<Tp, MI_U16>::value || std::is_same<Tp, MI_S16>::value>::type* = MI_NULL>
static AURA_VOID PyrUp5x5TwoRow(const Tp *src_p, const Tp *src_c, const Tp *src_n, MI_S32 iwidth,
                              Tp *dst_c0, Tp *dst_c1, const Kt *kernel)
{
    constexpr MI_S32 ELEM_COUNTS = AURA_HVLEN / sizeof(Tp);

    MI_S32 src_back_offset = iwidth - ELEM_COUNTS;
    MI_S32 ox = 0;
    MI_S32 ix = 0;

    HVX_Vector     vd16_src_p;
    HVX_Vector     vd16_src_c;
    HVX_Vector     vd16_src_n;
    HVX_Vector     vd16_result_c0x0, vd16_result_c0x1;
    HVX_Vector     vd16_result_c1x0, vd16_result_c1x1;
    HVX_VectorPair wd32_sum_c0x0, wd32_sum_c0x1, wd32_sum_c0x2;
    HVX_VectorPair wd32_sum_c1x0, wd32_sum_c1x1, wd32_sum_c1x2;

    // left border
    {
        ix = 0;
        vload(src_p + ix, vd16_src_p);
        vload(src_c + ix, vd16_src_c);
        vload(src_n + ix, vd16_src_n);

        HVX_Vector vd16_border_p = GetBorderVector<Tp, BORDER_TYPE, BorderArea::LEFT>(vd16_src_p, src_p[0], 0);
        HVX_Vector vd16_border_c = GetBorderVector<Tp, BORDER_TYPE, BorderArea::LEFT>(vd16_src_c, src_c[0], 0);
        HVX_Vector vd16_border_n = GetBorderVector<Tp, BORDER_TYPE, BorderArea::LEFT>(vd16_src_n, src_n[0], 0);

        Pyrup5x5VCore<Tp>(vd16_border_p, vd16_border_c, vd16_border_n, wd32_sum_c0x0, wd32_sum_c1x0, kernel);
        Pyrup5x5VCore<Tp>(vd16_src_p, vd16_src_c, vd16_src_n, wd32_sum_c0x1, wd32_sum_c1x1, kernel);
    }
    // main(0~n-2)
    for (ix = ELEM_COUNTS; ix <= src_back_offset; ix += ELEM_COUNTS)
    {
        ox = (ix - ELEM_COUNTS) * 2;
        vload(src_p + ix, vd16_src_p);
        vload(src_c + ix, vd16_src_c);
        vload(src_n + ix, vd16_src_n);

        Pyrup5x5VCore<Tp>(vd16_src_p, vd16_src_c, vd16_src_n, wd32_sum_c0x2, wd32_sum_c1x2, kernel);

        Pyrup5x5HCore<Tp, Kt>(wd32_sum_c0x0, wd32_sum_c0x1, wd32_sum_c0x2, vd16_result_c0x0, vd16_result_c0x1, kernel);
        Pyrup5x5HCore<Tp, Kt>(wd32_sum_c1x0, wd32_sum_c1x1, wd32_sum_c1x2, vd16_result_c1x0, vd16_result_c1x1, kernel);

        vstore(dst_c0 + ox,               vd16_result_c0x0);
        vstore(dst_c0 + ox + ELEM_COUNTS, vd16_result_c0x1);
        vstore(dst_c1 + ox,               vd16_result_c1x0);
        vstore(dst_c1 + ox + ELEM_COUNTS, vd16_result_c1x1);
    }
    // remain
    {
        MI_S32 src_rest = iwidth & (ELEM_COUNTS - 1);
        MI_S32 ox_last  = src_back_offset * 2;

        ox = (ix - ELEM_COUNTS) * 2;

        HVX_Vector vd16_last_c0x0, vd16_last_c0x1;
        HVX_Vector vd16_last_c1x0, vd16_last_c1x1;
        HVX_VectorPair wd32_sum_c0x3, wd32_sum_c1x3;

        vload(src_p + src_back_offset, vd16_src_p);
        vload(src_c + src_back_offset, vd16_src_c);
        vload(src_n + src_back_offset, vd16_src_n);

        HVX_Vector vd16_border_p = GetBorderVector<Tp, BorderType::REPLICATE, BorderArea::RIGHT>(vd16_src_p, src_p[iwidth - 1], 0);
        HVX_Vector vd16_border_c = GetBorderVector<Tp, BorderType::REPLICATE, BorderArea::RIGHT>(vd16_src_c, src_c[iwidth - 1], 0);
        HVX_Vector vd16_border_n = GetBorderVector<Tp, BorderType::REPLICATE, BorderArea::RIGHT>(vd16_src_n, src_n[iwidth - 1], 0);

        Pyrup5x5VCore<Tp>(vd16_src_p, vd16_src_c, vd16_src_n, wd32_sum_c0x2, wd32_sum_c1x2, kernel);
        Pyrup5x5VCore<Tp>(vd16_border_p, vd16_border_c, vd16_border_n, wd32_sum_c0x3, wd32_sum_c1x3, kernel);

        Pyrup5x5HCore<Tp, Kt>(wd32_sum_c0x0, wd32_sum_c0x1, wd32_sum_c0x2, wd32_sum_c0x3, vd16_result_c0x0,
                              vd16_result_c0x1, vd16_last_c0x0, vd16_last_c0x1, kernel, src_rest);
        Pyrup5x5HCore<Tp, Kt>(wd32_sum_c1x0, wd32_sum_c1x1, wd32_sum_c1x2, wd32_sum_c1x3, vd16_result_c1x0,
                              vd16_result_c1x1, vd16_last_c1x0, vd16_last_c1x1, kernel, src_rest);

        vstore(dst_c0 + ox,               vd16_result_c0x0);
        vstore(dst_c0 + ox + ELEM_COUNTS, vd16_result_c0x1);
        vstore(dst_c1 + ox,               vd16_result_c1x0);
        vstore(dst_c1 + ox + ELEM_COUNTS, vd16_result_c1x1);

        vstore(dst_c0 + ox_last,               vd16_last_c0x0);
        vstore(dst_c0 + ox_last + ELEM_COUNTS, vd16_last_c0x1);
        vstore(dst_c1 + ox_last,               vd16_last_c1x0);
        vstore(dst_c1 + ox_last + ELEM_COUNTS, vd16_last_c1x1);
    }
}

template <typename Tp, BorderType BORDER_TYPE>
static Status PyrUp5x5HvxImpl(const Mat &src, Mat &dst, const Mat &kmat, MI_S32 start_row, MI_S32 end_row)
{
    using Kt = typename PyrUpTraits<Tp>::KernelType;

    MI_S32 iwidth  = src.GetSizes().m_width;
    MI_S32 iheight = src.GetSizes().m_height;
    MI_S32 istride = src.GetStrides().m_width;

    const Kt *kernel = kmat.Ptr<Kt>(0);

    const Tp *src_p = src.Ptr<Tp, BORDER_TYPE>(start_row - 1, MI_NULL);
    const Tp *src_c = src.Ptr<Tp>(start_row);
    const Tp *src_n = src.Ptr<Tp, BorderType::REPLICATE>(start_row + 1, MI_NULL);
    Tp *dst_c0      = MI_NULL;
    Tp *dst_c1      = MI_NULL;
    MI_S32 dy       = 0;

    MI_U64 L2fetch_param = L2PfParam(istride, iwidth * ElemTypeSize(src.GetElemType()), 1, 0);
    for (MI_S32 sy = start_row; sy < end_row; sy++)
    {
        dy = sy << 1;
        if (sy + 2 < iheight)
        {
            L2Fetch(reinterpret_cast<MI_U32>(src.Ptr<Tp>(sy + 2)), L2fetch_param);
        }
        dst_c0 = dst.Ptr<Tp>(dy);
        dst_c1 = dst.Ptr<Tp>(dy + 1);

        PyrUp5x5TwoRow<Tp, BORDER_TYPE, Kt>(src_p, src_c, src_n, iwidth, dst_c0, dst_c1, kernel);

        src_p = src_c;
        src_c = src_n;
        src_n = src.Ptr<Tp, BorderType::REPLICATE>(sy + 2, MI_NULL);
    }

    return Status::OK;
}

template <typename Tp>
static Status PyrUp5x5HvxHelper(Context *ctx, const Mat &src, Mat &dst, const Mat &kmat, BorderType &border_type)
{
    Status ret = Status::ERROR;

    WorkerPool *wp = ctx->GetWorkerPool();
    if (MI_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerpool failed");
        return ret;
    }

    MI_S32 height = src.GetSizes().m_height;

    switch (border_type)
    {
        case BorderType::REPLICATE:
        {
            ret = wp->ParallelFor((MI_S32)0, height, PyrUp5x5HvxImpl<Tp, BorderType::REPLICATE>,
                                  std::cref(src), std::ref(dst), std::cref(kmat));
            break;
        }

        case BorderType::REFLECT_101:
        {
            ret = wp->ParallelFor((MI_S32)0, height, PyrUp5x5HvxImpl<Tp, BorderType::REFLECT_101>,
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

Status PyrUp5x5Hvx(Context *ctx, const Mat &src, Mat &dst, const Mat &kmat, BorderType border_type)
{
    Status ret = Status::ERROR;

    switch (src.GetElemType())
    {
        case ElemType::U8:
        {
            ret = PyrUp5x5HvxHelper<MI_U8>(ctx, src, dst, kmat, border_type);
            break;
        }

        case ElemType::U16:
        {
            ret = PyrUp5x5HvxHelper<MI_U16>(ctx, src, dst, kmat, border_type);
            break;
        }

        case ElemType::S16:
        {
            ret = PyrUp5x5HvxHelper<MI_S16>(ctx, src, dst, kmat, border_type);
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
