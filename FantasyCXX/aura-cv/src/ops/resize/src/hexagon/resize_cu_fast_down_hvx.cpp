#include "resize_impl.hpp"
#include "aura/runtime/worker_pool.h"
#include "aura/runtime/logger.h"

namespace aura
{

template <DT_S32 Size> struct ResizeCuDnCoefTraits;
template <> struct ResizeCuDnCoefTraits<1>
{
    static constexpr DT_S32 SHIFT_BITS  = 11;
    static constexpr DT_S32 COEF0       = -192;
    static constexpr DT_S32 COEF1       = 1216;
    static constexpr DT_S32 COEF_BORDER = 1024;
};

template <> struct ResizeCuDnCoefTraits<2>
{
    static constexpr DT_S32 SHIFT_BITS  = 15;
    static constexpr DT_S32 COEF0       = -3072;
    static constexpr DT_S32 COEF1       = 19456;
    static constexpr DT_S32 COEF_BORDER = 16384;
};

template <typename Tp, typename BufType>
AURA_ALWAYS_INLINE Tp ResizeCuSaturateCast(BufType val, DT_S32 bits)
{
    return SaturateCast<Tp>((val + (1 << (bits * 2 - 1))) >> (bits * 2));
}

template <typename Tp, typename HVX_VType, typename std::enable_if<std::is_same<DT_U8, Tp>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID ResizeCuDnX2HCore(HVX_Vector &vu8_x0_src, HVX_Vector &vu8_x1_src, HVX_Vector &vu8_x2_src,
                                             HVX_VType &ws32_result_l, HVX_VType &ws32_result_h, HVX_Vector &vs16_alpha0, HVX_Vector &vs16_alpha1)
{
    HVX_VectorPair wu16_x0_src = Q6_Wuh_vzxt_Vub(vu8_x0_src);
    HVX_VectorPair wu16_x1_src = Q6_Wuh_vzxt_Vub(vu8_x1_src);
    HVX_VectorPair wu16_x2_src = Q6_Wuh_vzxt_Vub(vu8_x2_src);

    HVX_Vector vu16_x0_even = Q6_V_lo_W(wu16_x0_src);
    HVX_Vector vu16_x0_odd  = Q6_V_hi_W(wu16_x0_src);
    HVX_Vector vu16_x1_even = Q6_V_lo_W(wu16_x1_src);
    HVX_Vector vu16_x1_odd  = Q6_V_hi_W(wu16_x1_src);
    HVX_Vector vu16_x2_even = Q6_V_lo_W(wu16_x2_src);
    HVX_Vector vu16_x2_odd  = Q6_V_hi_W(wu16_x2_src);

    HVX_Vector vu16_r0_even = Q6_V_valign_VVI(vu16_x1_even, vu16_x0_even, 2);
    HVX_Vector vu16_r0_odd  = Q6_V_valign_VVI(vu16_x1_odd,  vu16_x0_odd,  2);
    HVX_Vector vu16_r1_even = Q6_V_valign_VVI(vu16_x2_even, vu16_x1_even, 2);
    HVX_Vector vu16_r1_odd  = Q6_V_valign_VVI(vu16_x2_odd,  vu16_x1_odd,  2);

    HVX_Vector vu16_sum03_l = Q6_Vuh_vadd_VuhVuh_sat(vu16_x0_even, vu16_r0_odd);
    HVX_Vector vu16_sum12_l = Q6_Vuh_vadd_VuhVuh_sat(vu16_x0_odd,  vu16_r0_even);
    HVX_Vector vu16_sum03_h = Q6_Vuh_vadd_VuhVuh_sat(vu16_x1_even, vu16_r1_odd);
    HVX_Vector vu16_sum12_h = Q6_Vuh_vadd_VuhVuh_sat(vu16_x1_odd,  vu16_r1_even);

    ws32_result_l = Q6_Ww_vmpy_VhVuh(vs16_alpha0, vu16_sum03_l);
    ws32_result_h = Q6_Ww_vmpy_VhVuh(vs16_alpha0, vu16_sum03_h);
    ws32_result_l = Q6_Ww_vmpyacc_WwVhVuh(ws32_result_l, vs16_alpha1, vu16_sum12_l);
    ws32_result_h = Q6_Ww_vmpyacc_WwVhVuh(ws32_result_h, vs16_alpha1, vu16_sum12_h);
}

template <typename Tp, typename HVX_VType, typename std::enable_if<std::is_same<DT_S8, Tp>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID ResizeCuDnX2HCore(HVX_Vector &vs8_x0_src, HVX_Vector &vs8_x1_src, HVX_Vector &vs8_x2_src,
                                             HVX_VType &ws32_result_l, HVX_VType &ws32_result_h, HVX_Vector &vs16_alpha0, HVX_Vector &vs16_alpha1)
{
    HVX_VectorPair ws16_x0_src = Q6_Wh_vsxt_Vb(vs8_x0_src);
    HVX_VectorPair ws16_x1_src = Q6_Wh_vsxt_Vb(vs8_x1_src);
    HVX_VectorPair ws16_x2_src = Q6_Wh_vsxt_Vb(vs8_x2_src);

    HVX_Vector vs16_x0_even = Q6_V_lo_W(ws16_x0_src);
    HVX_Vector vs16_x0_odd  = Q6_V_hi_W(ws16_x0_src);
    HVX_Vector vs16_x1_even = Q6_V_lo_W(ws16_x1_src);
    HVX_Vector vs16_x1_odd  = Q6_V_hi_W(ws16_x1_src);
    HVX_Vector vs16_x2_even = Q6_V_lo_W(ws16_x2_src);
    HVX_Vector vs16_x2_odd  = Q6_V_hi_W(ws16_x2_src);

    HVX_Vector vs16_r0_even = Q6_V_valign_VVI(vs16_x1_even, vs16_x0_even, 2);
    HVX_Vector vs16_r0_odd  = Q6_V_valign_VVI(vs16_x1_odd,  vs16_x0_odd,  2);
    HVX_Vector vs16_r1_even = Q6_V_valign_VVI(vs16_x2_even, vs16_x1_even, 2);
    HVX_Vector vs16_r1_odd  = Q6_V_valign_VVI(vs16_x2_odd,  vs16_x1_odd,  2);

    HVX_Vector vs16_sum03_l = Q6_Vh_vadd_VhVh_sat(vs16_x0_even, vs16_r0_odd);
    HVX_Vector vs16_sum12_l = Q6_Vh_vadd_VhVh_sat(vs16_x0_odd,  vs16_r0_even);
    HVX_Vector vs16_sum03_h = Q6_Vh_vadd_VhVh_sat(vs16_x1_even, vs16_r1_odd);
    HVX_Vector vs16_sum12_h = Q6_Vh_vadd_VhVh_sat(vs16_x1_odd,  vs16_r1_even);

    ws32_result_l = Q6_Ww_vmpy_VhVh(vs16_alpha0, vs16_sum03_l);
    ws32_result_h = Q6_Ww_vmpy_VhVh(vs16_alpha0, vs16_sum03_h);
    ws32_result_l = Q6_Ww_vmpyacc_WwVhVh(ws32_result_l, vs16_alpha1, vs16_sum12_l);
    ws32_result_h = Q6_Ww_vmpyacc_WwVhVh(ws32_result_h, vs16_alpha1, vs16_sum12_h);
}

template <typename Tp, typename HVX_VType, typename std::enable_if<std::is_same<DT_U16, Tp>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID ResizeCuDnX2HCore(HVX_Vector &vu16_x0_src, HVX_Vector &vu16_x1_src, HVX_Vector &vu16_x2_src,
                                             HVX_VType &vs32_result_l, HVX_VType &vs32_result_h, HVX_Vector &vs16_alpha0, HVX_Vector &vs16_alpha1)
{
    HVX_Vector vu16_r0          = Q6_V_valign_VVI(vu16_x1_src, vu16_x0_src, 4);
    HVX_Vector vu16_r1          = Q6_V_valign_VVI(vu16_x2_src, vu16_x1_src, 4);
    HVX_VectorPair wu16_r01     = Q6_W_vdeal_VVR(vu16_x1_src, vu16_x0_src, -2);
    HVX_VectorPair wu16_r23     = Q6_W_vdeal_VVR(vu16_r1, vu16_r0, -2);
    HVX_VectorPair ws32_r03_sum = Q6_Ww_vadd_VuhVuh(Q6_V_lo_W(wu16_r01), Q6_V_hi_W(wu16_r23));
    HVX_VectorPair ws32_r12_sum = Q6_Ww_vadd_VuhVuh(Q6_V_hi_W(wu16_r01), Q6_V_lo_W(wu16_r23));

    HVX_Vector vs32_r03_mul_l = Q6_Vw_vmul32xhi16_VwVw(Q6_V_lo_W(ws32_r03_sum), vs16_alpha0);
    HVX_Vector vs32_r03_mul_h = Q6_Vw_vmul32xhi16_VwVw(Q6_V_hi_W(ws32_r03_sum), vs16_alpha0);
    HVX_Vector vs32_r12_mul_l = Q6_Vw_vmul32xhi16_VwVw(Q6_V_lo_W(ws32_r12_sum), vs16_alpha1);
    HVX_Vector vs32_r12_mul_h = Q6_Vw_vmul32xhi16_VwVw(Q6_V_hi_W(ws32_r12_sum), vs16_alpha1);

    vs32_result_l = Q6_Vw_vadd_VwVw(vs32_r03_mul_l, vs32_r12_mul_l);
    vs32_result_h = Q6_Vw_vadd_VwVw(vs32_r03_mul_h, vs32_r12_mul_h);
}

template <typename Tp, typename HVX_VType, typename std::enable_if<std::is_same<DT_S16, Tp>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID ResizeCuDnX2HCore(HVX_Vector &vs16_x0_src, HVX_Vector &vs16_x1_src, HVX_Vector &vs16_x2_src,
                                             HVX_VType &vs32_result_l, HVX_VType &vs32_result_h, HVX_Vector &vs16_alpha0, HVX_Vector &vs16_alpha1)
{
    AURA_UNUSED(vs16_alpha0);
    AURA_UNUSED(vs16_alpha1);

    HVX_Vector vs16_r0     = Q6_V_valign_VVI(vs16_x1_src, vs16_x0_src, 4);
    HVX_Vector vs16_r1     = Q6_V_valign_VVI(vs16_x2_src, vs16_x1_src, 4);
    HVX_Vector vs32_x0_r01 = Q6_Vw_vdmpy_VhRh_sat(vs16_x0_src, 0x4c00f400);
    HVX_Vector vs32_x1_r01 = Q6_Vw_vdmpy_VhRh_sat(vs16_x1_src, 0x4c00f400);

    vs32_result_l = Q6_Vw_vdmpyacc_VwVhRh_sat(vs32_x0_r01, vs16_r0, 0xf4004c00);
    vs32_result_h = Q6_Vw_vdmpyacc_VwVhRh_sat(vs32_x1_r01, vs16_r1, 0xf4004c00);
}

template <typename Tp, typename HVX_VType, typename std::enable_if<std::is_same<DT_U8, Tp>::value || std::is_same<DT_S8, Tp>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID ResizeCuDnX2BorderHCore(HVX_Vector &vd8_x0_src, HVX_Vector &vd8_x1_src, Tp border_value,
                                                   HVX_VType &ws32_result_l, HVX_VType &ws32_result_h, HVX_Vector &vs16_alpha0, HVX_Vector &vs16_alpha1)
{
    HVX_Vector vd8_x2_src = Q6_Vb_vsplat_R(border_value);
    ResizeCuDnX2HCore<Tp, HVX_VType>(vd8_x0_src, vd8_x1_src, vd8_x2_src, ws32_result_l, ws32_result_h, vs16_alpha0, vs16_alpha1);
}

template <typename Tp, typename HVX_VType, typename std::enable_if<std::is_same<DT_U16, Tp>::value || std::is_same<DT_S16, Tp>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID ResizeCuDnX2BorderHCore(HVX_Vector &vd16_x0_src, HVX_Vector &vd16_x1_src, Tp border_value,
                                                   HVX_VType &ws32_result_l, HVX_VType &ws32_result_h, HVX_Vector &vs16_alpha0, HVX_Vector &vs16_alpha1)
{
    HVX_Vector vd16_x2_src = Q6_Vh_vsplat_R(border_value);
    ResizeCuDnX2HCore<Tp, HVX_VType>(vd16_x0_src, vd16_x1_src, vd16_x2_src, ws32_result_l, ws32_result_h, vs16_alpha0, vs16_alpha1);
}

template <typename Tp, typename HVX_VType, typename std::enable_if<std::is_same<DT_U8, Tp>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID ResizeCuDnX2VCore(HVX_VType &ws32_n0_result_l, HVX_VType &ws32_n0_result_h,
                                             HVX_VType &ws32_n1_result_l, HVX_VType &ws32_n1_result_h,
                                             HVX_VType &ws32_n2_result_l, HVX_VType &ws32_n2_result_h,
                                             HVX_Vector &vu8_dst, DT_S32 beta0, DT_S32 beta1, DT_S32 beta2)
{
    HVX_Vector vs16_beta0 = Q6_Vh_vsplat_R(beta0);
    HVX_Vector vs16_beta1 = Q6_Vh_vsplat_R(beta1);
    HVX_Vector vs16_beta2 = Q6_Vh_vsplat_R(beta2);

    HVX_Vector vs32_result_ll = Q6_Vw_vmul32xhi16_VwVw(Q6_V_lo_W(ws32_n0_result_l), vs16_beta0);
    HVX_Vector vs32_result_lh = Q6_Vw_vmul32xhi16_VwVw(Q6_V_hi_W(ws32_n0_result_l), vs16_beta0);
    HVX_Vector vs32_result_hl = Q6_Vw_vmul32xhi16_VwVw(Q6_V_lo_W(ws32_n0_result_h), vs16_beta0);
    HVX_Vector vs32_result_hh = Q6_Vw_vmul32xhi16_VwVw(Q6_V_hi_W(ws32_n0_result_h), vs16_beta0);

    vs32_result_ll = Q6_Vw_vadd_VwVw(vs32_result_ll, Q6_Vw_vmul32xhi16_VwVw(Q6_V_lo_W(ws32_n1_result_l), vs16_beta1));
    vs32_result_lh = Q6_Vw_vadd_VwVw(vs32_result_lh, Q6_Vw_vmul32xhi16_VwVw(Q6_V_hi_W(ws32_n1_result_l), vs16_beta1));
    vs32_result_hl = Q6_Vw_vadd_VwVw(vs32_result_hl, Q6_Vw_vmul32xhi16_VwVw(Q6_V_lo_W(ws32_n1_result_h), vs16_beta1));
    vs32_result_hh = Q6_Vw_vadd_VwVw(vs32_result_hh, Q6_Vw_vmul32xhi16_VwVw(Q6_V_hi_W(ws32_n1_result_h), vs16_beta1));
    vs32_result_ll = Q6_Vw_vadd_VwVw(vs32_result_ll, Q6_Vw_vmul32xhi16_VwVw(Q6_V_lo_W(ws32_n2_result_l), vs16_beta2));
    vs32_result_lh = Q6_Vw_vadd_VwVw(vs32_result_lh, Q6_Vw_vmul32xhi16_VwVw(Q6_V_hi_W(ws32_n2_result_l), vs16_beta2));
    vs32_result_hl = Q6_Vw_vadd_VwVw(vs32_result_hl, Q6_Vw_vmul32xhi16_VwVw(Q6_V_lo_W(ws32_n2_result_h), vs16_beta2));
    vs32_result_hh = Q6_Vw_vadd_VwVw(vs32_result_hh, Q6_Vw_vmul32xhi16_VwVw(Q6_V_hi_W(ws32_n2_result_h), vs16_beta2));

    HVX_Vector vu16_result_l = Q6_V_hi_W(Q6_W_vshuff_VVR(vs32_result_lh, vs32_result_ll, 2));
    HVX_Vector vu16_result_h = Q6_V_hi_W(Q6_W_vshuff_VVR(vs32_result_hh, vs32_result_hl, 2));
    vu8_dst = Q6_Vb_vdeal_Vb(Q6_Vub_vasr_VhVhR_rnd_sat(vu16_result_h, vu16_result_l, 6));
}

template <typename Tp, typename HVX_VType, typename std::enable_if<std::is_same<DT_S8, Tp>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID ResizeCuDnX2VCore(HVX_VType &ws32_n0_result_l, HVX_VType &ws32_n0_result_h,
                                             HVX_VType &ws32_n1_result_l, HVX_VType &ws32_n1_result_h,
                                             HVX_VType &ws32_n2_result_l, HVX_VType &ws32_n2_result_h,
                                             HVX_Vector &vs8_dst, DT_S32 beta0, DT_S32 beta1, DT_S32 beta2)
{
    HVX_Vector vs16_beta0 = Q6_Vh_vsplat_R(beta0);
    HVX_Vector vs16_beta1 = Q6_Vh_vsplat_R(beta1);
    HVX_Vector vs16_beta2 = Q6_Vh_vsplat_R(beta2);

    HVX_Vector vs32_result_ll = Q6_Vw_vmul32xhi16_VwVw(Q6_V_lo_W(ws32_n0_result_l), vs16_beta0);
    HVX_Vector vs32_result_lh = Q6_Vw_vmul32xhi16_VwVw(Q6_V_hi_W(ws32_n0_result_l), vs16_beta0);
    HVX_Vector vs32_result_hl = Q6_Vw_vmul32xhi16_VwVw(Q6_V_lo_W(ws32_n0_result_h), vs16_beta0);
    HVX_Vector vs32_result_hh = Q6_Vw_vmul32xhi16_VwVw(Q6_V_hi_W(ws32_n0_result_h), vs16_beta0);

    vs32_result_ll = Q6_Vw_vadd_VwVw(vs32_result_ll, Q6_Vw_vmul32xhi16_VwVw(Q6_V_lo_W(ws32_n1_result_l), vs16_beta1));
    vs32_result_lh = Q6_Vw_vadd_VwVw(vs32_result_lh, Q6_Vw_vmul32xhi16_VwVw(Q6_V_hi_W(ws32_n1_result_l), vs16_beta1));
    vs32_result_hl = Q6_Vw_vadd_VwVw(vs32_result_hl, Q6_Vw_vmul32xhi16_VwVw(Q6_V_lo_W(ws32_n1_result_h), vs16_beta1));
    vs32_result_hh = Q6_Vw_vadd_VwVw(vs32_result_hh, Q6_Vw_vmul32xhi16_VwVw(Q6_V_hi_W(ws32_n1_result_h), vs16_beta1));
    vs32_result_ll = Q6_Vw_vadd_VwVw(vs32_result_ll, Q6_Vw_vmul32xhi16_VwVw(Q6_V_lo_W(ws32_n2_result_l), vs16_beta2));
    vs32_result_lh = Q6_Vw_vadd_VwVw(vs32_result_lh, Q6_Vw_vmul32xhi16_VwVw(Q6_V_hi_W(ws32_n2_result_l), vs16_beta2));
    vs32_result_hl = Q6_Vw_vadd_VwVw(vs32_result_hl, Q6_Vw_vmul32xhi16_VwVw(Q6_V_lo_W(ws32_n2_result_h), vs16_beta2));
    vs32_result_hh = Q6_Vw_vadd_VwVw(vs32_result_hh, Q6_Vw_vmul32xhi16_VwVw(Q6_V_hi_W(ws32_n2_result_h), vs16_beta2));

    HVX_Vector vu16_result_l = Q6_V_hi_W(Q6_W_vshuff_VVR(vs32_result_lh, vs32_result_ll, 2));
    HVX_Vector vu16_result_h = Q6_V_hi_W(Q6_W_vshuff_VVR(vs32_result_hh, vs32_result_hl, 2));
    vs8_dst = Q6_Vb_vdeal_Vb(Q6_Vb_vasr_VhVhR_rnd_sat(vu16_result_h, vu16_result_l, 6));
}

template <typename Tp, typename HVX_VType, typename std::enable_if<std::is_same<DT_U16, Tp>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID ResizeCuDnX2VCore(HVX_VType &vs32_n0_result_l, HVX_VType &vs32_n0_result_h,
                                             HVX_VType &vs32_n1_result_l, HVX_VType &vs32_n1_result_h,
                                             HVX_VType &vs32_n2_result_l, HVX_VType &vs32_n2_result_h,
                                             HVX_Vector &vu16_dst, DT_S32 beta0, DT_S32 beta1, DT_S32 beta2)
{
    HVX_Vector vs32_beta0   = Q6_V_vsplat_R(beta0);
    HVX_Vector vs32_beta1   = Q6_V_vsplat_R(beta1);
    HVX_Vector vs32_beta2   = Q6_V_vsplat_R(beta2);
    HVX_VectorPair ws64_rnd = Q6_W_vcombine_VV(Q6_V_vzero(), Q6_V_vsplat_R(1 << 29));

    HVX_VectorPair ws64_result_l = Q6_Wd_vmul_VwVw(vs32_n0_result_l, vs32_beta0);
    HVX_VectorPair ws64_result_h = Q6_Wd_vmul_VwVw(vs32_n0_result_h, vs32_beta0);
    ws64_result_l = Q6_Wd_vmulacc_WdVwVw(ws64_result_l, vs32_n1_result_l, vs32_beta1);
    ws64_result_h = Q6_Wd_vmulacc_WdVwVw(ws64_result_h, vs32_n1_result_h, vs32_beta1);
    ws64_result_l = Q6_Wd_vmulacc_WdVwVw(ws64_result_l, vs32_n2_result_l, vs32_beta2);
    ws64_result_h = Q6_Wd_vmulacc_WdVwVw(ws64_result_h, vs32_n2_result_h, vs32_beta2);
    ws64_result_l = Q6_Wud_vadd_WudWud(ws64_result_l, ws64_rnd);
    ws64_result_h = Q6_Wud_vadd_WudWud(ws64_result_h, ws64_rnd);

    HVX_Vector vu32_sum_l = Q6_V_vor_VV(Q6_Vw_vasl_VwR(Q6_V_hi_W(ws64_result_l), 2), Q6_Vuw_vlsr_VuwR(Q6_V_lo_W(ws64_result_l), 30));
    HVX_Vector vu32_sum_h = Q6_V_vor_VV(Q6_Vw_vasl_VwR(Q6_V_hi_W(ws64_result_h), 2), Q6_Vuw_vlsr_VuwR(Q6_V_lo_W(ws64_result_h), 30));
    vu32_sum_l = Q6_Vw_vmax_VwVw(vu32_sum_l, Q6_V_vzero());
    vu32_sum_h = Q6_Vw_vmax_VwVw(vu32_sum_h, Q6_V_vzero());

    vu16_dst = Q6_Vuh_vsat_VuwVuw(vu32_sum_h, vu32_sum_l);
}

template <typename Tp, typename HVX_VType, typename std::enable_if<std::is_same<DT_S16, Tp>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID ResizeCuDnX2VCore(HVX_VType &vs32_n0_result_l, HVX_VType &vs32_n0_result_h,
                                             HVX_VType &vs32_n1_result_l, HVX_VType &vs32_n1_result_h,
                                             HVX_VType &vs32_n2_result_l, HVX_VType &vs32_n2_result_h,
                                             HVX_Vector &vs16_dst, DT_S32 beta0, DT_S32 beta1, DT_S32 beta2)
{
    HVX_Vector vs32_beta0   = Q6_V_vsplat_R(beta0);
    HVX_Vector vs32_beta1   = Q6_V_vsplat_R(beta1);
    HVX_Vector vs32_beta2   = Q6_V_vsplat_R(beta2);
    HVX_VectorPair ws64_rnd = Q6_W_vcombine_VV(Q6_V_vzero(), Q6_V_vsplat_R(1 << 29));

    HVX_VectorPair ws64_result_l = Q6_Wd_vmul_VwVw(vs32_n0_result_l, vs32_beta0);
    HVX_VectorPair ws64_result_h = Q6_Wd_vmul_VwVw(vs32_n0_result_h, vs32_beta0);
    ws64_result_l = Q6_Wd_vmulacc_WdVwVw(ws64_result_l, vs32_n1_result_l, vs32_beta1);
    ws64_result_h = Q6_Wd_vmulacc_WdVwVw(ws64_result_h, vs32_n1_result_h, vs32_beta1);
    ws64_result_l = Q6_Wd_vmulacc_WdVwVw(ws64_result_l, vs32_n2_result_l, vs32_beta2);
    ws64_result_h = Q6_Wd_vmulacc_WdVwVw(ws64_result_h, vs32_n2_result_h, vs32_beta2);
    ws64_result_l = Q6_Wd_vadd_WdWd(ws64_result_l, ws64_rnd);
    ws64_result_h = Q6_Wd_vadd_WdWd(ws64_result_h, ws64_rnd);

    HVX_Vector vs32_sum_l = Q6_V_vor_VV(Q6_Vw_vasl_VwR(Q6_V_hi_W(ws64_result_l), 2), Q6_Vuw_vlsr_VuwR(Q6_V_lo_W(ws64_result_l), 30));
    HVX_Vector vs32_sum_h = Q6_V_vor_VV(Q6_Vw_vasl_VwR(Q6_V_hi_W(ws64_result_h), 2), Q6_Vuw_vlsr_VuwR(Q6_V_lo_W(ws64_result_h), 30));
    vs16_dst = Q6_Vh_vdeal_Vh(Q6_Vh_vsat_VwVw(vs32_sum_h, vs32_sum_l));
}

template <typename Tp, typename HVX_VType, typename std::enable_if<std::is_same<DT_U8, Tp>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID ResizeCuDnX2VCore(HVX_VType &ws32_n0_result_l, HVX_VType &ws32_n0_result_h,
                                             HVX_VType &ws32_n1_result_l, HVX_VType &ws32_n1_result_h,
                                             HVX_VType &ws32_n2_result_l, HVX_VType &ws32_n2_result_h,
                                             HVX_VType &ws32_n3_result_l, HVX_VType &ws32_n3_result_h,
                                             HVX_Vector &vu8_dst)
{
    HVX_Vector vs16_beta0 = Q6_Vh_vsplat_R(-192);
    HVX_Vector vs16_beta1 = Q6_Vh_vsplat_R(1216);

    HVX_VectorPair ws32_n03_sum_l = Q6_Ww_vadd_WwWw_sat(ws32_n0_result_l, ws32_n3_result_l);
    HVX_VectorPair ws32_n03_sum_h = Q6_Ww_vadd_WwWw_sat(ws32_n0_result_h, ws32_n3_result_h);
    HVX_VectorPair ws32_n12_sum_l = Q6_Ww_vadd_WwWw_sat(ws32_n1_result_l, ws32_n2_result_l);
    HVX_VectorPair ws32_n12_sum_h = Q6_Ww_vadd_WwWw_sat(ws32_n1_result_h, ws32_n2_result_h);

    HVX_Vector vs32_result_ll = Q6_Vw_vmul32xhi16_VwVw(Q6_V_lo_W(ws32_n03_sum_l), vs16_beta0);
    HVX_Vector vs32_result_lh = Q6_Vw_vmul32xhi16_VwVw(Q6_V_hi_W(ws32_n03_sum_l), vs16_beta0);
    HVX_Vector vs32_result_hl = Q6_Vw_vmul32xhi16_VwVw(Q6_V_lo_W(ws32_n03_sum_h), vs16_beta0);
    HVX_Vector vs32_result_hh = Q6_Vw_vmul32xhi16_VwVw(Q6_V_hi_W(ws32_n03_sum_h), vs16_beta0);

    vs32_result_ll = Q6_Vw_vadd_VwVw(vs32_result_ll, Q6_Vw_vmul32xhi16_VwVw(Q6_V_lo_W(ws32_n12_sum_l), vs16_beta1));
    vs32_result_lh = Q6_Vw_vadd_VwVw(vs32_result_lh, Q6_Vw_vmul32xhi16_VwVw(Q6_V_hi_W(ws32_n12_sum_l), vs16_beta1));
    vs32_result_hl = Q6_Vw_vadd_VwVw(vs32_result_hl, Q6_Vw_vmul32xhi16_VwVw(Q6_V_lo_W(ws32_n12_sum_h), vs16_beta1));
    vs32_result_hh = Q6_Vw_vadd_VwVw(vs32_result_hh, Q6_Vw_vmul32xhi16_VwVw(Q6_V_hi_W(ws32_n12_sum_h), vs16_beta1));

    HVX_Vector vu16_result_l = Q6_V_hi_W(Q6_W_vshuff_VVR(vs32_result_lh, vs32_result_ll, 2));
    HVX_Vector vu16_result_h = Q6_V_hi_W(Q6_W_vshuff_VVR(vs32_result_hh, vs32_result_hl, 2));
    vu8_dst = Q6_Vb_vdeal_Vb(Q6_Vub_vasr_VhVhR_rnd_sat(vu16_result_h, vu16_result_l, 6));
}

template <typename Tp, typename HVX_VType, typename std::enable_if<std::is_same<DT_S8, Tp>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID ResizeCuDnX2VCore(HVX_VType &ws32_n0_result_l, HVX_VType &ws32_n0_result_h,
                                             HVX_VType &ws32_n1_result_l, HVX_VType &ws32_n1_result_h,
                                             HVX_VType &ws32_n2_result_l, HVX_VType &ws32_n2_result_h,
                                             HVX_VType &ws32_n3_result_l, HVX_VType &ws32_n3_result_h,
                                             HVX_Vector &vs8_dst)
{
    HVX_Vector vs16_beta0 = Q6_Vh_vsplat_R(-192);
    HVX_Vector vs16_beta1 = Q6_Vh_vsplat_R(1216);

    HVX_VectorPair ws32_n03_sum_l = Q6_Ww_vadd_WwWw_sat(ws32_n0_result_l, ws32_n3_result_l);
    HVX_VectorPair ws32_n03_sum_h = Q6_Ww_vadd_WwWw_sat(ws32_n0_result_h, ws32_n3_result_h);
    HVX_VectorPair ws32_n12_sum_l = Q6_Ww_vadd_WwWw_sat(ws32_n1_result_l, ws32_n2_result_l);
    HVX_VectorPair ws32_n12_sum_h = Q6_Ww_vadd_WwWw_sat(ws32_n1_result_h, ws32_n2_result_h);

    HVX_Vector vs32_result_ll = Q6_Vw_vmul32xhi16_VwVw(Q6_V_lo_W(ws32_n03_sum_l), vs16_beta0);
    HVX_Vector vs32_result_lh = Q6_Vw_vmul32xhi16_VwVw(Q6_V_hi_W(ws32_n03_sum_l), vs16_beta0);
    HVX_Vector vs32_result_hl = Q6_Vw_vmul32xhi16_VwVw(Q6_V_lo_W(ws32_n03_sum_h), vs16_beta0);
    HVX_Vector vs32_result_hh = Q6_Vw_vmul32xhi16_VwVw(Q6_V_hi_W(ws32_n03_sum_h), vs16_beta0);

    vs32_result_ll = Q6_Vw_vadd_VwVw(vs32_result_ll, Q6_Vw_vmul32xhi16_VwVw(Q6_V_lo_W(ws32_n12_sum_l), vs16_beta1));
    vs32_result_lh = Q6_Vw_vadd_VwVw(vs32_result_lh, Q6_Vw_vmul32xhi16_VwVw(Q6_V_hi_W(ws32_n12_sum_l), vs16_beta1));
    vs32_result_hl = Q6_Vw_vadd_VwVw(vs32_result_hl, Q6_Vw_vmul32xhi16_VwVw(Q6_V_lo_W(ws32_n12_sum_h), vs16_beta1));
    vs32_result_hh = Q6_Vw_vadd_VwVw(vs32_result_hh, Q6_Vw_vmul32xhi16_VwVw(Q6_V_hi_W(ws32_n12_sum_h), vs16_beta1));

    HVX_Vector vs16_result_l = Q6_V_hi_W(Q6_W_vshuff_VVR(vs32_result_lh, vs32_result_ll, 2));
    HVX_Vector vs16_result_h = Q6_V_hi_W(Q6_W_vshuff_VVR(vs32_result_hh, vs32_result_hl, 2));
    vs8_dst = Q6_Vb_vdeal_Vb(Q6_Vb_vasr_VhVhR_rnd_sat(vs16_result_h, vs16_result_l, 6));
}

template <typename Tp, typename HVX_VType, typename std::enable_if<std::is_same<DT_U16, Tp>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID ResizeCuDnX2VCore(HVX_VType &vs32_n0_result_l, HVX_VType &vs32_n0_result_h,
                                             HVX_VType &vs32_n1_result_l, HVX_VType &vs32_n1_result_h,
                                             HVX_VType &vs32_n2_result_l, HVX_VType &vs32_n2_result_h,
                                             HVX_VType &vs32_n3_result_l, HVX_VType &vs32_n3_result_h,
                                             HVX_Vector &vu16_dst)
{
    HVX_Vector vs32_beta0   = Q6_V_vsplat_R(-3072);
    HVX_Vector vs32_beta1   = Q6_V_vsplat_R(19456);
    HVX_VectorPair ws64_rnd = Q6_W_vcombine_VV(Q6_V_vzero(), Q6_V_vsplat_R(1 << 29));

    HVX_Vector vs32_n03_result_l = Q6_Vw_vadd_VwVw(vs32_n0_result_l, vs32_n3_result_l);
    HVX_Vector vs32_n03_result_h = Q6_Vw_vadd_VwVw(vs32_n0_result_h, vs32_n3_result_h);
    HVX_Vector vs32_n12_result_l = Q6_Vw_vadd_VwVw(vs32_n1_result_l, vs32_n2_result_l);
    HVX_Vector vs32_n12_result_h = Q6_Vw_vadd_VwVw(vs32_n1_result_h, vs32_n2_result_h);

    HVX_VectorPair ws64_result_l = Q6_Wd_vmul_VwVw(vs32_n03_result_l, vs32_beta0);
    HVX_VectorPair ws64_result_h = Q6_Wd_vmul_VwVw(vs32_n03_result_h, vs32_beta0);
    ws64_result_l = Q6_Wd_vmulacc_WdVwVw(ws64_result_l, vs32_n12_result_l, vs32_beta1);
    ws64_result_h = Q6_Wd_vmulacc_WdVwVw(ws64_result_h, vs32_n12_result_h, vs32_beta1);
    ws64_result_l = Q6_Wud_vadd_WudWud(ws64_result_l, ws64_rnd);
    ws64_result_h = Q6_Wud_vadd_WudWud(ws64_result_h, ws64_rnd);

    HVX_Vector vu32_sum_l = Q6_V_vor_VV(Q6_Vw_vasl_VwR(Q6_V_hi_W(ws64_result_l), 2), Q6_Vuw_vlsr_VuwR(Q6_V_lo_W(ws64_result_l), 30));
    HVX_Vector vu32_sum_h = Q6_V_vor_VV(Q6_Vw_vasl_VwR(Q6_V_hi_W(ws64_result_h), 2), Q6_Vuw_vlsr_VuwR(Q6_V_lo_W(ws64_result_h), 30));
    vu32_sum_l = Q6_Vw_vmax_VwVw(vu32_sum_l, Q6_V_vzero());
    vu32_sum_h = Q6_Vw_vmax_VwVw(vu32_sum_h, Q6_V_vzero());
    vu16_dst = Q6_Vuh_vsat_VuwVuw(vu32_sum_h, vu32_sum_l);
}

template <typename Tp, typename HVX_VType, typename std::enable_if<std::is_same<DT_S16, Tp>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID ResizeCuDnX2VCore(HVX_VType &vs32_n0_result_l, HVX_VType &vs32_n0_result_h,
                                             HVX_VType &vs32_n1_result_l, HVX_VType &vs32_n1_result_h,
                                             HVX_VType &vs32_n2_result_l, HVX_VType &vs32_n2_result_h,
                                             HVX_VType &vs32_n3_result_l, HVX_VType &vs32_n3_result_h,
                                             HVX_Vector &vs16_dst)
{
    HVX_Vector vs32_beta0   = Q6_V_vsplat_R(-3072);
    HVX_Vector vs32_beta1   = Q6_V_vsplat_R(19456);
    HVX_VectorPair ws64_rnd = Q6_W_vcombine_VV(Q6_V_vzero(), Q6_V_vsplat_R(1 << 29));

    HVX_Vector vs32_n03_result_l = Q6_Vw_vadd_VwVw(vs32_n0_result_l, vs32_n3_result_l);
    HVX_Vector vs32_n03_result_h = Q6_Vw_vadd_VwVw(vs32_n0_result_h, vs32_n3_result_h);
    HVX_Vector vs32_n12_result_l = Q6_Vw_vadd_VwVw(vs32_n1_result_l, vs32_n2_result_l);
    HVX_Vector vs32_n12_result_h = Q6_Vw_vadd_VwVw(vs32_n1_result_h, vs32_n2_result_h);

    HVX_VectorPair ws64_result_l = Q6_Wd_vmul_VwVw(vs32_n03_result_l, vs32_beta0);
    HVX_VectorPair ws64_result_h = Q6_Wd_vmul_VwVw(vs32_n03_result_h, vs32_beta0);
    ws64_result_l = Q6_Wd_vmulacc_WdVwVw(ws64_result_l, vs32_n12_result_l, vs32_beta1);
    ws64_result_h = Q6_Wd_vmulacc_WdVwVw(ws64_result_h, vs32_n12_result_h, vs32_beta1);
    ws64_result_l = Q6_Wd_vadd_WdWd(ws64_result_l, ws64_rnd);
    ws64_result_h = Q6_Wd_vadd_WdWd(ws64_result_h, ws64_rnd);

    HVX_Vector vs32_sum_l = Q6_V_vor_VV(Q6_Vw_vasl_VwR(Q6_V_hi_W(ws64_result_l), 2), Q6_Vuw_vlsr_VuwR(Q6_V_lo_W(ws64_result_l), 30));
    HVX_Vector vs32_sum_h = Q6_V_vor_VV(Q6_Vw_vasl_VwR(Q6_V_hi_W(ws64_result_h), 2), Q6_Vuw_vlsr_VuwR(Q6_V_lo_W(ws64_result_h), 30));
    vs16_dst = Q6_Vh_vdeal_Vh(Q6_Vh_vsat_VwVw(vs32_sum_h, vs32_sum_l));
}

template <typename Tp, typename std::enable_if<std::is_same<DT_U8, Tp>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID ResizeCuDnX4Core(HVX_Vector &vu8_n0_x0_src, HVX_Vector &vu8_n0_x1_src, HVX_Vector &vu8_n0_x2_src, HVX_Vector &vu8_n0_x3_src,
                                            HVX_Vector &vu8_n1_x0_src, HVX_Vector &vu8_n1_x1_src, HVX_Vector &vu8_n1_x2_src, HVX_Vector &vu8_n1_x3_src,
                                            HVX_Vector &vu8_n2_x0_src, HVX_Vector &vu8_n2_x1_src, HVX_Vector &vu8_n2_x2_src, HVX_Vector &vu8_n2_x3_src,
                                            HVX_Vector &vu8_n3_x0_src, HVX_Vector &vu8_n3_x1_src, HVX_Vector &vu8_n3_x2_src, HVX_Vector &vu8_n3_x3_src,
                                            HVX_Vector &vu8_dst)
{
    HVX_Vector vs16_coef0 = Q6_Vh_vsplat_R(-192);
    HVX_Vector vs16_coef1 = Q6_Vh_vsplat_R(1216);

    // n0n3
    HVX_VectorPair wu16_n03_x0  = Q6_Wh_vadd_VubVub(vu8_n0_x0_src, vu8_n3_x0_src);
    HVX_VectorPair wu16_n03_x1  = Q6_Wh_vadd_VubVub(vu8_n0_x1_src, vu8_n3_x1_src);
    HVX_VectorPair wu16_n03_x2  = Q6_Wh_vadd_VubVub(vu8_n0_x2_src, vu8_n3_x2_src);
    HVX_VectorPair wu16_n03_x3  = Q6_Wh_vadd_VubVub(vu8_n0_x3_src, vu8_n3_x3_src);
    HVX_Vector vs32_n03_x0_even = Q6_Vw_vdmpy_VhRh_sat(Q6_V_lo_W(wu16_n03_x0), 0x04c0ff40);
    HVX_Vector vs32_n03_x1_even = Q6_Vw_vdmpy_VhRh_sat(Q6_V_lo_W(wu16_n03_x1), 0x04c0ff40);
    HVX_Vector vs32_n03_x2_even = Q6_Vw_vdmpy_VhRh_sat(Q6_V_lo_W(wu16_n03_x2), 0x04c0ff40);
    HVX_Vector vs32_n03_x3_even = Q6_Vw_vdmpy_VhRh_sat(Q6_V_lo_W(wu16_n03_x3), 0x04c0ff40);
    HVX_Vector vs32_n03_x0_sum  = Q6_Vw_vdmpyacc_VwVhRh_sat(vs32_n03_x0_even, Q6_V_hi_W(wu16_n03_x0), 0xff4004c0);
    HVX_Vector vs32_n03_x1_sum  = Q6_Vw_vdmpyacc_VwVhRh_sat(vs32_n03_x1_even, Q6_V_hi_W(wu16_n03_x1), 0xff4004c0);
    HVX_Vector vs32_n03_x2_sum  = Q6_Vw_vdmpyacc_VwVhRh_sat(vs32_n03_x2_even, Q6_V_hi_W(wu16_n03_x2), 0xff4004c0);
    HVX_Vector vs32_n03_x3_sum  = Q6_Vw_vdmpyacc_VwVhRh_sat(vs32_n03_x3_even, Q6_V_hi_W(wu16_n03_x3), 0xff4004c0);

    vs32_n03_x0_sum = Q6_Vw_vmul32xhi16_VwVw(vs32_n03_x0_sum, vs16_coef0);
    vs32_n03_x1_sum = Q6_Vw_vmul32xhi16_VwVw(vs32_n03_x1_sum, vs16_coef0);
    vs32_n03_x2_sum = Q6_Vw_vmul32xhi16_VwVw(vs32_n03_x2_sum, vs16_coef0);
    vs32_n03_x3_sum = Q6_Vw_vmul32xhi16_VwVw(vs32_n03_x3_sum, vs16_coef0);

    // n1n2
    HVX_VectorPair wu16_n12_x0  = Q6_Wh_vadd_VubVub(vu8_n1_x0_src, vu8_n2_x0_src);
    HVX_VectorPair wu16_n12_x1  = Q6_Wh_vadd_VubVub(vu8_n1_x1_src, vu8_n2_x1_src);
    HVX_VectorPair wu16_n12_x2  = Q6_Wh_vadd_VubVub(vu8_n1_x2_src, vu8_n2_x2_src);
    HVX_VectorPair wu16_n12_x3  = Q6_Wh_vadd_VubVub(vu8_n1_x3_src, vu8_n2_x3_src);
    HVX_Vector vs32_n12_x0_even = Q6_Vw_vdmpy_VhRh_sat(Q6_V_lo_W(wu16_n12_x0), 0x04c0ff40);
    HVX_Vector vs32_n12_x1_even = Q6_Vw_vdmpy_VhRh_sat(Q6_V_lo_W(wu16_n12_x1), 0x04c0ff40);
    HVX_Vector vs32_n12_x2_even = Q6_Vw_vdmpy_VhRh_sat(Q6_V_lo_W(wu16_n12_x2), 0x04c0ff40);
    HVX_Vector vs32_n12_x3_even = Q6_Vw_vdmpy_VhRh_sat(Q6_V_lo_W(wu16_n12_x3), 0x04c0ff40);
    HVX_Vector vs32_n12_x0_sum  = Q6_Vw_vdmpyacc_VwVhRh_sat(vs32_n12_x0_even, Q6_V_hi_W(wu16_n12_x0), 0xff4004c0);
    HVX_Vector vs32_n12_x1_sum  = Q6_Vw_vdmpyacc_VwVhRh_sat(vs32_n12_x1_even, Q6_V_hi_W(wu16_n12_x1), 0xff4004c0);
    HVX_Vector vs32_n12_x2_sum  = Q6_Vw_vdmpyacc_VwVhRh_sat(vs32_n12_x2_even, Q6_V_hi_W(wu16_n12_x2), 0xff4004c0);
    HVX_Vector vs32_n12_x3_sum  = Q6_Vw_vdmpyacc_VwVhRh_sat(vs32_n12_x3_even, Q6_V_hi_W(wu16_n12_x3), 0xff4004c0);

    vs32_n12_x0_sum = Q6_Vw_vmul32xhi16_VwVw(vs32_n12_x0_sum, vs16_coef1);
    vs32_n12_x1_sum = Q6_Vw_vmul32xhi16_VwVw(vs32_n12_x1_sum, vs16_coef1);
    vs32_n12_x2_sum = Q6_Vw_vmul32xhi16_VwVw(vs32_n12_x2_sum, vs16_coef1);
    vs32_n12_x3_sum = Q6_Vw_vmul32xhi16_VwVw(vs32_n12_x3_sum, vs16_coef1);

    // sum
    HVX_Vector vs32_x0_sum  = Q6_Vw_vadd_VwVw(vs32_n03_x0_sum, vs32_n12_x0_sum);
    HVX_Vector vs32_x1_sum  = Q6_Vw_vadd_VwVw(vs32_n03_x1_sum, vs32_n12_x1_sum);
    HVX_Vector vs32_x2_sum  = Q6_Vw_vadd_VwVw(vs32_n03_x2_sum, vs32_n12_x2_sum);
    HVX_Vector vs32_x3_sum  = Q6_Vw_vadd_VwVw(vs32_n03_x3_sum, vs32_n12_x3_sum);
    HVX_Vector vu16_result0 = Q6_V_hi_W(Q6_W_vdeal_VVR(vs32_x1_sum, vs32_x0_sum, -2));
    HVX_Vector vu16_result1 = Q6_V_hi_W(Q6_W_vdeal_VVR(vs32_x3_sum, vs32_x2_sum, -2));

    vu8_dst = Q6_Vb_vdeal_Vb(Q6_Vub_vasr_VhVhR_rnd_sat(vu16_result1, vu16_result0, 6));
}

template <typename Tp, typename std::enable_if<std::is_same<DT_S8, Tp>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID ResizeCuDnX4Core(HVX_Vector &vs8_n0_x0_src, HVX_Vector &vs8_n0_x1_src, HVX_Vector &vs8_n0_x2_src, HVX_Vector &vs8_n0_x3_src,
                                            HVX_Vector &vs8_n1_x0_src, HVX_Vector &vs8_n1_x1_src, HVX_Vector &vs8_n1_x2_src, HVX_Vector &vs8_n1_x3_src,
                                            HVX_Vector &vs8_n2_x0_src, HVX_Vector &vs8_n2_x1_src, HVX_Vector &vs8_n2_x2_src, HVX_Vector &vs8_n2_x3_src,
                                            HVX_Vector &vs8_n3_x0_src, HVX_Vector &vs8_n3_x1_src, HVX_Vector &vs8_n3_x2_src, HVX_Vector &vs8_n3_x3_src,
                                            HVX_Vector &vs8_dst)
{
    HVX_Vector vs16_coef0 = Q6_Vh_vsplat_R(-192);
    HVX_Vector vs16_coef1 = Q6_Vh_vsplat_R(1216);

    // n0n3
    HVX_VectorPair ws16_n03_x0  = Q6_Wh_vadd_WhWh(Q6_Wh_vsxt_Vb(vs8_n0_x0_src), Q6_Wh_vsxt_Vb(vs8_n3_x0_src));
    HVX_VectorPair ws16_n03_x1  = Q6_Wh_vadd_WhWh(Q6_Wh_vsxt_Vb(vs8_n0_x1_src), Q6_Wh_vsxt_Vb(vs8_n3_x1_src));
    HVX_VectorPair ws16_n03_x2  = Q6_Wh_vadd_WhWh(Q6_Wh_vsxt_Vb(vs8_n0_x2_src), Q6_Wh_vsxt_Vb(vs8_n3_x2_src));
    HVX_VectorPair ws16_n03_x3  = Q6_Wh_vadd_WhWh(Q6_Wh_vsxt_Vb(vs8_n0_x3_src), Q6_Wh_vsxt_Vb(vs8_n3_x3_src));
    HVX_Vector vs32_n03_x0_even = Q6_Vw_vdmpy_VhRh_sat(Q6_V_lo_W(ws16_n03_x0), 0x04c0ff40);
    HVX_Vector vs32_n03_x1_even = Q6_Vw_vdmpy_VhRh_sat(Q6_V_lo_W(ws16_n03_x1), 0x04c0ff40);
    HVX_Vector vs32_n03_x2_even = Q6_Vw_vdmpy_VhRh_sat(Q6_V_lo_W(ws16_n03_x2), 0x04c0ff40);
    HVX_Vector vs32_n03_x3_even = Q6_Vw_vdmpy_VhRh_sat(Q6_V_lo_W(ws16_n03_x3), 0x04c0ff40);
    HVX_Vector vs32_n03_x0_sum  = Q6_Vw_vdmpyacc_VwVhRh_sat(vs32_n03_x0_even, Q6_V_hi_W(ws16_n03_x0), 0xff4004c0);
    HVX_Vector vs32_n03_x1_sum  = Q6_Vw_vdmpyacc_VwVhRh_sat(vs32_n03_x1_even, Q6_V_hi_W(ws16_n03_x1), 0xff4004c0);
    HVX_Vector vs32_n03_x2_sum  = Q6_Vw_vdmpyacc_VwVhRh_sat(vs32_n03_x2_even, Q6_V_hi_W(ws16_n03_x2), 0xff4004c0);
    HVX_Vector vs32_n03_x3_sum  = Q6_Vw_vdmpyacc_VwVhRh_sat(vs32_n03_x3_even, Q6_V_hi_W(ws16_n03_x3), 0xff4004c0);

    vs32_n03_x0_sum = Q6_Vw_vmul32xhi16_VwVw(vs32_n03_x0_sum, vs16_coef0);
    vs32_n03_x1_sum = Q6_Vw_vmul32xhi16_VwVw(vs32_n03_x1_sum, vs16_coef0);
    vs32_n03_x2_sum = Q6_Vw_vmul32xhi16_VwVw(vs32_n03_x2_sum, vs16_coef0);
    vs32_n03_x3_sum = Q6_Vw_vmul32xhi16_VwVw(vs32_n03_x3_sum, vs16_coef0);

    // n1n2
    HVX_VectorPair ws16_n12_x0  = Q6_Wh_vadd_WhWh(Q6_Wh_vsxt_Vb(vs8_n1_x0_src), Q6_Wh_vsxt_Vb(vs8_n2_x0_src));
    HVX_VectorPair ws16_n12_x1  = Q6_Wh_vadd_WhWh(Q6_Wh_vsxt_Vb(vs8_n1_x1_src), Q6_Wh_vsxt_Vb(vs8_n2_x1_src));
    HVX_VectorPair ws16_n12_x2  = Q6_Wh_vadd_WhWh(Q6_Wh_vsxt_Vb(vs8_n1_x2_src), Q6_Wh_vsxt_Vb(vs8_n2_x2_src));
    HVX_VectorPair ws16_n12_x3  = Q6_Wh_vadd_WhWh(Q6_Wh_vsxt_Vb(vs8_n1_x3_src), Q6_Wh_vsxt_Vb(vs8_n2_x3_src));
    HVX_Vector vs32_n12_x0_even = Q6_Vw_vdmpy_VhRh_sat(Q6_V_lo_W(ws16_n12_x0), 0x04c0ff40);
    HVX_Vector vs32_n12_x1_even = Q6_Vw_vdmpy_VhRh_sat(Q6_V_lo_W(ws16_n12_x1), 0x04c0ff40);
    HVX_Vector vs32_n12_x2_even = Q6_Vw_vdmpy_VhRh_sat(Q6_V_lo_W(ws16_n12_x2), 0x04c0ff40);
    HVX_Vector vs32_n12_x3_even = Q6_Vw_vdmpy_VhRh_sat(Q6_V_lo_W(ws16_n12_x3), 0x04c0ff40);
    HVX_Vector vs32_n12_x0_sum  = Q6_Vw_vdmpyacc_VwVhRh_sat(vs32_n12_x0_even, Q6_V_hi_W(ws16_n12_x0), 0xff4004c0);
    HVX_Vector vs32_n12_x1_sum  = Q6_Vw_vdmpyacc_VwVhRh_sat(vs32_n12_x1_even, Q6_V_hi_W(ws16_n12_x1), 0xff4004c0);
    HVX_Vector vs32_n12_x2_sum  = Q6_Vw_vdmpyacc_VwVhRh_sat(vs32_n12_x2_even, Q6_V_hi_W(ws16_n12_x2), 0xff4004c0);
    HVX_Vector vs32_n12_x3_sum  = Q6_Vw_vdmpyacc_VwVhRh_sat(vs32_n12_x3_even, Q6_V_hi_W(ws16_n12_x3), 0xff4004c0);

    vs32_n12_x0_sum = Q6_Vw_vmul32xhi16_VwVw(vs32_n12_x0_sum, vs16_coef1);
    vs32_n12_x1_sum = Q6_Vw_vmul32xhi16_VwVw(vs32_n12_x1_sum, vs16_coef1);
    vs32_n12_x2_sum = Q6_Vw_vmul32xhi16_VwVw(vs32_n12_x2_sum, vs16_coef1);
    vs32_n12_x3_sum = Q6_Vw_vmul32xhi16_VwVw(vs32_n12_x3_sum, vs16_coef1);

    // sum
    HVX_Vector vs32_x0_sum  = Q6_Vw_vadd_VwVw(vs32_n03_x0_sum, vs32_n12_x0_sum);
    HVX_Vector vs32_x1_sum  = Q6_Vw_vadd_VwVw(vs32_n03_x1_sum, vs32_n12_x1_sum);
    HVX_Vector vs32_x2_sum  = Q6_Vw_vadd_VwVw(vs32_n03_x2_sum, vs32_n12_x2_sum);
    HVX_Vector vs32_x3_sum  = Q6_Vw_vadd_VwVw(vs32_n03_x3_sum, vs32_n12_x3_sum);
    HVX_Vector vs16_result0 = Q6_V_hi_W(Q6_W_vdeal_VVR(vs32_x1_sum, vs32_x0_sum, -2));
    HVX_Vector vs16_result1 = Q6_V_hi_W(Q6_W_vdeal_VVR(vs32_x3_sum, vs32_x2_sum, -2));

    vs8_dst = Q6_Vb_vdeal_Vb(Q6_Vb_vasr_VhVhR_rnd_sat(vs16_result1, vs16_result0, 6));
}

template <typename Tp, typename std::enable_if<std::is_same<DT_U16, Tp>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID ResizeCuDnX4Core(HVX_Vector &vu16_n0_x0_src, HVX_Vector &vu16_n0_x1_src, HVX_Vector &vu16_n0_x2_src, HVX_Vector &vu16_n0_x3_src,
                                            HVX_Vector &vu16_n1_x0_src, HVX_Vector &vu16_n1_x1_src, HVX_Vector &vu16_n1_x2_src, HVX_Vector &vu16_n1_x3_src,
                                            HVX_Vector &vu16_n2_x0_src, HVX_Vector &vu16_n2_x1_src, HVX_Vector &vu16_n2_x2_src, HVX_Vector &vu16_n2_x3_src,
                                            HVX_Vector &vu16_n3_x0_src, HVX_Vector &vu16_n3_x1_src, HVX_Vector &vu16_n3_x2_src, HVX_Vector &vu16_n3_x3_src,
                                            HVX_Vector &vu16_dst)
{
    HVX_Vector vs16_alpha0  = Q6_Vh_vsplat_R(-3072);
    HVX_Vector vs16_alpha1  = Q6_Vh_vsplat_R(19456);
    HVX_Vector vs32_beta0   = Q6_V_vsplat_R(-3072);
    HVX_Vector vs32_beta1   = Q6_V_vsplat_R(19456);
    HVX_VectorPair ws64_rnd = Q6_W_vcombine_VV(Q6_V_vzero(), Q6_V_vsplat_R(1 << 29));

    // n0n3
    HVX_VectorPair wu16_x01     = Q6_W_vdeal_VVR(vu16_n0_x1_src, vu16_n0_x0_src, -2);
    HVX_VectorPair wu16_x23     = Q6_W_vdeal_VVR(vu16_n0_x3_src, vu16_n0_x2_src, -2);
    HVX_VectorPair wu16_r02     = Q6_W_vdeal_VVR(Q6_V_lo_W(wu16_x23), Q6_V_lo_W(wu16_x01), -2);
    HVX_VectorPair wu16_r13     = Q6_W_vdeal_VVR(Q6_V_hi_W(wu16_x23), Q6_V_hi_W(wu16_x01), -2);
    HVX_VectorPair ws32_r0_mul  = Q6_Ww_vmpy_VhVuh(vs16_alpha0, Q6_V_lo_W(wu16_r02));
    HVX_VectorPair ws32_r1_mul  = Q6_Ww_vmpy_VhVuh(vs16_alpha1, Q6_V_lo_W(wu16_r13));
    HVX_VectorPair ws32_r2_mul  = Q6_Ww_vmpy_VhVuh(vs16_alpha1, Q6_V_hi_W(wu16_r02));
    HVX_VectorPair ws32_r3_mul  = Q6_Ww_vmpy_VhVuh(vs16_alpha0, Q6_V_hi_W(wu16_r13));
    HVX_VectorPair ws32_r03_sum = Q6_Ww_vadd_WwWw(ws32_r0_mul, ws32_r3_mul);
    HVX_VectorPair ws32_r12_sum = Q6_Ww_vadd_WwWw(ws32_r1_mul, ws32_r2_mul);
    HVX_VectorPair ws32_sum0    = Q6_Ww_vadd_WwWw(ws32_r03_sum, ws32_r12_sum);

    wu16_x01                    = Q6_W_vdeal_VVR(vu16_n3_x1_src, vu16_n3_x0_src, -2);
    wu16_x23                    = Q6_W_vdeal_VVR(vu16_n3_x3_src, vu16_n3_x2_src, -2);
    wu16_r02                    = Q6_W_vdeal_VVR(Q6_V_lo_W(wu16_x23), Q6_V_lo_W(wu16_x01), -2);
    wu16_r13                    = Q6_W_vdeal_VVR(Q6_V_hi_W(wu16_x23), Q6_V_hi_W(wu16_x01), -2);
    ws32_r0_mul                 = Q6_Ww_vmpy_VhVuh(vs16_alpha0, Q6_V_lo_W(wu16_r02));
    ws32_r1_mul                 = Q6_Ww_vmpy_VhVuh(vs16_alpha1, Q6_V_lo_W(wu16_r13));
    ws32_r2_mul                 = Q6_Ww_vmpy_VhVuh(vs16_alpha1, Q6_V_hi_W(wu16_r02));
    ws32_r3_mul                 = Q6_Ww_vmpy_VhVuh(vs16_alpha0, Q6_V_hi_W(wu16_r13));
    ws32_r03_sum                = Q6_Ww_vadd_WwWw(ws32_r0_mul, ws32_r3_mul);
    ws32_r12_sum                = Q6_Ww_vadd_WwWw(ws32_r1_mul, ws32_r2_mul);
    HVX_VectorPair ws32_sum1    = Q6_Ww_vadd_WwWw(ws32_r03_sum, ws32_r12_sum);

    HVX_VectorPair ws32_sum       = Q6_Ww_vadd_WwWw(ws32_sum0, ws32_sum1);
    HVX_VectorPair ws64_n03_sum_l = Q6_Wd_vmul_VwVw(Q6_V_lo_W(ws32_sum), vs32_beta0);
    HVX_VectorPair ws64_n03_sum_h = Q6_Wd_vmul_VwVw(Q6_V_hi_W(ws32_sum), vs32_beta0);

    // n1n2
    wu16_x01     = Q6_W_vdeal_VVR(vu16_n1_x1_src, vu16_n1_x0_src, -2);
    wu16_x23     = Q6_W_vdeal_VVR(vu16_n1_x3_src, vu16_n1_x2_src, -2);
    wu16_r02     = Q6_W_vdeal_VVR(Q6_V_lo_W(wu16_x23), Q6_V_lo_W(wu16_x01), -2);
    wu16_r13     = Q6_W_vdeal_VVR(Q6_V_hi_W(wu16_x23), Q6_V_hi_W(wu16_x01), -2);
    ws32_r0_mul  = Q6_Ww_vmpy_VhVuh(vs16_alpha0, Q6_V_lo_W(wu16_r02));
    ws32_r1_mul  = Q6_Ww_vmpy_VhVuh(vs16_alpha1, Q6_V_lo_W(wu16_r13));
    ws32_r2_mul  = Q6_Ww_vmpy_VhVuh(vs16_alpha1, Q6_V_hi_W(wu16_r02));
    ws32_r3_mul  = Q6_Ww_vmpy_VhVuh(vs16_alpha0, Q6_V_hi_W(wu16_r13));
    ws32_r03_sum = Q6_Ww_vadd_WwWw(ws32_r0_mul, ws32_r3_mul);
    ws32_r12_sum = Q6_Ww_vadd_WwWw(ws32_r1_mul, ws32_r2_mul);
    ws32_sum0    = Q6_Ww_vadd_WwWw(ws32_r03_sum, ws32_r12_sum);

    wu16_x01     = Q6_W_vdeal_VVR(vu16_n2_x1_src, vu16_n2_x0_src, -2);
    wu16_x23     = Q6_W_vdeal_VVR(vu16_n2_x3_src, vu16_n2_x2_src, -2);
    wu16_r02     = Q6_W_vdeal_VVR(Q6_V_lo_W(wu16_x23), Q6_V_lo_W(wu16_x01), -2);
    wu16_r13     = Q6_W_vdeal_VVR(Q6_V_hi_W(wu16_x23), Q6_V_hi_W(wu16_x01), -2);
    ws32_r0_mul  = Q6_Ww_vmpy_VhVuh(vs16_alpha0, Q6_V_lo_W(wu16_r02));
    ws32_r1_mul  = Q6_Ww_vmpy_VhVuh(vs16_alpha1, Q6_V_lo_W(wu16_r13));
    ws32_r2_mul  = Q6_Ww_vmpy_VhVuh(vs16_alpha1, Q6_V_hi_W(wu16_r02));
    ws32_r3_mul  = Q6_Ww_vmpy_VhVuh(vs16_alpha0, Q6_V_hi_W(wu16_r13));
    ws32_r03_sum = Q6_Ww_vadd_WwWw(ws32_r0_mul, ws32_r3_mul);
    ws32_r12_sum = Q6_Ww_vadd_WwWw(ws32_r1_mul, ws32_r2_mul);
    ws32_sum1    = Q6_Ww_vadd_WwWw(ws32_r03_sum, ws32_r12_sum);

    ws32_sum                      = Q6_Ww_vadd_WwWw(ws32_sum0, ws32_sum1);
    HVX_VectorPair ws64_n12_sum_l = Q6_Wd_vmul_VwVw(Q6_V_lo_W(ws32_sum), vs32_beta1);
    HVX_VectorPair ws64_n12_sum_h = Q6_Wd_vmul_VwVw(Q6_V_hi_W(ws32_sum), vs32_beta1);

    // sum
    HVX_VectorPair ws64_sum_l = Q6_Wd_vadd_WdWd(ws64_n03_sum_l, ws64_n12_sum_l);
    HVX_VectorPair ws64_sum_h = Q6_Wd_vadd_WdWd(ws64_n03_sum_h, ws64_n12_sum_h);
    ws64_sum_l = Q6_Wd_vadd_WdWd(ws64_sum_l, ws64_rnd);
    ws64_sum_h = Q6_Wd_vadd_WdWd(ws64_sum_h, ws64_rnd);

    HVX_Vector vu32_sum_l = Q6_V_vor_VV(Q6_Vw_vasl_VwR(Q6_V_hi_W(ws64_sum_l), 2), Q6_Vuw_vlsr_VuwR(Q6_V_lo_W(ws64_sum_l), 30));
    HVX_Vector vu32_sum_h = Q6_V_vor_VV(Q6_Vw_vasl_VwR(Q6_V_hi_W(ws64_sum_h), 2), Q6_Vuw_vlsr_VuwR(Q6_V_lo_W(ws64_sum_h), 30));
    vu32_sum_l = Q6_Vw_vmax_VwVw(vu32_sum_l, Q6_V_vzero());
    vu32_sum_h = Q6_Vw_vmax_VwVw(vu32_sum_h, Q6_V_vzero());
    vu16_dst = Q6_Vuh_vsat_VuwVuw(vu32_sum_h, vu32_sum_l);
}

template <typename Tp, typename std::enable_if<std::is_same<DT_S16, Tp>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID ResizeCuDnX4Core(HVX_Vector &vs16_n0_x0_src, HVX_Vector &vs16_n0_x1_src, HVX_Vector &vs16_n0_x2_src, HVX_Vector &vs16_n0_x3_src,
                                            HVX_Vector &vs16_n1_x0_src, HVX_Vector &vs16_n1_x1_src, HVX_Vector &vs16_n1_x2_src, HVX_Vector &vs16_n1_x3_src,
                                            HVX_Vector &vs16_n2_x0_src, HVX_Vector &vs16_n2_x1_src, HVX_Vector &vs16_n2_x2_src, HVX_Vector &vs16_n2_x3_src,
                                            HVX_Vector &vs16_n3_x0_src, HVX_Vector &vs16_n3_x1_src, HVX_Vector &vs16_n3_x2_src, HVX_Vector &vs16_n3_x3_src,
                                            HVX_Vector &vs16_dst)
{
    HVX_Vector vs32_beta0   = Q6_V_vsplat_R(-3072);
    HVX_Vector vs32_beta1   = Q6_V_vsplat_R(19456);
    HVX_VectorPair ws64_rnd = Q6_W_vcombine_VV(Q6_V_vzero(), Q6_V_vsplat_R(1 << 29));

    // n0n3
    HVX_VectorPair ws16_x01    = Q6_W_vdeal_VVR(vs16_n0_x1_src, vs16_n0_x0_src, -4);
    HVX_VectorPair ws16_x23    = Q6_W_vdeal_VVR(vs16_n0_x3_src, vs16_n0_x2_src, -4);
    HVX_Vector vs32_x01_even   = Q6_Vw_vdmpy_VhRh_sat(Q6_V_lo_W(ws16_x01), 0x4c00f400);
    HVX_Vector vs32_x23_even   = Q6_Vw_vdmpy_VhRh_sat(Q6_V_lo_W(ws16_x23), 0x4c00f400);
    HVX_Vector vs32_n0_x01_sum = Q6_Vw_vdmpyacc_VwVhRh_sat(vs32_x01_even, Q6_V_hi_W(ws16_x01), 0xf4004c00);
    HVX_Vector vs32_n0_x23_sum = Q6_Vw_vdmpyacc_VwVhRh_sat(vs32_x23_even, Q6_V_hi_W(ws16_x23), 0xf4004c00);

    ws16_x01                   = Q6_W_vdeal_VVR(vs16_n3_x1_src, vs16_n3_x0_src, -4);
    ws16_x23                   = Q6_W_vdeal_VVR(vs16_n3_x3_src, vs16_n3_x2_src, -4);
    vs32_x01_even              = Q6_Vw_vdmpy_VhRh_sat(Q6_V_lo_W(ws16_x01), 0x4c00f400);
    vs32_x23_even              = Q6_Vw_vdmpy_VhRh_sat(Q6_V_lo_W(ws16_x23), 0x4c00f400);
    HVX_Vector vs32_n3_x01_sum = Q6_Vw_vdmpyacc_VwVhRh_sat(vs32_x01_even, Q6_V_hi_W(ws16_x01), 0xf4004c00);
    HVX_Vector vs32_n3_x23_sum = Q6_Vw_vdmpyacc_VwVhRh_sat(vs32_x23_even, Q6_V_hi_W(ws16_x23), 0xf4004c00);

    HVX_VectorPair ws64_n03_x01_sum = Q6_Wd_vmul_VwVw(Q6_Vw_vadd_VwVw(vs32_n0_x01_sum, vs32_n3_x01_sum), vs32_beta0);
    HVX_VectorPair ws64_n03_x23_sum = Q6_Wd_vmul_VwVw(Q6_Vw_vadd_VwVw(vs32_n0_x23_sum, vs32_n3_x23_sum), vs32_beta0);

    // n1n2
    ws16_x01                   = Q6_W_vdeal_VVR(vs16_n1_x1_src, vs16_n1_x0_src, -4);
    ws16_x23                   = Q6_W_vdeal_VVR(vs16_n1_x3_src, vs16_n1_x2_src, -4);
    vs32_x01_even              = Q6_Vw_vdmpy_VhRh_sat(Q6_V_lo_W(ws16_x01), 0x4c00f400);
    vs32_x23_even              = Q6_Vw_vdmpy_VhRh_sat(Q6_V_lo_W(ws16_x23), 0x4c00f400);
    HVX_Vector vs32_n1_x01_sum = Q6_Vw_vdmpyacc_VwVhRh_sat(vs32_x01_even, Q6_V_hi_W(ws16_x01), 0xf4004c00);
    HVX_Vector vs32_n1_x23_sum = Q6_Vw_vdmpyacc_VwVhRh_sat(vs32_x23_even, Q6_V_hi_W(ws16_x23), 0xf4004c00);

    ws16_x01                   = Q6_W_vdeal_VVR(vs16_n2_x1_src, vs16_n2_x0_src, -4);
    ws16_x23                   = Q6_W_vdeal_VVR(vs16_n2_x3_src, vs16_n2_x2_src, -4);
    vs32_x01_even              = Q6_Vw_vdmpy_VhRh_sat(Q6_V_lo_W(ws16_x01), 0x4c00f400);
    vs32_x23_even              = Q6_Vw_vdmpy_VhRh_sat(Q6_V_lo_W(ws16_x23), 0x4c00f400);
    HVX_Vector vs32_n2_x01_sum = Q6_Vw_vdmpyacc_VwVhRh_sat(vs32_x01_even, Q6_V_hi_W(ws16_x01), 0xf4004c00);
    HVX_Vector vs32_n2_x23_sum = Q6_Vw_vdmpyacc_VwVhRh_sat(vs32_x23_even, Q6_V_hi_W(ws16_x23), 0xf4004c00);

    HVX_VectorPair ws64_n12_x01_sum = Q6_Wd_vmul_VwVw(Q6_Vw_vadd_VwVw(vs32_n1_x01_sum, vs32_n2_x01_sum), vs32_beta1);
    HVX_VectorPair ws64_n12_x23_sum = Q6_Wd_vmul_VwVw(Q6_Vw_vadd_VwVw(vs32_n1_x23_sum, vs32_n2_x23_sum), vs32_beta1);

    // sum
    HVX_VectorPair ws64_x01_sum = Q6_Wd_vadd_WdWd(ws64_n03_x01_sum, ws64_n12_x01_sum);
    HVX_VectorPair ws64_x23_sum = Q6_Wd_vadd_WdWd(ws64_n03_x23_sum, ws64_n12_x23_sum);
    ws64_x01_sum = Q6_Wd_vadd_WdWd(ws64_x01_sum, ws64_rnd);
    ws64_x23_sum = Q6_Wd_vadd_WdWd(ws64_x23_sum, ws64_rnd);

    HVX_Vector vs32_x01_sum = Q6_V_vor_VV(Q6_Vw_vasl_VwR(Q6_V_hi_W(ws64_x01_sum), 2), Q6_Vuw_vlsr_VuwR(Q6_V_lo_W(ws64_x01_sum), 30));
    HVX_Vector vs32_x23_sum = Q6_V_vor_VV(Q6_Vw_vasl_VwR(Q6_V_hi_W(ws64_x23_sum), 2), Q6_Vuw_vlsr_VuwR(Q6_V_lo_W(ws64_x23_sum), 30));
    vs16_dst = Q6_Vh_vdeal_Vh(Q6_Vh_vsat_VwVw(vs32_x23_sum, vs32_x01_sum));
}

template<typename Tp, DT_S32 C>
static DT_VOID ResizeCuDnX2ZeroRow(const Tp *src_c, Tp *dst_c, ResizeCuFastVtcmBuffer *vtcm_buffer, DT_S32 iwidth, DT_S32 owidth, DT_S32 istride)
{
    using MVType    = typename MVHvxVector<C>::Type;
    using HVX_VType = typename std::conditional<1 == sizeof(Tp), HVX_VectorPair, HVX_Vector>::type;
    using BufType   = typename std::conditional<1 == sizeof(Tp), DT_S32, DT_S64>::type;

    DT_S32 elem_counts  = AURA_HVLEN / sizeof(Tp);
    DT_S32 width_align  = (owidth - 2)  & (-elem_counts);
    DT_S32 row_pre_size = AURA_ALIGN(owidth * C * sizeof(DT_S32), AURA_HVLEN);

    constexpr DT_S32 SHIFT_BITS  = ResizeCuDnCoefTraits<sizeof(Tp)>::SHIFT_BITS;
    constexpr DT_S32 COEF0       = ResizeCuDnCoefTraits<sizeof(Tp)>::COEF0;
    constexpr DT_S32 COEF1       = ResizeCuDnCoefTraits<sizeof(Tp)>::COEF1;
    constexpr DT_S32 COEF_BORDER = ResizeCuDnCoefTraits<sizeof(Tp)>::COEF_BORDER;

    const Tp *src_n0 = src_c  + (istride / sizeof(Tp));
    const Tp *src_n1 = src_n0 + (istride / sizeof(Tp));

    DT_S32 *row2_ptr      = reinterpret_cast<DT_S32*>(vtcm_buffer->row2_ptr);
    DT_S32 *row3_ptr      = reinterpret_cast<DT_S32*>(vtcm_buffer->row3_ptr);
    BufType *row_head     = reinterpret_cast<BufType*>(vtcm_buffer->row_head);
    DT_S32 *row2_back_ptr = reinterpret_cast<DT_S32*>(vtcm_buffer->row2_ptr + row_pre_size);
    DT_S32 *row3_back_ptr = reinterpret_cast<DT_S32*>(vtcm_buffer->row3_ptr + row_pre_size);

    MVType mv_c_x0_src, mv_n0_x0_src, mv_n1_x0_src;
    MVType mv_c_x1_src, mv_n0_x1_src, mv_n1_x1_src;
    MVType mv_c_x2_src, mv_n0_x2_src, mv_n1_x2_src;
    MVType mv_c_dst;

    HVX_Vector v_coef0 = Q6_Vh_vsplat_R(COEF0);
    HVX_Vector v_coef1 = Q6_Vh_vsplat_R(COEF1);

    BufType *row2_head = row_head + C * 4;
    BufType *row3_head = row_head + C * 6;

    #pragma unroll(C)
    for (DT_S32 ch = 0; ch < C; ch++)
    {
        BufType row0_head;
        DT_S32 id0 = ch;
        DT_S32 id1 = C + ch;
        DT_S32 id2 = 2 * C + ch;

        row0_head      = src_c[id0]  * COEF_BORDER + src_c[id1]  * COEF1 + src_c[id2]  * COEF0;
        row2_head[id0] = src_n0[id0] * COEF_BORDER + src_n0[id1] * COEF1 + src_n0[id2] * COEF0;
        row3_head[id0] = src_n1[id0] * COEF_BORDER + src_n1[id1] * COEF1 + src_n1[id2] * COEF0;

        dst_c[id0] = ResizeCuSaturateCast<Tp, BufType>(row0_head * COEF_BORDER + row2_head[id0] * COEF1 + row3_head[id0] * COEF0, SHIFT_BITS);
    }

    HVX_VType *v_row2      = (HVX_VType *)(row2_ptr);
    HVX_VType *v_row3      = (HVX_VType *)(row3_ptr);
    HVX_VType *v_back_row2 = (HVX_VType *)(row2_back_ptr);
    HVX_VType *v_back_row3 = (HVX_VType *)(row3_back_ptr);
    HVX_VType vs32_c_result_l, vs32_n0_result_l, vs32_n1_result_l;
    HVX_VType vs32_c_result_h, vs32_n0_result_h, vs32_n1_result_h;

    vload(src_c  + C, mv_c_x0_src);
    vload(src_n0 + C, mv_n0_x0_src);
    vload(src_n1 + C, mv_n1_x0_src);

    for (DT_S32 x = 0; x < width_align; x += elem_counts)
    {
        vload(src_c  + (x * 2 + elem_counts + 1) * C, mv_c_x1_src);
        vload(src_n0 + (x * 2 + elem_counts + 1) * C, mv_n0_x1_src);
        vload(src_n1 + (x * 2 + elem_counts + 1) * C, mv_n1_x1_src);

        vload(src_c  + (x * 2 + (elem_counts << 1) + 1) * C, mv_c_x2_src);
        vload(src_n0 + (x * 2 + (elem_counts << 1) + 1) * C, mv_n0_x2_src);
        vload(src_n1 + (x * 2 + (elem_counts << 1) + 1) * C, mv_n1_x2_src);

        #pragma unroll(C)
        for (DT_S32 ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuDnX2HCore<Tp, HVX_VType>(mv_c_x0_src.val[ch],  mv_c_x1_src.val[ch],  mv_c_x2_src.val[ch],  vs32_c_result_l,  vs32_c_result_h,  v_coef0, v_coef1);
            ResizeCuDnX2HCore<Tp, HVX_VType>(mv_n0_x0_src.val[ch], mv_n0_x1_src.val[ch], mv_n0_x2_src.val[ch], vs32_n0_result_l, vs32_n0_result_h, v_coef0, v_coef1);
            ResizeCuDnX2HCore<Tp, HVX_VType>(mv_n1_x0_src.val[ch], mv_n1_x1_src.val[ch], mv_n1_x2_src.val[ch], vs32_n1_result_l, vs32_n1_result_h, v_coef0, v_coef1);

            *v_row2++ = vs32_n0_result_l;
            *v_row2++ = vs32_n0_result_h;
            *v_row3++ = vs32_n1_result_l;
            *v_row3++ = vs32_n1_result_h;

            // col
            ResizeCuDnX2VCore<Tp, HVX_VType>(vs32_c_result_l, vs32_c_result_h, vs32_n0_result_l, vs32_n0_result_h, vs32_n1_result_l, vs32_n1_result_h,
                                             mv_c_dst.val[ch], COEF_BORDER, COEF1, COEF0);
        }

        vstore(dst_c + (x + 1) * C, mv_c_dst);

        mv_c_x0_src  = mv_c_x2_src;
        mv_n0_x0_src = mv_n0_x2_src;
        mv_n1_x0_src = mv_n1_x2_src;
    }

    if (width_align < owidth - 2)
    {
        DT_S32 dx = owidth - 1 - elem_counts;
        DT_S32 sx = iwidth - 3 - (elem_counts << 1);

        vload(src_c  + sx * C, mv_c_x0_src);
        vload(src_n0 + sx * C, mv_n0_x0_src);
        vload(src_n1 + sx * C, mv_n1_x0_src);

        vload(src_c  + (sx + elem_counts) * C, mv_c_x1_src);
        vload(src_n0 + (sx + elem_counts) * C, mv_n0_x1_src);
        vload(src_n1 + (sx + elem_counts) * C, mv_n1_x1_src);

        #pragma unroll(C)
        for (DT_S32 ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuDnX2BorderHCore<Tp, HVX_VType>(mv_c_x0_src.val[ch],  mv_c_x1_src.val[ch],  src_c[(iwidth - 2)  * C + ch], vs32_c_result_l,  vs32_c_result_h,  v_coef0, v_coef1);
            ResizeCuDnX2BorderHCore<Tp, HVX_VType>(mv_n0_x0_src.val[ch], mv_n0_x1_src.val[ch], src_n0[(iwidth - 2) * C + ch], vs32_n0_result_l, vs32_n0_result_h, v_coef0, v_coef1);
            ResizeCuDnX2BorderHCore<Tp, HVX_VType>(mv_n1_x0_src.val[ch], mv_n1_x1_src.val[ch], src_n1[(iwidth - 2) * C + ch], vs32_n1_result_l, vs32_n1_result_h, v_coef0, v_coef1);

            *v_back_row2++ = vs32_n0_result_l;
            *v_back_row2++ = vs32_n0_result_h;
            *v_back_row3++ = vs32_n1_result_l;
            *v_back_row3++ = vs32_n1_result_h;

            // col
            ResizeCuDnX2VCore<Tp, HVX_VType>(vs32_c_result_l, vs32_c_result_h, vs32_n0_result_l, vs32_n0_result_h, vs32_n1_result_l, vs32_n1_result_h,
                                             mv_c_dst.val[ch], COEF_BORDER, COEF1, COEF0);
        }

        vstore(dst_c + dx * C, mv_c_dst);
    }

    #pragma unroll(C)
    for (DT_S32 ch = 0; ch < C; ch++)
    {
        BufType row0_head;
        DT_S32 id0 = (iwidth - 1) * C + ch;
        DT_S32 id1 = (iwidth - 2) * C + ch;
        DT_S32 id2 = (iwidth - 3) * C + ch;

        row0_head         = src_c[id2]  * COEF0 + src_c[id1]  * COEF1 + src_c[id0]  * COEF_BORDER;
        row2_head[C + ch] = src_n0[id2] * COEF0 + src_n0[id1] * COEF1 + src_n0[id0] * COEF_BORDER;
        row3_head[C + ch] = src_n1[id2] * COEF0 + src_n1[id1] * COEF1 + src_n1[id0] * COEF_BORDER;

        dst_c[(owidth - 1) * C + ch] = ResizeCuSaturateCast<Tp, BufType>(row0_head * COEF_BORDER + row2_head[C + ch] * COEF1 + row3_head[C + ch] * COEF0, SHIFT_BITS);
    }
}

template<typename Tp, DT_S32 C>
static DT_VOID ResizeCuDnX2UpBorder(const Tp *src_c, Tp *dst_c, ResizeCuFastVtcmBuffer *vtcm_buffer, DT_S32 iwidth, DT_S32 owidth, DT_S32 istride)
{
    using MVType    = typename MVHvxVector<C>::Type;
    using HVX_VType = typename std::conditional<1 == sizeof(Tp), HVX_VectorPair, HVX_Vector>::type;
    using BufType   = typename std::conditional<1 == sizeof(Tp), DT_S32, DT_S64>::type;

    DT_S32 elem_counts  = AURA_HVLEN / sizeof(Tp);
    DT_S32 width_align  = (owidth - 2)  & (-elem_counts);
    DT_S32 row_pre_size = AURA_ALIGN(owidth * C * sizeof(DT_S32), AURA_HVLEN);

    constexpr DT_S32 SHIFT_BITS  = ResizeCuDnCoefTraits<sizeof(Tp)>::SHIFT_BITS;
    constexpr DT_S32 COEF0       = ResizeCuDnCoefTraits<sizeof(Tp)>::COEF0;
    constexpr DT_S32 COEF1       = ResizeCuDnCoefTraits<sizeof(Tp)>::COEF1;
    constexpr DT_S32 COEF_BORDER = ResizeCuDnCoefTraits<sizeof(Tp)>::COEF_BORDER;

    const Tp *src_n0 = src_c  + (istride / sizeof(Tp));
    const Tp *src_n1 = src_n0 + (istride / sizeof(Tp));
    const Tp *src_n2 = src_n1 + (istride / sizeof(Tp));

    DT_S32 *row2_ptr      = reinterpret_cast<DT_S32*>(vtcm_buffer->row2_ptr);
    DT_S32 *row3_ptr      = reinterpret_cast<DT_S32*>(vtcm_buffer->row3_ptr);
    BufType *row_head     = reinterpret_cast<BufType*>(vtcm_buffer->row_head);
    DT_S32 *row2_back_ptr = reinterpret_cast<DT_S32*>(vtcm_buffer->row2_ptr + row_pre_size);
    DT_S32 *row3_back_ptr = reinterpret_cast<DT_S32*>(vtcm_buffer->row3_ptr + row_pre_size);

    MVType mv_c_x0_src, mv_n0_x0_src, mv_n1_x0_src, mv_n2_x0_src;
    MVType mv_c_x1_src, mv_n0_x1_src, mv_n1_x1_src, mv_n2_x1_src;
    MVType mv_c_x2_src, mv_n0_x2_src, mv_n1_x2_src, mv_n2_x2_src;
    MVType mv_c_dst;
    HVX_Vector v_coef0 = Q6_Vh_vsplat_R(COEF0);
    HVX_Vector v_coef1 = Q6_Vh_vsplat_R(COEF1);

    BufType *row2_head = row_head + C * 4;
    BufType *row3_head = row_head + C * 6;

    #pragma unroll(C)
    for (DT_S32 ch = 0; ch < C; ch++)
    {
        BufType row0_head, row1_head;
        DT_S32 id0 = ch;
        DT_S32 id1 = C + ch;
        DT_S32 id2 = 2 * C + ch;

        row0_head      = src_c[id0]  * COEF_BORDER + src_c[id1]  * COEF1 + src_c[id2]  * COEF0;
        row1_head      = src_n0[id0] * COEF_BORDER + src_n0[id1] * COEF1 + src_n0[id2] * COEF0;
        row2_head[id0] = src_n1[id0] * COEF_BORDER + src_n1[id1] * COEF1 + src_n1[id2] * COEF0;
        row3_head[id0] = src_n2[id0] * COEF_BORDER + src_n2[id1] * COEF1 + src_n2[id2] * COEF0;

        dst_c[id0] = ResizeCuSaturateCast<Tp, BufType>((row0_head + row3_head[id0]) * COEF0 + (row1_head + row2_head[id0]) * COEF1, SHIFT_BITS);
    }

    HVX_VType *v_row2      = (HVX_VType *)(row2_ptr);
    HVX_VType *v_row3      = (HVX_VType *)(row3_ptr);
    HVX_VType *v_back_row2 = (HVX_VType *)(row2_back_ptr);
    HVX_VType *v_back_row3 = (HVX_VType *)(row3_back_ptr);
    HVX_VType vs32_c_result_l, vs32_n0_result_l, vs32_n1_result_l, ws32_n2_result_l;
    HVX_VType vs32_c_result_h, vs32_n0_result_h, vs32_n1_result_h, ws32_n2_result_h;

    vload(src_c  + C, mv_c_x0_src);
    vload(src_n0 + C, mv_n0_x0_src);
    vload(src_n1 + C, mv_n1_x0_src);
    vload(src_n2 + C, mv_n2_x0_src);

    for (DT_S32 x = 0; x < width_align; x += elem_counts)
    {
        vload(src_c  + (x * 2 + elem_counts + 1) * C, mv_c_x1_src);
        vload(src_n0 + (x * 2 + elem_counts + 1) * C, mv_n0_x1_src);
        vload(src_n1 + (x * 2 + elem_counts + 1) * C, mv_n1_x1_src);
        vload(src_n2 + (x * 2 + elem_counts + 1) * C, mv_n2_x1_src);

        vload(src_c  + (x * 2 + (elem_counts << 1) + 1) * C, mv_c_x2_src);
        vload(src_n0 + (x * 2 + (elem_counts << 1) + 1) * C, mv_n0_x2_src);
        vload(src_n1 + (x * 2 + (elem_counts << 1) + 1) * C, mv_n1_x2_src);
        vload(src_n2 + (x * 2 + (elem_counts << 1) + 1) * C, mv_n2_x2_src);

        #pragma unroll(C)
        for (DT_S32 ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuDnX2HCore<Tp, HVX_VType>(mv_c_x0_src.val[ch],  mv_c_x1_src.val[ch],  mv_c_x2_src.val[ch],  vs32_c_result_l,  vs32_c_result_h,  v_coef0, v_coef1);
            ResizeCuDnX2HCore<Tp, HVX_VType>(mv_n0_x0_src.val[ch], mv_n0_x1_src.val[ch], mv_n0_x2_src.val[ch], vs32_n0_result_l, vs32_n0_result_h, v_coef0, v_coef1);
            ResizeCuDnX2HCore<Tp, HVX_VType>(mv_n1_x0_src.val[ch], mv_n1_x1_src.val[ch], mv_n1_x2_src.val[ch], vs32_n1_result_l, vs32_n1_result_h, v_coef0, v_coef1);
            ResizeCuDnX2HCore<Tp, HVX_VType>(mv_n2_x0_src.val[ch], mv_n2_x1_src.val[ch], mv_n2_x2_src.val[ch], ws32_n2_result_l, ws32_n2_result_h, v_coef0, v_coef1);

            *v_row2++ = vs32_n1_result_l;
            *v_row2++ = vs32_n1_result_h;
            *v_row3++ = ws32_n2_result_l;
            *v_row3++ = ws32_n2_result_h;

            // col
            ResizeCuDnX2VCore<Tp, HVX_VType>(vs32_c_result_l, vs32_c_result_h, vs32_n0_result_l, vs32_n0_result_h, vs32_n1_result_l, vs32_n1_result_h,
                                             ws32_n2_result_l, ws32_n2_result_h, mv_c_dst.val[ch]);
        }

        vstore(dst_c + (x + 1) * C, mv_c_dst);

        mv_c_x0_src  = mv_c_x2_src;
        mv_n0_x0_src = mv_n0_x2_src;
        mv_n1_x0_src = mv_n1_x2_src;
        mv_n2_x0_src = mv_n2_x2_src;
    }

    if (width_align < owidth - 2)
    {
        DT_S32 dx = owidth - 1 - elem_counts;
        DT_S32 sx = iwidth - 3 - (elem_counts << 1);

        vload(src_c  + sx * C, mv_c_x0_src);
        vload(src_n0 + sx * C, mv_n0_x0_src);
        vload(src_n1 + sx * C, mv_n1_x0_src);
        vload(src_n2 + sx * C, mv_n2_x0_src);

        vload(src_c  + (sx + elem_counts) * C, mv_c_x1_src);
        vload(src_n0 + (sx + elem_counts) * C, mv_n0_x1_src);
        vload(src_n1 + (sx + elem_counts) * C, mv_n1_x1_src);
        vload(src_n2 + (sx + elem_counts) * C, mv_n2_x1_src);

        #pragma unroll(C)
        for (DT_S32 ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuDnX2BorderHCore<Tp, HVX_VType>(mv_c_x0_src.val[ch],  mv_c_x1_src.val[ch],  src_c[(iwidth - 2)  * C + ch], vs32_c_result_l,  vs32_c_result_h,  v_coef0, v_coef1);
            ResizeCuDnX2BorderHCore<Tp, HVX_VType>(mv_n0_x0_src.val[ch], mv_n0_x1_src.val[ch], src_n0[(iwidth - 2) * C + ch], vs32_n0_result_l, vs32_n0_result_h, v_coef0, v_coef1);
            ResizeCuDnX2BorderHCore<Tp, HVX_VType>(mv_n1_x0_src.val[ch], mv_n1_x1_src.val[ch], src_n1[(iwidth - 2) * C + ch], vs32_n1_result_l, vs32_n1_result_h, v_coef0, v_coef1);
            ResizeCuDnX2BorderHCore<Tp, HVX_VType>(mv_n2_x0_src.val[ch], mv_n2_x1_src.val[ch], src_n2[(iwidth - 2) * C + ch], ws32_n2_result_l, ws32_n2_result_h, v_coef0, v_coef1);

            *v_back_row2++ = vs32_n1_result_l;
            *v_back_row2++ = vs32_n1_result_h;
            *v_back_row3++ = ws32_n2_result_l;
            *v_back_row3++ = ws32_n2_result_h;

            // col
            ResizeCuDnX2VCore<Tp, HVX_VType>(vs32_c_result_l, vs32_c_result_h, vs32_n0_result_l, vs32_n0_result_h, vs32_n1_result_l, vs32_n1_result_h,
                                             ws32_n2_result_l, ws32_n2_result_h, mv_c_dst.val[ch]);
        }

        vstore(dst_c + dx * C, mv_c_dst);
    }

    #pragma unroll(C)
    for (DT_S32 ch = 0; ch < C; ch++)
    {
        BufType row0_head, row1_head;
        DT_S32 id0 = (iwidth - 1) * C + ch;
        DT_S32 id1 = (iwidth - 2) * C + ch;
        DT_S32 id2 = (iwidth - 3) * C + ch;

        row0_head         = src_c[id2]  * (COEF0) + src_c[id1]  * COEF1 + src_c[id0]  * COEF_BORDER;
        row1_head         = src_n0[id2] * (COEF0) + src_n0[id1] * COEF1 + src_n0[id0] * COEF_BORDER;
        row2_head[C + ch] = src_n1[id2] * (COEF0) + src_n1[id1] * COEF1 + src_n1[id0] * COEF_BORDER;
        row3_head[C + ch] = src_n2[id2] * (COEF0) + src_n2[id1] * COEF1 + src_n2[id0] * COEF_BORDER;

        dst_c[(owidth - 1) * C + ch] = ResizeCuSaturateCast<Tp, BufType>((row0_head + row3_head[C + ch]) * COEF0 + (row1_head + row2_head[C + ch]) * COEF1, SHIFT_BITS);
    }
}

template<typename Tp, DT_S32 C>
static DT_VOID ResizeCuDnX2Row(const Tp *src_c, Tp *dst_c, ResizeCuFastVtcmBuffer *vtcm_buffer, DT_S32 iwidth, DT_S32 owidth, DT_S32 istride)
{
    using MVType    = typename MVHvxVector<C>::Type;
    using HVX_VType = typename std::conditional<1 == sizeof(Tp), HVX_VectorPair, HVX_Vector>::type;
    using BufType   = typename std::conditional<1 == sizeof(Tp), DT_S32, DT_S64>::type;

    DT_S32 elem_counts  = AURA_HVLEN / sizeof(Tp);
    DT_S32 width_align  = (owidth - 2)  & (-elem_counts);
    DT_S32 row_pre_size = AURA_ALIGN(owidth * C * sizeof(DT_S32), AURA_HVLEN);

    constexpr DT_S32 SHIFT_BITS  = ResizeCuDnCoefTraits<sizeof(Tp)>::SHIFT_BITS;
    constexpr DT_S32 COEF0       = ResizeCuDnCoefTraits<sizeof(Tp)>::COEF0;
    constexpr DT_S32 COEF1       = ResizeCuDnCoefTraits<sizeof(Tp)>::COEF1;
    constexpr DT_S32 COEF_BORDER = ResizeCuDnCoefTraits<sizeof(Tp)>::COEF_BORDER;

    const Tp *src_n0      = src_c + (istride / sizeof(Tp));
    DT_S32 *row0_ptr      = reinterpret_cast<DT_S32*>(vtcm_buffer->row2_ptr);
    DT_S32 *row1_ptr      = reinterpret_cast<DT_S32*>(vtcm_buffer->row3_ptr);
    DT_S32 *row2_ptr      = reinterpret_cast<DT_S32*>(vtcm_buffer->row0_ptr);
    DT_S32 *row3_ptr      = reinterpret_cast<DT_S32*>(vtcm_buffer->row1_ptr);
    BufType *row_head     = reinterpret_cast<BufType*>(vtcm_buffer->row_head);
    DT_S32 *row0_back_ptr = reinterpret_cast<DT_S32*>(vtcm_buffer->row2_ptr + row_pre_size);
    DT_S32 *row1_back_ptr = reinterpret_cast<DT_S32*>(vtcm_buffer->row3_ptr + row_pre_size);
    DT_S32 *row2_back_ptr = reinterpret_cast<DT_S32*>(vtcm_buffer->row0_ptr + row_pre_size);
    DT_S32 *row3_back_ptr = reinterpret_cast<DT_S32*>(vtcm_buffer->row1_ptr + row_pre_size);
    vtcm_buffer->row0_ptr = reinterpret_cast<DT_U8*>(row0_ptr);
    vtcm_buffer->row1_ptr = reinterpret_cast<DT_U8*>(row1_ptr);
    vtcm_buffer->row2_ptr = reinterpret_cast<DT_U8*>(row2_ptr);
    vtcm_buffer->row3_ptr = reinterpret_cast<DT_U8*>(row3_ptr);

    MVType mv_c_x0_src, mv_n0_x0_src;
    MVType mv_c_x1_src, mv_n0_x1_src;
    MVType mv_c_x2_src, mv_n0_x2_src;
    MVType mv_c_dst;
    HVX_Vector v_coef0 = Q6_Vh_vsplat_R(COEF0);
    HVX_Vector v_coef1 = Q6_Vh_vsplat_R(COEF1);

    BufType *row0_head = row_head;
    BufType *row1_head = row_head + C * 2;
    BufType *row2_head = row_head + C * 4;
    BufType *row3_head = row_head + C * 6;

    #pragma unroll(C)
    for (DT_S32 ch = 0; ch < C; ch++)
    {
        DT_S32 id0 = ch;
        DT_S32 id1 = C + ch;
        DT_S32 id2 = 2 * C + ch;

        row0_head[id0] = row2_head[id0];
        row1_head[id0] = row3_head[id0];
        row2_head[id0] = src_c[id0]  * COEF_BORDER + src_c[id1]  * COEF1 + src_c[id2]  * COEF0;
        row3_head[id0] = src_n0[id0] * COEF_BORDER + src_n0[id1] * COEF1 + src_n0[id2] * COEF0;

        dst_c[id0] = ResizeCuSaturateCast<Tp, BufType>((row0_head[id0] + row3_head[id0]) * COEF0 + (row1_head[id0] + row2_head[id0]) * COEF1, SHIFT_BITS);
    }

    HVX_VType *v_row0      = (HVX_VType *)(row0_ptr);
    HVX_VType *v_row1      = (HVX_VType *)(row1_ptr);
    HVX_VType *v_row2      = (HVX_VType *)(row2_ptr);
    HVX_VType *v_row3      = (HVX_VType *)(row3_ptr);
    HVX_VType *v_back_row0 = (HVX_VType *)(row0_back_ptr);
    HVX_VType *v_back_row1 = (HVX_VType *)(row1_back_ptr);
    HVX_VType *v_back_row2 = (HVX_VType *)(row2_back_ptr);
    HVX_VType *v_back_row3 = (HVX_VType *)(row3_back_ptr);
    HVX_VType ws32_p1_result_l, ws32_p0_result_l, vs32_c_result_l, vs32_n0_result_l;
    HVX_VType ws32_p1_result_h, ws32_p0_result_h, vs32_c_result_h, vs32_n0_result_h;

    vload(src_c  + C, mv_c_x0_src);
    vload(src_n0 + C, mv_n0_x0_src);

    for (DT_S32 x = 0; x < width_align; x += elem_counts)
    {
        vload(src_c  + (x * 2 + elem_counts + 1) * C, mv_c_x1_src);
        vload(src_n0 + (x * 2 + elem_counts + 1) * C, mv_n0_x1_src);
        vload(src_c  + (x * 2 + (elem_counts << 1) + 1) * C, mv_c_x2_src);
        vload(src_n0 + (x * 2 + (elem_counts << 1) + 1) * C, mv_n0_x2_src);

        #pragma unroll(C)
        for (DT_S32 ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuDnX2HCore<Tp, HVX_VType>(mv_c_x0_src.val[ch],  mv_c_x1_src.val[ch],  mv_c_x2_src.val[ch],  vs32_c_result_l,  vs32_c_result_h,  v_coef0, v_coef1);
            ResizeCuDnX2HCore<Tp, HVX_VType>(mv_n0_x0_src.val[ch], mv_n0_x1_src.val[ch], mv_n0_x2_src.val[ch], vs32_n0_result_l, vs32_n0_result_h, v_coef0, v_coef1);

            ws32_p1_result_l = *v_row0++;
            ws32_p1_result_h = *v_row0++;
            ws32_p0_result_l = *v_row1++;
            ws32_p0_result_h = *v_row1++;
            *v_row2++ = vs32_c_result_l;
            *v_row2++ = vs32_c_result_h;
            *v_row3++ = vs32_n0_result_l;
            *v_row3++ = vs32_n0_result_h;

            // col
            ResizeCuDnX2VCore<Tp, HVX_VType>(ws32_p1_result_l, ws32_p1_result_h, ws32_p0_result_l, ws32_p0_result_h, vs32_c_result_l, vs32_c_result_h,
                                             vs32_n0_result_l, vs32_n0_result_h, mv_c_dst.val[ch]);
        }

        vstore(dst_c + (x + 1) * C, mv_c_dst);

        mv_c_x0_src  = mv_c_x2_src;
        mv_n0_x0_src = mv_n0_x2_src;
    }

    if (width_align < owidth - 2)
    {
        DT_S32 dx = owidth - 1 - elem_counts;
        DT_S32 sx = iwidth - 3 - elem_counts * 2;

        vload(src_c  + sx * C, mv_c_x0_src);
        vload(src_n0 + sx * C, mv_n0_x0_src);
        vload(src_c  + (sx + elem_counts) * C, mv_c_x1_src);
        vload(src_n0 + (sx + elem_counts) * C, mv_n0_x1_src);

        #pragma unroll(C)
        for (DT_S32 ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuDnX2BorderHCore<Tp, HVX_VType>(mv_c_x0_src.val[ch],  mv_c_x1_src.val[ch],  src_c[(iwidth - 2)  * C + ch], vs32_c_result_l,  vs32_c_result_h,  v_coef0, v_coef1);
            ResizeCuDnX2BorderHCore<Tp, HVX_VType>(mv_n0_x0_src.val[ch], mv_n0_x1_src.val[ch], src_n0[(iwidth - 2) * C + ch], vs32_n0_result_l, vs32_n0_result_h, v_coef0, v_coef1);

            ws32_p1_result_l = *v_back_row0++;
            ws32_p1_result_h = *v_back_row0++;
            ws32_p0_result_l = *v_back_row1++;
            ws32_p0_result_h = *v_back_row1++;
            *v_back_row2++ = vs32_c_result_l;
            *v_back_row2++ = vs32_c_result_h;
            *v_back_row3++ = vs32_n0_result_l;
            *v_back_row3++ = vs32_n0_result_h;

            // col
            ResizeCuDnX2VCore<Tp, HVX_VType>(ws32_p1_result_l, ws32_p1_result_h, ws32_p0_result_l, ws32_p0_result_h, vs32_c_result_l, vs32_c_result_h,
                                             vs32_n0_result_l, vs32_n0_result_h, mv_c_dst.val[ch]);
        }

        vstore(dst_c + dx * C, mv_c_dst);
    }

    #pragma unroll(C)
    for (DT_S32 ch = 0; ch < C; ch++)
    {
        DT_S32 id0 = (iwidth - 1) * C + ch;
        DT_S32 id1 = (iwidth - 2) * C + ch;
        DT_S32 id2 = (iwidth - 3) * C + ch;

        row0_head[C + ch] = row2_head[C + ch];
        row1_head[C + ch] = row3_head[C + ch];
        row2_head[C + ch] = src_c[id2]  * COEF0 + src_c[id1]  * COEF1 + src_c[id0]  * COEF_BORDER;
        row3_head[C + ch] = src_n0[id2] * COEF0 + src_n0[id1] * COEF1 + src_n0[id0] * COEF_BORDER;

        dst_c[(owidth - 1) * C + ch] = ResizeCuSaturateCast<Tp, BufType>((row0_head[C + ch] + row3_head[C + ch]) * COEF0 + (row1_head[C + ch] + row2_head[C + ch]) * COEF1, SHIFT_BITS);
    }
}

template<typename Tp, DT_S32 C>
static DT_VOID ResizeCuDnX2BottomBorder(const Tp *src_c, Tp *dst_c, ResizeCuFastVtcmBuffer *vtcm_buffer, DT_S32 iwidth, DT_S32 owidth)
{
    using MVType    = typename MVHvxVector<C>::Type;
    using HVX_VType = typename std::conditional<1 == sizeof(Tp), HVX_VectorPair, HVX_Vector>::type;
    using BufType   = typename std::conditional<1 == sizeof(Tp), DT_S32, DT_S64>::type;

    DT_S32 elem_counts  = AURA_HVLEN / sizeof(Tp);
    DT_S32 width_align  = (owidth - 2)  & (-elem_counts);
    DT_S32 row_pre_size = AURA_ALIGN(owidth * C * sizeof(DT_S32), AURA_HVLEN);

    constexpr DT_S32 SHIFT_BITS  = ResizeCuDnCoefTraits<sizeof(Tp)>::SHIFT_BITS;
    constexpr DT_S32 COEF0       = ResizeCuDnCoefTraits<sizeof(Tp)>::COEF0;
    constexpr DT_S32 COEF1       = ResizeCuDnCoefTraits<sizeof(Tp)>::COEF1;
    constexpr DT_S32 COEF_BORDER = ResizeCuDnCoefTraits<sizeof(Tp)>::COEF_BORDER;

    DT_S32 *row1_ptr      = reinterpret_cast<DT_S32*>(vtcm_buffer->row2_ptr);
    DT_S32 *row2_ptr      = reinterpret_cast<DT_S32*>(vtcm_buffer->row3_ptr);
    BufType *row_head     = reinterpret_cast<BufType*>(vtcm_buffer->row_head);
    DT_S32 *row1_back_ptr = reinterpret_cast<DT_S32*>(vtcm_buffer->row2_ptr + row_pre_size);
    DT_S32 *row2_back_ptr = reinterpret_cast<DT_S32*>(vtcm_buffer->row3_ptr + row_pre_size);

    MVType mv_c_x0_src, mv_c_x1_src, mv_c_x2_src;
    MVType mv_c_dst;
    HVX_Vector v_coef0 = Q6_Vh_vsplat_R(COEF0);
    HVX_Vector v_coef1 = Q6_Vh_vsplat_R(COEF1);

    BufType *row1_head = row_head + C * 2;
    BufType *row2_head = row_head + C * 4;
    BufType *row3_head = row_head + C * 6;

    #pragma unroll(C)
    for (DT_S32 ch = 0; ch < C; ch++)
    {
        DT_S32 id0 = ch;
        DT_S32 id1 = C + ch;
        DT_S32 id2 = 2 * C + ch;

        row1_head[id0] = row2_head[id0];
        row2_head[id0] = row3_head[id0];
        row3_head[id0] = src_c[id0] * COEF_BORDER + src_c[id1] * COEF1 + src_c[id2] * COEF0;

        dst_c[id0] = ResizeCuSaturateCast<Tp, BufType>(row1_head[id0] * COEF0 + row2_head[id0] * COEF1 + row3_head[id0] * COEF_BORDER, SHIFT_BITS);
    }

    HVX_VType *v_row1      = (HVX_VType *)(row1_ptr);
    HVX_VType *v_row2      = (HVX_VType *)(row2_ptr);
    HVX_VType *v_back_row1 = (HVX_VType *)(row1_back_ptr);
    HVX_VType *v_back_row2 = (HVX_VType *)(row2_back_ptr);
    HVX_VType ws32_p1_result_l, ws32_p0_result_l, vs32_c_result_l;
    HVX_VType ws32_p1_result_h, ws32_p0_result_h, vs32_c_result_h;

    vload(src_c + C , mv_c_x0_src);

    for (DT_S32 x = 0; x < width_align; x += elem_counts)
    {
        vload(src_c  + (x * 2 + elem_counts + 1) * C, mv_c_x1_src);
        vload(src_c  + (x * 2 + (elem_counts << 1) + 1) * C, mv_c_x2_src);

        #pragma unroll(C)
        for (DT_S32 ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuDnX2HCore<Tp, HVX_VType>(mv_c_x0_src.val[ch],  mv_c_x1_src.val[ch],  mv_c_x2_src.val[ch],  vs32_c_result_l,  vs32_c_result_h,  v_coef0, v_coef1);

            ws32_p1_result_l = *v_row1++;
            ws32_p1_result_h = *v_row1++;
            ws32_p0_result_l = *v_row2++;
            ws32_p0_result_h = *v_row2++;

            // col
            ResizeCuDnX2VCore<Tp, HVX_VType>(ws32_p1_result_l, ws32_p1_result_h, ws32_p0_result_l, ws32_p0_result_h, vs32_c_result_l, vs32_c_result_h,
                                             mv_c_dst.val[ch], COEF0, COEF1, COEF_BORDER);
        }

        vstore(dst_c + (x + 1) * C, mv_c_dst);
        mv_c_x0_src  = mv_c_x2_src;
    }

    if (width_align < owidth - 2)
    {
        DT_S32 dx = owidth - 1 - elem_counts;
        DT_S32 sx = iwidth - 3 - elem_counts * 2;

        vload(src_c + sx * C, mv_c_x0_src);
        vload(src_c + (sx + elem_counts) * C, mv_c_x1_src);

        #pragma unroll(C)
        for (DT_S32 ch = 0; ch < C; ch++)
        {
            // row
            ResizeCuDnX2BorderHCore<Tp, HVX_VType>(mv_c_x0_src.val[ch], mv_c_x1_src.val[ch], src_c[(iwidth - 2) * C + ch], vs32_c_result_l, vs32_c_result_h, v_coef0, v_coef1);

            ws32_p1_result_l = *v_back_row1++;
            ws32_p1_result_h = *v_back_row1++;
            ws32_p0_result_l = *v_back_row2++;
            ws32_p0_result_h = *v_back_row2++;

            // col
            ResizeCuDnX2VCore<Tp, HVX_VType>(ws32_p1_result_l, ws32_p1_result_h, ws32_p0_result_l, ws32_p0_result_h, vs32_c_result_l, vs32_c_result_h,
                                             mv_c_dst.val[ch], COEF0, COEF1, COEF_BORDER);
        }

        vstore(dst_c + dx * C, mv_c_dst);
    }

    #pragma unroll(C)
    for (DT_S32 ch = 0; ch < C; ch++)
    {
        DT_S32 id0 = (iwidth - 1) * C + ch;
        DT_S32 id1 = (iwidth - 2) * C + ch;
        DT_S32 id2 = (iwidth - 3) * C + ch;

        row1_head[C + ch] = row2_head[C + ch];
        row2_head[C + ch] = row3_head[C + ch];
        row3_head[C + ch] = src_c[id2] * COEF0 + src_c[id1] * COEF1 + src_c[id0] * COEF_BORDER;

        dst_c[(owidth - 1) * C + ch] = ResizeCuSaturateCast<Tp, BufType>(row1_head[C + ch] * COEF0 + row2_head[C + ch] * COEF1 + row3_head[C + ch] * COEF_BORDER, SHIFT_BITS);
    }
}

template<typename Tp, DT_S32 C>
static DT_VOID ResizeCuDnX4Row(const Tp *src_c, Tp *dst_c, DT_S32 owidth, DT_S32 istride)
{
    using MVType = typename MVHvxVector<C>::Type;
    DT_S32 elem_counts = AURA_HVLEN / sizeof(Tp);
    DT_S32 width_align = owidth & (-elem_counts);

    const Tp *src_n0 = src_c  + (istride / sizeof(Tp));
    const Tp *src_n1 = src_n0 + (istride / sizeof(Tp));
    const Tp *src_n2 = src_n1 + (istride / sizeof(Tp));

    MVType mv_c_x0_src, mv_n0_x0_src, mv_n1_x0_src, mv_n2_x0_src;
    MVType mv_c_x1_src, mv_n0_x1_src, mv_n1_x1_src, mv_n2_x1_src;
    MVType mv_c_x2_src, mv_n0_x2_src, mv_n1_x2_src, mv_n2_x2_src;
    MVType mv_c_x3_src, mv_n0_x3_src, mv_n1_x3_src, mv_n2_x3_src;
    MVType mv_c_dst;

    for (DT_S32 x = 0; x < width_align; x += elem_counts)
    {
        vload(src_c + x * 4 * C, mv_c_x0_src);
        vload(src_c + (x * 4 + 1 * elem_counts) * C, mv_c_x1_src);
        vload(src_c + (x * 4 + 2 * elem_counts) * C, mv_c_x2_src);
        vload(src_c + (x * 4 + 3 * elem_counts) * C, mv_c_x3_src);
        vload(src_n0 + x * 4 * C, mv_n0_x0_src);
        vload(src_n0 + (x * 4 + 1 * elem_counts) * C, mv_n0_x1_src);
        vload(src_n0 + (x * 4 + 2 * elem_counts) * C, mv_n0_x2_src);
        vload(src_n0 + (x * 4 + 3 * elem_counts) * C, mv_n0_x3_src);
        vload(src_n1 + x * 4 * C, mv_n1_x0_src);
        vload(src_n1 + (x * 4 + 1 * elem_counts) * C, mv_n1_x1_src);
        vload(src_n1 + (x * 4 + 2 * elem_counts) * C, mv_n1_x2_src);
        vload(src_n1 + (x * 4 + 3 * elem_counts) * C, mv_n1_x3_src);
        vload(src_n2 + x * 4 * C, mv_n2_x0_src);
        vload(src_n2 + (x * 4 + 1 * elem_counts) * C, mv_n2_x1_src);
        vload(src_n2 + (x * 4 + 2 * elem_counts) * C, mv_n2_x2_src);
        vload(src_n2 + (x * 4 + 3 * elem_counts) * C, mv_n2_x3_src);

        #pragma unroll(C)
        for (DT_S32 ch = 0; ch < C; ch++)
        {
            ResizeCuDnX4Core<Tp>(mv_c_x0_src.val[ch],  mv_c_x1_src.val[ch],  mv_c_x2_src.val[ch],  mv_c_x3_src.val[ch],
                                 mv_n0_x0_src.val[ch], mv_n0_x1_src.val[ch], mv_n0_x2_src.val[ch], mv_n0_x3_src.val[ch],
                                 mv_n1_x0_src.val[ch], mv_n1_x1_src.val[ch], mv_n1_x2_src.val[ch], mv_n1_x3_src.val[ch],
                                 mv_n2_x0_src.val[ch], mv_n2_x1_src.val[ch], mv_n2_x2_src.val[ch], mv_n2_x3_src.val[ch],
                                 mv_c_dst.val[ch]);
        }

        vstore(dst_c + x * C, mv_c_dst);
    }

    if (width_align < owidth)
    {
        DT_S32 x = owidth - elem_counts;
        vload(src_c + x * 4 * C, mv_c_x0_src);
        vload(src_c + (x * 4 + 1 * elem_counts) * C, mv_c_x1_src);
        vload(src_c + (x * 4 + 2 * elem_counts) * C, mv_c_x2_src);
        vload(src_c + (x * 4 + 3 * elem_counts) * C, mv_c_x3_src);
        vload(src_n0 + x * 4 * C, mv_n0_x0_src);
        vload(src_n0 + (x * 4 + 1 * elem_counts) * C, mv_n0_x1_src);
        vload(src_n0 + (x * 4 + 2 * elem_counts) * C, mv_n0_x2_src);
        vload(src_n0 + (x * 4 + 3 * elem_counts) * C, mv_n0_x3_src);
        vload(src_n1 + x * 4 * C, mv_n1_x0_src);
        vload(src_n1 + (x * 4 + 1 * elem_counts) * C, mv_n1_x1_src);
        vload(src_n1 + (x * 4 + 2 * elem_counts) * C, mv_n1_x2_src);
        vload(src_n1 + (x * 4 + 3 * elem_counts) * C, mv_n1_x3_src);
        vload(src_n2 + x * 4 * C, mv_n2_x0_src);
        vload(src_n2 + (x * 4 + 1 * elem_counts) * C, mv_n2_x1_src);
        vload(src_n2 + (x * 4 + 2 * elem_counts) * C, mv_n2_x2_src);
        vload(src_n2 + (x * 4 + 3 * elem_counts) * C, mv_n2_x3_src);

        #pragma unroll(C)
        for (DT_S32 ch = 0; ch < C; ch++)
        {
            ResizeCuDnX4Core<Tp>(mv_c_x0_src.val[ch],  mv_c_x1_src.val[ch],  mv_c_x2_src.val[ch],  mv_c_x3_src.val[ch],
                                 mv_n0_x0_src.val[ch], mv_n0_x1_src.val[ch], mv_n0_x2_src.val[ch], mv_n0_x3_src.val[ch],
                                 mv_n1_x0_src.val[ch], mv_n1_x1_src.val[ch], mv_n1_x2_src.val[ch], mv_n1_x3_src.val[ch],
                                 mv_n2_x0_src.val[ch], mv_n2_x1_src.val[ch], mv_n2_x2_src.val[ch], mv_n2_x3_src.val[ch],
                                 mv_c_dst.val[ch]);
        }

        vstore(dst_c + x * C, mv_c_dst);
    }

    return;
}

template<typename Tp, DT_S32 C>
static Status ResizeCuDnX2HvxImpl(const Mat &src, Mat &dst, ResizeCuFastVtcmBuffer vtcm_buffer, DT_S32 thread_num, DT_S32 start_row, DT_S32 end_row)
{
    DT_S32 iwidth    = src.GetSizes().m_width;
    DT_S32 iheight   = src.GetSizes().m_height;
    DT_S32 owidth    = dst.GetSizes().m_width;
    DT_S32 oheight   = dst.GetSizes().m_height;
    DT_S32 istride   = src.GetStrides().m_width;
    DT_S32 thread_id = SaturateCast<DT_S32>(static_cast<DT_F32>(start_row) * thread_num / oheight);
    DT_S32 back_size = AURA_HVLEN * 2 * C * sizeof(DT_S32);
    DT_S32 row_size  = AURA_ALIGN(owidth * C * sizeof(DT_S32), AURA_HVLEN) + back_size;

    vtcm_buffer.row0_ptr = vtcm_buffer.row0_ptr + row_size * thread_id;
    vtcm_buffer.row1_ptr = vtcm_buffer.row1_ptr + row_size * thread_id;
    vtcm_buffer.row2_ptr = vtcm_buffer.row2_ptr + row_size * thread_id;
    vtcm_buffer.row3_ptr = vtcm_buffer.row3_ptr + row_size * thread_id;
    vtcm_buffer.row_head = vtcm_buffer.row_head + AURA_HVLEN * sizeof(Tp) * thread_id;

    DT_U64 l2fetch_param = L2PfParam(istride, iwidth * sizeof(Tp) * C, 4, 0);
    DT_S32 y        = start_row;
    DT_S32 loop_row = end_row - 1 * (oheight == end_row);

    if (0 == y)
    {
        const Tp *src_c = src.Ptr<Tp>(0);
        Tp *dst_c       = dst.Ptr<Tp>(0);
        L2Fetch(reinterpret_cast<DT_U32>(src_c), l2fetch_param);
        ResizeCuDnX2ZeroRow<Tp, C>(src_c, dst_c, &vtcm_buffer, iwidth, owidth, istride);
        y = 1;
    }
    else
    {
        const Tp *src_c = src.Ptr<Tp>((y << 1) - 1);
        Tp *dst_c       = dst.Ptr<Tp>(y);
        L2Fetch(reinterpret_cast<DT_U32>(src_c), l2fetch_param);
        ResizeCuDnX2UpBorder<Tp, C>(src_c, dst_c, &vtcm_buffer, iwidth, owidth, istride);
        y += 1;
    }

    DT_S32 yofs   = (y << 1) + 1;
    l2fetch_param = L2PfParam(istride, iwidth * sizeof(Tp) * C, 2, 0);
    L2Fetch(reinterpret_cast<DT_U32>(src.Ptr<Tp>(yofs)), l2fetch_param);

    for (; y < loop_row; y++)
    {
        if (y + 1 < loop_row)
        {
            L2Fetch(reinterpret_cast<DT_U32>(src.Ptr<Tp>(yofs + 2)), l2fetch_param);
        }

        const Tp *src_c = src.Ptr<Tp>(yofs);
        Tp *dst_c       = dst.Ptr<Tp>(y);
        ResizeCuDnX2Row<Tp, C>(src_c, dst_c, &vtcm_buffer, iwidth, owidth, istride);

        yofs += 2;
    }

    if (oheight == end_row)
    {
        const Tp *src_c = src.Ptr<Tp>(iheight - 1);
        Tp *dst_c       = dst.Ptr<Tp>(oheight - 1);
        ResizeCuDnX2BottomBorder<Tp, C>(src_c, dst_c, &vtcm_buffer, iwidth, owidth);
    }

    return Status::OK;
}

template<typename Tp, DT_S32 C>
static Status ResizeCuDnX4HvxImpl(const Mat &src, Mat &dst, DT_S32 start_row, DT_S32 end_row)
{
    DT_S32 owidth  = dst.GetSizes().m_width;
    DT_S32 istride = src.GetStrides().m_width;
    DT_U64 l2fetch_param = L2PfParam(istride, src.GetSizes().m_width * sizeof(Tp) * C, 4, 0);

    for (DT_S32 y = start_row; y < end_row; y++)
    {
        DT_S32 yofs = (y << 2);
        if (y + 1 < end_row)
        {
            L2Fetch(reinterpret_cast<DT_U32>(src.Ptr<Tp>(yofs + 4)), l2fetch_param);
        }

        const Tp *src_c = src.Ptr<Tp>(yofs);
        Tp *dst_c       = dst.Ptr<Tp>(y);
        ResizeCuDnX4Row<Tp, C>(src_c, dst_c, owidth, istride);
    }

    return Status::OK;
}

template <typename Tp, DT_S32 C>
static Status ResizeCuFastDownHvxHelper(Context *ctx, const Mat &src, Mat &dst)
{
    WorkerPool *wp = ctx->GetWorkerPool();
    if (DT_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerPool fail");
        return Status::ERROR;
    }

    Status ret      = Status::ERROR;
    DT_S32 iwidth   = src.GetSizes().m_width;
    DT_S32 owidth   = dst.GetSizes().m_width;
    DT_U8 *vtcm_mem = DT_NULL;

    if (iwidth == 2 * owidth)
    {
        DT_S32 thread_num        = wp->GetComputeThreadNum();
        DT_S32 head_buffer_size  = AURA_HVLEN * sizeof(Tp) * thread_num;
        DT_S32 back_buffer_size  = AURA_HVLEN * 2 * C * sizeof(DT_S32);
        DT_S32 row_buffer_size   = (AURA_ALIGN(owidth * C * sizeof(DT_S32), AURA_HVLEN) + back_buffer_size) * thread_num;
        DT_S32 total_buffer_size = head_buffer_size + row_buffer_size * 4;

        vtcm_mem = static_cast<DT_U8*>(AURA_ALLOC_PARAM(ctx, AURA_MEM_VTCM, total_buffer_size, AURA_HVLEN));
        if (DT_NULL == vtcm_mem)
        {
            AURA_ADD_ERROR_STRING(ctx, "alloc vtcm memory failed");
            AURA_FREE(ctx, vtcm_mem);
            return Status::ABORT;
        }

        struct ResizeCuFastVtcmBuffer vtcm_buffer;
        vtcm_buffer.row0_ptr = vtcm_mem;
        vtcm_buffer.row1_ptr = vtcm_buffer.row0_ptr + row_buffer_size;
        vtcm_buffer.row2_ptr = vtcm_buffer.row1_ptr + row_buffer_size;
        vtcm_buffer.row3_ptr = vtcm_buffer.row2_ptr + row_buffer_size;
        vtcm_buffer.row_head = vtcm_buffer.row3_ptr + row_buffer_size;

        ret = wp->ParallelFor((DT_S32)0, dst.GetSizes().m_height, ResizeCuDnX2HvxImpl<Tp, C>, std::cref(src), std::ref(dst), vtcm_buffer, thread_num);
    }
    else if (iwidth == 4 * owidth)
    {
        ret = wp->ParallelFor((DT_S32)0, dst.GetSizes().m_height, ResizeCuDnX4HvxImpl<Tp, C>, std::cref(src), std::ref(dst));
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "only support scale 2, 4");
        return Status::ERROR;
    }

    AURA_FREE(ctx, vtcm_mem);
    AURA_RETURN(ctx, ret);
}

template <typename Tp>
static Status ResizeCuFastDownHvxHelper(Context *ctx, const Mat &src, Mat &dst)
{
    Status ret     = Status::ERROR;
    DT_S32 channel = src.GetSizes().m_channel;

    switch (channel)
    {
        case 1:
        {
            ret = ResizeCuFastDownHvxHelper<Tp, 1>(ctx, src, dst);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeCuFastDownHvxHelper failed for c1");
            }
            break;
        }

        case 2:
        {
            ret = ResizeCuFastDownHvxHelper<Tp, 2>(ctx, src, dst);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeCuFastDownHvxHelper failed for c2");
            }
            break;
        }

        case 3:
        {
            ret = ResizeCuFastDownHvxHelper<Tp, 3>(ctx, src, dst);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeCuFastDownHvxHelper failed for c3");
            }
            break;
        }

        default:
        {
            ret = Status::ERROR;
            AURA_ADD_ERROR_STRING(ctx, "only support channel 1,2,3");
        }
    }

    AURA_RETURN(ctx, ret);
}

Status ResizeCuFastDnHvx(Context *ctx, const Mat &src, Mat &dst)
{
    Status ret = Status::ERROR;

    switch (src.GetElemType())
    {
        case ElemType::U8:
        {
            ret = ResizeCuFastDownHvxHelper<DT_U8>(ctx, src, dst);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeCuFastDownHvxHelper run failed, type: DT_U8");
            }
            break;
        }

        case ElemType::S8:
        {
            ret = ResizeCuFastDownHvxHelper<DT_S8>(ctx, src, dst);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeCuFastDownHvxHelper run failed, type: DT_S8");
            }
            break;
        }

        case ElemType::U16:
        {
            ret = ResizeCuFastDownHvxHelper<DT_U16>(ctx, src, dst);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeCuFastDownHvxHelper run failed, type: DT_U16");
            }
            break;
        }

        case ElemType::S16:
        {
            ret = ResizeCuFastDownHvxHelper<DT_S16>(ctx, src, dst);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ResizeCuFastDownHvxHelper run failed, type: DT_S16");
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