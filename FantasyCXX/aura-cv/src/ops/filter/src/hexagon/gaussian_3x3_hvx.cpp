#include "gaussian_impl.hpp"
#include "aura/runtime/worker_pool.h"
#include "aura/runtime/logger.h"

namespace aura
{

/**
 * TODO 这个抵挡就是在device设备侧：DSP侧进行执行
 **/
// using Tp = DT_U8
template <typename Tp, typename std::enable_if<std::is_same<Tp, DT_U8>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID Gaussian3x3VCore(HVX_Vector &vu8_src_p0, HVX_Vector &vu8_src_c, HVX_Vector &vu8_src_n0,
                                            HVX_VectorPair &wu16_sum, const DT_U16 *kernel)
{
    DT_U32 k0k0k0k0 = Q6_R_vsplatb_R(kernel[0]);
    DT_U32 k1k1k1k1 = Q6_R_vsplatb_R(kernel[1]);

    wu16_sum = Q6_Wh_vmpa_WubRub(Q6_W_vcombine_VV(vu8_src_p0, vu8_src_n0), k0k0k0k0);
    wu16_sum = Q6_Wuh_vmpyacc_WuhVubRub(wu16_sum, vu8_src_c, k1k1k1k1);
}

// using Tp = DT_U8
template <typename Tp, typename std::enable_if<std::is_same<Tp, DT_U8>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID Gaussian3x3HCore(HVX_VectorPair &wu16_sum_x0, HVX_VectorPair &wu16_sum_x1, HVX_VectorPair &wu16_sum_x2,
                                            HVX_Vector &vu8_result, const DT_U16 *kernel)
{
    DT_U32 k0k0k0k0   = Q6_R_vsplatb_R(kernel[0]);
    DT_U32 k1k1       = (kernel[1] << 16) | kernel[1];
    DT_S32 align_size = sizeof(DT_U16);

    HVX_Vector wu16_sum_x1_lo = Q6_V_lo_W(wu16_sum_x1);
    HVX_Vector wu16_sum_x1_hi = Q6_V_hi_W(wu16_sum_x1);

    HVX_Vector vu16_sum_l0_hi = Q6_V_vlalign_VVR(wu16_sum_x1_hi, Q6_V_hi_W(wu16_sum_x0), align_size);
    HVX_Vector vu16_sum_c_lo  = wu16_sum_x1_lo;
    HVX_Vector vu16_sum_c_hi  = wu16_sum_x1_hi;
    HVX_Vector vu16_sum_r0_lo = Q6_V_valign_VVR(Q6_V_lo_W(wu16_sum_x2), wu16_sum_x1_lo, align_size);

    HVX_VectorPair wu32_sum_lo, wu32_sum_hi;
    wu32_sum_lo = Q6_Ww_vmpa_WuhRb(Q6_W_vcombine_VV(vu16_sum_l0_hi, vu16_sum_c_hi), k0k0k0k0);
    wu32_sum_lo = Q6_Wuw_vmpyacc_WuwVuhRuh(wu32_sum_lo, vu16_sum_c_lo, k1k1);
    wu32_sum_hi = Q6_Ww_vmpa_WuhRb(Q6_W_vcombine_VV(vu16_sum_c_lo, vu16_sum_r0_lo), k0k0k0k0);
    wu32_sum_hi = Q6_Wuw_vmpyacc_WuwVuhRuh(wu32_sum_hi, vu16_sum_c_hi, k1k1);

    HVX_Vector vu16_result_lo = Q6_Vuh_vround_VuwVuw_sat(Q6_V_hi_W(wu32_sum_lo), Q6_V_lo_W(wu32_sum_lo));
    HVX_Vector vu16_result_hi = Q6_Vuh_vround_VuwVuw_sat(Q6_V_hi_W(wu32_sum_hi), Q6_V_lo_W(wu32_sum_hi));

    vu8_result = Q6_Vub_vsat_VhVh(vu16_result_hi, vu16_result_lo);
}

// using Tp = DT_U16
template <typename Tp, typename std::enable_if<std::is_same<Tp, DT_U16>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID Gaussian3x3VCore(HVX_Vector &vu16_src_p0, HVX_Vector &vu16_src_c, HVX_Vector &vu16_src_n0,
                                            HVX_VectorPair &wu32_sum, const DT_U32 *kernel)
{
    DT_U32 k0k0 = (kernel[0] << 16) | kernel[0];
    DT_U32 k1k1 = (kernel[1] << 16) | kernel[1];

    wu32_sum = Q6_Wuw_vmpy_VuhRuh(vu16_src_p0, k0k0);
    wu32_sum = Q6_Wuw_vmpyacc_WuwVuhRuh(wu32_sum, vu16_src_c,  k1k1);
    wu32_sum = Q6_Wuw_vmpyacc_WuwVuhRuh(wu32_sum, vu16_src_n0, k0k0);
}

// using Tp = DT_U16
template <typename Tp, typename std::enable_if<std::is_same<Tp, DT_U16>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID Gaussian3x3HCore(HVX_VectorPair &wu32_sum_x0, HVX_VectorPair &wu32_sum_x1, HVX_VectorPair &wu32_sum_x2,
                                            HVX_Vector &vu16_result, const DT_U32 *kernel)
{
    HVX_Vector vu32_k0 = Q6_V_vsplat_R(kernel[0]);
    HVX_Vector vu32_k1 = Q6_V_vsplat_R(kernel[1]);
    DT_S32 align_size  = sizeof(DT_U32);

    HVX_Vector vu32_sum_x1_lo = Q6_V_lo_W(wu32_sum_x1);
    HVX_Vector vu32_sum_x1_hi = Q6_V_hi_W(wu32_sum_x1);

    HVX_Vector vu32_sum_l0_hi = Q6_V_vlalign_VVR(vu32_sum_x1_hi, Q6_V_hi_W(wu32_sum_x0), align_size);
    HVX_Vector vu32_sum_c_lo  = vu32_sum_x1_lo;
    HVX_Vector vu32_sum_c_hi  = vu32_sum_x1_hi;
    HVX_Vector vu32_sum_r0_lo = Q6_V_valign_VVR(Q6_V_lo_W(wu32_sum_x2), vu32_sum_x1_lo, align_size);

    HVX_VectorPair wu64_sum;
    HVX_Vector vu32_sum_lo, vu32_sum_hi;
    wu64_sum    = Q6_Wd_vmul_VwVw(vu32_sum_l0_hi, vu32_k0);
    wu64_sum    = Q6_Wd_vmulacc_WdVwVw(wu64_sum, vu32_sum_c_lo, vu32_k1);
    wu64_sum    = Q6_Wd_vmulacc_WdVwVw(wu64_sum, vu32_sum_c_hi, vu32_k0);
    wu64_sum    = Q6_Wud_vadd_WudWud(wu64_sum, Q6_W_vcombine_VV(Q6_V_vzero(), Q6_V_vsplat_R(1 << 27)));
    vu32_sum_lo = Q6_V_vor_VV(Q6_Vw_vasl_VwR(Q6_V_hi_W(wu64_sum), 4), Q6_Vuw_vlsr_VuwR(Q6_V_lo_W(wu64_sum), 28));

    wu64_sum    = Q6_Wd_vmul_VwVw(vu32_sum_c_lo, vu32_k0);
    wu64_sum    = Q6_Wd_vmulacc_WdVwVw(wu64_sum, vu32_sum_c_hi,  vu32_k1);
    wu64_sum    = Q6_Wd_vmulacc_WdVwVw(wu64_sum, vu32_sum_r0_lo, vu32_k0);
    wu64_sum    = Q6_Wud_vadd_WudWud(wu64_sum, Q6_W_vcombine_VV(Q6_V_vzero(), Q6_V_vsplat_R(1 << 27)));
    vu32_sum_hi = Q6_V_vor_VV(Q6_Vw_vasl_VwR(Q6_V_hi_W(wu64_sum), 4), Q6_Vuw_vlsr_VuwR(Q6_V_lo_W(wu64_sum), 28));

    vu16_result = Q6_Vuh_vsat_VuwVuw(vu32_sum_hi, vu32_sum_lo);
}

// using Tp = DT_S16
template <typename Tp, typename std::enable_if<std::is_same<Tp, DT_S16>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID Gaussian3x3VCore(HVX_Vector &vs16_src_p0, HVX_Vector &vs16_src_c, HVX_Vector &vs16_src_n0,
                                            HVX_VectorPair &ws32_sum, const DT_S32 *kernel)
{
    DT_S32 k0k0 = (kernel[0] << 16) | kernel[0];
    DT_S32 k1k1 = (kernel[1] << 16) | kernel[1];

    ws32_sum = Q6_Ww_vmpy_VhRh(vs16_src_p0, k0k0);
    ws32_sum = Q6_Ww_vmpyacc_WwVhRh(ws32_sum, vs16_src_c,  k1k1);
    ws32_sum = Q6_Ww_vmpyacc_WwVhRh(ws32_sum, vs16_src_n0, k0k0);
}

// using Tp = DT_S16
template <typename Tp, typename std::enable_if<std::is_same<Tp, DT_S16>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID Gaussian3x3HCore(HVX_VectorPair &ws32_sum_x0, HVX_VectorPair &ws32_sum_x1, HVX_VectorPair &ws32_sum_x2,
                                            HVX_Vector &vs16_result, const DT_S32 *kernel)
{
    HVX_Vector vs32_k0 = Q6_V_vsplat_R(kernel[0]);
    HVX_Vector vs32_k1 = Q6_V_vsplat_R(kernel[1]);
    DT_S32 align_size  = sizeof(DT_S32);

    HVX_Vector vs32_sum_x1_lo = Q6_V_lo_W(ws32_sum_x1);
    HVX_Vector vs32_sum_x1_hi = Q6_V_hi_W(ws32_sum_x1);

    HVX_Vector vs32_sum_l0_hi = Q6_V_vlalign_VVR(vs32_sum_x1_hi, Q6_V_hi_W(ws32_sum_x0), align_size);
    HVX_Vector vs32_sum_c_lo  = vs32_sum_x1_lo;
    HVX_Vector vs32_sum_c_hi  = vs32_sum_x1_hi;
    HVX_Vector vs32_sum_r0_lo = Q6_V_valign_VVR(Q6_V_lo_W(ws32_sum_x2), vs32_sum_x1_lo, align_size);

    HVX_VectorPair ws64_sum;
    HVX_Vector vs32_sum_lo, vs32_sum_hi;
    ws64_sum    = Q6_Wd_vmul_VwVw(vs32_sum_l0_hi, vs32_k0);
    ws64_sum    = Q6_Wd_vmulacc_WdVwVw(ws64_sum, vs32_sum_c_lo, vs32_k1);
    ws64_sum    = Q6_Wd_vmulacc_WdVwVw(ws64_sum, vs32_sum_c_hi, vs32_k0);
    ws64_sum    = Q6_Wd_vadd_WdWd(ws64_sum, Q6_W_vcombine_VV(Q6_V_vzero(), Q6_V_vsplat_R(1 << 27)));
    vs32_sum_lo = Q6_V_vor_VV(Q6_Vw_vasl_VwR(Q6_V_hi_W(ws64_sum), 4), Q6_Vuw_vlsr_VuwR(Q6_V_lo_W(ws64_sum), 28));

    ws64_sum    = Q6_Wd_vmul_VwVw(vs32_sum_c_lo, vs32_k0);
    ws64_sum    = Q6_Wd_vmulacc_WdVwVw(ws64_sum, vs32_sum_c_hi,  vs32_k1);
    ws64_sum    = Q6_Wd_vmulacc_WdVwVw(ws64_sum, vs32_sum_r0_lo, vs32_k0);
    ws64_sum    = Q6_Wd_vadd_WdWd(ws64_sum, Q6_W_vcombine_VV(Q6_V_vzero(), Q6_V_vsplat_R(1 << 27)));
    vs32_sum_hi = Q6_V_vor_VV(Q6_Vw_vasl_VwR(Q6_V_hi_W(ws64_sum), 4), Q6_Vuw_vlsr_VuwR(Q6_V_lo_W(ws64_sum), 28));

    vs16_result = Q6_Vh_vsat_VwVw(vs32_sum_hi, vs32_sum_lo);
}

// using Tp = DT_U32
template <typename Tp, typename std::enable_if<std::is_same<Tp, DT_U32>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID Gaussian3x3VCore(HVX_Vector &vu32_src_p0, HVX_Vector &vu32_src_c, HVX_Vector &vu32_src_n0,
                                            HVX_VectorPair &wu64_sum, const DT_U32 *kernel)
{
    HVX_Vector vu16_k0 = Q6_V_vsplat_R(kernel[0]);
    HVX_Vector vu16_k1 = Q6_V_vsplat_R(kernel[1]);

    wu64_sum = Q6_Wud_vmul32xlo16_VuwVuw(vu32_src_p0, vu16_k0);
    wu64_sum = Q6_Wud_vadd_WudWud(wu64_sum, Q6_Wud_vmul32xlo16_VuwVuw(vu32_src_c,  vu16_k1));
    wu64_sum = Q6_Wud_vadd_WudWud(wu64_sum, Q6_Wud_vmul32xlo16_VuwVuw(vu32_src_n0, vu16_k0));
}

// using Tp = DT_U32 DT_S32
template <typename Tp, typename Kt, typename std::enable_if<sizeof(Tp) == 4>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID Gaussian3x3HCore(HVX_VectorPair &wd64_sum_x0, HVX_VectorPair &wd64_sum_x1, HVX_VectorPair &wd64_sum_x2,
                                            HVX_Vector &vd32_result, const Kt *kernel)
{
    HVX_Vector vu16_k0   = Q6_V_vsplat_R(kernel[0]);
    HVX_Vector vu16_k1   = Q6_V_vsplat_R(kernel[1]);
    DT_S32 align_size    = sizeof(DT_U32);
    HVX_Vector vu32_half = Q6_V_vsplat_R(1 << 27);

    HVX_Vector vd64_sum_x1_lo = Q6_V_lo_W(wd64_sum_x1);
    HVX_Vector vd64_sum_x1_hi = Q6_V_hi_W(wd64_sum_x1);

    HVX_Vector vd64_sum_l0_lo = Q6_V_vlalign_VVR(vd64_sum_x1_lo, Q6_V_lo_W(wd64_sum_x0), align_size);
    HVX_Vector vd64_sum_l0_hi = Q6_V_vlalign_VVR(vd64_sum_x1_hi, Q6_V_hi_W(wd64_sum_x0), align_size);
    HVX_Vector vd64_sum_c_lo  = vd64_sum_x1_lo;
    HVX_Vector vd64_sum_c_hi  = vd64_sum_x1_hi;
    HVX_Vector vd64_sum_r0_lo = Q6_V_valign_VVR(Q6_V_lo_W(wd64_sum_x2), vd64_sum_x1_lo, align_size);
    HVX_Vector vd64_sum_r0_hi = Q6_V_valign_VVR(Q6_V_hi_W(wd64_sum_x2), vd64_sum_x1_hi, align_size);

    HVX_VectorPair wd64_sum;
    HVX_Vector vd32_sum_lo, vd32_sum_hi;

    wd64_sum    = Q6_Wud_vmul32xlo16_VuwVuw(vd64_sum_l0_lo, vu16_k0);
    wd64_sum    = Q6_Wud_vadd_WudWud(wd64_sum, Q6_Wud_vmul32xlo16_VuwVuw(vd64_sum_c_lo,  vu16_k1));
    wd64_sum    = Q6_Wud_vadd_WudWud(wd64_sum, Q6_Wud_vmul32xlo16_VuwVuw(vd64_sum_r0_lo, vu16_k0));
    wd64_sum    = Q6_Wud_vadd_WudWud(wd64_sum, Q6_W_vcombine_VV(Q6_V_vzero(), vu32_half));
    vd32_sum_hi = Q6_Vw_vmpyie_VwVuh(vd64_sum_l0_hi, vu16_k0);
    vd32_sum_hi = Q6_Vw_vadd_VwVw(vd32_sum_hi, Q6_Vw_vmpyie_VwVuh(vd64_sum_c_hi,  vu16_k1));
    vd32_sum_hi = Q6_Vw_vadd_VwVw(vd32_sum_hi, Q6_Vw_vmpyie_VwVuh(vd64_sum_r0_hi, vu16_k0));
    vd32_sum_hi = Q6_Vw_vadd_VwVw(vd32_sum_hi, Q6_V_hi_W(wd64_sum));

    // shift hi
    vd32_sum_hi = Q6_Vw_vasl_VwR(vd32_sum_hi, 4);

    // shift lo
    vd32_sum_lo = Q6_Vuw_vlsr_VuwR(Q6_V_lo_W(wd64_sum), 28);

    vd32_result = Q6_Vw_vadd_VwVw(vd32_sum_hi, vd32_sum_lo);
}

// using Tp = DT_S32
template <typename Tp, typename std::enable_if<std::is_same<Tp, DT_S32>::value>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID Gaussian3x3VCore(HVX_Vector &vs32_src_p0, HVX_Vector &vs32_src_c, HVX_Vector &vs32_src_n0,
                                            HVX_VectorPair &ws64_sum, const DT_S32 *kernel)
{
    HVX_Vector vs32_k0 = Q6_V_vsplat_R(kernel[0]);
    HVX_Vector vs32_k1 = Q6_V_vsplat_R(kernel[1]);

    ws64_sum = Q6_Wd_vmul_VwVw(vs32_src_p0, vs32_k0);
    ws64_sum = Q6_Wd_vmulacc_WdVwVw(ws64_sum, vs32_src_c,  vs32_k1);
    ws64_sum = Q6_Wd_vmulacc_WdVwVw(ws64_sum, vs32_src_n0, vs32_k0);
}

// Tp = DT_U8 DT_U16 DT_S16
template <typename Tp, typename Kt, typename std::enable_if<sizeof(Tp) != 4>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID Gaussian3x3HCore(HVX_VectorPair &w_sum_x0, HVX_VectorPair &w_sum_x1, HVX_VectorPair &w_sum_x2, HVX_VectorPair &w_sum_x3,
                                            HVX_Vector &v_result_x0, HVX_Vector &v_result_x1, const Kt *kernel, DT_S32 rest)
{
    HVX_Vector v_sum_x0_lo = Q6_V_lo_W(w_sum_x0);
    HVX_Vector v_sum_x0_hi = Q6_V_hi_W(w_sum_x0);
    HVX_Vector v_sum_x1_lo = Q6_V_lo_W(w_sum_x1);
    HVX_Vector v_sum_x1_hi = Q6_V_hi_W(w_sum_x1);
    HVX_Vector v_sum_x2_lo = Q6_V_lo_W(w_sum_x2);
    HVX_Vector v_sum_x2_hi = Q6_V_hi_W(w_sum_x2);
    HVX_Vector v_sum_x3_lo = Q6_V_lo_W(w_sum_x3);
    HVX_Vector v_sum_x3_hi = Q6_V_hi_W(w_sum_x3);

    HVX_Vector v_sum_l0_lo, v_sum_l0_hi, v_sum_r0_lo, v_sum_r0_hi;
    if (rest & 1)
    {
        DT_S32 align_size0 = (rest / 2) * sizeof(Kt);
        DT_S32 align_size1 = align_size0 + sizeof(Kt);
        v_sum_r0_lo = Q6_V_vlalign_safe_VVR(v_sum_x3_hi, v_sum_x2_hi, align_size1);
        v_sum_r0_hi = Q6_V_vlalign_safe_VVR(v_sum_x3_lo, v_sum_x2_lo, align_size0);
        v_sum_l0_lo = Q6_V_valign_safe_VVR(v_sum_x1_hi, v_sum_x0_hi, align_size0);
        v_sum_l0_hi = Q6_V_valign_safe_VVR(v_sum_x1_lo, v_sum_x0_lo, align_size1);
    }
    else
    {
        DT_S32 align_size = (rest / 2) * sizeof(Kt);
        v_sum_r0_lo = Q6_V_vlalign_safe_VVR(v_sum_x3_lo, v_sum_x2_lo, align_size);
        v_sum_r0_hi = Q6_V_vlalign_safe_VVR(v_sum_x3_hi, v_sum_x2_hi, align_size);
        v_sum_l0_lo = Q6_V_valign_safe_VVR(v_sum_x1_lo, v_sum_x0_lo, align_size);
        v_sum_l0_hi = Q6_V_valign_safe_VVR(v_sum_x1_hi, v_sum_x0_hi, align_size);
    }

    HVX_VectorPair w_sum_r0 = Q6_W_vcombine_VV(v_sum_r0_hi, v_sum_r0_lo);
    HVX_VectorPair w_sum_l0 = Q6_W_vcombine_VV(v_sum_l0_hi, v_sum_l0_lo);

    Gaussian3x3HCore<Tp>(w_sum_x0, w_sum_x1, w_sum_r0, v_result_x0, kernel);
    Gaussian3x3HCore<Tp>(w_sum_l0, w_sum_x2, w_sum_x3, v_result_x1, kernel);
}

// Tp = DT_U32 DT_S32
template <typename Tp, typename Kt, typename std::enable_if<sizeof(Tp) == 4>::type* = DT_NULL>
AURA_ALWAYS_INLINE DT_VOID Gaussian3x3HCore(HVX_VectorPair &wd64_sum_x0, HVX_VectorPair &wd64_sum_x1, HVX_VectorPair &wd64_sum_x2, HVX_VectorPair &wd64_sum_x3,
                                            HVX_Vector &vd32_result_x0, HVX_Vector &vd32_result_x1, const Kt *kernel, DT_S32 rest)
{
    DT_S32 align_size = rest * sizeof(Kt);

    HVX_Vector vd32_sum_x0_lo = Q6_V_lo_W(wd64_sum_x0);
    HVX_Vector vd32_sum_x0_hi = Q6_V_hi_W(wd64_sum_x0);
    HVX_Vector vd32_sum_x1_lo = Q6_V_lo_W(wd64_sum_x1);
    HVX_Vector vd32_sum_x1_hi = Q6_V_hi_W(wd64_sum_x1);
    HVX_Vector vd32_sum_x2_lo = Q6_V_lo_W(wd64_sum_x2);
    HVX_Vector vd32_sum_x2_hi = Q6_V_hi_W(wd64_sum_x2);
    HVX_Vector vd32_sum_x3_lo = Q6_V_lo_W(wd64_sum_x3);
    HVX_Vector vd32_sum_x3_hi = Q6_V_hi_W(wd64_sum_x3);

    HVX_Vector vd32_sum_r0_lo = Q6_V_vlalign_safe_VVR(vd32_sum_x3_lo, vd32_sum_x2_lo, align_size);
    HVX_Vector vd32_sum_r0_hi = Q6_V_vlalign_safe_VVR(vd32_sum_x3_hi, vd32_sum_x2_hi, align_size);
    HVX_Vector vd32_sum_l0_lo = Q6_V_valign_safe_VVR(vd32_sum_x1_lo, vd32_sum_x0_lo, align_size);
    HVX_Vector vd32_sum_l0_hi = Q6_V_valign_safe_VVR(vd32_sum_x1_hi, vd32_sum_x0_hi, align_size);

    HVX_VectorPair wd64_sum_r0 = Q6_W_vcombine_VV(vd32_sum_r0_hi, vd32_sum_r0_lo);
    HVX_VectorPair wd64_sum_l0 = Q6_W_vcombine_VV(vd32_sum_l0_hi, vd32_sum_l0_lo);

    Gaussian3x3HCore<Tp>(wd64_sum_x0, wd64_sum_x1, wd64_sum_r0, vd32_result_x0, kernel);
    Gaussian3x3HCore<Tp>(wd64_sum_l0, wd64_sum_x2, wd64_sum_x3, vd32_result_x1, kernel);
}

/**
 * 高斯滤波函数的每一行的逻辑处理
 * @tparam Tp
 * @tparam BORDER_TYPE
 * @tparam C
 * @tparam Kt
 * @param src_p0
 * @param src_c
 * @param src_n0
 * @param dst
 * @param kernel
 * @param border_value
 * @param width
 */
template <typename Tp, BorderType BORDER_TYPE, DT_S32 C, typename Kt>
static DT_VOID Gaussian3x3Row(const Tp *src_p0, const Tp *src_c, const Tp *src_n0, Tp *dst,
                              const Kt *kernel, const std::vector<Tp> &border_value, DT_S32 width)
{
    ///
    using MVType = typename MVHvxVector<C>::Type;
    using MWType = typename MWHvxVector<C>::Type;

    DT_S32 elem_counts = AURA_HVLEN / sizeof(Tp);
    DT_S32 back_offset = width - elem_counts;

    MVType mv_src_p0, mv_src_c, mv_src_n0;
    MWType mw_sum_x0, mw_sum_x1, mw_sum_x2;
    MVType mv_result;

    // left border
    {
        vload(src_p0, mv_src_p0);
        vload(src_c,  mv_src_c);
        vload(src_n0, mv_src_n0);

        #pragma unroll(C)
        for (DT_S32 ch = 0; ch < C; ch++)
        {
            HVX_Vector v_border_p0 = GetBorderVector<Tp, BORDER_TYPE, BorderArea::LEFT>(mv_src_p0.val[ch], src_p0[ch], border_value[ch]);
            HVX_Vector v_border_c  = GetBorderVector<Tp, BORDER_TYPE, BorderArea::LEFT>(mv_src_c.val[ch],  src_c[ch],  border_value[ch]);
            HVX_Vector v_border_n0 = GetBorderVector<Tp, BORDER_TYPE, BorderArea::LEFT>(mv_src_n0.val[ch], src_n0[ch], border_value[ch]);

            Gaussian3x3VCore<Tp>(v_border_p0, v_border_c, v_border_n0, mw_sum_x0.val[ch], kernel);
            Gaussian3x3VCore<Tp>(mv_src_p0.val[ch], mv_src_c.val[ch], mv_src_n0.val[ch], mw_sum_x1.val[ch], kernel);
        }
    }

    // main(0 ~ n-2)
    {
        for (DT_S32 x = elem_counts; x <= back_offset; x += elem_counts)
        {
            vload(src_p0 + C * x, mv_src_p0);
            vload(src_c  + C * x, mv_src_c);
            vload(src_n0 + C * x, mv_src_n0);

            #pragma unroll(C)
            for (DT_S32 ch = 0; ch < C; ch++)
            {
                Gaussian3x3VCore<Tp>(mv_src_p0.val[ch], mv_src_c.val[ch], mv_src_n0.val[ch], mw_sum_x2.val[ch], kernel);
                Gaussian3x3HCore<Tp>(mw_sum_x0.val[ch], mw_sum_x1.val[ch], mw_sum_x2.val[ch], mv_result.val[ch], kernel);
            }

            vstore(dst + C * (x - elem_counts), mv_result);

            mw_sum_x0 = mw_sum_x1;
            mw_sum_x1 = mw_sum_x2;
        }
    }

    // remain
    {
        DT_S32 last = C * (width - 1);
        DT_S32 rest = width % elem_counts;
        MVType mv_last;

        vload(src_p0 + C * back_offset, mv_src_p0);
        vload(src_c  + C * back_offset, mv_src_c);
        vload(src_n0 + C * back_offset, mv_src_n0);

        #pragma unroll(C)
        for (DT_S32 ch = 0; ch < C; ch++)
        {
            HVX_Vector v_border_p0 = GetBorderVector<Tp, BORDER_TYPE, BorderArea::RIGHT>(mv_src_p0.val[ch], src_p0[last + ch], border_value[ch]);
            HVX_Vector v_border_c  = GetBorderVector<Tp, BORDER_TYPE, BorderArea::RIGHT>(mv_src_c.val[ch],  src_c[last + ch],  border_value[ch]);
            HVX_Vector v_border_n0 = GetBorderVector<Tp, BORDER_TYPE, BorderArea::RIGHT>(mv_src_n0.val[ch], src_n0[last + ch], border_value[ch]);

            HVX_VectorPair w_sum_x2, w_sum_x3;
            Gaussian3x3VCore<Tp>(mv_src_p0.val[ch], mv_src_c.val[ch], mv_src_n0.val[ch], w_sum_x2, kernel);
            Gaussian3x3VCore<Tp>(v_border_p0, v_border_c, v_border_n0, w_sum_x3, kernel);
            Gaussian3x3HCore<Tp, Kt>(mw_sum_x0.val[ch], mw_sum_x1.val[ch], w_sum_x2, w_sum_x3, mv_result.val[ch], mv_last.val[ch], kernel, rest);
        }

        vstore(dst + C * (back_offset - rest), mv_result);
        vstore(dst + C * back_offset, mv_last);
    }
}

/**
 * 重点：这里需要注意start_row、end_row 就是我们ParallelFor里面的start和end
 * @tparam Tp
 * @tparam BORDER_TYPE
 * @tparam C
 * @param src
 * @param dst
 * @param kmat
 * @param border_value
 * @param border_buffer
 * @param start_row
 * @param end_row
 * @return
 */
template <typename Tp, BorderType BORDER_TYPE, DT_S32 C>
static Status Gaussian3x3HvxImpl(const Mat &src, Mat &dst, const Mat &kmat, const std::vector<Tp> &border_value,
                                 const Tp *border_buffer, DT_S32 start_row, DT_S32 end_row)
{
    /// 如果数据是32位，则默认就是32位。如果不是32位。则提升
    using Kt = typename std::conditional<sizeof(Tp) == 4, Tp, typename Promote<Tp>::Type>::type;

    ///
    DT_S32 width  = src.GetSizes().m_width;
    DT_S32 height = src.GetSizes().m_height;
    /// 数据步长，默认就是数据的宽度
    DT_S32 stride = src.GetStrides().m_width;

    /// kmat.Ptr<Kt> 将数据结构mat转换成为指针(模仿opencv的mat的写法)
    /// 将指针指向高斯核向量的第0个数据
    const Kt *kernel = kmat.Ptr<Kt>(0);

    ///
    const Tp *src_p0 = src.Ptr<Tp, BORDER_TYPE>(start_row - 1, border_buffer);
    const Tp *src_c  = src.Ptr<Tp>(start_row);
    const Tp *src_n0 = src.Ptr<Tp, BORDER_TYPE>(start_row + 1, border_buffer);

    /// TODO 这个函数的设计？ 是什么？？
    /// 用于生成L2缓存预取参数（L2缓存是介于CPU和主存之间的）
    ///
    DT_U64 L2fetch_param = L2PfParam(stride, width * C * ElemTypeSize(src.GetElemType()), 1, 0);
    for (DT_S32 y = start_row; y < end_row; y++)
    {
        if (y + 2 < height)
        {
            L2Fetch(reinterpret_cast<DT_U32>(src.Ptr<Tp>(y + 2)), L2fetch_param);
        }

        Tp *dst_row = dst.Ptr<Tp>(y);
        Gaussian3x3Row<Tp, BORDER_TYPE, C, Kt>(src_p0, src_c, src_n0, dst_row, kernel, border_value, width);

        src_p0 = src_c;
        src_c  = src_n0;
        src_n0 = src.Ptr<Tp, BORDER_TYPE>(y + 2, border_buffer);
    }

    return Status::OK;
}

/**
 * 模板函数，传入的数据类型
 * @tparam Tp
 * @tparam BORDER_TYPE
 * @param ctx
 * @param src
 * @param dst
 * @param kmat
 * @param border_value
 * @param border_buffer
 * @return
 */
template<typename Tp, BorderType BORDER_TYPE>
static Status Gaussian3x3HvxHelper(Context *ctx, const Mat &src, Mat &dst, const Mat &kmat,
                                   const std::vector<Tp> &border_value, const Tp *border_buffer)
{
    Status ret = Status::ERROR;

    WorkerPool *wp = ctx->GetWorkerPool();
    if (DT_NULL == wp)
    {
        AURA_ADD_ERROR_STRING(ctx, "GetWorkerpool failed");
        return ret;
    }

    DT_S32 height  = src.GetSizes().m_height;
    DT_S32 channel = src.GetSizes().m_channel;

    /// 判断输入数据的通融倒数
    switch (channel)
    {
        /// 假设是单通道的灰度图
        case 1:
        {
            /// 这个是由框架进行划分同等的线程进行多线程计算的
            /// start 是0, height是数据的高低。也就是行数。 多线程会进行划分相同的行数来进行多线程处理
            /// 具体的函数的实现：Gaussian3x3HvxImpl
            ret = wp->ParallelFor((DT_S32)0, height, Gaussian3x3HvxImpl<Tp, BORDER_TYPE, 1>,
                                  std::cref(src), std::ref(dst), std::cref(kmat), std::cref(border_value), border_buffer);
            break;
        }

        case 2:
        {
            ret = wp->ParallelFor((DT_S32)0, height, Gaussian3x3HvxImpl<Tp, BORDER_TYPE, 2>,
                                  std::cref(src), std::ref(dst), std::cref(kmat), std::cref(border_value), border_buffer);
            break;
        }

        case 3:
        {
            ret = wp->ParallelFor((DT_S32)0, height, Gaussian3x3HvxImpl<Tp, BORDER_TYPE, 3>,
                                  std::cref(src), std::ref(dst), std::cref(kmat), std::cref(border_value), border_buffer);
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

/**
 * 模板函数。传入的模板数据可能是：DT_U8、DT_U16、DT_S16、DT_U32、DT_S32
 * @tparam Tp
 * @param ctx
 * @param src
 * @param dst
 * @param kmat
 * @param border_type
 * @param border_value
 * @return
 */
template <typename Tp>
static Status Gaussian3x3HvxHelper(Context *ctx, const Mat &src, Mat &dst, const Mat &kmat,
                                   BorderType border_type, const Scalar &border_value)
{
    Status ret = Status::ERROR;

    Tp *border_buffer = DT_NULL;
    /// 将传递尽力的border数据，传入进来之后我们将其转化成为向量
    std::vector<Tp> vec_border_value = border_value.ToVector<Tp>();

    DT_S32 width   = dst.GetSizes().m_width;
    DT_S32 channel = dst.GetSizes().m_channel;

    switch (border_type)
    {
        case BorderType::CONSTANT:
        {
            // 创建边界的buffer 我们看一下的这个边界的buffer是怎么处理的。
            // 如果
            border_buffer = CreateBorderBuffer(ctx, width, channel, vec_border_value);
            if (DT_NULL == border_buffer)
            {
                AURA_ADD_ERROR_STRING(ctx, "CreateBorderBuffer failed");
                return Status::ERROR;
            }

            ret = Gaussian3x3HvxHelper<Tp, BorderType::CONSTANT>(ctx, src, dst, kmat, vec_border_value, border_buffer);

            AURA_FREE(ctx, border_buffer);
            break;
        }

        case BorderType::REPLICATE:
        {
            ret = Gaussian3x3HvxHelper<Tp, BorderType::REPLICATE>(ctx, src, dst, kmat, vec_border_value, border_buffer);
            break;
        }
        /// 如果是101反射的高斯模糊
        case BorderType::REFLECT_101:
        {
            /// 重点：注意在101反射的逻辑中,border_buffer为空。
            ret = Gaussian3x3HvxHelper<Tp, BorderType::REFLECT_101>(ctx, src, dst, kmat, vec_border_value, border_buffer);
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

Status Gaussian3x3Hvx(Context *ctx, const Mat &src, Mat &dst, const Mat &kmat,
                      BorderType border_type, const Scalar &border_value)
{
    Status ret = Status::ERROR;

    switch (src.GetElemType())
    {
        // 看看原始数据
        case ElemType::U8:
        {
            ret = Gaussian3x3HvxHelper<DT_U8>(ctx, src, dst, kmat, border_type, border_value);
            break;
        }

        case ElemType::U16:
        {
            ret = Gaussian3x3HvxHelper<DT_U16>(ctx, src, dst, kmat, border_type, border_value);
            break;
        }

        case ElemType::S16:
        {
            ret = Gaussian3x3HvxHelper<DT_S16>(ctx, src, dst, kmat, border_type, border_value);
            break;
        }

        case ElemType::U32:
        {
            ret = Gaussian3x3HvxHelper<DT_U32>(ctx, src, dst, kmat, border_type, border_value);
            break;
        }

        case ElemType::S32:
        {
            ret = Gaussian3x3HvxHelper<DT_S32>(ctx, src, dst, kmat, border_type, border_value);
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