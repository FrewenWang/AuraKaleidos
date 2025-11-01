#include "gaussian_impl.hpp"
#include "aura/runtime/worker_pool.h"
#include "aura/runtime/logger.h"

namespace aura
{

// using Tp = MI_U8
template <typename Tp, typename std::enable_if<std::is_same<Tp, MI_U8>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID Gaussian9x9VCore(HVX_Vector &vu8_src_p3, HVX_Vector &vu8_src_p2, HVX_Vector &vu8_src_p1,
                                            HVX_Vector &vu8_src_p0, HVX_Vector &vu8_src_c, HVX_Vector &vu8_src_n0,
                                            HVX_Vector &vu8_src_n1, HVX_Vector &vu8_src_n2, HVX_Vector &vu8_src_n3,
                                            HVX_VectorPair &wu16_sum, const MI_U16 *kernel)
{
    MI_U32 k0k0k0k0 = Q6_R_vsplatb_R(kernel[0]);
    MI_U32 k1k1k1k1 = Q6_R_vsplatb_R(kernel[1]);
    MI_U32 k2k2k2k2 = Q6_R_vsplatb_R(kernel[2]);
    MI_U32 k3k3k3k3 = Q6_R_vsplatb_R(kernel[3]);
    MI_U32 k4k4k4k4 = Q6_R_vsplatb_R(kernel[4]);

    wu16_sum = Q6_Wh_vmpa_WubRub(Q6_W_vcombine_VV(vu8_src_p3, vu8_src_n3), k0k0k0k0);
    wu16_sum = Q6_Wh_vmpaacc_WhWubRub(wu16_sum, Q6_W_vcombine_VV(vu8_src_p2, vu8_src_n2), k1k1k1k1);
    wu16_sum = Q6_Wh_vmpaacc_WhWubRub(wu16_sum, Q6_W_vcombine_VV(vu8_src_p1, vu8_src_n1), k2k2k2k2);
    wu16_sum = Q6_Wh_vmpaacc_WhWubRub(wu16_sum, Q6_W_vcombine_VV(vu8_src_p0, vu8_src_n0), k3k3k3k3);
    wu16_sum = Q6_Wuh_vmpyacc_WuhVubRub(wu16_sum, vu8_src_c, k4k4k4k4);
}

// using Tp = MI_U8
template <typename Tp, typename std::enable_if<std::is_same<Tp, MI_U8>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID Gaussian9x9HCore(HVX_VectorPair &wu16_sum_x0, HVX_VectorPair &wu16_sum_x1, HVX_VectorPair &wu16_sum_x2,
                                            HVX_Vector &vu8_result, const MI_U16 *kernel)
{
    MI_U32 k0k0k0k0   = Q6_R_vsplatb_R(kernel[0]);
    MI_U32 k1k1k1k1   = Q6_R_vsplatb_R(kernel[1]);
    MI_U32 k2k2k2k2   = Q6_R_vsplatb_R(kernel[2]);
    MI_U32 k3k3k3k3   = Q6_R_vsplatb_R(kernel[3]);
    MI_U32 k4k4       = (kernel[4] << 16) | kernel[4];
    MI_S32 align_size = sizeof(MI_U16);

    HVX_Vector vu16_sum_x0_lo = Q6_V_lo_W(wu16_sum_x0);
    HVX_Vector vu16_sum_x0_hi = Q6_V_hi_W(wu16_sum_x0);
    HVX_Vector vu16_sum_x1_lo = Q6_V_lo_W(wu16_sum_x1);
    HVX_Vector vu16_sum_x1_hi = Q6_V_hi_W(wu16_sum_x1);
    HVX_Vector vu16_sum_x2_lo = Q6_V_lo_W(wu16_sum_x2);
    HVX_Vector vu16_sum_x2_hi = Q6_V_hi_W(wu16_sum_x2);

    HVX_Vector vu16_sum_l1_lo = Q6_V_vlalign_VVR(vu16_sum_x1_lo, vu16_sum_x0_lo, 2 * align_size);
    HVX_Vector vu16_sum_l1_hi = Q6_V_vlalign_VVR(vu16_sum_x1_hi, vu16_sum_x0_hi, 2 * align_size);
    HVX_Vector vu16_sum_l0_lo = Q6_V_vlalign_VVR(vu16_sum_x1_lo, vu16_sum_x0_lo, align_size);
    HVX_Vector vu16_sum_l0_hi = Q6_V_vlalign_VVR(vu16_sum_x1_hi, vu16_sum_x0_hi, align_size);
    HVX_Vector vu16_sum_c_lo  = vu16_sum_x1_lo;
    HVX_Vector vu16_sum_c_hi  = vu16_sum_x1_hi;
    HVX_Vector vu16_sum_r0_lo = Q6_V_valign_VVR(vu16_sum_x2_lo, vu16_sum_x1_lo, align_size);
    HVX_Vector vu16_sum_r0_hi = Q6_V_valign_VVR(vu16_sum_x2_hi, vu16_sum_x1_hi, align_size);
    HVX_Vector vu16_sum_r1_lo = Q6_V_valign_VVR(vu16_sum_x2_lo, vu16_sum_x1_lo, 2 * align_size);
    HVX_Vector vu16_sum_r1_hi = Q6_V_valign_VVR(vu16_sum_x2_hi, vu16_sum_x1_hi, 2 * align_size);

    HVX_VectorPair wu32_sum_lo, wu32_sum_hi;
    wu32_sum_lo = Q6_Ww_vmpa_WuhRb(Q6_W_vcombine_VV(vu16_sum_l1_lo, vu16_sum_r1_lo), k0k0k0k0);
    wu32_sum_lo = Q6_Ww_vmpaacc_WwWuhRb(wu32_sum_lo, Q6_W_vcombine_VV(vu16_sum_l1_hi, vu16_sum_r0_hi), k1k1k1k1);
    wu32_sum_lo = Q6_Ww_vmpaacc_WwWuhRb(wu32_sum_lo, Q6_W_vcombine_VV(vu16_sum_l0_lo, vu16_sum_r0_lo), k2k2k2k2);
    wu32_sum_lo = Q6_Ww_vmpaacc_WwWuhRb(wu32_sum_lo, Q6_W_vcombine_VV(vu16_sum_l0_hi, vu16_sum_c_hi), k3k3k3k3);
    wu32_sum_lo = Q6_Wuw_vmpyacc_WuwVuhRuh(wu32_sum_lo, vu16_sum_c_lo, k4k4);

    wu32_sum_hi = Q6_Ww_vmpa_WuhRb(Q6_W_vcombine_VV(vu16_sum_l1_hi, vu16_sum_r1_hi), k0k0k0k0);
    wu32_sum_hi = Q6_Ww_vmpaacc_WwWuhRb(wu32_sum_hi, Q6_W_vcombine_VV(vu16_sum_l0_lo, vu16_sum_r1_lo), k1k1k1k1);
    wu32_sum_hi = Q6_Ww_vmpaacc_WwWuhRb(wu32_sum_hi, Q6_W_vcombine_VV(vu16_sum_l0_hi, vu16_sum_r0_hi), k2k2k2k2);
    wu32_sum_hi = Q6_Ww_vmpaacc_WwWuhRb(wu32_sum_hi, Q6_W_vcombine_VV(vu16_sum_c_lo, vu16_sum_r0_lo), k3k3k3k3);
    wu32_sum_hi = Q6_Wuw_vmpyacc_WuwVuhRuh(wu32_sum_hi, vu16_sum_c_hi, k4k4);

    HVX_Vector vu16_result_lo = Q6_Vuh_vround_VuwVuw_sat(Q6_V_hi_W(wu32_sum_lo), Q6_V_lo_W(wu32_sum_lo));
    HVX_Vector vu16_result_hi = Q6_Vuh_vround_VuwVuw_sat(Q6_V_hi_W(wu32_sum_hi), Q6_V_lo_W(wu32_sum_hi));

    vu8_result = Q6_Vub_vsat_VhVh(vu16_result_hi, vu16_result_lo);
}

// using Tp = MI_U16
template <typename Tp, typename std::enable_if<std::is_same<Tp, MI_U16>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID Gaussian9x9VCore(HVX_Vector &vu16_src_p3, HVX_Vector &vu16_src_p2, HVX_Vector &vu16_src_p1,
                                            HVX_Vector &vu16_src_p0, HVX_Vector &vu16_src_c, HVX_Vector &vu16_src_n0,
                                            HVX_Vector &vu16_src_n1, HVX_Vector &vu16_src_n2, HVX_Vector &vu16_src_n3,
                                            HVX_VectorPair &wu32_sum, const MI_U32 *kernel)
{
    MI_U32 k0k0 = (kernel[0] << 16) | kernel[0];
    MI_U32 k1k1 = (kernel[1] << 16) | kernel[1];
    MI_U32 k2k2 = (kernel[2] << 16) | kernel[2];
    MI_U32 k3k3 = (kernel[3] << 16) | kernel[3];
    MI_U32 k4k4 = (kernel[4] << 16) | kernel[4];

    wu32_sum = Q6_Wuw_vmpy_VuhRuh(vu16_src_p3, k0k0);
    wu32_sum = Q6_Wuw_vmpyacc_WuwVuhRuh(wu32_sum, vu16_src_p2, k1k1);
    wu32_sum = Q6_Wuw_vmpyacc_WuwVuhRuh(wu32_sum, vu16_src_p1, k2k2);
    wu32_sum = Q6_Wuw_vmpyacc_WuwVuhRuh(wu32_sum, vu16_src_p0, k3k3);
    wu32_sum = Q6_Wuw_vmpyacc_WuwVuhRuh(wu32_sum, vu16_src_c,  k4k4);
    wu32_sum = Q6_Wuw_vmpyacc_WuwVuhRuh(wu32_sum, vu16_src_n0, k3k3);
    wu32_sum = Q6_Wuw_vmpyacc_WuwVuhRuh(wu32_sum, vu16_src_n1, k2k2);
    wu32_sum = Q6_Wuw_vmpyacc_WuwVuhRuh(wu32_sum, vu16_src_n2, k1k1);
    wu32_sum = Q6_Wuw_vmpyacc_WuwVuhRuh(wu32_sum, vu16_src_n3, k0k0);
}

// using Tp = MI_U16
template <typename Tp, typename std::enable_if<std::is_same<Tp, MI_U16>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID Gaussian9x9HCore(HVX_VectorPair &wu32_sum_x0, HVX_VectorPair &wu32_sum_x1, HVX_VectorPair &wu32_sum_x2,
                                            HVX_Vector &vu16_result, const MI_U32 *kernel)
{
    HVX_Vector vu32_k0 = Q6_V_vsplat_R(kernel[0]);
    HVX_Vector vu32_k1 = Q6_V_vsplat_R(kernel[1]);
    HVX_Vector vu32_k2 = Q6_V_vsplat_R(kernel[2]);
    HVX_Vector vu32_k3 = Q6_V_vsplat_R(kernel[3]);
    HVX_Vector vu32_k4 = Q6_V_vsplat_R(kernel[4]);
    MI_S32 align_size  = sizeof(MI_U32);

    HVX_Vector vu32_sum_x0_lo = Q6_V_lo_W(wu32_sum_x0);
    HVX_Vector vu32_sum_x0_hi = Q6_V_hi_W(wu32_sum_x0);
    HVX_Vector vu32_sum_x1_lo = Q6_V_lo_W(wu32_sum_x1);
    HVX_Vector vu32_sum_x1_hi = Q6_V_hi_W(wu32_sum_x1);
    HVX_Vector vu32_sum_x2_lo = Q6_V_lo_W(wu32_sum_x2);
    HVX_Vector vu32_sum_x2_hi = Q6_V_hi_W(wu32_sum_x2);

    HVX_Vector vu32_sum_l1_lo = Q6_V_vlalign_VVR(vu32_sum_x1_lo, vu32_sum_x0_lo, 2 * align_size);
    HVX_Vector vu32_sum_l1_hi = Q6_V_vlalign_VVR(vu32_sum_x1_hi, vu32_sum_x0_hi, 2 * align_size);
    HVX_Vector vu32_sum_l0_lo = Q6_V_vlalign_VVR(vu32_sum_x1_lo, vu32_sum_x0_lo, align_size);
    HVX_Vector vu32_sum_l0_hi = Q6_V_vlalign_VVR(vu32_sum_x1_hi, vu32_sum_x0_hi, align_size);
    HVX_Vector vu32_sum_c_lo  = vu32_sum_x1_lo;
    HVX_Vector vu32_sum_c_hi  = vu32_sum_x1_hi;
    HVX_Vector vu32_sum_r0_lo = Q6_V_valign_VVR(vu32_sum_x2_lo, vu32_sum_x1_lo, align_size);
    HVX_Vector vu32_sum_r0_hi = Q6_V_valign_VVR(vu32_sum_x2_hi, vu32_sum_x1_hi, align_size);
    HVX_Vector vu32_sum_r1_lo = Q6_V_valign_VVR(vu32_sum_x2_lo, vu32_sum_x1_lo, 2 * align_size);
    HVX_Vector vu32_sum_r1_hi = Q6_V_valign_VVR(vu32_sum_x2_hi, vu32_sum_x1_hi, 2 * align_size);

    HVX_VectorPair wu64_sum;
    HVX_Vector vu32_sum_lo, vu32_sum_hi;
    wu64_sum    = Q6_Wd_vmul_VwVw(vu32_sum_l1_lo, vu32_k0);
    wu64_sum    = Q6_Wd_vmulacc_WdVwVw(wu64_sum, vu32_sum_l1_hi, vu32_k1);
    wu64_sum    = Q6_Wd_vmulacc_WdVwVw(wu64_sum, vu32_sum_l0_lo, vu32_k2);
    wu64_sum    = Q6_Wd_vmulacc_WdVwVw(wu64_sum, vu32_sum_l0_hi, vu32_k3);
    wu64_sum    = Q6_Wd_vmulacc_WdVwVw(wu64_sum, vu32_sum_c_lo,  vu32_k4);
    wu64_sum    = Q6_Wd_vmulacc_WdVwVw(wu64_sum, vu32_sum_c_hi,  vu32_k3);
    wu64_sum    = Q6_Wd_vmulacc_WdVwVw(wu64_sum, vu32_sum_r0_lo, vu32_k2);
    wu64_sum    = Q6_Wd_vmulacc_WdVwVw(wu64_sum, vu32_sum_r0_hi, vu32_k1);
    wu64_sum    = Q6_Wd_vmulacc_WdVwVw(wu64_sum, vu32_sum_r1_lo, vu32_k0);
    wu64_sum    = Q6_Wud_vadd_WudWud(wu64_sum, Q6_W_vcombine_VV(Q6_V_vzero(), Q6_V_vsplat_R(1 << 27)));
    vu32_sum_lo = Q6_V_vor_VV(Q6_Vw_vasl_VwR(Q6_V_hi_W(wu64_sum), 4), Q6_Vuw_vlsr_VuwR(Q6_V_lo_W(wu64_sum), 28));

    wu64_sum    = Q6_Wd_vmul_VwVw(vu32_sum_l1_hi, vu32_k0);
    wu64_sum    = Q6_Wd_vmulacc_WdVwVw(wu64_sum, vu32_sum_l0_lo, vu32_k1);
    wu64_sum    = Q6_Wd_vmulacc_WdVwVw(wu64_sum, vu32_sum_l0_hi, vu32_k2);
    wu64_sum    = Q6_Wd_vmulacc_WdVwVw(wu64_sum, vu32_sum_c_lo,  vu32_k3);
    wu64_sum    = Q6_Wd_vmulacc_WdVwVw(wu64_sum, vu32_sum_c_hi,  vu32_k4);
    wu64_sum    = Q6_Wd_vmulacc_WdVwVw(wu64_sum, vu32_sum_r0_lo, vu32_k3);
    wu64_sum    = Q6_Wd_vmulacc_WdVwVw(wu64_sum, vu32_sum_r0_hi, vu32_k2);
    wu64_sum    = Q6_Wd_vmulacc_WdVwVw(wu64_sum, vu32_sum_r1_lo, vu32_k1);
    wu64_sum    = Q6_Wd_vmulacc_WdVwVw(wu64_sum, vu32_sum_r1_hi, vu32_k0);
    wu64_sum    = Q6_Wud_vadd_WudWud(wu64_sum, Q6_W_vcombine_VV(Q6_V_vzero(), Q6_V_vsplat_R(1 << 27)));
    vu32_sum_hi = Q6_V_vor_VV(Q6_Vw_vasl_VwR(Q6_V_hi_W(wu64_sum), 4), Q6_Vuw_vlsr_VuwR(Q6_V_lo_W(wu64_sum), 28));

    vu16_result = Q6_Vuh_vsat_VuwVuw(vu32_sum_hi, vu32_sum_lo);
}

// using Tp = MI_S16
template <typename Tp, typename std::enable_if<std::is_same<Tp, MI_S16>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID Gaussian9x9VCore(HVX_Vector &vs16_src_p3, HVX_Vector &vs16_src_p2, HVX_Vector &vs16_src_p1,
                                            HVX_Vector &vs16_src_p0, HVX_Vector &vs16_src_c, HVX_Vector &vs16_src_n0,
                                            HVX_Vector &vs16_src_n1, HVX_Vector &vs16_src_n2, HVX_Vector &vs16_src_n3,
                                            HVX_VectorPair &ws32_sum, const MI_S32 *kernel)
{
    MI_S32 k0k0 = (kernel[0] << 16) | kernel[0];
    MI_S32 k1k1 = (kernel[1] << 16) | kernel[1];
    MI_S32 k2k2 = (kernel[2] << 16) | kernel[2];
    MI_S32 k3k3 = (kernel[3] << 16) | kernel[3];
    MI_S32 k4k4 = (kernel[4] << 16) | kernel[4];

    ws32_sum = Q6_Ww_vmpy_VhRh(vs16_src_p3, k0k0);
    ws32_sum = Q6_Ww_vmpyacc_WwVhRh(ws32_sum, vs16_src_p2, k1k1);
    ws32_sum = Q6_Ww_vmpyacc_WwVhRh(ws32_sum, vs16_src_p1, k2k2);
    ws32_sum = Q6_Ww_vmpyacc_WwVhRh(ws32_sum, vs16_src_p0, k3k3);
    ws32_sum = Q6_Ww_vmpyacc_WwVhRh(ws32_sum, vs16_src_c,  k4k4);
    ws32_sum = Q6_Ww_vmpyacc_WwVhRh(ws32_sum, vs16_src_n0, k3k3);
    ws32_sum = Q6_Ww_vmpyacc_WwVhRh(ws32_sum, vs16_src_n1, k2k2);
    ws32_sum = Q6_Ww_vmpyacc_WwVhRh(ws32_sum, vs16_src_n2, k1k1);
    ws32_sum = Q6_Ww_vmpyacc_WwVhRh(ws32_sum, vs16_src_n3, k0k0);
}

// using Tp = MI_S16
template <typename Tp, typename std::enable_if<std::is_same<Tp, MI_S16>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID Gaussian9x9HCore(HVX_VectorPair &ws32_sum_x0, HVX_VectorPair &ws32_sum_x1, HVX_VectorPair &ws32_sum_x2,
                                            HVX_Vector &vs16_result, const MI_S32 *kernel)
{
    HVX_Vector vs32_k0 = Q6_V_vsplat_R(kernel[0]);
    HVX_Vector vs32_k1 = Q6_V_vsplat_R(kernel[1]);
    HVX_Vector vs32_k2 = Q6_V_vsplat_R(kernel[2]);
    HVX_Vector vs32_k3 = Q6_V_vsplat_R(kernel[3]);
    HVX_Vector vs32_k4 = Q6_V_vsplat_R(kernel[4]);
    MI_S32 align_size  = sizeof(MI_S32);

    HVX_Vector vs32_sum_x0_lo = Q6_V_lo_W(ws32_sum_x0);
    HVX_Vector vs32_sum_x0_hi = Q6_V_hi_W(ws32_sum_x0);
    HVX_Vector vs32_sum_x1_lo = Q6_V_lo_W(ws32_sum_x1);
    HVX_Vector vs32_sum_x1_hi = Q6_V_hi_W(ws32_sum_x1);
    HVX_Vector vs32_sum_x2_lo = Q6_V_lo_W(ws32_sum_x2);
    HVX_Vector vs32_sum_x2_hi = Q6_V_hi_W(ws32_sum_x2);

    HVX_Vector vs32_sum_l1_lo = Q6_V_vlalign_VVR(vs32_sum_x1_lo, vs32_sum_x0_lo, 2 * align_size);
    HVX_Vector vs32_sum_l1_hi = Q6_V_vlalign_VVR(vs32_sum_x1_hi, vs32_sum_x0_hi, 2 * align_size);
    HVX_Vector vs32_sum_l0_lo = Q6_V_vlalign_VVR(vs32_sum_x1_lo, Q6_V_lo_W(ws32_sum_x0), align_size);
    HVX_Vector vs32_sum_l0_hi = Q6_V_vlalign_VVR(vs32_sum_x1_hi, Q6_V_hi_W(ws32_sum_x0), align_size);
    HVX_Vector vs32_sum_c_lo  = vs32_sum_x1_lo;
    HVX_Vector vs32_sum_c_hi  = vs32_sum_x1_hi;
    HVX_Vector vs32_sum_r0_lo = Q6_V_valign_VVR(Q6_V_lo_W(ws32_sum_x2), vs32_sum_x1_lo, align_size);
    HVX_Vector vs32_sum_r0_hi = Q6_V_valign_VVR(Q6_V_hi_W(ws32_sum_x2), vs32_sum_x1_hi, align_size);
    HVX_Vector vs32_sum_r1_lo = Q6_V_valign_VVR(vs32_sum_x2_lo, vs32_sum_x1_lo, 2 * align_size);
    HVX_Vector vs32_sum_r1_hi = Q6_V_valign_VVR(vs32_sum_x2_hi, vs32_sum_x1_hi, 2 * align_size);

    HVX_VectorPair ws64_sum;
    HVX_Vector vs32_sum_lo, vs32_sum_hi;
    ws64_sum    = Q6_Wd_vmul_VwVw(vs32_sum_l1_lo, vs32_k0);
    ws64_sum    = Q6_Wd_vmulacc_WdVwVw(ws64_sum, vs32_sum_l1_hi, vs32_k1);
    ws64_sum    = Q6_Wd_vmulacc_WdVwVw(ws64_sum, vs32_sum_l0_lo, vs32_k2);
    ws64_sum    = Q6_Wd_vmulacc_WdVwVw(ws64_sum, vs32_sum_l0_hi, vs32_k3);
    ws64_sum    = Q6_Wd_vmulacc_WdVwVw(ws64_sum, vs32_sum_c_lo,  vs32_k4);
    ws64_sum    = Q6_Wd_vmulacc_WdVwVw(ws64_sum, vs32_sum_c_hi,  vs32_k3);
    ws64_sum    = Q6_Wd_vmulacc_WdVwVw(ws64_sum, vs32_sum_r0_lo, vs32_k2);
    ws64_sum    = Q6_Wd_vmulacc_WdVwVw(ws64_sum, vs32_sum_r0_hi, vs32_k1);
    ws64_sum    = Q6_Wd_vmulacc_WdVwVw(ws64_sum, vs32_sum_r1_lo, vs32_k0);
    ws64_sum    = Q6_Wd_vadd_WdWd(ws64_sum, Q6_W_vcombine_VV(Q6_V_vzero(), Q6_V_vsplat_R(1 << 27)));
    vs32_sum_lo = Q6_V_vor_VV(Q6_Vw_vasl_VwR(Q6_V_hi_W(ws64_sum), 4), Q6_Vuw_vlsr_VuwR(Q6_V_lo_W(ws64_sum), 28));

    ws64_sum    = Q6_Wd_vmul_VwVw(vs32_sum_l1_hi, vs32_k0);
    ws64_sum    = Q6_Wd_vmulacc_WdVwVw(ws64_sum, vs32_sum_l0_lo, vs32_k1);
    ws64_sum    = Q6_Wd_vmulacc_WdVwVw(ws64_sum, vs32_sum_l0_hi, vs32_k2);
    ws64_sum    = Q6_Wd_vmulacc_WdVwVw(ws64_sum, vs32_sum_c_lo,  vs32_k3);
    ws64_sum    = Q6_Wd_vmulacc_WdVwVw(ws64_sum, vs32_sum_c_hi,  vs32_k4);
    ws64_sum    = Q6_Wd_vmulacc_WdVwVw(ws64_sum, vs32_sum_r0_lo, vs32_k3);
    ws64_sum    = Q6_Wd_vmulacc_WdVwVw(ws64_sum, vs32_sum_r0_hi, vs32_k2);
    ws64_sum    = Q6_Wd_vmulacc_WdVwVw(ws64_sum, vs32_sum_r1_lo, vs32_k1);
    ws64_sum    = Q6_Wd_vmulacc_WdVwVw(ws64_sum, vs32_sum_r1_hi, vs32_k0);
    ws64_sum    = Q6_Wd_vadd_WdWd(ws64_sum, Q6_W_vcombine_VV(Q6_V_vzero(), Q6_V_vsplat_R(1 << 27)));
    vs32_sum_hi = Q6_V_vor_VV(Q6_Vw_vasl_VwR(Q6_V_hi_W(ws64_sum), 4), Q6_Vuw_vlsr_VuwR(Q6_V_lo_W(ws64_sum), 28));

    vs16_result = Q6_Vh_vsat_VwVw(vs32_sum_hi, vs32_sum_lo);
}

// using Tp = MI_U32
template <typename Tp, typename std::enable_if<std::is_same<Tp, MI_U32>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID Gaussian9x9VCore(HVX_Vector &vu32_src_p3, HVX_Vector &vu32_src_p2, HVX_Vector &vu32_src_p1,
                                            HVX_Vector &vu32_src_p0, HVX_Vector &vu32_src_c, HVX_Vector &vu32_src_n0,
                                            HVX_Vector &vu32_src_n1, HVX_Vector &vu32_src_n2, HVX_Vector &vu32_src_n3,
                                            HVX_VectorPair &wu64_sum, const MI_U32 *kernel)
{
    HVX_Vector vu32_k0 = Q6_V_vsplat_R(kernel[0]);
    HVX_Vector vu32_k1 = Q6_V_vsplat_R(kernel[1]);
    HVX_Vector vu32_k2 = Q6_V_vsplat_R(kernel[2]);
    HVX_Vector vu32_k3 = Q6_V_vsplat_R(kernel[3]);
    HVX_Vector vu32_k4 = Q6_V_vsplat_R(kernel[4]);

    wu64_sum = Q6_Wud_vmul32xlo16_VuwVuw(vu32_src_p3, vu32_k0);
    wu64_sum = Q6_Wud_vadd_WudWud(wu64_sum, Q6_Wud_vmul32xlo16_VuwVuw(vu32_src_p2, vu32_k1));
    wu64_sum = Q6_Wud_vadd_WudWud(wu64_sum, Q6_Wud_vmul32xlo16_VuwVuw(vu32_src_p1, vu32_k2));
    wu64_sum = Q6_Wud_vadd_WudWud(wu64_sum, Q6_Wud_vmul32xlo16_VuwVuw(vu32_src_p0, vu32_k3));
    wu64_sum = Q6_Wud_vadd_WudWud(wu64_sum, Q6_Wud_vmul32xlo16_VuwVuw(vu32_src_c,  vu32_k4));
    wu64_sum = Q6_Wud_vadd_WudWud(wu64_sum, Q6_Wud_vmul32xlo16_VuwVuw(vu32_src_n0, vu32_k3));
    wu64_sum = Q6_Wud_vadd_WudWud(wu64_sum, Q6_Wud_vmul32xlo16_VuwVuw(vu32_src_n1, vu32_k2));
    wu64_sum = Q6_Wud_vadd_WudWud(wu64_sum, Q6_Wud_vmul32xlo16_VuwVuw(vu32_src_n2, vu32_k1));
    wu64_sum = Q6_Wud_vadd_WudWud(wu64_sum, Q6_Wud_vmul32xlo16_VuwVuw(vu32_src_n3, vu32_k0));
}

// using Tp = MI_S32
template <typename Tp, typename std::enable_if<std::is_same<Tp, MI_S32>::value>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID Gaussian9x9VCore(HVX_Vector &vs32_src_p3, HVX_Vector &vs32_src_p2, HVX_Vector &vs32_src_p1,
                                            HVX_Vector &vs32_src_p0, HVX_Vector &vs32_src_c, HVX_Vector &vs32_src_n0,
                                            HVX_Vector &vs32_src_n1, HVX_Vector &vs32_src_n2, HVX_Vector &vs32_src_n3,
                                            HVX_VectorPair &ws64_sum, const MI_S32 *kernel)
{
    HVX_Vector vs32_k0 = Q6_V_vsplat_R(kernel[0]);
    HVX_Vector vs32_k1 = Q6_V_vsplat_R(kernel[1]);
    HVX_Vector vs32_k2 = Q6_V_vsplat_R(kernel[2]);
    HVX_Vector vs32_k3 = Q6_V_vsplat_R(kernel[3]);
    HVX_Vector vs32_k4 = Q6_V_vsplat_R(kernel[4]);

    ws64_sum = Q6_Wd_vmul_VwVw(vs32_src_p3, vs32_k0);
    ws64_sum = Q6_Wd_vmulacc_WdVwVw(ws64_sum, vs32_src_p2, vs32_k1);
    ws64_sum = Q6_Wd_vmulacc_WdVwVw(ws64_sum, vs32_src_p1, vs32_k2);
    ws64_sum = Q6_Wd_vmulacc_WdVwVw(ws64_sum, vs32_src_p0, vs32_k3);
    ws64_sum = Q6_Wd_vmulacc_WdVwVw(ws64_sum, vs32_src_c,  vs32_k4);
    ws64_sum = Q6_Wd_vmulacc_WdVwVw(ws64_sum, vs32_src_n0, vs32_k3);
    ws64_sum = Q6_Wd_vmulacc_WdVwVw(ws64_sum, vs32_src_n1, vs32_k2);
    ws64_sum = Q6_Wd_vmulacc_WdVwVw(ws64_sum, vs32_src_n2, vs32_k1);
    ws64_sum = Q6_Wd_vmulacc_WdVwVw(ws64_sum, vs32_src_n3, vs32_k0);
}

// Tp = MI_U32 MI_S32
template <typename Tp, typename Kt, typename std::enable_if<sizeof(Tp) == 4>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID Gaussian9x9HCore(HVX_VectorPair &wd64_sum_x0, HVX_VectorPair &wd64_sum_x1, HVX_VectorPair &wd64_sum_x2,
                                            HVX_Vector &vd32_result, const Kt *kernel)
{
    HVX_Vector vu32_k0   = Q6_V_vsplat_R(kernel[0]);
    HVX_Vector vu32_k1   = Q6_V_vsplat_R(kernel[1]);
    HVX_Vector vu32_k2   = Q6_V_vsplat_R(kernel[2]);
    HVX_Vector vu32_k3   = Q6_V_vsplat_R(kernel[3]);
    HVX_Vector vu32_k4   = Q6_V_vsplat_R(kernel[4]);
    MI_S32 align_size    = sizeof(MI_U32);
    HVX_Vector vu32_half = Q6_V_vsplat_R(1 << 27);

    HVX_Vector vd64_sum_x1_lo = Q6_V_lo_W(wd64_sum_x1);
    HVX_Vector vd64_sum_x1_hi = Q6_V_hi_W(wd64_sum_x1);

    HVX_Vector vd64_sum_l3_lo = Q6_V_vlalign_VVR(vd64_sum_x1_lo, Q6_V_lo_W(wd64_sum_x0), 4 * align_size);
    HVX_Vector vd64_sum_l3_hi = Q6_V_vlalign_VVR(vd64_sum_x1_hi, Q6_V_hi_W(wd64_sum_x0), 4 * align_size);
    HVX_Vector vd64_sum_l2_lo = Q6_V_vlalign_VVR(vd64_sum_x1_lo, Q6_V_lo_W(wd64_sum_x0), 3 * align_size);
    HVX_Vector vd64_sum_l2_hi = Q6_V_vlalign_VVR(vd64_sum_x1_hi, Q6_V_hi_W(wd64_sum_x0), 3 * align_size);
    HVX_Vector vd64_sum_l1_lo = Q6_V_vlalign_VVR(vd64_sum_x1_lo, Q6_V_lo_W(wd64_sum_x0), 2 * align_size);
    HVX_Vector vd64_sum_l1_hi = Q6_V_vlalign_VVR(vd64_sum_x1_hi, Q6_V_hi_W(wd64_sum_x0), 2 * align_size);
    HVX_Vector vd64_sum_l0_lo = Q6_V_vlalign_VVR(vd64_sum_x1_lo, Q6_V_lo_W(wd64_sum_x0), align_size);
    HVX_Vector vd64_sum_l0_hi = Q6_V_vlalign_VVR(vd64_sum_x1_hi, Q6_V_hi_W(wd64_sum_x0), align_size);
    HVX_Vector vd64_sum_c_lo  = vd64_sum_x1_lo;
    HVX_Vector vd64_sum_c_hi  = vd64_sum_x1_hi;
    HVX_Vector vd64_sum_r0_lo = Q6_V_valign_VVR(Q6_V_lo_W(wd64_sum_x2), vd64_sum_x1_lo, align_size);
    HVX_Vector vd64_sum_r0_hi = Q6_V_valign_VVR(Q6_V_hi_W(wd64_sum_x2), vd64_sum_x1_hi, align_size);
    HVX_Vector vd64_sum_r1_lo = Q6_V_valign_VVR(Q6_V_lo_W(wd64_sum_x2), vd64_sum_x1_lo, 2 * align_size);
    HVX_Vector vd64_sum_r1_hi = Q6_V_valign_VVR(Q6_V_hi_W(wd64_sum_x2), vd64_sum_x1_hi, 2 * align_size);
    HVX_Vector vd64_sum_r2_lo = Q6_V_valign_VVR(Q6_V_lo_W(wd64_sum_x2), vd64_sum_x1_lo, 3 * align_size);
    HVX_Vector vd64_sum_r2_hi = Q6_V_valign_VVR(Q6_V_hi_W(wd64_sum_x2), vd64_sum_x1_hi, 3 * align_size);
    HVX_Vector vd64_sum_r3_lo = Q6_V_valign_VVR(Q6_V_lo_W(wd64_sum_x2), vd64_sum_x1_lo, 4 * align_size);
    HVX_Vector vd64_sum_r3_hi = Q6_V_valign_VVR(Q6_V_hi_W(wd64_sum_x2), vd64_sum_x1_hi, 4 * align_size);

    HVX_VectorPair wd64_sum;
    HVX_Vector vd32_sum_lo, vd32_sum_hi;
    wd64_sum    = Q6_Wud_vmul32xlo16_VuwVuw(vd64_sum_l3_lo, vu32_k0);
    wd64_sum    = Q6_Wud_vadd_WudWud(wd64_sum, Q6_Wud_vmul32xlo16_VuwVuw(vd64_sum_l2_lo, vu32_k1));
    wd64_sum    = Q6_Wud_vadd_WudWud(wd64_sum, Q6_Wud_vmul32xlo16_VuwVuw(vd64_sum_l1_lo, vu32_k2));
    wd64_sum    = Q6_Wud_vadd_WudWud(wd64_sum, Q6_Wud_vmul32xlo16_VuwVuw(vd64_sum_l0_lo, vu32_k3));
    wd64_sum    = Q6_Wud_vadd_WudWud(wd64_sum, Q6_Wud_vmul32xlo16_VuwVuw(vd64_sum_c_lo,  vu32_k4));
    wd64_sum    = Q6_Wud_vadd_WudWud(wd64_sum, Q6_Wud_vmul32xlo16_VuwVuw(vd64_sum_r0_lo, vu32_k3));
    wd64_sum    = Q6_Wud_vadd_WudWud(wd64_sum, Q6_Wud_vmul32xlo16_VuwVuw(vd64_sum_r1_lo, vu32_k2));
    wd64_sum    = Q6_Wud_vadd_WudWud(wd64_sum, Q6_Wud_vmul32xlo16_VuwVuw(vd64_sum_r2_lo, vu32_k1));
    wd64_sum    = Q6_Wud_vadd_WudWud(wd64_sum, Q6_Wud_vmul32xlo16_VuwVuw(vd64_sum_r3_lo, vu32_k0));
    wd64_sum    = Q6_Wud_vadd_WudWud(wd64_sum, Q6_W_vcombine_VV(Q6_V_vzero(), vu32_half));

    vd32_sum_hi = Q6_Vw_vmpyie_VwVuh(vd64_sum_l3_hi, vu32_k0);
    vd32_sum_hi = Q6_Vw_vadd_VwVw(vd32_sum_hi, Q6_Vw_vmpyie_VwVuh(vd64_sum_l2_hi, vu32_k1));
    vd32_sum_hi = Q6_Vw_vadd_VwVw(vd32_sum_hi, Q6_Vw_vmpyie_VwVuh(vd64_sum_l1_hi, vu32_k2));
    vd32_sum_hi = Q6_Vw_vadd_VwVw(vd32_sum_hi, Q6_Vw_vmpyie_VwVuh(vd64_sum_l0_hi, vu32_k3));
    vd32_sum_hi = Q6_Vw_vadd_VwVw(vd32_sum_hi, Q6_Vw_vmpyie_VwVuh(vd64_sum_c_hi,  vu32_k4));
    vd32_sum_hi = Q6_Vw_vadd_VwVw(vd32_sum_hi, Q6_Vw_vmpyie_VwVuh(vd64_sum_r0_hi, vu32_k3));
    vd32_sum_hi = Q6_Vw_vadd_VwVw(vd32_sum_hi, Q6_Vw_vmpyie_VwVuh(vd64_sum_r1_hi, vu32_k2));
    vd32_sum_hi = Q6_Vw_vadd_VwVw(vd32_sum_hi, Q6_Vw_vmpyie_VwVuh(vd64_sum_r2_hi, vu32_k1));
    vd32_sum_hi = Q6_Vw_vadd_VwVw(vd32_sum_hi, Q6_Vw_vmpyie_VwVuh(vd64_sum_r3_hi, vu32_k0));

    vd32_sum_hi = Q6_Vw_vadd_VwVw(vd32_sum_hi, Q6_V_hi_W(wd64_sum));

    // shift hi
    vd32_sum_hi = Q6_Vw_vasl_VwR(vd32_sum_hi, 4);

    // shift lo
    vd32_sum_lo = Q6_Vuw_vlsr_VuwR(Q6_V_lo_W(wd64_sum), 28);

    // vadd or vor
    vd32_result = Q6_Vw_vadd_VwVw(vd32_sum_hi, vd32_sum_lo);
}

// Tp = MI_U8 MI_U16 MI_S16
template <typename Tp, typename Kt, typename std::enable_if<sizeof(Tp) != 4>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID Gaussian9x9HCore(HVX_VectorPair &w_sum_x0, HVX_VectorPair &w_sum_x1, HVX_VectorPair &w_sum_x2, HVX_VectorPair &w_sum_x3,
                                            HVX_Vector &v_result_x0, HVX_Vector &v_result_x1, const Kt *kernel, MI_S32 rest)
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
        MI_S32 align_size0 = (rest / 2) * sizeof(Kt);
        MI_S32 align_size1 = align_size0 + sizeof(Kt);
        v_sum_r0_lo = Q6_V_vlalign_safe_VVR(v_sum_x3_hi, v_sum_x2_hi, align_size1);
        v_sum_r0_hi = Q6_V_vlalign_safe_VVR(v_sum_x3_lo, v_sum_x2_lo, align_size0);
        v_sum_l0_lo = Q6_V_valign_safe_VVR(v_sum_x1_hi, v_sum_x0_hi, align_size0);
        v_sum_l0_hi = Q6_V_valign_safe_VVR(v_sum_x1_lo, v_sum_x0_lo, align_size1);
    }
    else
    {
        MI_S32 align_size = (rest / 2) * sizeof(Kt);
        v_sum_r0_lo = Q6_V_vlalign_safe_VVR(v_sum_x3_lo, v_sum_x2_lo, align_size);
        v_sum_r0_hi = Q6_V_vlalign_safe_VVR(v_sum_x3_hi, v_sum_x2_hi, align_size);
        v_sum_l0_lo = Q6_V_valign_safe_VVR(v_sum_x1_lo, v_sum_x0_lo, align_size);
        v_sum_l0_hi = Q6_V_valign_safe_VVR(v_sum_x1_hi, v_sum_x0_hi, align_size);
    }

    HVX_VectorPair w_sum_r0 = Q6_W_vcombine_VV(v_sum_r0_hi, v_sum_r0_lo);
    HVX_VectorPair w_sum_l0 = Q6_W_vcombine_VV(v_sum_l0_hi, v_sum_l0_lo);

    Gaussian9x9HCore<Tp>(w_sum_x0, w_sum_x1, w_sum_r0, v_result_x0, kernel);
    Gaussian9x9HCore<Tp>(w_sum_l0, w_sum_x2, w_sum_x3, v_result_x1, kernel);
}

// Tp = MI_U32 MI_S32
template <typename Tp, typename Kt, typename std::enable_if<sizeof(Tp) == 4>::type* = MI_NULL>
AURA_ALWAYS_INLINE AURA_VOID Gaussian9x9HCore(HVX_VectorPair &wd64_sum_x0, HVX_VectorPair &wd64_sum_x1, HVX_VectorPair &wd64_sum_x2, HVX_VectorPair &wd64_sum_x3,
                                            HVX_Vector &vd32_result_x0, HVX_Vector &vd32_result_x1, const Kt *kernel, MI_S32 rest)
{
    MI_S32 align_size = rest * sizeof(Kt);

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

    Gaussian9x9HCore<Tp>(wd64_sum_x0, wd64_sum_x1, wd64_sum_r0, vd32_result_x0, kernel);
    Gaussian9x9HCore<Tp>(wd64_sum_l0, wd64_sum_x2, wd64_sum_x3, vd32_result_x1, kernel);
}

template <typename Tp, BorderType BORDER_TYPE, MI_S32 C, typename Kt>
AURA_VOID Gaussian9x9Row(const Tp *src_p3, const Tp *src_p2, const Tp *src_p1, const Tp *src_p0, const Tp *src_c,
                       const Tp *src_n0, const Tp *src_n1, const Tp *src_n2, const Tp *src_n3, Tp *dst, MI_S32 width,
                       const Kt *kernel, const std::vector<Tp> &border_value)
{
    using MVType = typename MVHvxVector<C>::Type;
    using MWType = typename MWHvxVector<C>::Type;

    MI_S32 elem_counts = AURA_HVLEN / sizeof(Tp);
    MI_S32 back_offset = width - elem_counts;

    MVType mv_src_p3, mv_src_p2, mv_src_p1, mv_src_p0, mv_src_c, mv_src_n0, mv_src_n1, mv_src_n2, mv_src_n3;
    MWType mw_sum_x0, mw_sum_x1, mw_sum_x2;
    MVType mv_result;

    // left border
    {
        vload(src_p3, mv_src_p3);
        vload(src_p2, mv_src_p2);
        vload(src_p1, mv_src_p1);
        vload(src_p0, mv_src_p0);
        vload(src_c,  mv_src_c);
        vload(src_n0, mv_src_n0);
        vload(src_n1, mv_src_n1);
        vload(src_n2, mv_src_n2);
        vload(src_n3, mv_src_n3);

        #pragma unroll(C)
        for (MI_S32 ch = 0; ch < C; ch++)
        {
            HVX_Vector v_border_p3 = GetBorderVector<Tp, BORDER_TYPE, BorderArea::LEFT>(mv_src_p3.val[ch], src_p3[ch], border_value[ch]);
            HVX_Vector v_border_p2 = GetBorderVector<Tp, BORDER_TYPE, BorderArea::LEFT>(mv_src_p2.val[ch], src_p2[ch], border_value[ch]);
            HVX_Vector v_border_p1 = GetBorderVector<Tp, BORDER_TYPE, BorderArea::LEFT>(mv_src_p1.val[ch], src_p1[ch], border_value[ch]);
            HVX_Vector v_border_p0 = GetBorderVector<Tp, BORDER_TYPE, BorderArea::LEFT>(mv_src_p0.val[ch], src_p0[ch], border_value[ch]);
            HVX_Vector v_border_c  = GetBorderVector<Tp, BORDER_TYPE, BorderArea::LEFT>(mv_src_c.val[ch],  src_c[ch],  border_value[ch]);
            HVX_Vector v_border_n0 = GetBorderVector<Tp, BORDER_TYPE, BorderArea::LEFT>(mv_src_n0.val[ch], src_n0[ch], border_value[ch]);
            HVX_Vector v_border_n1 = GetBorderVector<Tp, BORDER_TYPE, BorderArea::LEFT>(mv_src_n1.val[ch], src_n1[ch], border_value[ch]);
            HVX_Vector v_border_n2 = GetBorderVector<Tp, BORDER_TYPE, BorderArea::LEFT>(mv_src_n2.val[ch], src_n2[ch], border_value[ch]);
            HVX_Vector v_border_n3 = GetBorderVector<Tp, BORDER_TYPE, BorderArea::LEFT>(mv_src_n3.val[ch], src_n3[ch], border_value[ch]);

            Gaussian9x9VCore<Tp>(v_border_p3, v_border_p2, v_border_p1, v_border_p0, v_border_c, v_border_n0,
                                 v_border_n1, v_border_n2, v_border_n3, mw_sum_x0.val[ch], kernel);
            Gaussian9x9VCore<Tp>(mv_src_p3.val[ch], mv_src_p2.val[ch], mv_src_p1.val[ch], mv_src_p0.val[ch], mv_src_c.val[ch], mv_src_n0.val[ch],
                                 mv_src_n1.val[ch], mv_src_n2.val[ch], mv_src_n3.val[ch], mw_sum_x1.val[ch], kernel);
        }
    }

    // main(0 ~ n-2)
    {
        for (MI_S32 x = elem_counts; x <= back_offset; x += elem_counts)
        {
            vload(src_p3 + C * x, mv_src_p3);
            vload(src_p2 + C * x, mv_src_p2);
            vload(src_p1 + C * x, mv_src_p1);
            vload(src_p0 + C * x, mv_src_p0);
            vload(src_c  + C * x, mv_src_c);
            vload(src_n0 + C * x, mv_src_n0);
            vload(src_n1 + C * x, mv_src_n1);
            vload(src_n2 + C * x, mv_src_n2);
            vload(src_n3 + C * x, mv_src_n3);

            #pragma unroll(C)
            for (MI_S32 ch = 0; ch < C; ch++)
            {
                Gaussian9x9VCore<Tp>(mv_src_p3.val[ch], mv_src_p2.val[ch], mv_src_p1.val[ch], mv_src_p0.val[ch], mv_src_c.val[ch], mv_src_n0.val[ch],
                                     mv_src_n1.val[ch], mv_src_n2.val[ch], mv_src_n3.val[ch], mw_sum_x2.val[ch],kernel);
                Gaussian9x9HCore<Tp>(mw_sum_x0.val[ch], mw_sum_x1.val[ch], mw_sum_x2.val[ch], mv_result.val[ch], kernel);
            }
            vstore(dst + C * (x - elem_counts), mv_result);

            mw_sum_x0 = mw_sum_x1;
            mw_sum_x1 = mw_sum_x2;
        }
    }

    // remain
    {
        MI_S32 last = C * (width - 1);
        MI_S32 rest = width % elem_counts;
        MVType mv_last;

        vload(src_p3 + C * back_offset, mv_src_p3);
        vload(src_p2 + C * back_offset, mv_src_p2);
        vload(src_p1 + C * back_offset, mv_src_p1);
        vload(src_p0 + C * back_offset, mv_src_p0);
        vload(src_c  + C * back_offset, mv_src_c);
        vload(src_n0 + C * back_offset, mv_src_n0);
        vload(src_n1 + C * back_offset, mv_src_n1);
        vload(src_n2 + C * back_offset, mv_src_n2);
        vload(src_n3 + C * back_offset, mv_src_n3);

        #pragma unroll(C)
        for (MI_S32 ch = 0; ch < C; ch++)
        {
            HVX_Vector v_border_p3 = GetBorderVector<Tp, BORDER_TYPE, BorderArea::RIGHT>(mv_src_p3.val[ch], src_p3[ch + last], border_value[ch]);
            HVX_Vector v_border_p2 = GetBorderVector<Tp, BORDER_TYPE, BorderArea::RIGHT>(mv_src_p2.val[ch], src_p2[ch + last], border_value[ch]);
            HVX_Vector v_border_p1 = GetBorderVector<Tp, BORDER_TYPE, BorderArea::RIGHT>(mv_src_p1.val[ch], src_p1[ch + last], border_value[ch]);
            HVX_Vector v_border_p0 = GetBorderVector<Tp, BORDER_TYPE, BorderArea::RIGHT>(mv_src_p0.val[ch], src_p0[ch + last], border_value[ch]);
            HVX_Vector v_border_c  = GetBorderVector<Tp, BORDER_TYPE, BorderArea::RIGHT>(mv_src_c.val[ch],  src_c[ch + last],  border_value[ch]);
            HVX_Vector v_border_n0 = GetBorderVector<Tp, BORDER_TYPE, BorderArea::RIGHT>(mv_src_n0.val[ch], src_n0[ch + last], border_value[ch]);
            HVX_Vector v_border_n1 = GetBorderVector<Tp, BORDER_TYPE, BorderArea::RIGHT>(mv_src_n1.val[ch], src_n1[ch + last], border_value[ch]);
            HVX_Vector v_border_n2 = GetBorderVector<Tp, BORDER_TYPE, BorderArea::RIGHT>(mv_src_n2.val[ch], src_n2[ch + last], border_value[ch]);
            HVX_Vector v_border_n3 = GetBorderVector<Tp, BORDER_TYPE, BorderArea::RIGHT>(mv_src_n3.val[ch], src_n3[ch + last], border_value[ch]);

            HVX_VectorPair w_sum_x2, w_sum_x3;
            Gaussian9x9VCore<Tp>(mv_src_p3.val[ch], mv_src_p2.val[ch], mv_src_p1.val[ch], mv_src_p0.val[ch], mv_src_c.val[ch],
                                 mv_src_n0.val[ch], mv_src_n1.val[ch], mv_src_n2.val[ch], mv_src_n3.val[ch], w_sum_x2, kernel);
            Gaussian9x9VCore<Tp>(v_border_p3, v_border_p2, v_border_p1, v_border_p0, v_border_c, v_border_n0,
                                 v_border_n1, v_border_n2, v_border_n3, w_sum_x3, kernel);

            Gaussian9x9HCore<Tp, Kt>(mw_sum_x0.val[ch], mw_sum_x1.val[ch], w_sum_x2, w_sum_x3, mv_result.val[ch], mv_last.val[ch], kernel, rest);
        }

        vstore(dst + C * (back_offset - rest), mv_result);
        vstore(dst + C * back_offset, mv_last);
    }
}

template <typename Tp, BorderType BORDER_TYPE, MI_S32 C>
static Status Gaussian9x9HvxImpl(const Mat &src, Mat &dst, const Mat &kmat, const std::vector<Tp> &border_value,
                                 const Tp *border_buffer, MI_S32 start_row, MI_S32 end_row)
{
    using Kt = typename std::conditional<sizeof(Tp) == 4, Tp, typename Promote<Tp>::Type>::type;

    MI_S32 width  = src.GetSizes().m_width;
    MI_S32 height = src.GetSizes().m_height;
    MI_S32 stride = src.GetStrides().m_width;

    const Kt *kernel = kmat.Ptr<Kt>(0);

    const Tp *src_p3 = src.Ptr<Tp, BORDER_TYPE>(start_row - 4, border_buffer);
    const Tp *src_p2 = src.Ptr<Tp, BORDER_TYPE>(start_row - 3, border_buffer);
    const Tp *src_p1 = src.Ptr<Tp, BORDER_TYPE>(start_row - 2, border_buffer);
    const Tp *src_p0 = src.Ptr<Tp, BORDER_TYPE>(start_row - 1, border_buffer);
    const Tp *src_c  = src.Ptr<Tp>(start_row);
    const Tp *src_n0 = src.Ptr<Tp, BORDER_TYPE>(start_row + 1, border_buffer);
    const Tp *src_n1 = src.Ptr<Tp, BORDER_TYPE>(start_row + 2, border_buffer);
    const Tp *src_n2 = src.Ptr<Tp, BORDER_TYPE>(start_row + 3, border_buffer);
    const Tp *src_n3 = src.Ptr<Tp, BORDER_TYPE>(start_row + 4, border_buffer);

    MI_U64 L2fetch_param = L2PfParam(stride, width * C * ElemTypeSize(src.GetElemType()), 1, 0);
    for (MI_S32 y = start_row; y < end_row; y++)
    {
        if (y + 5 < height)
        {
            L2Fetch(reinterpret_cast<MI_U32>(src.Ptr<Tp>(y + 5)), L2fetch_param);
        }

        Tp *dst_row  = dst.Ptr<Tp>(y);
        Gaussian9x9Row<Tp, BORDER_TYPE, C, Kt>(src_p3, src_p2, src_p1, src_p0, src_c, src_n0, src_n1, src_n2, src_n3, dst_row, width, kernel, border_value);

        src_p3 = src_p2;
        src_p2 = src_p1;
        src_p1 = src_p0;
        src_p0 = src_c;
        src_c  = src_n0;
        src_n0 = src_n1;
        src_n1 = src_n2;
        src_n2 = src_n3;
        src_n3 = src.Ptr<Tp, BORDER_TYPE>(y + 5, border_buffer);
    }

    return Status::OK;
}

template<typename Tp, BorderType BORDER_TYPE>
static Status Gaussian9x9HvxHelper(Context *ctx, const Mat &src, Mat &dst, const Mat &kmat,
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
            ret = wp->ParallelFor((MI_S32)0, height, Gaussian9x9HvxImpl<Tp, BORDER_TYPE, 1>,
                                  std::cref(src), std::ref(dst), std::cref(kmat), std::cref(border_value), border_buffer);
            break;
        }

        case 2:
        {
            ret = wp->ParallelFor((MI_S32)0, height, Gaussian9x9HvxImpl<Tp, BORDER_TYPE, 2>,
                                  std::cref(src), std::ref(dst), std::cref(kmat), std::cref(border_value), border_buffer);
            break;
        }

        case 3:
        {
            ret = wp->ParallelFor((MI_S32)0, height, Gaussian9x9HvxImpl<Tp, BORDER_TYPE, 3>,
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

template <typename Tp>
static Status Gaussian9x9HvxHelper(Context *ctx, const Mat &src, Mat &dst, const Mat &kmat,
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

            ret = Gaussian9x9HvxHelper<Tp, BorderType::CONSTANT>(ctx, src, dst, kmat, vec_border_value, border_buffer);

            AURA_FREE(ctx, border_buffer);
            break;
        }

        case BorderType::REPLICATE:
        {
            ret = Gaussian9x9HvxHelper<Tp, BorderType::REPLICATE>(ctx, src, dst, kmat, vec_border_value, border_buffer);
            break;
        }

        case BorderType::REFLECT_101:
        {
            ret = Gaussian9x9HvxHelper<Tp, BorderType::REFLECT_101>(ctx, src, dst, kmat, vec_border_value, border_buffer);
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

Status Gaussian9x9Hvx(Context *ctx, const Mat &src, Mat &dst, const Mat &kmat,
                      BorderType border_type, const Scalar &border_value)
{
    Status ret = Status::ERROR;

    switch (src.GetElemType())
    {
        case ElemType::U8:
        {
            ret = Gaussian9x9HvxHelper<MI_U8>(ctx, src, dst, kmat, border_type, border_value);
            break;
        }

        case ElemType::U16:
        {
            ret = Gaussian9x9HvxHelper<MI_U16>(ctx, src, dst, kmat, border_type, border_value);
            break;
        }

        case ElemType::S16:
        {
            ret = Gaussian9x9HvxHelper<MI_S16>(ctx, src, dst, kmat, border_type, border_value);
            break;
        }

        case ElemType::U32:
        {
            ret = Gaussian9x9HvxHelper<MI_U32>(ctx, src, dst, kmat, border_type, border_value);
            break;
        }

        case ElemType::S32:
        {
            ret = Gaussian9x9HvxHelper<MI_S32>(ctx, src, dst, kmat, border_type, border_value);
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