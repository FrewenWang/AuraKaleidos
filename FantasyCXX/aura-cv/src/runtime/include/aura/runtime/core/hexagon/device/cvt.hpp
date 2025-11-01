#ifndef AURA_RUNTIME_CORE_HEXAGON_DEVICE_CVT_HPP__
#define AURA_RUNTIME_CORE_HEXAGON_DEVICE_CVT_HPP__

#include "aura/runtime/core/types.h"

#include "hexagon_types.h"

namespace aura
{

// u32 -> f32
AURA_INLINE HVX_Vector Q6_Vsf_vcvt_Vuw(HVX_Vector vu32)
{
    HVX_Vector vd32_const_31   = Q6_V_vsplat_R(31);
    HVX_Vector vd32_const_127  = Q6_V_vsplat_R(127);
    HVX_Vector vd32_const_1    = Q6_V_vsplat_R(1);
    HVX_Vector vd32_const_0f   = Q6_V_vsplat_R(0x00000000);
    HVX_Vector vd32_const_0fix = Q6_V_vsplat_R(0xff800000);

    HVX_Vector vd32_count = Q6_Vuw_vcl0_Vuw(vu32);
    HVX_Vector vu32_e = Q6_Vuw_vadd_VuwVuw_sat(Q6_Vw_vsub_VwVw(vd32_const_31, vd32_count), vd32_const_127);
    vu32_e = Q6_Vw_vasl_VwR(vu32_e, 23);

    HVX_Vector v_u32_m = Q6_Vw_vasl_VwVw(vu32, Q6_Vuw_vadd_VuwVuw_sat(vd32_count, vd32_const_1));
    v_u32_m = Q6_Vuw_vlsr_VuwR(v_u32_m, 9);

    vu32 = Q6_V_vor_VV(v_u32_m, vu32_e);
    HVX_VectorPred q = Q6_Q_vcmp_eq_VwVw(vu32, vd32_const_0fix);
    
    return Q6_V_vmux_QVV(q, vd32_const_0f, vu32);
}

// s32 -> f32
AURA_INLINE HVX_Vector Q6_Vsf_vcvt_Vw(HVX_Vector vs32)
{
    HVX_Vector vd32_const_31   = Q6_V_vsplat_R(31);
    HVX_Vector vd32_const_127  = Q6_V_vsplat_R(127);
    HVX_Vector vd32_const_1    = Q6_V_vsplat_R(1);
    HVX_Vector vd32_const_0f   = Q6_V_vsplat_R(0x00000000);
    HVX_Vector vd32_const_0fix = Q6_V_vsplat_R(0xff800000);

    HVX_Vector vd32_const_sign = Q6_V_vsplat_R(0x80000000);
    vd32_const_sign = Q6_V_vand_VV(vd32_const_sign, vs32);

    vs32 = Q6_Vw_vabs_Vw(vs32);

    HVX_Vector vs32_count = Q6_Vuw_vcl0_Vuw(vs32);
    HVX_Vector vs32_e = Q6_Vuw_vadd_VuwVuw_sat(Q6_Vw_vsub_VwVw(vd32_const_31, vs32_count), vd32_const_127);
    vs32_e = Q6_Vw_vasl_VwR(vs32_e, 23);

    HVX_Vector vs32_m = Q6_Vw_vasl_VwVw(vs32, Q6_Vuw_vadd_VuwVuw_sat(vs32_count, vd32_const_1));
    vs32_m = Q6_Vuw_vlsr_VuwR(vs32_m, 9);

    vs32 = Q6_V_vor_VV(vs32_m, vs32_e);
    HVX_VectorPred q = Q6_Q_vcmp_eq_VwVw(vs32, vd32_const_0fix);
    vs32 = Q6_V_vmux_QVV(q, vd32_const_0f, vs32);

    return Q6_V_vor_VV(vd32_const_sign, vs32);
}

// f32 -> u32
// limitation: saturation is not supported, make sure 0 < f32 < 4294967295.f(max value of s32)
AURA_INLINE HVX_Vector Q6_Vuw_vcvt_Vsf(HVX_Vector vf32)
{
    HVX_Vector vd32_m_and  = Q6_V_vsplat_R(0x7fffff);
    HVX_Vector vd32_150    = Q6_V_vsplat_R(0x96);
    HVX_Vector vd32_127    = Q6_V_vsplat_R(0x7f);
    HVX_Vector vd32_1      = Q6_V_vsplat_R(0x1);
    HVX_Vector vd32_check0 = Q6_V_vsplat_R(127);
    HVX_Vector vd32_checkn = Q6_V_vsplat_R(128);

    HVX_Vector vd32_m = Q6_V_vand_VV(vf32, vd32_m_and);
    HVX_Vector vd32_e = Q6_Vw_vasr_VwR(vf32, 23);

    HVX_Vector vd32_8_t = Q6_Vw_vasr_VwR(vf32, 24);

    HVX_VectorPred q_0 = Q6_Q_vcmp_gt_VuwVuw(vd32_check0, vd32_e);
    HVX_VectorPred q_1 = Q6_Q_vcmp_gt_VuwVuw(vd32_8_t, vd32_checkn);
    q_0 = Q6_Q_or_QQ(q_0, q_1);

    HVX_VectorPred vd32_out0 = Q6_Vw_vadd_VwVw(Q6_Vw_vasr_VwVw(vd32_m, Q6_Vw_vsub_VwVw(vd32_150, vd32_e)),
                               Q6_Vw_vasl_VwVw(vd32_1, Q6_Vw_vsub_VwVw(vd32_e, vd32_127)));
    return Q6_V_vmux_QVV(q_0, Q6_V_vzero(), vd32_out0);
}

// f32 -> s32
// limitation:  make sure -2147483648(min value of s32) < f32 < 2147483647.f (max value of s32).
// the result maybe not equal to C. the behavior On C is undefined, it is due to the compiler
AURA_INLINE HVX_Vector Q6_Vw_vcvt_Vsf(HVX_Vector vf32)
{
    HVX_Vector vd32_m_and = Q6_V_vsplat_R(0x7FFFFF);
    HVX_Vector vd32_150    = Q6_V_vsplat_R(0x96);
    HVX_Vector vd32_127    = Q6_V_vsplat_R(0x7F);
    HVX_Vector vd32_1      = Q6_V_vsplat_R(0x1);
    HVX_Vector vd32_check0 = Q6_V_vsplat_R(127);
    HVX_Vector vd32_const_sign = Q6_V_vsplat_R(0x80000000);
    HVX_Vector vd32_const_sign_tmp = Q6_V_vand_VV(vd32_const_sign, vf32);

    HVX_VectorPred q = Q6_Q_vcmp_eq_VwVw(vd32_const_sign_tmp, vd32_const_sign);
    HVX_Vector vd32_m = Q6_V_vand_VV(vf32, vd32_m_and);

    vf32 = Q6_Vw_vasl_VwR(vf32, 1);
    HVX_Vector vd32_e = Q6_Vuw_vlsr_VuwR(vf32, 24);

    HVX_VectorPred q_0 = Q6_Q_vcmp_gt_VuwVuw(vd32_check0, vd32_e);

    HVX_VectorPred vd32_out0 = Q6_Vw_vadd_VwVw(Q6_Vw_vasr_VwVw(vd32_m, Q6_Vw_vsub_VwVw(vd32_150, vd32_e)),
                               Q6_Vw_vasl_VwVw(vd32_1, Q6_Vw_vsub_VwVw(vd32_e, vd32_127)));
    vd32_out0 = Q6_V_vmux_QVV(q_0, Q6_V_vzero(), vd32_out0);
    return Q6_V_vmux_QVV(q, Q6_Vw_vsub_VwVw(Q6_V_vzero(), vd32_out0), vd32_out0);
}

// f32 -> s32
AURA_INLINE HVX_Vector Q6_Vw_vcvt_Vsf_rnd(HVX_Vector vf32)
{
    HVX_Vector vd32_m_and  = Q6_V_vsplat_R(0x7FFFFF);
    HVX_Vector vd32_150    = Q6_V_vsplat_R(0x96);
    HVX_Vector vd32_127    = Q6_V_vsplat_R(0x7F);
    HVX_Vector vd32_1      = Q6_V_vsplat_R(0x1);
    HVX_Vector vd32_check0 = Q6_V_vsplat_R(127);
    HVX_Vector vd32_const_sign = Q6_V_vsplat_R(0x80000000);
    HVX_Vector vd32_const_sign_tmp = Q6_V_vand_VV(vd32_const_sign, vf32);

    HVX_VectorPred q = Q6_Q_vcmp_eq_VwVw(vd32_const_sign_tmp, vd32_const_sign);
    vf32 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(vf32, Q6_V_vmux_QVV(q, Q6_V_vsplat_R(0xbF000000), Q6_V_vsplat_R(0x3F000000))));

    HVX_Vector vd32_m = Q6_V_vand_VV(vf32, vd32_m_and);

    vf32 = Q6_Vw_vasl_VwR(vf32, 1);
    HVX_Vector vd32_e = Q6_Vuw_vlsr_VuwR(vf32, 24);

    HVX_VectorPred q_0 = Q6_Q_vcmp_gt_VuwVuw(vd32_check0, vd32_e);
        
    HVX_VectorPred vd32_out0 = Q6_Vw_vadd_VwVw(Q6_Vw_vasr_VwVw(vd32_m, Q6_Vw_vsub_VwVw(vd32_150, vd32_e)), 
                               Q6_Vw_vasl_VwVw(vd32_1, Q6_Vw_vsub_VwVw(vd32_e, vd32_127)));
    vd32_out0 = Q6_V_vmux_QVV(q_0, Q6_V_vzero(), vd32_out0);
    return Q6_V_vmux_QVV(q, Q6_Vw_vsub_VwVw(Q6_V_vzero(), vd32_out0), vd32_out0);
}

} // namespace aura

#endif // AURA_RUNTIME_CORE_HEXAGON_DEVICE_CVT_HPP__