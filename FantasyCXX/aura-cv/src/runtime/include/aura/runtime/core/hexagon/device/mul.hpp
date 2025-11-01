#ifndef AURA_RUNTIME_CORE_HEXAGON_DEVICE_MUL_HPP__
#define AURA_RUNTIME_CORE_HEXAGON_DEVICE_MUL_HPP__

#include "aura/runtime/core/types.h"
#include "aura/runtime/core/hexagon/device/add.hpp"

#include "hexagon_types.h"

namespace aura
{

// s64 = s32 * s32
AURA_INLINE HVX_VectorPair Q6_Wd_vmul_VwVw(HVX_Vector vs32_u, HVX_Vector vs32_v)
{
    HVX_VectorPair wd64_tmp = Q6_W_vmpye_VwVuh(vs32_u, vs32_v);
    return Q6_W_vmpyoacc_WVwVh(wd64_tmp, vs32_u, vs32_v);
}

// u64 = u32 * u32
AURA_INLINE HVX_VectorPair Q6_Wud_vmul_VuwVuw(HVX_Vector vu32_u, HVX_Vector vu32_v)
{
    HVX_VectorPair wd64_tmp0 = Q6_Wuw_vmpy_VuhVuh(vu32_u, vu32_v);
    HVX_Vector vd32_tmp0 = Q6_Vuw_vlsr_VuwR(vu32_u, 16);
    HVX_VectorPred q_0 = Q6_Q_vsetq_R(0);
    HVX_VectorPair wd64_tmp1 = Q6_Wh_vshuffoe_VhVh(vu32_u, vd32_tmp0);
    wd64_tmp1 = Q6_Wuw_vmpy_VuhVuh(Q6_V_lo_W(wd64_tmp1), vu32_v);
    vd32_tmp0 = Q6_Vw_vadd_VwVwQ_carry(Q6_V_hi_W(wd64_tmp1), Q6_V_lo_W(wd64_tmp1), &q_0);
    HVX_Vector vd32_tmp1 = Q6_Vuw_vlsr_VuwR(vd32_tmp0, 16);

    HVX_VectorPred q_1 = Q6_V_vsplat_R(0x00000000);
    HVX_Vector vd32_conth1 = Q6_V_vsplat_R(0x00010000);
    vd32_tmp0 = Q6_Vw_vadd_VwVwQ_carry(Q6_V_lo_W(wd64_tmp0), Q6_Vw_vasl_VwR(vd32_tmp0, 16), &q_1);
    vd32_tmp1 = Q6_Vw_vadd_VwVwQ_carry(Q6_V_hi_W(wd64_tmp0), vd32_tmp1, &q_1);
    vd32_tmp1 = Q6_Vw_condacc_QVwVw(q_0, vd32_tmp1, vd32_conth1);
    return Q6_W_vcombine_VV(vd32_tmp1, vd32_tmp0);
}

// u64 += u32 * u32
AURA_INLINE HVX_VectorPair Q6_Wud_vmulacc_WudVuwVuw(HVX_VectorPair wu64_acc, HVX_Vector vu32_u, HVX_Vector vu32_v)
{
    HVX_VectorPair wu64_tmp = Q6_Wud_vmul_VuwVuw(vu32_u, vu32_v);
    return Q6_Wud_vadd_WudWud(wu64_tmp, wu64_acc);
}

// s64 += s32 * s32
AURA_INLINE HVX_VectorPair Q6_Wd_vmulacc_WdVwVw(HVX_VectorPair ws64_acc, HVX_Vector vs32_u, HVX_Vector vs32_v)
{
    HVX_VectorPair ws64_tmp = Q6_Wd_vmul_VwVw(vs32_u, vs32_v);
    return Q6_Wd_vadd_WdWd(ws64_tmp, ws64_acc);
}

// s32 = s32 * s16
AURA_INLINE HVX_Vector Q6_Vw_vmul32xhi16_VwVw(HVX_Vector vs32_u, HVX_Vector vs16_v)
{
    return Q6_Vw_vmpyio_VwVh(vs32_u, vs16_v);
}

// s32 = s32 * u16
AURA_INLINE HVX_Vector Q6_Vw_vmul32xlo16_VwVuw(HVX_Vector vs32_u, HVX_Vector vu16_v)
{
    return Q6_Vw_vmpyie_VwVuh(vs32_u, vu16_v);
}

// s32 = (s32 * u16) >> 16
AURA_INLINE HVX_Vector Q6_Vw_vmul32xlo16lsr16_VwVuw(HVX_Vector vs32_u, HVX_Vector vu16_v)
{   
    return Q6_Vw_vmpye_VwVuh(vs32_u, vu16_v);
}

// U32 = (U32 * u16) >> 16
AURA_INLINE HVX_Vector Q6_Vuw_vmul32xlo16lsr16_VuwVuw(HVX_Vector vu32_u, HVX_Vector vu16_v)
{   
    HVX_VectorPair wd64_tmp0 = Q6_Wuw_vmpy_VuhVuh(vu32_u, vu16_v);
    HVX_Vector vd32_tmp0 = Q6_Vuw_vlsr_VuwR(vu32_u, 16);
    HVX_VectorPair wd64_tmp1 = Q6_Wuw_vmpy_VuhVuh(vd32_tmp0, vu16_v);
    HVX_Vector vd32_tmp1 = Q6_Vuw_vlsr_VuwR(Q6_V_lo_W(wd64_tmp0), 16);
    return Q6_Vw_vadd_VwVw(vd32_tmp1, Q6_V_lo_W(wd64_tmp1));
}

// U64 = U32 * u16
AURA_INLINE HVX_VectorPair Q6_Wud_vmul32xlo16_VuwVuw(HVX_Vector vu32_u, HVX_Vector vu16_v)
{   
    HVX_VectorPair wd64_tmp0 = Q6_Wuw_vmpy_VuhVuh(vu32_u, vu16_v);
    HVX_Vector vd32_tmp0 = Q6_Vuw_vlsr_VuwR(vu32_u, 16);
    HVX_VectorPair wd64_tmp1 = Q6_Wuw_vmpy_VuhVuh(vd32_tmp0, vu16_v);
    HVX_VectorPred q = Q6_Q_vsetq_R(0);
    HVX_Vector vd32_tmp1 = Q6_Vw_vasl_VwR(Q6_V_lo_W(wd64_tmp1), 16);
    vd32_tmp1 = Q6_Vw_vadd_VwVwQ_carry(vd32_tmp1, Q6_V_lo_W(wd64_tmp0), &q);
    vd32_tmp0 = Q6_Vuw_vlsr_VuwR(Q6_V_lo_W(wd64_tmp1), 16);
    vd32_tmp0 = Q6_Vw_vadd_VwVwQ_carry(vd32_tmp0, Q6_V_vzero(), &q);
    return Q6_W_vcombine_VV(vd32_tmp0, vd32_tmp1);
}

} // namespace aura

#endif // AURA_RUNTIME_CORE_HEXAGON_DEVICE_MUL_HPP__