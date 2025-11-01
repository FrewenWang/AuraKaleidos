#ifndef AURA_RUNTIME_CORE_HEXAGON_DEVICE_ADD_HPP__
#define AURA_RUNTIME_CORE_HEXAGON_DEVICE_ADD_HPP__

#include "aura/runtime/core/types.h"

#include "hexagon_types.h"

namespace aura
{

// u64 = u32 + u32
AURA_INLINE HVX_VectorPair Q6_Wud_vadd_VuwVuw(HVX_Vector vu32_u, HVX_Vector vu32_v)
{
    HVX_VectorPred q = Q6_V_vsplat_R(0x00000000);
    HVX_Vector vu64_sum_l = Q6_Vw_vadd_VwVwQ_carry(vu32_u, vu32_v, &q);
    HVX_Vector vu64_sum_h = Q6_Vw_vadd_VwVwQ_carry(Q6_V_vzero(), Q6_V_vzero(), &q);
    return Q6_W_vcombine_VV(vu64_sum_h, vu64_sum_l);
}

// s64 = s32 + s32
AURA_INLINE HVX_VectorPair Q6_Wd_vadd_VwVw(HVX_Vector vs32_u, HVX_Vector vs32_v)
{
    HVX_VectorPred q = Q6_V_vsplat_R(0x00000000);
    HVX_Vector vs64_sum_l = Q6_Vw_vadd_VwVwQ_carry(vs32_u, vs32_v, &q);
    HVX_Vector vs64_tmp0 = Q6_Vw_vasr_VwR(vs32_u, 31);
    HVX_Vector vs64_tmp1 = Q6_Vw_vasr_VwR(vs32_v, 31);
    HVX_Vector vs64_sum_h = Q6_Vw_vadd_VwVwQ_carry(vs64_tmp0, vs64_tmp1, &q);
    return Q6_W_vcombine_VV(vs64_sum_h, vs64_sum_l);
}

// s64 = s64 + s64
AURA_INLINE HVX_VectorPair Q6_Wd_vadd_WdWd(HVX_VectorPair vs64_u, HVX_VectorPair vs64_v)
{
    HVX_VectorPred q = Q6_V_vsplat_R(0x00000000);
    HVX_Vector vs64_sum_l = Q6_Vw_vadd_VwVwQ_carry(Q6_V_lo_W(vs64_u), Q6_V_lo_W(vs64_v), &q);
    HVX_Vector vs64_sum_h = Q6_Vw_vadd_VwVwQ_carry(Q6_V_hi_W(vs64_u), Q6_V_hi_W(vs64_v), &q);
    return Q6_W_vcombine_VV(vs64_sum_h, vs64_sum_l);
}

// u64 = u64 + u64
AURA_INLINE HVX_VectorPair Q6_Wud_vadd_WudWud(HVX_VectorPair vu64_u, HVX_VectorPair vu64_v)
{
    return Q6_Wd_vadd_WdWd(vu64_u, vu64_v);
}

} // namespace aura

#endif // AURA_RUNTIME_CORE_HEXAGON_DEVICE_ADD_HPP__