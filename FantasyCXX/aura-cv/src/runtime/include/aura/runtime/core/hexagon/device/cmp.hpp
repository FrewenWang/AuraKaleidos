#ifndef AURA_RUNTIME_CORE_HEXAGON_DEVICE_CMP_HPP__
#define AURA_RUNTIME_CORE_HEXAGON_DEVICE_CMP_HPP__

#include "aura/runtime/core/types.h"

#include "hexagon_types.h"

namespace aura
{

// vs8_u >= vs8_v ? 1 : 0
AURA_INLINE HVX_VectorPred Q6_Q_vcmp_ge_VbVb(HVX_Vector vs8_u, HVX_Vector vs8_v)
{
    HVX_VectorPred q = Q6_Q_vcmp_eq_VbVb(vs8_u, vs8_v);
    return Q6_Q_vcmp_gtor_QVbVb(q, vs8_u, vs8_v);
}

// vu8_u >= vu8_v ? 1 : 0
AURA_INLINE HVX_VectorPred Q6_Q_vcmp_ge_VubVub(HVX_Vector vu8_u, HVX_Vector vu8_v)
{
    HVX_VectorPred q = Q6_Q_vcmp_eq_VbVb(vu8_u, vu8_v);
    return Q6_Q_vcmp_gtor_QVubVub(q, vu8_u, vu8_v);
}

// vs16_u >= vs16_v ? 1 : 0
AURA_INLINE HVX_VectorPred Q6_Q_vcmp_ge_VhVh(HVX_Vector vs16_u, HVX_Vector vs16_v)
{
    HVX_VectorPred q = Q6_Q_vcmp_eq_VhVh(vs16_u, vs16_v);
    return Q6_Q_vcmp_gtor_QVhVh(q, vs16_u, vs16_v);
}

// vu16_u >= vu16_v ? 1 : 0
AURA_INLINE HVX_VectorPred Q6_Q_vcmp_ge_VuhVuh(HVX_Vector vu16_u, HVX_Vector vu16_v)
{
    HVX_VectorPred q = Q6_Q_vcmp_eq_VhVh(vu16_u, vu16_v);
    return Q6_Q_vcmp_gtor_QVuhVuh(q, vu16_u, vu16_v);
}

// vs32_u >= vs32_v ? 1 : 0
AURA_INLINE HVX_VectorPred Q6_Q_vcmp_ge_VwVw(HVX_Vector vs32_u, HVX_Vector vs32_v)
{
    HVX_VectorPred q = Q6_Q_vcmp_eq_VwVw(vs32_u, vs32_v);
    return Q6_Q_vcmp_gtor_QVwVw(q, vs32_u, vs32_v);
}

// vu32_u >= vu32_v ? 1 : 0
AURA_INLINE HVX_VectorPred Q6_Q_vcmp_ge_VuwVuw(HVX_Vector vu32_u, HVX_Vector vu32_v)
{
    HVX_VectorPred q = Q6_Q_vcmp_eq_VwVw(vu32_u, vu32_v);
    return Q6_Q_vcmp_gtor_QVuwVuw(q, vu32_u, vu32_v);
}

} // namespace aura

#endif // AURA_RUNTIME_CORE_HEXAGON_DEVICE_CMP_HPP__