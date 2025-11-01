#ifndef AURA_RUNTIME_CORE_HEXAGON_DEVICE_MINMAX_HPP__
#define AURA_RUNTIME_CORE_HEXAGON_DEVICE_MINMAX_HPP__

#include "aura/runtime/core/types.h"
#include "hexagon_types.h"

namespace aura
{

AURA_ALWAYS_INLINE AURA_VOID Q6_vminmax_VubVub(HVX_Vector &vu8_u, HVX_Vector &vu8_v)
{
    HVX_Vector vu8_min = Q6_Vub_vmin_VubVub(vu8_u, vu8_v);
    HVX_Vector vu8_max = Q6_Vub_vmax_VubVub(vu8_u, vu8_v);
    vu8_u = vu8_min;
    vu8_v = vu8_max;
}

AURA_ALWAYS_INLINE AURA_VOID Q6_vminmax_VbVb(HVX_Vector &vs8_u, HVX_Vector &vs8_v)
{
    HVX_Vector vs8_min = Q6_Vb_vmin_VbVb(vs8_u, vs8_v);
    HVX_Vector vs8_max = Q6_Vb_vmax_VbVb(vs8_u, vs8_v);
    vs8_u = vs8_min;
    vs8_v = vs8_max;
}

AURA_ALWAYS_INLINE AURA_VOID Q6_vminmax_VuhVuh(HVX_Vector &vu16_u, HVX_Vector &vu16_v)
{
    HVX_Vector vu16_min = Q6_Vuh_vmin_VuhVuh(vu16_u, vu16_v);
    HVX_Vector vu16_max = Q6_Vuh_vmax_VuhVuh(vu16_u, vu16_v);
    vu16_u = vu16_min;
    vu16_v = vu16_max;
}

AURA_ALWAYS_INLINE AURA_VOID Q6_vminmax_VhVh(HVX_Vector &vs16_u, HVX_Vector &vs16_v)
{
    HVX_Vector vs16_min = Q6_Vh_vmin_VhVh(vs16_u, vs16_v);
    HVX_Vector vs16_max = Q6_Vh_vmax_VhVh(vs16_u, vs16_v);
    vs16_u = vs16_min;
    vs16_v = vs16_max;
}

} // namespace aura

#endif // AURA_RUNTIME_CORE_HEXAGON_DEVICE_MINMAX_HPP__