#ifndef AURA_RUNTIME_CORE_HEXAGON_DEVICE_ALIGN_HPP__
#define AURA_RUNTIME_CORE_HEXAGON_DEVICE_ALIGN_HPP__

#include "aura/runtime/core/hexagon/comm.hpp"
#include "aura/runtime/core/types.h"

#include "hexagon_types.h"

namespace aura
{

AURA_INLINE HVX_Vector Q6_V_vlalign_safe_VVR(HVX_Vector v_u, HVX_Vector v_v, MI_S32 align)
{
    return (AURA_HVLEN == align) ? v_v : Q6_V_vlalign_VVR(v_u, v_v, align);
}

AURA_INLINE HVX_Vector Q6_V_valign_safe_VVR(HVX_Vector v_u, HVX_Vector v_v, MI_S32 align)
{
    return (AURA_HVLEN == align) ? v_u : Q6_V_valign_VVR(v_u, v_v, align);
}

} // namespace aura

#endif // AURA_RUNTIME_CORE_HEXAGON_DEVICE_ALIGN_HPP__