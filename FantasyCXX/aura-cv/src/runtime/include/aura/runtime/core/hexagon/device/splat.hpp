#ifndef AURA_RUNTIME_CORE_HEXAGON_DEVICE_SPLAT_HPP__
#define AURA_RUNTIME_CORE_HEXAGON_DEVICE_SPLAT_HPP__

#include "aura/runtime/core/types.h"

#include "hexagon_types.h"

namespace aura
{

template <typename St, typename std::enable_if<std::is_same<St, MI_U8>::value || std::is_same<St, MI_S8>::value>::type* = MI_NULL>
AURA_INLINE HVX_Vector vsplat(St value)
{
    return Q6_Vb_vsplat_R(value);
}

template <typename St, typename std::enable_if<std::is_same<St, MI_U16>::value || std::is_same<St, MI_S16>::value>::type* = MI_NULL>
AURA_INLINE HVX_Vector vsplat(St value)
{
    return Q6_Vh_vsplat_R(value);
}

template <typename St, typename std::enable_if<std::is_same<St, MI_U32>::value || std::is_same<St, MI_S32>::value>::type* = MI_NULL>
AURA_INLINE HVX_Vector vsplat(St value)
{
    return Q6_V_vsplat_R(value);
}

template <typename St, typename std::enable_if<std::is_same<St, MI_F32>::value>::type* = MI_NULL>
AURA_INLINE HVX_Vector vsplat(St value)
{
    return Q6_V_vsplat_R(*reinterpret_cast<MI_S32*>(&value));
}

} // namespace aura

#endif // AURA_RUNTIME_CORE_HEXAGON_DEVICE_SPLAT_HPP__