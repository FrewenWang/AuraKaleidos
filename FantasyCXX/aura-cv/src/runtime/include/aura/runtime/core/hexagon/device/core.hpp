#ifndef AURA_RUNTIME_CORE_HEXAGON_DEVICE_CORE_HPP__
#define AURA_RUNTIME_CORE_HEXAGON_DEVICE_CORE_HPP__

#include "aura/runtime/core/hexagon/comm.hpp"
#include "aura/runtime/core/types.h"

#include "hexagon_types.h"

#define L2PfParam(s, w, h, d)   static_cast<MI_U64>(HEXAGON_V64_CREATE_H((d), (s), (w), (h)))

namespace aura
{

AURA_ALWAYS_INLINE void L2Fetch(MI_U32 addr, MI_U64 param)
{
    __asm__ __volatile__ ("l2fetch(%0,%1)" : : "r"(addr), "r"(param));
}

} // namespace aura

#endif // AURA_RUNTIME_CORE_HEXAGON_DEVICE_CORE_HPP__