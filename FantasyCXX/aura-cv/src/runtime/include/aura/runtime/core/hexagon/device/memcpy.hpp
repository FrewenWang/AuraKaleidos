#ifndef AURA_RUNTIME_CORE_HEXAGON_DEVICE_MEMCPY_HPP__
#define AURA_RUNTIME_CORE_HEXAGON_DEVICE_MEMCPY_HPP__

#include "aura/runtime/core/types.h"

#if defined(__cplusplus)
extern "C"
{
#endif // __cplusplus

AURA_VOID AuraMemCopy(AURA_VOID *dst, const AURA_VOID *src, MI_S32 len);

#if defined(__cplusplus)
}
#endif // __cplusplus

#endif // AURA_RUNTIME_CORE_HEXAGON_DEVICE_MEMCPY_HPP__