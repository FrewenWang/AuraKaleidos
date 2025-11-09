#ifndef AURA_RUNTIME_CORE_HEXAGON_DEVICE_MEMCPY_HPP__
#define AURA_RUNTIME_CORE_HEXAGON_DEVICE_MEMCPY_HPP__

#include "aura/runtime/core/types.h"

#if defined(__cplusplus)
extern "C"
{
#endif // __cplusplus

DT_VOID AuraMemCopy(DT_VOID *dst, const DT_VOID *src, DT_S32 len);

#if defined(__cplusplus)
}
#endif // __cplusplus

#endif // AURA_RUNTIME_CORE_HEXAGON_DEVICE_MEMCPY_HPP__