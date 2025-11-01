#ifndef AURA_RUNTIME_CORE_NEON_MAX_HPP__
#define AURA_RUNTIME_CORE_NEON_MAX_HPP__

#include <arm_neon.h>

namespace aura
{

namespace neon
{

#define DECLFUN(dtype, vtype, prefix, postfix)        \
    inline vtype vmax(const vtype &v, const vtype &u) \
    {                                                 \
        return prefix##_##postfix(v, u);              \
    }

DECLFUN(uint8x8_t,   uint8x8_t,   vmax,  u8)
DECLFUN(uint8x16_t,   uint8x16_t, vmaxq, u8)
DECLFUN(int8x8_t,    int8x8_t,    vmax,  s8)
DECLFUN(int8x16_t,    int8x16_t,  vmaxq, s8)
DECLFUN(uint16x4_t,  uint16x4_t,  vmax,  u16)
DECLFUN(uint16x8_t,  uint16x8_t,  vmaxq, u16)
DECLFUN(int16x4_t,   int16x4_t,   vmax,  s16)
DECLFUN(int16x8_t,   int16x8_t,   vmaxq, s16)
DECLFUN(uint32x2_t,  uint32x2_t,  vmax,  u32)
DECLFUN(uint32x4_t,  uint32x4_t,  vmaxq, u32)
DECLFUN(int32x2_t,   int32x2_t,   vmax,  s32)
DECLFUN(int32x4_t,   int32x4_t,   vmaxq, s32)
DECLFUN(float32x2_t, float32x2_t, vmax,  f32)
DECLFUN(float32x4_t, float32x4_t, vmaxq, f32)

#  if defined(AURA_ENABLE_NEON_FP16)
DECLFUN(float16x4_t,   float16x4_t, vmax, f16)
DECLFUN(float16x8_t,   float16x8_t, vmaxq, f16)
#  endif
#undef DECLFUN

#define DECLFUN(dtype, vtype, prefix, postfix)         \
    inline vtype vpmax(const vtype &v, const vtype &u) \
    {                                                  \
        return prefix##_##postfix(v, u);               \
    }

DECLFUN(uint8x8_t,   uint8x8_t,   vpmax,  u8)
DECLFUN(int8x8_t,    int8x8_t,    vpmax,  s8)
DECLFUN(uint16x4_t,  uint16x4_t,  vpmax,  u16)
DECLFUN(int16x4_t,   int16x4_t,   vpmax,  s16)
DECLFUN(uint32x2_t,  uint32x2_t,  vpmax,  u32)
DECLFUN(int32x2_t,   int32x2_t,   vpmax,  s32)
DECLFUN(float32x2_t, float32x2_t, vpmax,  s32)
#undef DECLFUN

} // namespace neon

} // namespace aura

#endif // AURA_RUNTIME_CORE_NEON_MAX_HPP__