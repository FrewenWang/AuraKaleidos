#ifndef AURA_RUNTIME_CORE_NEON_MIN_HPP__
#define AURA_RUNTIME_CORE_NEON_MIN_HPP__

#include <arm_neon.h>

namespace aura
{

namespace neon
{

#define DECLFUN(dtype, vtype, prefix, postfix)        \
    inline vtype vmin(const vtype &v, const vtype &u) \
    {                                                 \
        return prefix##_##postfix(v, u);              \
    }

DECLFUN(uint8x8_t,   uint8x8_t,   vmin,  u8)
DECLFUN(uint8x16_t,   uint8x16_t, vminq, u8)
DECLFUN(int8x8_t,    int8x8_t,    vmin,  s8)
DECLFUN(int8x16_t,    int8x16_t,  vminq, s8)
DECLFUN(uint16x4_t,  uint16x4_t,  vmin,  u16)
DECLFUN(uint16x8_t,  uint16x8_t,  vminq, u16)
DECLFUN(int16x4_t,   int16x4_t,   vmin,  s16)
DECLFUN(int16x8_t,   int16x8_t,   vminq, s16)
DECLFUN(uint32x2_t,  uint32x2_t,  vmin,  u32)
DECLFUN(uint32x4_t,  uint32x4_t,  vminq, u32)
DECLFUN(int32x2_t,   int32x2_t,   vmin,  s32)
DECLFUN(int32x4_t,   int32x4_t,   vminq, s32)
DECLFUN(float32x2_t, float32x2_t, vmin,  f32)
DECLFUN(float32x4_t, float32x4_t, vminq, f32)

#  if defined(AURA_ENABLE_NEON_FP16)
DECLFUN(float16x4_t,   float16x4_t, vmin, f16)
DECLFUN(float16x8_t,   float16x8_t, vminq, f16)
#  endif
#undef DECLFUN

#define DECLFUN(dtype, vtype, prefix, postfix)         \
    inline vtype vpmin(const vtype &v, const vtype &u) \
    {                                                  \
        return prefix##_##postfix(v, u);               \
    }

DECLFUN(uint8x8_t,   uint8x8_t,   vpmin,  u8)
DECLFUN(int8x8_t,    int8x8_t,    vpmin,  s8)
DECLFUN(uint16x4_t,  uint16x4_t,  vpmin,  u16)
DECLFUN(int16x4_t,   int16x4_t,   vpmin,  s16)
DECLFUN(uint32x2_t,  uint32x2_t,  vpmin,  u32)
DECLFUN(int32x2_t,   int32x2_t,   vpmin,  s32)
DECLFUN(float32x2_t, float32x2_t, vpmin,  s32)
#undef DECLFUN

} // namespace neon

} // namespace aura

#endif // AURA_RUNTIME_CORE_NEON_MIN_HPP__