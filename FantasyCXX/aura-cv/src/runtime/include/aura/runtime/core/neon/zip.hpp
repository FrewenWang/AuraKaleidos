#ifndef AURA_RUNTIME_CORE_NEON_ZIP_HPP__
#define AURA_RUNTIME_CORE_NEON_ZIP_HPP__

#include <arm_neon.h>

namespace aura
{

namespace neon
{

#define DECLFUN(dtype, vtype, prefix, postfix)        \
    inline dtype vzip(const vtype &v, const vtype &u) \
    {                                                 \
        return prefix##_##postfix(v, u);              \
    }

DECLFUN(uint8x8x2_t,    uint8x8_t,   vzip,  u8)
DECLFUN(uint8x16x2_t,   uint8x16_t,  vzipq, u8)
DECLFUN(int8x8x2_t,     int8x8_t,    vzip,  s8)
DECLFUN(int8x16x2_t,    int8x16_t,   vzipq, s8)
DECLFUN(uint16x4x2_t,   uint16x4_t,  vzip,  u16)
DECLFUN(uint16x8x2_t,   uint16x8_t,  vzipq, u16)
DECLFUN(int16x4x2_t,    int16x4_t,   vzip,  s16)
DECLFUN(int16x8x2_t,    int16x8_t,   vzipq, s16)
DECLFUN(uint32x2x2_t,   uint32x2_t,  vzip,  u32)
DECLFUN(uint32x4x2_t,   uint32x4_t,  vzipq, u32)
DECLFUN(int32x2x2_t,    int32x2_t,   vzip,  s32)
DECLFUN(int32x4x2_t,    int32x4_t,   vzipq, s32)
DECLFUN(float32x2x2_t,  float32x2_t, vzip,  f32)
DECLFUN(float32x4x2_t,  float32x4_t, vzipq, f32)

#  if defined(AURA_ENABLE_NEON_FP16)
DECLFUN(float16x4x2_t,  float16x4_t, vzip,  f16)
DECLFUN(float16x8x2_t,  float16x8_t, vzipq, f16)
#  endif

#undef DECLFUN

#define DECLFUN(dtype, vtype, prefix, postfix)        \
    inline dtype vuzp(const vtype &v, const vtype &u) \
    {                                                 \
        return prefix##_##postfix(v, u);              \
    }

DECLFUN(uint8x8x2_t,    uint8x8_t,   vuzp,  u8)
DECLFUN(uint8x16x2_t,   uint8x16_t,  vuzpq, u8)
DECLFUN(int8x8x2_t,     int8x8_t,    vuzp,  s8)
DECLFUN(int8x16x2_t,    int8x16_t,   vuzpq, s8)
DECLFUN(uint16x4x2_t,   uint16x4_t,  vuzp,  u16)
DECLFUN(uint16x8x2_t,   uint16x8_t,  vuzpq, u16)
DECLFUN(int16x4x2_t,    int16x4_t,   vuzp,  s16)
DECLFUN(int16x8x2_t,    int16x8_t,   vuzpq, s16)
DECLFUN(uint32x2x2_t,   uint32x2_t,  vuzp,  u32)
DECLFUN(uint32x4x2_t,   uint32x4_t,  vuzpq, u32)
DECLFUN(int32x2x2_t,    int32x2_t,   vuzp,  s32)
DECLFUN(int32x4x2_t,    int32x4_t,   vuzpq, s32)
DECLFUN(float32x2x2_t,  float32x2_t, vuzp,  f32)
DECLFUN(float32x4x2_t,  float32x4_t, vuzpq, f32)

#  if defined(AURA_ENABLE_NEON_FP16)
DECLFUN(float16x4x2_t,  float16x4_t, vuzp,  f16)
DECLFUN(float16x8x2_t,  float16x8_t, vuzpq, f16)
#  endif

#undef DECLFUN

#if defined(__aarch64__)

#define DECLFUN(dtype, vtype, prefix, postfix)        \
    inline dtype vzip1(const vtype &v, const vtype &u) \
    {                                                  \
        return prefix##_##postfix(v, u);               \
    }

DECLFUN(uint8x8_t,    uint8x8_t,   vzip1,  u8)
DECLFUN(uint8x16_t,   uint8x16_t,  vzip1q, u8)
DECLFUN(int8x8_t,     int8x8_t,    vzip1,  s8)
DECLFUN(int8x16_t,    int8x16_t,   vzip1q, s8)
DECLFUN(uint16x4_t,   uint16x4_t,  vzip1,  u16)
DECLFUN(uint16x8_t,   uint16x8_t,  vzip1q, u16)
DECLFUN(int16x4_t,    int16x4_t,   vzip1,  s16)
DECLFUN(int16x8_t,    int16x8_t,   vzip1q, s16)
DECLFUN(uint32x2_t,   uint32x2_t,  vzip1,  u32)
DECLFUN(uint32x4_t,   uint32x4_t,  vzip1q, u32)
DECLFUN(int32x2_t,    int32x2_t,   vzip1,  s32)
DECLFUN(int32x4_t,    int32x4_t,   vzip1q, s32)
DECLFUN(float32x2_t,  float32x2_t, vzip1,  f32)
DECLFUN(float32x4_t,  float32x4_t, vzip1q, f32)

#  if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
DECLFUN(float16x4_t,  float16x4_t, vzip1,  f16)
DECLFUN(float16x8_t,  float16x8_t, vzip1q, f16)
#  endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

#undef DECLFUN
#endif // __aarch64__

} // namespace neon

} // namespace aura

#endif // AURA_RUNTIME_CORE_NEON_ZIP_HPP__