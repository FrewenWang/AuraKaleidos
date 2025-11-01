#ifndef AURA_RUNTIME_CORE_NEON_REV_HPP__
#define AURA_RUNTIME_CORE_NEON_REV_HPP__

#include <arm_neon.h>

namespace aura
{

namespace neon
{

#define DECLFUN(dtype, vtype, prefix, postfix) \
    inline vtype vrev64(const vtype &v)        \
    {                                          \
        return prefix##_##postfix(v);          \
    }

DECLFUN(uint8x8_t,    uint8x8_t,   vrev64,  u8)
DECLFUN(uint8x16_t,   uint8x16_t,  vrev64q, u8)
DECLFUN(int8x8_t,     int8x8_t,    vrev64,  s8)
DECLFUN(int8x16_t,    int8x16_t,   vrev64q, s8)
DECLFUN(uint16x4_t,   uint16x4_t,  vrev64,  u16)
DECLFUN(uint16x8_t,   uint16x8_t,  vrev64q, u16)
DECLFUN(int16x4_t,    int16x4_t,   vrev64,  s16)
DECLFUN(int16x8_t,    int16x8_t,   vrev64q, s16)
DECLFUN(uint32x2_t,   uint32x2_t,  vrev64,  u32)
DECLFUN(uint32x4_t,   uint32x4_t,  vrev64q, u32)
DECLFUN(int32x2_t,    int32x2_t,   vrev64,  s32)
DECLFUN(int32x4_t,    int32x4_t,   vrev64q, s32)
DECLFUN(float32x2_t,  float32x2_t, vrev64,  f32)
DECLFUN(float32x4_t,  float32x4_t, vrev64q, f32)

#  if defined(AURA_ENABLE_NEON_FP16)
DECLFUN(float16x4_t,  float16x4_t, vrev64,  f16)
DECLFUN(float16x8_t,  float16x8_t, vrev64q, f16)
#  endif

#undef DECLFUN

#define DECLFUN(dtype, vtype, prefix, postfix) \
    inline vtype vrev32(const vtype &v)        \
    {                                          \
        return prefix##_##postfix(v);          \
    }

DECLFUN(uint8x8_t,    uint8x8_t,   vrev32,  u8)
DECLFUN(uint8x16_t,   uint8x16_t,  vrev32q, u8)
DECLFUN(int8x8_t,     int8x8_t,    vrev32,  s8)
DECLFUN(int8x16_t,    int8x16_t,   vrev32q, s8)
DECLFUN(uint16x4_t,   uint16x4_t,  vrev32,  u16)
DECLFUN(uint16x8_t,   uint16x8_t,  vrev32q, u16)
DECLFUN(int16x4_t,    int16x4_t,   vrev32,  s16)
DECLFUN(int16x8_t,    int16x8_t,   vrev32q, s16)
#undef DECLFUN

#define DECLFUN(dtype, vtype, prefix, postfix) \
    inline vtype vrev16(const vtype &v)        \
    {                                          \
        return prefix##_##postfix(v);          \
    }

DECLFUN(uint8x8_t,    uint8x8_t,   vrev16,  u8)
DECLFUN(uint8x16_t,   uint8x16_t,  vrev16q, u8)
DECLFUN(int8x8_t,     int8x8_t,    vrev16,  s8)
DECLFUN(int8x16_t,    int8x16_t,   vrev16q, s8)
#undef DECLFUN

} // namespace neon

} // namespace aura

#endif // AURA_RUNTIME_CORE_NEON_REV_HPP__