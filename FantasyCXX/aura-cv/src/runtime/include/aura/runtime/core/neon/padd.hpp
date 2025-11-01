#ifndef AURA_RUNTIME_CORE_NEON_PADD_HPP__
#define AURA_RUNTIME_CORE_NEON_PADD_HPP__

#include <arm_neon.h>

namespace aura
{

namespace neon
{

#define DECLFUN(dtype, vtype, prefix, postfix)            \
    inline vtype vpadd(const vtype &v, const vtype &u)    \
    {                                                     \
        return prefix##_##postfix(v, u);                  \
    }

DECLFUN(uint8x8_t,   uint8x8_t,   vpadd,  u8)
DECLFUN(int8x8_t,    int8x8_t,    vpadd,  s8)
DECLFUN(uint16x4_t,  uint16x4_t,  vpadd,  u16)
DECLFUN(int16x4_t,   int16x4_t,   vpadd,  s16)
DECLFUN(uint32x2_t,  uint32x2_t,  vpadd,  u32)
DECLFUN(int32x2_t,   int32x2_t,   vpadd,  s32)
DECLFUN(float32x2_t, float32x2_t, vpadd,  f32)

#  if defined(AURA_ENABLE_NEON_FP16)
DECLFUN(float16x4_t,   float16x4_t, vpadd, f16)
#  endif
#undef DECLFUN

#define DECLFUN(dtype, vtype, prefix, postfix) \
    inline dtype vpaddl(const vtype &v)        \
    {                                          \
        return prefix##_##postfix(v);          \
    }

DECLFUN(uint16x4_t,  uint8x8_t,   vpaddl,  u8)
DECLFUN(uint16x8_t,  uint8x16_t,  vpaddlq, u8)
DECLFUN(int16x4_t,   int8x8_t,    vpaddl,  s8)
DECLFUN(int16x8_t,   int8x16_t,   vpaddlq, s8)
DECLFUN(uint32x2_t,  uint16x4_t,  vpaddl,  u16)
DECLFUN(uint32x4_t,  uint16x8_t,  vpaddlq, u16)
DECLFUN(int32x2_t,   int16x4_t,   vpaddl,  s16)
DECLFUN(int32x4_t,   int16x8_t,   vpaddlq, s16)
DECLFUN(uint64x1_t,  uint32x2_t,  vpaddl,  u32)
DECLFUN(uint64x2_t,  uint32x4_t,  vpaddlq, u32)
DECLFUN(int64x1_t,   int32x2_t,   vpaddl,  s32)
DECLFUN(int64x2_t,   int32x4_t,   vpaddlq, s32)
#undef DECLFUN

#define DECLFUN(dtype, vtype, prefix, postfix)          \
    inline dtype vpadal(const dtype &v, const vtype &u) \
    {                                                   \
        return prefix##_##postfix(v, u);                \
    }

DECLFUN(uint16x4_t,  uint8x8_t,   vpadal,  u8)
DECLFUN(uint16x8_t,  uint8x16_t,  vpadalq, u8)
DECLFUN(int16x4_t,   int8x8_t,    vpadal,  s8)
DECLFUN(int16x8_t,   int8x16_t,   vpadalq, s8)
DECLFUN(uint32x2_t,  uint16x4_t,  vpadal,  u16)
DECLFUN(uint32x4_t,  uint16x8_t,  vpadalq, u16)
DECLFUN(int32x2_t,   int16x4_t,   vpadal,  s16)
DECLFUN(int32x4_t,   int16x8_t,   vpadalq, s16)
DECLFUN(uint64x1_t,  uint32x2_t,  vpadal,  u32)
DECLFUN(uint64x2_t,  uint32x4_t,  vpadalq, u32)
DECLFUN(int64x1_t,   int32x2_t,   vpadal,  s32)
DECLFUN(int64x2_t,   int32x4_t,   vpadalq, s32)
#undef DECLFUN

} // namespace neon

} // namespace aura

#endif // AURA_RUNTIME_CORE_NEON_PADD_HPP__