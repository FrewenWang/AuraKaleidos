#ifndef AURA_RUNTIME_CORE_NEON_ABD_HPP__
#define AURA_RUNTIME_CORE_NEON_ABD_HPP__

#include <arm_neon.h>

namespace aura
{

namespace neon
{

#define DECLFUN(dtype, vtype, prefix, postfix)        \
    inline vtype vabd(const vtype &v, const vtype &u) \
    {                                                 \
        return prefix##_##postfix(v, u);              \
    }

DECLFUN(uint8x8_t,   uint8x8_t,   vabd,  u8)
DECLFUN(int8x8_t,    int8x8_t,    vabd,  s8)
DECLFUN(uint8x16_t,  uint8x16_t,  vabdq, u8)
DECLFUN(int8x16_t,   int8x16_t,   vabdq, s8)
DECLFUN(uint16x4_t,  uint16x4_t,  vabd,  u16)
DECLFUN(int16x4_t,   int16x4_t,   vabd,  s16)
DECLFUN(uint16x8_t,  uint16x8_t,  vabdq, u16)
DECLFUN(int16x8_t,   int16x8_t,   vabdq, s16)
DECLFUN(uint32x2_t,  uint32x2_t,  vabd,  u32)
DECLFUN(int32x2_t,   int32x2_t,   vabd,  s32)
DECLFUN(uint32x4_t,  uint32x4_t,  vabdq, u32)
DECLFUN(int32x4_t,   int32x4_t,   vabdq, s32)
DECLFUN(float32x2_t, float32x2_t, vabd,  f32)
DECLFUN(float32x4_t, float32x4_t, vabdq, f32)

#  if defined(AURA_ENABLE_NEON_FP16)
DECLFUN(float16x4_t,   float16x4_t, vabd, f16)
DECLFUN(float16x8_t,   float16x8_t, vabdq, f16)
#  endif
#undef DECLFUN

#define DECLFUN(dtype, vtype, prefix, postfix)         \
    inline dtype vabdl(const vtype &v, const vtype &u) \
    {                                                  \
        return prefix##_##postfix(v, u);               \
    }
DECLFUN(uint16x8_t,  uint8x8_t,  vabdl, u8)
DECLFUN(int16x8_t,   int8x8_t,   vabdl, s8)
DECLFUN(uint32x4_t,  uint16x4_t, vabdl, u16)
DECLFUN(int32x4_t,   int16x4_t,  vabdl, s16)
DECLFUN(uint64x2_t,  uint32x2_t, vabdl, u32)
DECLFUN(int64x2_t,   int32x2_t,  vabdl, s32)
#undef DECLFUN

} // namespace neon

} // namespace aura

#endif // AURA_RUNTIME_CORE_NEON_ABD_HPP__