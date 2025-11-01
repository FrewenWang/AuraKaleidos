#ifndef AURA_RUNTIME_CORE_NEON_NEG_HPP__
#define AURA_RUNTIME_CORE_NEON_NEG_HPP__

#include <arm_neon.h>

namespace aura
{

namespace neon
{

#define DECLFUN(dtype, vtype, prefix, postfix) \
    inline vtype vneg(const vtype &v)          \
    {                                          \
        return prefix##_##postfix(v);          \
    }

DECLFUN(int8x8_t,    int8x8_t,    vneg,  s8)
DECLFUN(int8x16_t,   int8x16_t,   vnegq, s8)
DECLFUN(int16x4_t,   int16x4_t,   vneg,  s16)
DECLFUN(int16x8_t,   int16x8_t,   vnegq, s16)
DECLFUN(int32x2_t,   int32x2_t,   vneg,  s32)
DECLFUN(int32x4_t,   int32x4_t,   vnegq, s32)
DECLFUN(float32x2_t, float32x2_t, vneg,  f32)
DECLFUN(float32x4_t, float32x4_t, vnegq, f32)

#  if defined(AURA_ENABLE_NEON_FP16)
DECLFUN(float16x4_t,   float16x4_t, vneg, f16)
DECLFUN(float16x8_t,   float16x8_t, vnegq, f16)
#  endif
#undef DECLFUN

#define DECLFUN(dtype, vtype, prefix, postfix) \
    inline vtype vqneg(const vtype &v)         \
    {                                          \
        return prefix##_##postfix(v);          \
    }

DECLFUN(int8x8_t,    int8x8_t,    vqneg,  s8)
DECLFUN(int8x16_t,   int8x16_t,   vqnegq, s8)
DECLFUN(int16x4_t,   int16x4_t,   vqneg,  s16)
DECLFUN(int16x8_t,   int16x8_t,   vqnegq, s16)
DECLFUN(int32x2_t,   int32x2_t,   vqneg,  s32)
DECLFUN(int32x4_t,   int32x4_t,   vqnegq, s32)
#undef DECLFUN

} // namespace neon

} // namespace aura

#endif // AURA_RUNTIME_CORE_NEON_NEG_HPP__