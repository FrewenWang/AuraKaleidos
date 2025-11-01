#ifndef AURA_RUNTIME_CORE_NEON_ABS_HPP__
#define AURA_RUNTIME_CORE_NEON_ABS_HPP__

#include <arm_neon.h>

namespace aura
{

namespace neon
{

#define DECLFUN(dtype, vtype, prefix, postfix) \
    inline vtype vabs(const vtype &v)          \
    {                                          \
        return prefix##_##postfix(v);          \
    }

DECLFUN(int8x8_t,    int8x8_t,    vabs,  s8)
DECLFUN(int8x16_t,   int8x16_t,   vabsq, s8)
DECLFUN(int16x4_t,   int16x4_t,   vabs,  s16)
DECLFUN(int16x8_t,   int16x8_t,   vabsq, s16)
DECLFUN(int32x2_t,   int32x2_t,   vabs,  s32)
DECLFUN(int32x4_t,   int32x4_t,   vabsq, s32)
DECLFUN(float32x2_t, float32x2_t, vabs,  f32)
DECLFUN(float32x4_t, float32x4_t, vabsq, f32)

#  if defined(AURA_ENABLE_NEON_FP16)
DECLFUN(float16x4_t,   float16x4_t, vabs, f16)
DECLFUN(float16x8_t,   float16x8_t, vabsq, f16)
#  endif
#undef DECLFUN

#define DECLFUN(dtype, vtype, prefix, postfix) \
    inline vtype vqabs(const vtype &v)         \
    {                                          \
        return prefix##_##postfix(v);          \
    }

DECLFUN(int8x8_t,    int8x8_t,    vqabs,  s8)
DECLFUN(int8x16_t,   int8x16_t,   vqabsq, s8)
DECLFUN(int16x4_t,   int16x4_t,   vqabs,  s16)
DECLFUN(int16x8_t,   int16x8_t,   vqabsq, s16)
DECLFUN(int32x2_t,   int32x2_t,   vqabs,  s32)
DECLFUN(int32x4_t,   int32x4_t,   vqabsq, s32)
#undef DECLFUN

#define DECLFUN(dtype)                \
    inline dtype vabs(const dtype &v) \
    {                                 \
        return v;                     \
    }

DECLFUN(uint8x8_t);
DECLFUN(uint8x16_t);
DECLFUN(uint16x4_t);
DECLFUN(uint16x8_t);
DECLFUN(uint32x2_t);
DECLFUN(uint32x4_t);
#undef DECLFUN

} // namespace neon

} // namespace aura

#endif // AURA_RUNTIME_CORE_NEON_ABS_HPP__