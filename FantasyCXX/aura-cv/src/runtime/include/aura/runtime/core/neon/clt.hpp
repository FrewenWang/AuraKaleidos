#ifndef AURA_RUNTIME_CORE_NEON_CLT_HPP__
#define AURA_RUNTIME_CORE_NEON_CLT_HPP__

#include <arm_neon.h>

namespace aura
{

namespace neon
{

#define DECLFUN(dtype, vtype, prefix, postfix)            \
    inline vtype vclt(const vtype &v, const vtype &u)     \
    {                                                     \
        return prefix##_##postfix(v, u);                  \
    }

DECLFUN(uint8x8_t,  uint8x8_t,   vclt,  u8)
DECLFUN(uint8x16_t, uint8x16_t,  vcltq, u8)
DECLFUN(uint8x8_t,  int8x8_t,    vclt,  s8)
DECLFUN(uint8x16_t, int8x16_t,   vcltq, s8)
DECLFUN(uint16x4_t, uint16x4_t,  vclt,  u16)
DECLFUN(uint16x8_t, uint16x8_t,  vcltq, u16)
DECLFUN(uint16x4_t, int16x4_t,   vclt,  s16)
DECLFUN(uint16x8_t, int16x8_t,   vcltq, s16)
DECLFUN(uint32x2_t, uint32x2_t,  vclt,  u32)
DECLFUN(uint32x4_t, uint32x4_t,  vcltq, u32)
DECLFUN(uint32x2_t, int32x2_t,   vclt,  s32)
DECLFUN(uint32x4_t, int32x4_t,   vcltq, s32)
DECLFUN(uint32x2_t, float32x2_t, vclt,  f32)
DECLFUN(uint32x4_t, float32x4_t, vcltq, f32)

#  if defined(AURA_ENABLE_NEON_FP16)
DECLFUN(uint16x8_t, float16x8_t, vcltq, f16)
DECLFUN(uint16x4_t, float16x4_t, vclt,  f16)
#  endif
#undef DECLFUN

} // namespace neon

} // namespace aura

#endif // AURA_RUNTIME_CORE_NEON_CLT_HPP__