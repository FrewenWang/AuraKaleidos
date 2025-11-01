#ifndef AURA_RUNTIME_CORE_NEON_CGE_HPP__
#define AURA_RUNTIME_CORE_NEON_CGE_HPP__

#include <arm_neon.h>

namespace aura
{

namespace neon
{

#define DECLFUN(dtype, vtype, prefix, postfix)            \
    inline vtype vcge(const vtype &v, const vtype &u)     \
    {                                                     \
        return prefix##_##postfix(v, u);                  \
    }

DECLFUN(uint8x8_t,  uint8x8_t,   vcge,  u8)
DECLFUN(uint8x16_t, uint8x16_t,  vcgeq, u8)
DECLFUN(uint8x8_t,  int8x8_t,    vcge,  s8)
DECLFUN(uint8x16_t, int8x16_t,   vcgeq, s8)
DECLFUN(uint16x4_t, uint16x4_t,  vcge,  u16)
DECLFUN(uint16x8_t, uint16x8_t,  vcgeq, u16)
DECLFUN(uint16x4_t, int16x4_t,   vcge,  s16)
DECLFUN(uint16x8_t, int16x8_t,   vcgeq, s16)
DECLFUN(uint32x2_t, uint32x2_t,  vcge,  u32)
DECLFUN(uint32x4_t, uint32x4_t,  vcgeq, u32)
DECLFUN(uint32x2_t, int32x2_t,   vcge,  s32)
DECLFUN(uint32x4_t, int32x4_t,   vcgeq, s32)
DECLFUN(uint32x2_t, float32x2_t, vcge,  f32)
DECLFUN(uint32x4_t, float32x4_t, vcgeq, f32)

#  if defined(AURA_ENABLE_NEON_FP16)
DECLFUN(uint16x4_t, float16x4_t, vcge,  f16)
DECLFUN(uint16x8_t, float16x8_t, vcgeq, f16)
#  endif
#undef DECLFUN

} // namespace neon

} // namespace aura

#endif // AURA_RUNTIME_CORE_NEON_CGE_HPP__