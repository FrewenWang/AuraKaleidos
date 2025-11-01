#ifndef AURA_RUNTIME_CORE_NEON_CGT_HPP__
#define AURA_RUNTIME_CORE_NEON_CGT_HPP__

#include <arm_neon.h>

namespace aura
{

namespace neon
{

#define DECLFUN(dtype, vtype, prefix, postfix)            \
    inline vtype vcgt(const vtype &v, const vtype &u)     \
    {                                                     \
        return prefix##_##postfix(v, u);                  \
    }

DECLFUN(uint8x8_t,  uint8x8_t,   vcgt,  u8)
DECLFUN(uint8x16_t, uint8x16_t,  vcgtq, u8)
DECLFUN(uint8x8_t,  int8x8_t,    vcgt,  s8)
DECLFUN(uint8x16_t, int8x16_t,   vcgtq, s8)
DECLFUN(uint16x4_t, uint16x4_t,  vcgt,  u16)
DECLFUN(uint16x8_t, uint16x8_t,  vcgtq, u16)
DECLFUN(uint16x4_t, int16x4_t,   vcgt,  s16)
DECLFUN(uint16x8_t, int16x8_t,   vcgtq, s16)
DECLFUN(uint32x2_t, uint32x2_t,  vcgt,  u32)
DECLFUN(uint32x4_t, uint32x4_t,  vcgtq, u32)
DECLFUN(uint32x2_t, int32x2_t,   vcgt,  s32)
DECLFUN(uint32x4_t, int32x4_t,   vcgtq, s32)
DECLFUN(uint32x2_t, float32x2_t, vcgt,  f32)
DECLFUN(uint32x4_t, float32x4_t, vcgtq, f32)

#  if defined(AURA_ENABLE_NEON_FP16)
DECLFUN(uint16x4_t, float16x4_t, vcgt,  f16)
DECLFUN(uint16x8_t, float16x8_t, vcgtq, f16)
#  endif
#undef DECLFUN

} // namespace neon

} // namespace aura

#endif // AURA_RUNTIME_CORE_NEON_CGT_HPP__