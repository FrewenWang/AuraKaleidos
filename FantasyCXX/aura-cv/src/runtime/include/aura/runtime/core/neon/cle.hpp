#ifndef AURA_RUNTIME_CORE_NEON_CLE_HPP__
#define AURA_RUNTIME_CORE_NEON_CLE_HPP__

#include <arm_neon.h>

namespace aura
{

namespace neon
{

#define DECLFUN(dtype, vtype, prefix, postfix)            \
    inline vtype vcle(const vtype &v, const vtype &u)     \
    {                                                     \
        return prefix##_##postfix(v, u);                  \
    }

DECLFUN(uint8x8_t,  uint8x8_t,   vcle,  u8)
DECLFUN(uint8x16_t, uint8x16_t,  vcleq, u8)
DECLFUN(uint8x8_t,  int8x8_t,    vcle,  s8)
DECLFUN(uint8x16_t, int8x16_t,   vcleq, s8)
DECLFUN(uint16x4_t, uint16x4_t,  vcle,  u16)
DECLFUN(uint16x8_t, uint16x8_t,  vcleq, u16)
DECLFUN(uint16x4_t, int16x4_t,   vcle,  s16)
DECLFUN(uint16x8_t, int16x8_t,   vcleq, s16)
DECLFUN(uint32x2_t, uint32x2_t,  vcle,  u32)
DECLFUN(uint32x4_t, uint32x4_t,  vcleq, u32)
DECLFUN(uint32x2_t, int32x2_t,   vcle,  s32)
DECLFUN(uint32x4_t, int32x4_t,   vcleq, s32)
DECLFUN(uint32x2_t, float32x2_t, vcle,  f32)
DECLFUN(uint32x4_t, float32x4_t, vcleq, f32)

#  if defined(AURA_ENABLE_NEON_FP16)
DECLFUN(unit16x4_t, float16x4_t, vcle,  f16)
DECLFUN(unit16x8_t, float16x8_t, vcleq, f16)
#  endif
#undef DECLFUN

} // namespace neon

} // namespace aura

#endif // AURA_RUNTIME_CORE_NEON_CLE_HPP__