#ifndef AURA_RUNTIME_CORE_NEON_TRN_HPP__
#define AURA_RUNTIME_CORE_NEON_TRN_HPP__

#include <arm_neon.h>

namespace aura
{

namespace neon
{

#define DECLFUN(dtype, vtype, prefix, postfix)        \
    inline dtype vtrn(const vtype &v, const vtype &u) \
    {                                                 \
        return prefix##_##postfix(v, u);              \
    }

DECLFUN(uint8x8x2_t,    uint8x8_t,   vtrn,  u8)
DECLFUN(uint8x16x2_t,   uint8x16_t,  vtrnq, u8)
DECLFUN(int8x8x2_t,     int8x8_t,    vtrn,  s8)
DECLFUN(int8x16x2_t,    int8x16_t,   vtrnq, s8)
DECLFUN(uint16x4x2_t,   uint16x4_t,  vtrn,  u16)
DECLFUN(uint16x8x2_t,   uint16x8_t,  vtrnq, u16)
DECLFUN(int16x4x2_t,    int16x4_t,   vtrn,  s16)
DECLFUN(int16x8x2_t,    int16x8_t,   vtrnq, s16)
DECLFUN(uint32x2x2_t,   uint32x2_t,  vtrn,  u32)
DECLFUN(uint32x4x2_t,   uint32x4_t,  vtrnq, u32)
DECLFUN(int32x2x2_t,    int32x2_t,   vtrn,  s32)
DECLFUN(int32x4x2_t,    int32x4_t,   vtrnq, s32)
DECLFUN(float32x2x2_t,  float32x2_t, vtrn,  f32)
DECLFUN(float32x4x2_t,  float32x4_t, vtrnq, f32)

#  if defined(AURA_ENABLE_NEON_FP16)
DECLFUN(float16x4x2_t,  float16x4_t, vtrn,  f16)
DECLFUN(float16x8x2_t,  float16x8_t, vtrnq, f16)
#  endif

#undef DECLFUN

} // namespace neon

} // namespace aura

#endif // AURA_RUNTIME_CORE_NEON_TRN_HPP__