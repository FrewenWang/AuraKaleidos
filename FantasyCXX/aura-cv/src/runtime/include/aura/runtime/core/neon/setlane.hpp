#ifndef AURA_RUNTIME_CORE_NEON_SETLANE_HPP__
#define AURA_RUNTIME_CORE_NEON_SETLANE_HPP__

#include <arm_neon.h>

namespace aura
{

namespace neon
{

#define DECLFUN(stype, atype, vtype, prefix, postfix)               \
    template <int n>                                                \
    inline stype vsetlane(const atype value, const vtype vector)    \
    {                                                               \
        return prefix##_lane_##postfix(value, vector, n);           \
    }

DECLFUN(uint8x8_t,   DT_U8,  uint8x8_t,   vset,  u8 )
DECLFUN(int8x8_t,    DT_S8,  int8x8_t,    vset,  s8 )
DECLFUN(uint8x16_t,  DT_U8,  uint8x16_t,  vsetq, u8 )
DECLFUN(int8x16_t,   DT_S8,  int8x16_t,   vsetq, s8 )
DECLFUN(uint16x4_t,  DT_U16, uint16x4_t,  vset,  u16)
DECLFUN(int16x4_t,   DT_S16, int16x4_t,   vset,  s16)
DECLFUN(uint16x8_t,  DT_U16, uint16x8_t,  vsetq, u16)
DECLFUN(int16x8_t,   DT_S16, int16x8_t,   vsetq, s16)
DECLFUN(uint32x2_t,  DT_U32, uint32x2_t,  vset,  u32)
DECLFUN(int32x2_t,   DT_S32, int32x2_t,   vset,  s32)
DECLFUN(float32x2_t, DT_F32, float32x2_t, vset,  f32)
DECLFUN(uint32x4_t,  DT_U32, uint32x4_t,  vsetq, u32)
DECLFUN(int32x4_t,   DT_S32, int32x4_t,   vsetq, s32)
DECLFUN(float32x4_t, DT_F32, float32x4_t, vsetq, f32)

#  if defined(AURA_ENABLE_NEON_FP16)
DECLFUN(float16x4_t, float16_t, float16x4_t, vset,  f16)
DECLFUN(float16x8_t, float16_t, float16x8_t, vsetq, f16)
#  endif
#undef DECLFUN

} // namespace neon

} // namespace aura

#endif // AURA_RUNTIME_CORE_NEON_SETLANE_HPP__