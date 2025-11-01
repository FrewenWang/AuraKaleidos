#ifndef AURA_RUNTIME_CORE_NEON_GETLANE_HPP__
#define AURA_RUNTIME_CORE_NEON_GETLANE_HPP__

#include <arm_neon.h>

namespace aura
{

namespace neon
{

#define DECLFUN(stype, vtype, prefix, postfix)      \
    template <int n>                                \
    inline stype vgetlane(const vtype vector)       \
    {                                               \
        return prefix##_lane_##postfix(vector, n);  \
    }

DECLFUN(MI_U8,  uint8x8_t,   vget,  u8 )
DECLFUN(MI_S8,  int8x8_t,    vget,  s8 )
DECLFUN(MI_U8,  uint8x16_t,  vgetq, u8 )
DECLFUN(MI_S8,  int8x16_t,   vgetq, s8 )
DECLFUN(MI_U16, uint16x4_t,  vget,  u16)
DECLFUN(MI_S16, int16x4_t,   vget,  s16)
DECLFUN(MI_U16, uint16x8_t,  vgetq, u16)
DECLFUN(MI_S16, int16x8_t,   vgetq, s16)
DECLFUN(MI_U32, uint32x2_t,  vget,  u32)
DECLFUN(MI_S32, int32x2_t,   vget,  s32)
DECLFUN(MI_F32, float32x2_t, vget,  f32)
DECLFUN(MI_U32, uint32x4_t,  vgetq, u32)
DECLFUN(MI_S32, int32x4_t,   vgetq, s32)
DECLFUN(MI_F32, float32x4_t, vgetq, f32)
DECLFUN(MI_U64, uint64x2_t,  vgetq, u64)
DECLFUN(MI_S64, int64x2_t,   vgetq, s64)

#  if defined(AURA_ENABLE_NEON_FP16)
DECLFUN(float16_t, float16x4_t, vget,  f16)
DECLFUN(float16_t, float16x8_t, vgetq, f16)
#  endif
#undef DECLFUN

} // namespace neon

} // namespace aura

#endif // AURA_RUNTIME_CORE_NEON_GETLANE_HPP__