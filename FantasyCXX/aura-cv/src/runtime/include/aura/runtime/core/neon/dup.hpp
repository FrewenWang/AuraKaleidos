#ifndef AURA_RUNTIME_CORE_NEON_DUP_HPP__
#define AURA_RUNTIME_CORE_NEON_DUP_HPP__

#include <arm_neon.h>

namespace aura
{

namespace neon
{

#define DECLFUN(stype, vtype, prefix, postfix)      \
    inline DT_VOID vdup(vtype &v, const stype s)    \
    {                                               \
        v = prefix##_##postfix(s);                  \
    }

DECLFUN(DT_U8,  uint8x8_t,   vdup_n,  u8 )
DECLFUN(DT_S8,  int8x8_t,    vdup_n,  s8 )
DECLFUN(DT_U16, uint16x4_t,  vdup_n,  u16)
DECLFUN(DT_S16, int16x4_t,   vdup_n,  s16)
DECLFUN(DT_U32, uint32x2_t,  vdup_n,  u32)
DECLFUN(DT_S32, int32x2_t,   vdup_n,  s32)
DECLFUN(DT_U64, uint64x1_t,  vdup_n,  u64)
DECLFUN(DT_S64, int64x1_t,   vdup_n,  s64)
DECLFUN(DT_F32, float32x2_t, vdup_n,  f32)
DECLFUN(DT_U8,  uint8x16_t,  vdupq_n, u8 )
DECLFUN(DT_S8,  int8x16_t,   vdupq_n, s8 )
DECLFUN(DT_U16, uint16x8_t,  vdupq_n, u16)
DECLFUN(DT_S16, int16x8_t,   vdupq_n, s16)
DECLFUN(DT_U32, uint32x4_t,  vdupq_n, u32)
DECLFUN(DT_S32, int32x4_t,   vdupq_n, s32)
DECLFUN(DT_U64, uint64x2_t,  vdupq_n, u64)
DECLFUN(DT_S64, int64x2_t,   vdupq_n, s64)
DECLFUN(DT_F32, float32x4_t, vdupq_n, f32)
#  if defined(AURA_ENABLE_NEON_FP16)
DECLFUN(MI_F16, float16x4_t, vdup_n,  f16)
DECLFUN(MI_F16, float16x8_t, vdupq_n, f16)
#  endif
#undef DECLFUN

#define DECLFUN(vtype, prefix, postfix)        \
    template <int n>                           \
    inline vtype vduplane(const vtype &v)      \
    {                                          \
        return prefix##_##postfix(v, n);       \
    }

DECLFUN(uint8x8_t,   vdup_lane,  u8)
DECLFUN(int8x8_t,    vdup_lane,  s8)
DECLFUN(uint16x4_t,  vdup_lane,  u16)
DECLFUN(int16x4_t,   vdup_lane,  s16)
DECLFUN(uint32x2_t,  vdup_lane,  u32)
DECLFUN(int32x2_t,   vdup_lane,  s32)
DECLFUN(uint64x1_t,  vdup_lane,  u64)
DECLFUN(int64x1_t,   vdup_lane,  s64)
DECLFUN(float32x2_t, vdup_lane,  f32)
#  if defined(AURA_ENABLE_NEON_FP16)
DECLFUN(float16x4_t, vdup_lane,  f16)
#  endif
#undef DECLFUN

#define DECLFUN(stype, vtype, prefix, postfix)    \
    template <int n>                              \
    inline vtype vduplaneq(const stype &v)        \
    {                                             \
        return prefix##_##postfix(v, n);          \
    }

DECLFUN(uint8x8_t,   uint8x16_t,  vdupq_lane,  u8)
DECLFUN(int8x8_t,    int8x16_t,   vdupq_lane,  s8)
DECLFUN(uint16x4_t,  uint16x8_t,  vdupq_lane,  u16)
DECLFUN(int16x4_t,   int16x8_t,   vdupq_lane,  s16)
DECLFUN(uint32x2_t,  uint32x4_t,  vdupq_lane,  u32)
DECLFUN(int32x2_t,   int32x4_t,   vdupq_lane,  s32)
DECLFUN(float32x2_t, float32x4_t, vdupq_lane,  f32)
DECLFUN(uint64x1_t,  uint64x2_t,  vdupq_lane,  u64)
DECLFUN(int64x1_t,   int64x2_t,   vdupq_lane,  s64)

#undef DECLFUN

} // namespace neon

} // namespace aura

#endif // AURA_RUNTIME_CORE_NEON_DUP_HPP__