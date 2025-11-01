#ifndef AURA_RUNTIME_CORE_NEON_MUL_HPP__
#define AURA_RUNTIME_CORE_NEON_MUL_HPP__

#include <arm_neon.h>

namespace aura
{

namespace neon
{

#define DECLFUN(stype, vtype, prefix, postfix)        \
    inline vtype vmul(const vtype &v, const vtype &u) \
    {                                                 \
        return prefix##_##postfix(v, u);              \
    }

DECLFUN(uint8x8_t,   uint8x8_t,   vmul,  u8 )
DECLFUN(uint8x16_t,  uint8x16_t,  vmulq, u8 )
DECLFUN(int8x8_t,    int8x8_t,    vmul,  s8 )
DECLFUN(int8x16_t,   int8x16_t,   vmulq, s8 )
DECLFUN(uint16x4_t,  uint16x4_t,  vmul,  u16)
DECLFUN(uint16x8_t,  uint16x8_t,  vmulq, u16)
DECLFUN(int16x4_t,   int16x4_t,   vmul,  s16)
DECLFUN(int16x8_t,   int16x8_t,   vmulq, s16)
DECLFUN(uint32x2_t,  uint32x2_t,  vmul,  u32)
DECLFUN(uint32x4_t,  uint32x4_t,  vmulq, u32)
DECLFUN(int32x2_t,   int32x2_t,   vmul,  s32)
DECLFUN(int32x4_t,   int32x4_t,   vmulq, s32)
DECLFUN(float32x2_t, float32x2_t, vmul,  f32)
DECLFUN(float32x4_t, float32x4_t, vmulq, f32)

#  if defined(AURA_ENABLE_NEON_FP16)
DECLFUN(float16x4_t, float16x4_t, vmul, f16)
DECLFUN(float16x8_t, float16x8_t, vmulq, f16)
#  endif
#undef DECLFUN

#define DECLFUN(stype, vtype, prefix, postfix)       \
    inline vtype vmul(const vtype &v, const stype u) \
    {                                                \
        return prefix##_##postfix(v, u);             \
    }

DECLFUN(MI_U16,  uint16x4_t,  vmul_n,  u16)
DECLFUN(MI_U16,  uint16x8_t,  vmulq_n, u16)
DECLFUN(MI_S16,  int16x4_t,   vmul_n,  s16)
DECLFUN(MI_S16,  int16x8_t,   vmulq_n, s16)
DECLFUN(MI_U32,  uint32x2_t,  vmul_n,  u32)
DECLFUN(MI_U32,  uint32x4_t,  vmulq_n, u32)
DECLFUN(MI_S32,  int32x2_t,   vmul_n,  s32)
DECLFUN(MI_S32,  int32x4_t,   vmulq_n, s32)
DECLFUN(MI_F32,  float32x2_t, vmul_n,  f32)
DECLFUN(MI_F32,  float32x4_t, vmulq_n, f32)

#  if defined(AURA_ENABLE_NEON_FP16)
DECLFUN(float16_t, float16x4_t, vmul_n, f16)
DECLFUN(float16_t, float16x8_t, vmulq_n, f16)
#  endif
#undef DECLFUN

#define DECLFUN(dtype, vtype, prefix, postfix)         \
    inline dtype vmull(const vtype &v, const vtype &u) \
    {                                                  \
        return prefix##_##postfix(v, u);               \
    }

DECLFUN(uint16x8_t, uint8x8_t,  vmull, u8 )
DECLFUN(int16x8_t,  int8x8_t,   vmull, s8 )
DECLFUN(uint32x4_t, uint16x4_t, vmull, u16)
DECLFUN(int32x4_t,  int16x4_t,  vmull, s16)
DECLFUN(uint64x2_t, uint32x2_t, vmull, u32)
DECLFUN(int64x2_t,  int32x2_t,  vmull, s32)
#undef DECLFUN

#define DECLFUN(dtype, stype, vtype, prefix, postfix)  \
    inline dtype vmull(const vtype &v, const stype &u) \
    {                                                  \
        return prefix##_##postfix(v, u);               \
    }

DECLFUN(uint32x4_t, MI_U16, uint16x4_t, vmull_n, u16)
DECLFUN(int32x4_t,  MI_S16, int16x4_t,  vmull_n, s16)
DECLFUN(uint64x2_t, MI_U32, uint32x2_t, vmull_n, u32)
DECLFUN(int64x2_t,  MI_S32, int32x2_t,  vmull_n, s32)
#undef DECLFUN

#define DECLFUN(dtype, vtype, prefix, postfix)              \
    template <int n>                                        \
    inline dtype vmull_lane(const vtype &v, const vtype &u) \
    {                                                       \
        return prefix##_##postfix(v, u, n);                 \
    }

DECLFUN(uint32x4_t, uint16x4_t, vmull_lane, u16)
DECLFUN(int32x4_t,  int16x4_t,  vmull_lane, s16)
DECLFUN(uint64x2_t, uint32x2_t, vmull_lane, u32)
DECLFUN(int64x2_t,  int32x2_t,  vmull_lane, s32)
#undef DECLFUN
} // namespace neon

} // namespace aura
#endif // AURA_RUNTIME_CORE_NEON_MUL_HPP__
