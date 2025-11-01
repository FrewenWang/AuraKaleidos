#ifndef AURA_RUNTIME_CORE_NEON_MOV_HPP__
#define AURA_RUNTIME_CORE_NEON_MOV_HPP__

#include <arm_neon.h>

namespace aura
{

namespace neon
{

#define DECLFUN(stype, vtype, prefix, postfix) \
    inline vtype vmov(const stype &v)          \
    {                                          \
        return prefix##_##postfix(v);          \
    }

DECLFUN(MI_U8,    uint8x8_t,   vmov_n,  u8)
DECLFUN(MI_S8,    int8x8_t,    vmov_n,  s8)
DECLFUN(MI_U16,   uint16x4_t,  vmov_n,  u16)
DECLFUN(MI_S16,   int16x4_t,   vmov_n,  s16)
DECLFUN(MI_U32,   uint32x2_t,  vmov_n,  u32)
DECLFUN(MI_S32,   int32x2_t,   vmov_n,  s32)
DECLFUN(MI_U64,   uint64x1_t,  vmov_n,  u64)
DECLFUN(MI_S64,   int64x1_t,   vmov_n,  s64)
DECLFUN(MI_F32,   float32x2_t, vmov_n,  f32)
#  if defined(AURA_ENABLE_NEON_FP16)
DECLFUN(float16_t,   float16x4_t, vmov_n,  f16)
#  endif
#undef DECLFUN

#define DECLFUN(stype, vtype, prefix, postfix) \
    inline vtype vmovq(const stype &v)         \
    {                                          \
        return prefix##_##postfix(v);          \
    }

DECLFUN(MI_U8,    uint8x16_t,  vmovq_n,  u8)
DECLFUN(MI_S8,    int8x16_t,   vmovq_n,  s8)
DECLFUN(MI_U16,   uint16x8_t,  vmovq_n,  u16)
DECLFUN(MI_S16,   int16x8_t,   vmovq_n,  s16)
DECLFUN(MI_U32,   uint32x4_t,  vmovq_n,  u32)
DECLFUN(MI_S32,   int32x4_t,   vmovq_n,  s32)
DECLFUN(MI_U64,   uint64x2_t,  vmovq_n,  u64)
DECLFUN(MI_S64,   int64x2_t,   vmovq_n,  s64)
DECLFUN(MI_F32,   float32x4_t, vmovq_n,  f32)
#  if defined(AURA_ENABLE_NEON_FP16)
DECLFUN(float16_t,   float16x8_t, vmovq_n,  f16)
#  endif
#undef DECLFUN

} // namespace neon

} // namespace aura

#endif // AURA_RUNTIME_CORE_NEON_MOV_HPP__