#ifndef AURA_RUNTIME_CORE_NEON_CREATE_HPP__
#define AURA_RUNTIME_CORE_NEON_CREATE_HPP__

#include <arm_neon.h>

namespace aura
{

namespace neon
{

#define DECLFUN(stype, vtype, prefix, postfix) \
    inline vtype vcreate(const stype v)        \
    {                                          \
        return prefix##_##postfix(v);          \
    }

DECLFUN(DT_U8,    uint8x8_t,   vcreate,  u8)
DECLFUN(DT_S8,    int8x8_t,    vcreate,  s8)
DECLFUN(DT_U16,   uint16x4_t,  vcreate,  u16)
DECLFUN(DT_S16,   int16x4_t,   vcreate,  s16)
DECLFUN(DT_U32,   uint32x2_t,  vcreate,  u32)
DECLFUN(DT_S32,   int32x2_t,   vcreate,  s32)
DECLFUN(DT_U64,   uint64x1_t,  vcreate,  u64)
DECLFUN(DT_S64,   int64x1_t,   vcreate,  s64)
DECLFUN(DT_F32,   float32x2_t, vcreate,  f32)
#  if defined(AURA_ENABLE_NEON_FP16)
DECLFUN(float16_t,   float16x4_t, vcreate,  f16)
#  endif
#undef DECLFUN

} // namespace neon

} // namespace aura

#endif // AURA_RUNTIME_CORE_NEON_CREATE_HPP__