#ifndef AURA_RUNTIME_CORE_NEON_BSL_HPP__
#define AURA_RUNTIME_CORE_NEON_BSL_HPP__

#include <arm_neon.h>

namespace aura
{

namespace neon
{

#define DECLFUN(dtype, vtype, prefix, postfix)                        \
    inline vtype vbsl(const dtype &v, const vtype &u, const vtype &p) \
    {                                                                 \
        return prefix##_##postfix(v, u, p);                           \
    }

DECLFUN(uint8x8_t,   uint8x8_t,   vbsl,  u8)
DECLFUN(uint8x16_t,  uint8x16_t,  vbslq, u8)
DECLFUN(uint8x8_t,   int8x8_t,    vbsl,  s8)
DECLFUN(uint8x16_t,  int8x16_t,   vbslq, s8)
DECLFUN(uint16x4_t,  uint16x4_t,  vbsl,  u16)
DECLFUN(uint16x8_t,  uint16x8_t,  vbslq, u16)
DECLFUN(uint16x4_t,  int16x4_t,   vbsl,  s16)
DECLFUN(uint16x8_t,  int16x8_t,   vbslq, s16)
DECLFUN(uint32x2_t,  uint32x2_t,  vbsl,  u32)
DECLFUN(uint32x4_t,  uint32x4_t,  vbslq, u32)
DECLFUN(uint32x2_t,  int32x2_t,   vbsl,  s32)
DECLFUN(uint32x4_t,  int32x4_t,   vbslq, s32)
DECLFUN(uint64x1_t,  uint64x1_t,  vbsl,  u64)
DECLFUN(uint64x2_t,  uint64x2_t,  vbslq, u64)
DECLFUN(uint64x1_t,  int64x1_t,   vbsl,  s64)
DECLFUN(uint64x2_t,  int64x2_t,   vbslq, s64)
DECLFUN(uint32x2_t,  float32x2_t, vbsl,  f32)
DECLFUN(uint32x4_t,  float32x4_t, vbslq, f32)

#  if defined(AURA_ENABLE_NEON_FP16)
DECLFUN(float16x4_t,   float16x4_t, vbsl, f16)
DECLFUN(float16x8_t,   float16x8_t, vbslq, f16)
#  endif
#undef DECLFUN

} // namespace neon

} // namespace aura

#endif // AURA_RUNTIME_CORE_NEON_BSL_HPP__
