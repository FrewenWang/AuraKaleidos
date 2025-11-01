#ifndef AURA_RUNTIME_CORE_NEON_CEQ_HPP__
#define AURA_RUNTIME_CORE_NEON_CEQ_HPP__

#include <arm_neon.h>

namespace aura
{

namespace neon
{

#define DECLFUN(dtype, vtype, prefix, postfix)            \
    inline vtype vceq(const vtype &v, const vtype &u)     \
    {                                                     \
        return prefix##_##postfix(v, u);                  \
    }

DECLFUN(uint8x8_t,  uint8x8_t,   vceq,  u8)
DECLFUN(uint8x16_t, uint8x16_t,  vceqq, u8)
DECLFUN(uint8x8_t,  int8x8_t,    vceq,  s8)
DECLFUN(uint8x16_t, int8x16_t,   vceqq, s8)
DECLFUN(uint16x4_t, uint16x4_t,  vceq,  u16)
DECLFUN(uint16x8_t, uint16x8_t,  vceqq, u16)
DECLFUN(uint16x4_t, int16x4_t,   vceq,  s16)
DECLFUN(uint16x8_t, int16x8_t,   vceqq, s16)
DECLFUN(uint32x2_t, uint32x2_t,  vceq,  u32)
DECLFUN(uint32x4_t, uint32x4_t,  vceqq, u32)
DECLFUN(uint32x2_t, int32x2_t,   vceq,  s32)
DECLFUN(uint32x4_t, int32x4_t,   vceqq, s32)
DECLFUN(uint32x2_t, float32x2_t, vceq,  f32)
DECLFUN(uint32x4_t, float32x4_t, vceqq, f32)

#  if defined(AURA_ENABLE_NEON_FP16)
DECLFUN(uint16x4_t, float16x4_t, vceq,  f16)
DECLFUN(uint16x8_t, float16x8_t, vceqq, f16)
#  endif
#undef DECLFUN

} // namespace neon

} // namespace aura

#endif // AURA_RUNTIME_CORE_NEON_CEQ_HPP__