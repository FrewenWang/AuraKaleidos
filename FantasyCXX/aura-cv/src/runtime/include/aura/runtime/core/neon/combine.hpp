#ifndef AURA_RUNTIME_CORE_NEON_COMBINE_HPP__
#define AURA_RUNTIME_CORE_NEON_COMBINE_HPP__

#include <arm_neon.h>

namespace aura
{

namespace neon
{

#define DECLFUN(dtype, vtype, postfix)                    \
    inline dtype vcombine(const vtype &v, const vtype &u) \
    {                                                     \
        return vcombine_##postfix(v, u);                  \
    }

DECLFUN(uint8x16_t,  uint8x8_t,   u8)
DECLFUN(int8x16_t,   int8x8_t,    s8)
DECLFUN(uint16x8_t,  uint16x4_t,  u16)
DECLFUN(int16x8_t,   int16x4_t,   s16)
DECLFUN(uint32x4_t,  uint32x2_t,  u32)
DECLFUN(int32x4_t,   int32x2_t,   s32)
DECLFUN(uint64x2_t,  uint64x1_t,  u64)
DECLFUN(int64x2_t,   int64x1_t,   s64)
DECLFUN(float32x4_t, float32x2_t, f32)
#  if defined(AURA_ENABLE_NEON_FP16)
DECLFUN(float16x8_t,   float16x4_t, f16)
#  endif
#undef DECLFUN

} // namespace neon

} // namespace aura

#endif // AURA_RUNTIME_CORE_NEON_COMBINE_HPP__