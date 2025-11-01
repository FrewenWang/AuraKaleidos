#ifndef AURA_RUNTIME_CORE_NEON_EXT_HPP__
#define AURA_RUNTIME_CORE_NEON_EXT_HPP__

#include <arm_neon.h>

namespace aura
{

namespace neon
{

#define DECLFUN(vtype, prefix, postfix)     \
    template <int n>                        \
    inline vtype vext(vtype &v, vtype &u)   \
    {                                       \
        return prefix##_##postfix(v, u, n); \
    }

DECLFUN(uint8x8_t,   vext,  u8 )
DECLFUN(uint8x16_t,  vextq, u8 )
DECLFUN(int8x8_t,    vext,  s8 )
DECLFUN(int8x16_t,   vextq, s8 )
DECLFUN(uint16x4_t,  vext,  u16)
DECLFUN(uint16x8_t,  vextq, u16)
DECLFUN(int16x4_t,   vext,  s16)
DECLFUN(int16x8_t,   vextq, s16)
DECLFUN(uint32x2_t,  vext,  u32)
DECLFUN(uint32x4_t,  vextq, u32)
DECLFUN(int32x2_t,   vext,  s32)
DECLFUN(int32x4_t,   vextq, s32)
DECLFUN(uint64x1_t,  vext,  u64)
DECLFUN(uint64x2_t,  vextq, u64)
DECLFUN(int64x1_t,   vext,  s64)
DECLFUN(int64x2_t,   vextq, s64)
DECLFUN(float32x2_t, vext,  f32)
DECLFUN(float32x4_t, vextq, f32)

#  if defined(AURA_ENABLE_NEON_FP16)
DECLFUN(float16x4_t, vext,  f16)
DECLFUN(float16x8_t, vextq, f16)
#  endif

#  if defined(__aarch64__)
DECLFUN(float64x1_t, vext,  f64)
DECLFUN(float64x2_t, vextq, f64)
#  endif // __aarch64__

#undef DECLFUN

} // namespace neon

} // namespace aura

#endif // AURA_RUNTIME_CORE_NEON_EXT_HPP__