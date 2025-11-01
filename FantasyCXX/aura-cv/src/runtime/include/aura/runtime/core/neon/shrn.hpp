#ifndef AURA_RUNTIME_CORE_NEON_SHRN_HPP__
#define AURA_RUNTIME_CORE_NEON_SHRN_HPP__

#include <arm_neon.h>

namespace aura
{

namespace neon
{

#define DECLFUN(vtype, prefix, postfix)     \
    template <int n>                        \
    inline vtype vshr_n(const vtype &v)     \
    {                                       \
        return prefix##_##postfix(v, n);    \
    }

DECLFUN(uint8x8_t,  vshr_n,  u8 )
DECLFUN(uint8x16_t, vshrq_n, u8 )
DECLFUN(int8x8_t,   vshr_n,  s8 )
DECLFUN(int8x16_t,  vshrq_n, s8 )
DECLFUN(uint16x4_t, vshr_n,  u16)
DECLFUN(uint16x8_t, vshrq_n, u16)
DECLFUN(int16x4_t,  vshr_n,  s16)
DECLFUN(int16x8_t,  vshrq_n, s16)
DECLFUN(uint32x2_t, vshr_n,  u32)
DECLFUN(uint32x4_t, vshrq_n, u32)
DECLFUN(int32x2_t,  vshr_n,  s32)
DECLFUN(int32x4_t,  vshrq_n, s32)
DECLFUN(uint64x1_t, vshr_n,  u64)
DECLFUN(uint64x2_t, vshrq_n, u64)
DECLFUN(int64x1_t,  vshr_n,  s64)
DECLFUN(int64x2_t,  vshrq_n, s64)
#undef DECLFUN

#define DECLFUN(dtype, vtype, prefix, postfix)  \
    template <int n>                            \
    inline dtype vshrn_n(const vtype &v)        \
    {                                           \
        return prefix##_##postfix(v, n);        \
    }

DECLFUN(int8x8_t,   int16x8_t,  vshrn_n, s16)
DECLFUN(int16x4_t,  int32x4_t,  vshrn_n, s32)
DECLFUN(int32x2_t,  int64x2_t,  vshrn_n, s64)
DECLFUN(uint8x8_t,  uint16x8_t, vshrn_n, u16)
DECLFUN(uint16x4_t, uint32x4_t, vshrn_n, u32)
DECLFUN(uint32x2_t, uint64x2_t, vshrn_n, u64)
#undef DECLFUN

} // namespace neon

} // namespace aura

#endif // AURA_RUNTIME_CORE_NEON_SHRN_HPP__