#ifndef AURA_RUNTIME_CORE_NEON_SHLN_HPP__
#define AURA_RUNTIME_CORE_NEON_SHLN_HPP__

#include <arm_neon.h>

namespace aura
{

namespace neon
{

#define DECLFUN(vtype, prefix, postfix)     \
    template <int n>                        \
    inline vtype vshln(const vtype &v)      \
    {                                       \
        return prefix##_##postfix(v, n);    \
    }

DECLFUN(uint8x8_t,   vshl_n,  u8 )
DECLFUN(uint8x16_t,  vshlq_n, u8 )
DECLFUN(int8x8_t,    vshl_n,  s8 )
DECLFUN(int8x16_t,   vshlq_n, s8 )
DECLFUN(uint16x4_t,  vshl_n,  u16)
DECLFUN(uint16x8_t,  vshlq_n, u16)
DECLFUN(int16x4_t,   vshl_n,  s16)
DECLFUN(int16x8_t,   vshlq_n, s16)
DECLFUN(uint32x2_t,  vshl_n,  u32)
DECLFUN(uint32x4_t,  vshlq_n, u32)
DECLFUN(int32x2_t,   vshl_n,  s32)
DECLFUN(int32x4_t,   vshlq_n, s32)
DECLFUN(uint64x1_t,  vshl_n,  u64)
DECLFUN(uint64x2_t,  vshlq_n, u64)
#undef DECLFUN

#define DECLFUN(dtype, vtype, postfix)      \
    template <int n>                        \
    inline dtype vshll_n(const vtype &v)    \
    {                                       \
        return vshll_n_##postfix(v, n);     \
    }

DECLFUN(uint16x8_t, uint8x8_t,  u8 )
DECLFUN(int16x8_t,  int8x8_t,   s8 )
DECLFUN(uint32x4_t, uint16x4_t, u16)
DECLFUN(int32x4_t,  int16x4_t,  s16)
DECLFUN(uint64x2_t, uint32x2_t, u32)
DECLFUN(int64x2_t,  int32x2_t,  s32)
#undef DECLFUN

} // namespace neon

} // namespace aura

#endif // AURA_RUNTIME_CORE_NEON_SHLN_HPP__