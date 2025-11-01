#ifndef AURA_RUNTIME_CORE_NEON_RSHRN_HPP__
#define AURA_RUNTIME_CORE_NEON_RSHRN_HPP__

#include <arm_neon.h>

namespace aura
{

namespace neon
{

#define DECLFUN(vtype, prefix, postfix)     \
    template <int n>                        \
    inline vtype vrshr_n(const vtype &v)    \
    {                                       \
        return prefix##_##postfix(v, n);    \
    }

DECLFUN(uint8x8_t,  vrshr_n,  u8)
DECLFUN(uint8x16_t, vrshrq_n, u8)
DECLFUN(int8x8_t,   vrshr_n,  s8)
DECLFUN(int8x16_t,  vrshrq_n, s8)
DECLFUN(uint16x4_t, vrshr_n,  u16)
DECLFUN(uint16x8_t, vrshrq_n, u16)
DECLFUN(int16x4_t,  vrshr_n,  s16)
DECLFUN(int16x8_t,  vrshrq_n, s16)
DECLFUN(uint32x2_t, vrshr_n,  u32)
DECLFUN(uint32x4_t, vrshrq_n, u32)
DECLFUN(int32x2_t,  vrshr_n,  s32)
DECLFUN(int32x4_t,  vrshrq_n, s32)
#undef DECLFUN

#define DECLFUN(dtype, vtype, prefix, postfix)  \
    template <int n>                            \
    inline dtype vrshrn_n(const vtype &v)       \
    {                                           \
        return prefix##_##postfix(v, n);        \
    }

DECLFUN(uint8x8_t,  uint16x8_t, vrshrn_n, u16)
DECLFUN(int8x8_t,   int16x8_t,  vrshrn_n, s16)
DECLFUN(uint16x4_t, uint32x4_t, vrshrn_n, u32)
DECLFUN(int16x4_t,  int32x4_t,  vrshrn_n, s32)
DECLFUN(uint32x2_t, uint64x2_t, vrshrn_n, u64)
DECLFUN(int32x2_t,  int64x2_t,  vrshrn_n, s64)
#undef DECLFUN

} // namespace neon

} // namespace aura

#endif // AURA_RUNTIME_CORE_NEON_RSHRN_HPP__