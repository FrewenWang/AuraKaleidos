#ifndef AURA_RUNTIME_CORE_NEON_QSHLN_HPP__
#define AURA_RUNTIME_CORE_NEON_QSHLN_HPP__

#include <arm_neon.h>

namespace aura
{

namespace neon
{

#define DECLFUN(dtype, vtype, prefix, postfix)  \
    template <int n>                            \
    inline vtype vqshl_n(const vtype &v)        \
    {                                           \
        return prefix##_##postfix(v, n);        \
    }

DECLFUN(int8x8_t,   int8x8_t,   vqshl_n, s8 )
DECLFUN(uint8x8_t,  uint8x8_t,  vqshl_n, u8 )
DECLFUN(int16x4_t,  int16x4_t,  vqshl_n, s16)
DECLFUN(uint16x4_t, uint16x4_t, vqshl_n, u16)
DECLFUN(int32x2_t,  int32x2_t,  vqshl_n, s32)
DECLFUN(uint32x2_t, uint32x2_t, vqshl_n, u32)
DECLFUN(int64x1_t,  int64x1_t,  vqshl_n, s64)
DECLFUN(uint64x1_t, uint64x1_t, vqshl_n, u64)

DECLFUN(int8x16_t,  int8x16_t,  vqshlq_n, s8 )
DECLFUN(uint8x16_t, uint8x16_t, vqshlq_n, u8 )
DECLFUN(int16x8_t,  int16x8_t,  vqshlq_n, s16)
DECLFUN(uint16x8_t, uint16x8_t, vqshlq_n, u16)
DECLFUN(int32x4_t,  int32x4_t,  vqshlq_n, s32)
DECLFUN(uint32x4_t, uint32x4_t, vqshlq_n, u32)
DECLFUN(int64x2_t,  int64x2_t,  vqshlq_n, s64)
DECLFUN(uint64x2_t, uint64x2_t, vqshlq_n, u64)
#undef DECLFUN

} // namespace neon

} // namespace aura

#endif // AURA_RUNTIME_CORE_NEON_QSHLN_HPP__