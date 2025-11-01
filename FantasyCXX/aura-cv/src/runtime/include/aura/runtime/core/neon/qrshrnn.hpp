#ifndef AURA_RUNTIME_CORE_NEON_QRSHRNN_HPP__
#define AURA_RUNTIME_CORE_NEON_QRSHRNN_HPP__

#include <arm_neon.h>

namespace aura
{

namespace neon
{

#define DECLFUN(dtype, vtype, prefix, postfix)  \
    template <int n>                            \
    inline dtype vqrshrn_n(const vtype &v)      \
    {                                           \
        return prefix##_##postfix(v, n);        \
    }

DECLFUN(int8x8_t,   int16x8_t,  vqrshrn_n, s16)
DECLFUN(uint8x8_t,  uint16x8_t, vqrshrn_n, u16)
DECLFUN(int16x4_t,  int32x4_t,  vqrshrn_n, s32)
DECLFUN(uint16x4_t, uint32x4_t, vqrshrn_n, u32)
DECLFUN(int32x2_t,  int64x2_t,  vqrshrn_n, s64)
DECLFUN(uint32x2_t, uint64x2_t, vqrshrn_n, u64)
#undef DECLFUN

} // namespace neon

} // namespace aura

#endif // AURA_RUNTIME_CORE_NEON_QRSHRNN_HPP__