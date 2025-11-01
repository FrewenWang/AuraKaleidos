#ifndef AURA_RUNTIME_CORE_NEON_QSHLUN_HPP__
#define AURA_RUNTIME_CORE_NEON_QSHLUN_HPP__

#include <arm_neon.h>

namespace aura
{

namespace neon
{

#define DECLFUN(dtype, vtype, prefix, postfix)  \
    template <int n>                            \
    inline vtype vqshlu_n(const vtype &v)       \
    {                                           \
        return prefix##_##postfix(v, n);        \
    }

DECLFUN(int8x8_t,  int8x8_t,  vqshlu_n,  s8 )
DECLFUN(int8x16_t, int8x16_t, vqshluq_n, s8 )
DECLFUN(int16x4_t, int16x4_t, vqshlu_n,  s16)
DECLFUN(int16x8_t, int16x8_t, vqshluq_n, s16)
DECLFUN(int32x2_t, int32x2_t, vqshlu_n,  s32)
DECLFUN(int32x4_t, int32x4_t, vqshluq_n, s32)
DECLFUN(int64x1_t, int64x1_t, vqshlu_n,  s64)
DECLFUN(int64x2_t, int64x2_t, vqshluq_n, s64)
#undef DECLFUN

} // namespace neon

} // namespace aura

#endif // AURA_RUNTIME_CORE_NEON_QSHLUN_HPP__