#ifndef AURA_RUNTIME_CORE_NEON_QSHRUN_HPP__
#define AURA_RUNTIME_CORE_NEON_QSHRUN_HPP__

#include <arm_neon.h>

namespace aura
{

namespace neon
{

#define DECLFUN(dtype, vtype, postfix)      \
    template <int n>                        \
    inline dtype vqshru_n(const vtype &v)   \
    {                                       \
        return vqshrun_n_##postfix(v, n);   \
    }

DECLFUN(int8x8_t,  int16x8_t, s16)
DECLFUN(int16x4_t, int32x4_t, s32)
DECLFUN(int32x2_t, int64x2_t, s64)
#undef DECLFUN

} // namespace neon

} // namespace aura

#endif // AURA_RUNTIME_CORE_NEON_QSHRUN_HPP__