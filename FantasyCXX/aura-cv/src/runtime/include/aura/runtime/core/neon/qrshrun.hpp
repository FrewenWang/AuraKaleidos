#ifndef AURA_RUNTIME_CORE_NEON_QRSHRUN_HPP__
#define AURA_RUNTIME_CORE_NEON_QRSHRUN_HPP__

#include <arm_neon.h>

namespace aura
{

namespace neon
{

#define DECLFUN(dtype, vtype, postfix)      \
    template <int n>                        \
    inline dtype vqrshrun_n(const vtype &v) \
    {                                       \
        return vqrshrun_n_##postfix(v, n);  \
    }

DECLFUN(uint8x8_t,  int16x8_t, s16)
DECLFUN(uint16x4_t, int32x4_t, s32)
DECLFUN(uint32x2_t, int64x2_t, s64)
#undef DECLFUN

} // namespace neon

} // namespace aura

#endif // AURA_RUNTIME_CORE_NEON_QRSHRUN_HPP__