#ifndef AURA_RUNTIME_CORE_NEON_CLS_HPP__
#define AURA_RUNTIME_CORE_NEON_CLS_HPP__

#include <arm_neon.h>

namespace aura
{

namespace neon
{

#define DECLFUN(dtype, vtype, prefix, postfix) \
    inline vtype vcls(const vtype &v)          \
    {                                          \
        return prefix##_##postfix(v);          \
    }

DECLFUN(int8x8_t,    int8x8_t,    vcls,  s8)
DECLFUN(int8x16_t,   int8x16_t,   vclsq, s8)
DECLFUN(int16x4_t,   int16x4_t,   vcls,  s16)
DECLFUN(int16x8_t,   int16x8_t,   vclsq, s16)
DECLFUN(int32x2_t,   int32x2_t,   vcls,  s32)
DECLFUN(int32x4_t,   int32x4_t,   vclsq, s32)
#undef DECLFUN

} // namespace neon

} // namespace aura

#endif // AURA_RUNTIME_CORE_NEON_CLS_HPP__