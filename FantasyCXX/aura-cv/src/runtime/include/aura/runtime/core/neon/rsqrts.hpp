#ifndef AURA_RUNTIME_CORE_NEON_RSQRTS_HPP__
#define AURA_RUNTIME_CORE_NEON_RSQRTS_HPP__

#include <arm_neon.h>

namespace aura
{

namespace neon
{

#define DECLFUN(dtype, vtype, prefix, postfix)           \
    inline vtype vrsqrts(const vtype &v, const vtype &u) \
    {                                                    \
        return prefix##_##postfix(v, u);                 \
    }

DECLFUN(float32x2_t, float32x2_t, vrsqrts,  f32)
DECLFUN(float32x4_t, float32x4_t, vrsqrtsq, f32)

#  if defined(AURA_ENABLE_NEON_FP16)
DECLFUN(float16x4_t, float16x4_t, vrsqrts,  f16)
DECLFUN(float16x8_t, float16x8_t, vrsqrtsq, f16)
#  endif

#undef DECLFUN

} // namespace neon

} // namespace aura

#endif // AURA_RUNTIME_CORE_NEON_RSQRTS_HPP__