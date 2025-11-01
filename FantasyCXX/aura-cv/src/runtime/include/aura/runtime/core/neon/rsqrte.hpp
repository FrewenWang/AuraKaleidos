#ifndef AURA_RUNTIME_CORE_NEON_RSQRTE_HPP__
#define AURA_RUNTIME_CORE_NEON_RSQRTE_HPP__

#include <arm_neon.h>

namespace aura
{

namespace neon
{

#define DECLFUN(dtype, vtype, prefix, postfix)  \
    inline vtype vrsqrte(const vtype &v)        \
    {                                           \
        return prefix##_##postfix(v);           \
    }

DECLFUN(uint32x2_t,  uint32x2_t,  vrsqrte,  u32)
DECLFUN(uint32x4_t,  uint32x4_t,  vrsqrteq, u32)
DECLFUN(float32x2_t, float32x2_t, vrsqrte,  f32)
DECLFUN(float32x4_t, float32x4_t, vrsqrteq, f32)

#  if defined(AURA_ENABLE_NEON_FP16)
DECLFUN(float16x4_t, float16x4_t, vrsqrte,  f16)
DECLFUN(float16x8_t, float16x8_t, vrsqrteq, f16)
#  endif

#undef DECLFUN

} // namespace neon

} // namespace aura

#endif // AURA_RUNTIME_CORE_NEON_RSQRTE_HPP__