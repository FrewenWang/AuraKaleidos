#ifndef AURA_RUNTIME_CORE_NEON_DIV_HPP__
#define AURA_RUNTIME_CORE_NEON_DIV_HPP__

#include <arm_neon.h>

namespace aura
{

namespace neon
{
#if defined(__aarch64__)

#  define DECLFUN(vtype, prefix, postfix)               \
      inline vtype vdiv(const vtype &v, const vtype &u) \
      {                                                 \
          return prefix##_##postfix(v, u);              \
      }

#  if defined(AURA_ENABLE_NEON_FP16)
DECLFUN(float16x4_t, vdiv,  f16)
DECLFUN(float16x8_t, vdivq, f16)
#  endif

DECLFUN(float32x2_t, vdiv,  f32)
DECLFUN(float32x4_t, vdivq, f32)

DECLFUN(float64x1_t, vdiv,  f64)
DECLFUN(float64x2_t, vdivq, f64)

#  undef DECLFUN

#else

inline float32x2_t vdiv(const float32x2_t &vdf32_v, const float32x2_t &vdf32_u)
{
    float32x2_t vdf32_reciprocal = vrecpe_f32(vdf32_u);
    vdf32_reciprocal             = vmul_f32(vrecps_f32(vdf32_u, vdf32_reciprocal), vdf32_reciprocal);
    return vmul_f32(vdf32_v, vdf32_reciprocal);
}

inline float32x4_t vdiv(const float32x4_t &vqf32_v, const float32x4_t &vqf32_u)
{
    float32x4_t vqf32_reciprocal = vrecpeq_f32(vqf32_u);
    vqf32_reciprocal             = vmulq_f32(vrecpsq_f32(vqf32_u, vqf32_reciprocal), vqf32_reciprocal);
    return vmulq_f32(vqf32_v, vqf32_reciprocal);
}

#endif //(__aarch64__)

} // namespace neon

} // namespace aura

#endif // AURA_RUNTIME_CORE_NEON_DIV_HPP__