#ifndef AURA_RUNTIME_CORE_NEON_UTILS_HPP__
#define AURA_RUNTIME_CORE_NEON_UTILS_HPP__

#include <arm_neon.h>

namespace aura
{

namespace neon
{

// Newton's method to find the reciprocal of a floating point number
inline float32x2_t vreciprocal_newton(const float32x2_t &vdf32_src)
{
    float32x2_t vdf32_dst;
    vdf32_dst = vrecpe_f32(vdf32_src);
    vdf32_dst = vmul_f32(vrecps_f32(vdf32_src, vdf32_dst), vdf32_dst);
    vdf32_dst = vmul_f32(vrecps_f32(vdf32_src, vdf32_dst), vdf32_dst);
    return vdf32_dst;
}

inline float32x4_t vreciprocal_newton(const float32x4_t &vqf32_src)
{
    float32x4_t vqf32_dst;
    vqf32_dst = vrecpeq_f32(vqf32_src);
    vqf32_dst = vmulq_f32(vrecpsq_f32(vqf32_src, vqf32_dst), vqf32_dst);
    vqf32_dst = vmulq_f32(vrecpsq_f32(vqf32_src, vqf32_dst), vqf32_dst);
    return vqf32_dst;
}

} // namespace neon

} // namespace aura

#endif // AURA_RUNTIME_CORE_NEON_UTILS_HPP__