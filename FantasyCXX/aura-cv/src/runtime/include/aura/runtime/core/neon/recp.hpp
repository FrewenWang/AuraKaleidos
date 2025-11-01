#ifndef AURA_RUNTIME_CORE_NEON_RECP_HPP__
#define AURA_RUNTIME_CORE_NEON_RECP_HPP__

#include <arm_neon.h>

namespace aura
{

namespace neon
{

#define DECLFUN(dtype, vtype, prefix, postfix) \
    inline dtype vrecpe(const vtype &v)        \
    {                                          \
        return prefix##_##postfix(v);          \
    }

DECLFUN(uint32x2_t,   uint32x2_t,   vrecpe,  u32)
DECLFUN(uint32x4_t,   uint32x4_t,   vrecpeq, u32)
DECLFUN(float32x2_t,  float32x2_t,  vrecpe,  f32)
DECLFUN(float32x4_t,  float32x4_t,  vrecpeq, f32)
#undef DECLFUN

template <typename T>
inline typename std::enable_if<std::is_same<T, MI_F32>::value, float32x2_t>::type
vrecps(const float32x2_t &v, const float32x2_t &u)
{
    return vrecps_f32(v, u);
}

template <typename T>
inline typename std::enable_if<std::is_same<T, MI_F32>::value, float32x4_t>::type
vrecps(const float32x4_t &v, const float32x4_t &u)
{
    return vrecpsq_f32(v, u);
}

#if defined(AURA_ENABLE_NEON_FP16)
template <typename T>
inline typename std::enable_if<std::is_same<T, float16_t>::value, float16x4_t>::type
vrecps(const float16x4_t &v, const float16x4_t &u)
{
    return vrecps_f16(v, u);
}

template <typename T>
inline typename std::enable_if<std::is_same<T, float16_t>::value, float16x8_t>::type
vrecps(const float16x8_t &v, const float16x8_t &u)
{
    return vrecpsq_f16(v, u);
}
#endif

} // namespace neon

} // namespace aura

#endif // AURA_RUNTIME_CORE_NEON_RECP_HPP__