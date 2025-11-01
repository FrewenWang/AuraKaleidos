#ifndef AURA_RUNTIME_CORE_NEON_CAG_HPP__
#define AURA_RUNTIME_CORE_NEON_CAG_HPP__

namespace aura
{

namespace neon
{

#define DECLFUN(dtype, vtype, prefix, postfix)                                        \
    template <typename T>                                                             \
    inline typename std::enable_if<std::is_same<T, MI_F32>::value, dtype>::type       \
    vcage(const vtype &v, const vtype &u)                                             \
    {                                                                                 \
        return prefix##_##postfix(v, u);                                              \
    }

DECLFUN(uint32x2_t, float32x2_t, vcage,  f32)
DECLFUN(uint32x4_t, float32x4_t, vcageq, f32)
#undef DECLFUN

#define DECLFUN(dtype, vtype, prefix, postfix)                                        \
    template <typename T>                                                             \
    inline typename std::enable_if<std::is_same<T, MI_F32>::value, dtype>::type       \
    vcagt(const vtype &v, const vtype &u)                                             \
    {                                                                                 \
        return prefix##_##postfix(v, u);                                              \
    }

DECLFUN(uint32x2_t, float32x2_t, vcagt,  f32)
DECLFUN(uint32x4_t, float32x4_t, vcagtq, f32)
#undef DECLFUN

#if defined(AURA_ENABLE_NEON_FP16)
#  define DECLFUN(dtype, vtype, prefix, postfix)                                      \
    template <typename T>                                                             \
    inline typename std::enable_if<std::is_same<T, float16_t>::value, dtype>::type    \
    vcage(const vtype &v, const vtype &u)                                             \
    {                                                                                 \
        return prefix##_##postfix(v, u);                                              \
    }

DECLFUN(uint16x4_t, float16x4_t, vcage,  f16)
DECLFUN(uint16x8_t, float16x8_t, vcageq, f16)
#  undef DECLFUN

#  define DECLFUN(dtype, vtype, prefix, postfix)                                      \
    template <typename T>                                                             \
    inline typename std::enable_if<std::is_same<T, float16_t>::value, dtype>::type    \
    vcagt(const vtype &v, const vtype &u)                                             \
    {                                                                                 \
        return prefix##_##postfix(v, u);                                              \
    }

DECLFUN(uint16x4_t, float16x4_t, vcagt,  f16)
DECLFUN(uint16x8_t, float16x8_t, vcagtq, f16)
#  undef DECLFUN
#endif

} // namespace neon

} // namespace aura

#endif // AURA_RUNTIME_CORE_NEON_CAG_HPP__