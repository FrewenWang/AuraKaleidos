#ifndef AURA_RUNTIME_CORE_NEON_CAL_HPP__
#define AURA_RUNTIME_CORE_NEON_CAL_HPP__

namespace aura
{

namespace neon
{

#define DECLFUN(dtype, vtype, prefix, postfix)                                        \
    template <typename T>                                                             \
    inline typename std::enable_if<std::is_same<T, MI_F32>::value, dtype>::type       \
    vcale(const vtype &v, const vtype &u)                                             \
    {                                                                                 \
        return prefix##_##postfix(v, u);                                              \
    }

DECLFUN(uint32x2_t, float32x2_t, vcale,  f32)
DECLFUN(uint32x4_t, float32x4_t, vcaleq, f32)
#undef DECLFUN

#define DECLFUN(dtype, vtype, prefix, postfix)                                        \
    template <typename T>                                                             \
    inline typename std::enable_if<std::is_same<T, MI_F32>::value, dtype>::type       \
    vcalt(const vtype &v, const vtype &u)                                             \
    {                                                                                 \
        return prefix##_##postfix(v, u);                                              \
    }

DECLFUN(uint32x2_t, float32x2_t, vcalt,  f32)
DECLFUN(uint32x4_t, float32x4_t, vcaltq, f32)
#undef DECLFUN

#if defined(AURA_ENABLE_NEON_FP16)
#  define DECLFUN(dtype, vtype, prefix, postfix)                                      \
    template <typename T>                                                             \
    inline typename std::enable_if<std::is_same<T, float16_t>::value, dtype>::type    \
    vcale(const vtype &v, const vtype &u)                                             \
    {                                                                                 \
        return prefix##_##postfix(v, u);                                              \
    }

DECLFUN(uint16x4_t, float16x4_t, vcale,  f16)
DECLFUN(uint16x8_t, float16x8_t, vcaleq, f16)
#  undef DECLFUN

#  define DECLFUN(dtype, vtype, prefix, postfix)                                      \
    template <typename T>                                                             \
    inline typename std::enable_if<std::is_same<T, float16_t>::value, dtype>::type    \
    vcalt(const vtype &v, const vtype &u)                                             \
    {                                                                                 \
        return prefix##_##postfix(v, u);                                              \
    }

DECLFUN(uint16x4_t, float16x4_t, vcalt,  f16)
DECLFUN(uint16x8_t, float16x8_t, vcaltq, f16)
#  undef DECLFUN
#endif

} // namespace neon

} // namespace aura

#endif // AURA_RUNTIME_CORE_NEON_CAL_HPP__