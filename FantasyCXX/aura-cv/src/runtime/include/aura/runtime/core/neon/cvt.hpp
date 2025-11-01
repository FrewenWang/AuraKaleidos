#ifndef AURA_RUNTIME_CORE_NEON_CVT_HPP__
#define AURA_RUNTIME_CORE_NEON_CVT_HPP__

namespace aura
{

namespace neon
{

#define DECLFUN(ptype, vtype, prefix, postfix)                                        \
    template <typename T>                                                             \
    inline typename std::enable_if<std::is_same<T, MI_F32>::value, ptype>::type       \
    vcvt(const vtype &v)                                                              \
    {                                                                                 \
        return prefix##_f32_##postfix(v);                                             \
    }

DECLFUN(float32x2_t, uint32x2_t, vcvt,  u32)
DECLFUN(float32x2_t, int32x2_t,  vcvt,  s32)
DECLFUN(float32x4_t, uint32x4_t, vcvtq, u32)
DECLFUN(float32x4_t, int32x4_t,  vcvtq, s32)
#  if defined(AURA_ENABLE_NEON_FP16)
DECLFUN(float32x4_t, float16x4_t, vcvt, f16)
#  endif
#undef DECLFUN

#define DECLFUN(ptype, vtype, prefix, postfix)                                        \
    template <typename T>                                                             \
    inline typename std::enable_if<std::is_same<T, float16_t>::value, ptype>::type    \
    vcvt(const vtype &v)                                                              \
    {                                                                                 \
        return prefix##_f16_##postfix(v);                                             \
    }

#  if defined(AURA_ENABLE_NEON_FP16)
DECLFUN(float16x4_t, float32x4_t, vcvt, f32)
#  endif
#undef DECLFUN

#define DECLFUN(ptype, vtype, prefix, postfix)                                        \
    template <typename T>                                                             \
    inline typename std::enable_if<std::is_same<T, MI_U32>::value, ptype>::type       \
    vcvt(const vtype &v)                                                              \
    {                                                                                 \
        return prefix##_u32_##postfix(v);                                             \
    }

DECLFUN(uint32x2_t, float32x2_t, vcvt,  f32)
DECLFUN(uint32x4_t, float32x4_t, vcvtq, f32)
#undef DECLFUN

#define DECLFUN(ptype, vtype, prefix, postfix)                                        \
    template <typename T>                                                             \
    inline typename std::enable_if<std::is_same<T, MI_S32>::value, ptype>::type       \
    vcvt(const vtype &v)                                                              \
    {                                                                                 \
        return prefix##_s32_##postfix(v);                                             \
    }

DECLFUN(int32x2_t, float32x2_t, vcvt,  f32)
DECLFUN(int32x4_t, float32x4_t, vcvtq, f32)
#undef DECLFUN

#if defined(__arm__)
inline int32x2_t vcvtn_s32_f32(float32x2_t v)
{
    const int32x2_t vds32_sign = vdup_n_s32(1 << 31);
    const int32x2_t vds32_05 = vreinterpret_s32_f32(vdup_n_f32(0.5f));
    int32x2_t vds32_add = vorr_s32(vds32_05, vand_s32(vds32_sign, vreinterpret_s32_f32(v)));
    return int32x2_t(vcvt_s32_f32(vadd_f32(v, vreinterpret_f32_s32(vds32_add))));
}

inline int32x4_t vcvtnq_s32_f32(float32x4_t v)
{
    const int32x4_t vqs32_sign = vdupq_n_s32(1 << 31);
    const int32x4_t vqs32_05 = vreinterpretq_s32_f32(vdupq_n_f32(0.5f));
    int32x4_t vqs32_add = vorrq_s32(vqs32_05, vandq_s32(vqs32_sign, vreinterpretq_s32_f32(v)));
    return int32x4_t(vcvtq_s32_f32(vaddq_f32(v, vreinterpretq_f32_s32(vqs32_add))));
}
#endif // __arm__

#define DECLFUN(ptype, vtype, prefix, postfix)                                        \
    template <typename T>                                                             \
    inline typename std::enable_if<std::is_same<T, MI_S32>::value, ptype>::type       \
    vcvtn(const vtype &v)                                                             \
    {                                                                                 \
        return prefix##_s32_##postfix(v);                                             \
    }

DECLFUN(int32x2_t, float32x2_t, vcvtn,  f32)
DECLFUN(int32x4_t, float32x4_t, vcvtnq, f32)
#undef DECLFUN

} // namespace neon

} // namespace aura

#endif // AURA_RUNTIME_CORE_NEON_CVT_HPP__