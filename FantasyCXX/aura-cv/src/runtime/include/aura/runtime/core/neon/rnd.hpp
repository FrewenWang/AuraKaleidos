#ifndef AURA_RUNTIME_CORE_NEON_RND_HPP__
#define AURA_RUNTIME_CORE_NEON_RND_HPP__

#include <arm_neon.h>

namespace aura
{

namespace neon
{
// aarch32 must set AURA_ARM82=ON
#if defined(__aarch64__)
#  define DECLFUN(dtype, vtype, prefix, postfix) \
    inline vtype vrnd(const vtype &v)            \
    {                                            \
        return prefix##_##postfix(v);            \
    }

DECLFUN(float32x2_t,  float32x2_t, vrnd,  f32)
DECLFUN(float32x4_t,  float32x4_t, vrndq, f32)
#  undef DECLFUN
#endif // __aarch64__

inline float32x4_t vrndn(const float32x4_t &val)
{
#if defined(__aarch64__)
    return vrndnq_f32(val);
#else  // __aarch64__
    static const float32x4_t vqf32_half  = vdupq_n_f32(0.5f);
    static const float32x4_t vqf32_1     = vdupq_n_f32(1.f);
    static const int32x4_t   vqs32_1     = vdupq_n_s32(1);
    const int32x4_t          vqs32_z     = vcvtq_s32_f32(val);
    const float32x4_t        vqf32_r     = vcvtq_f32_s32(vqs32_z);
    const float32x4_t        vqf32_floor = vbslq_f32(vcgtq_f32(vqf32_r, val), vsubq_f32(vqf32_r, vqf32_1), vqf32_r);
    const float32x4_t        vqf32_diff  = vsubq_f32(val, vqf32_floor);

    return vbslq_f32(vorrq_u32(vcltq_f32(vqf32_diff, vqf32_half), vandq_u32(vceqq_f32(vqf32_diff, vqf32_half),
                     vmvnq_u32(vtstq_s32(vandq_s32(vcvtq_s32_f32(vqf32_floor), vqs32_1), vqs32_1)))),
                     vqf32_floor, vaddq_f32(vqf32_floor, vqf32_1));
#endif // __aarch64__
}

inline float32x2_t vrndn(const float32x2_t &val)
{
#if defined(__aarch64__)
    return vrndn_f32(val);
#else  // __aarch64__
    static const float32x2_t vf32_half  = vdup_n_f32(0.5f);
    static const float32x2_t vf32_1     = vdup_n_f32(1.f);
    static const int32x2_t   vs32_1     = vdup_n_s32(1);
    const int32x2_t          vs32_z     = vcvt_s32_f32(val);
    const float32x2_t        vf32_r     = vcvt_f32_s32(vs32_z);
    const float32x2_t        vf32_floor = vbsl_f32(vcgt_f32(vf32_r, val), vsub_f32(vf32_r, vf32_1), vf32_r);
    const float32x2_t        vf32_diff  = vsub_f32(val, vf32_floor);

    return vbsl_f32(vorr_u32(vclt_f32(vf32_diff, vf32_half), vand_u32(vceq_f32(vf32_diff, vf32_half),
                    vmvn_u32(vtst_s32(vand_s32(vcvt_s32_f32(vf32_floor), vs32_1), vs32_1)))),
                    vf32_floor, vadd_f32(vf32_floor, vf32_1));
#endif // __aarch64__
}

#if defined(AURA_ENABLE_NEON_FP16)
#  define DECLFUN(dtype, vtype, prefix, postfix) \
    inline vtype vrndn(const vtype &v)         \
    {                                          \
        return prefix##_##postfix(v);          \
    }

#    if defined(__aarch64__)
DECLFUN(float16x4_t,  float16x4_t, vrndn,  f16)
DECLFUN(float16x8_t,  float16x8_t, vrndnq, f16)
#    endif // __aarch64__

#  undef DECLFUN
#endif
} // namespace neon

} // namespace aura

#endif // AURA_RUNTIME_CORE_NEON_RND_HPP__