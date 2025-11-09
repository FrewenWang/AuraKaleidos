#ifndef AURA_RUNTIME_CORE_NEON_QDMULH_HPP__
#define AURA_RUNTIME_CORE_NEON_QDMULH_HPP__

#include <arm_neon.h>

namespace aura
{

namespace neon
{

#define DECLFUN(dtype, vtype, prefix, postfix)           \
    inline vtype vqdmulh(const vtype &v, const vtype &u) \
    {                                                    \
        return prefix##_##postfix(v, u);                 \
    }

DECLFUN(int16x4_t,   int16x4_t,   vqdmulh,  s16)
DECLFUN(int16x8_t,   int16x8_t,   vqdmulhq, s16)
DECLFUN(int32x2_t,   int32x2_t,   vqdmulh,  s32)
DECLFUN(int32x4_t,   int32x4_t,   vqdmulhq, s32)
#undef DECLFUN

#define DECLFUN(dtype, stype, vtype, prefix, postfix)    \
    inline dtype vqdmulh(const vtype &v, const stype &u) \
    {                                                    \
        return prefix##_##postfix(v, u);                 \
    }

DECLFUN(int16x4_t, DT_S16, int16x4_t,  vqdmulh_n,  s16)
DECLFUN(int16x8_t, DT_S16, int16x8_t,  vqdmulhq_n, s16)
DECLFUN(int32x2_t, DT_S32, int32x2_t,  vqdmulh_n,  s32)
DECLFUN(int32x4_t, DT_S32, int32x4_t,  vqdmulhq_n, s32)
#undef DECLFUN

#define DECLFUN(dtype, vtype, prefix, postfix)                  \
    template <int n>                                            \
    inline dtype vqdmulh_lane(const dtype &v, const vtype &u)   \
    {                                                           \
        return prefix##_##postfix(v, u, n);                     \
    }

DECLFUN(int16x4_t, int16x4_t, vqdmulh_lane,  s16)
DECLFUN(int16x8_t, int16x4_t, vqdmulhq_lane, s16)
DECLFUN(int32x2_t, int32x2_t, vqdmulh_lane,  s32)
DECLFUN(int32x4_t, int32x2_t, vqdmulhq_lane, s32)
#undef DECLFUN

#define DECLFUN(dtype, vtype, prefix, postfix)            \
    inline vtype vqrdmulh(const vtype &v, const vtype &u) \
    {                                                     \
        return prefix##_##postfix(v, u);                  \
    }

DECLFUN(int16x4_t,   int16x4_t,   vqrdmulh,  s16)
DECLFUN(int16x8_t,   int16x8_t,   vqrdmulhq, s16)
DECLFUN(int32x2_t,   int32x2_t,   vqrdmulh,  s32)
DECLFUN(int32x4_t,   int32x4_t,   vqrdmulhq, s32)
#undef DECLFUN

#define DECLFUN(dtype, stype, vtype, prefix, postfix)     \
    inline dtype vqrdmulh(const vtype &v, const stype &u) \
    {                                                     \
        return prefix##_##postfix(v, u);                  \
    }

DECLFUN(int16x4_t, DT_S16, int16x4_t,  vqrdmulh_n,  s16)
DECLFUN(int16x8_t, DT_S16, int16x8_t,  vqrdmulhq_n, s16)
DECLFUN(int32x2_t, DT_S32, int32x2_t,  vqrdmulh_n,  s32)
DECLFUN(int32x4_t, DT_S32, int32x4_t,  vqrdmulhq_n, s32)
#undef DECLFUN

#define DECLFUN(dtype, vtype, prefix, postfix)                  \
    template <int n>                                            \
    inline dtype vqrdmulh_lane(const dtype &v, const vtype &u)  \
    {                                                           \
        return prefix##_##postfix(v, u, n);                     \
    }

DECLFUN(int16x4_t, int16x4_t, vqrdmulh_lane,  s16)
DECLFUN(int16x8_t, int16x4_t, vqrdmulhq_lane, s16)
DECLFUN(int32x2_t, int32x2_t, vqrdmulh_lane,  s32)
DECLFUN(int32x4_t, int32x2_t, vqrdmulhq_lane, s32)
#undef DECLFUN

} // namespace neon

} // namespace aura
#endif // AURA_RUNTIME_CORE_NEON_QDMULH_HPP__
