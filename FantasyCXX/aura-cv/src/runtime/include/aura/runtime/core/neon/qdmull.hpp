#ifndef AURA_RUNTIME_CORE_NEON_QDMULL_HPP__
#define AURA_RUNTIME_CORE_NEON_QDMULL_HPP__

#include <arm_neon.h>

namespace aura
{

namespace neon
{

#define DECLFUN(dtype, vtype, prefix, postfix)           \
    inline dtype vqdmull(const vtype &v, const vtype &u) \
    {                                                    \
        return prefix##_##postfix(v, u);                 \
    }

DECLFUN(int32x4_t,   int16x4_t,   vqdmull,  s16)
DECLFUN(int64x2_t,   int32x2_t,   vqdmull,  s32)
#undef DECLFUN

#define DECLFUN(dtype, vtype, stype, prefix, postfix)      \
    inline dtype vqdmull_n(const vtype &v, const stype &u) \
    {                                                      \
        return prefix##_##postfix(v, u);                   \
    }

DECLFUN(int32x4_t, int16x4_t, DT_S16, vqdmull_n, s16)
DECLFUN(int64x2_t, int32x2_t, DT_S32, vqdmull_n, s32)
#undef DECLFUN

#define DECLFUN(dtype, vtype, prefix, postfix)                  \
    template <int n>                                            \
    inline dtype vqdmull_lane(const vtype &v, const vtype &u)   \
    {                                                           \
        return prefix##_##postfix(v, u, n);                     \
    }

DECLFUN(int32x4_t, int16x4_t, vqdmull_lane, s16)
DECLFUN(int64x2_t, int32x2_t, vqdmull_lane, s32)
#undef DECLFUN

} // namespace neon

} // namespace aura

#endif // AURA_RUNTIME_CORE_NEON_QDMULL_HPP__
