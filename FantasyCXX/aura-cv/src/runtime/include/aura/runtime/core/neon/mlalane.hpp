#ifndef AURA_RUNTIME_CORE_NEON_MLALANE_HPP__
#define AURA_RUNTIME_CORE_NEON_MLALANE_HPP__

#include <arm_neon.h>

namespace aura
{

namespace neon
{

#define DECLFUN(dtype, vtype, prefix, postfix)                              \
    template <int n>                                                        \
    inline dtype vmla_lane(const dtype &v, const dtype &u, const vtype &m)  \
    {                                                                       \
        return prefix##_##postfix(v, u, m, n);                              \
    }

DECLFUN(int16x4_t,   int16x4_t,   vmla_lane,  s16)
DECLFUN(uint16x4_t,  uint16x4_t,  vmla_lane,  u16)
DECLFUN(int16x8_t,   int16x4_t,   vmlaq_lane, s16)
DECLFUN(uint16x8_t,  uint16x4_t,  vmlaq_lane, u16)
DECLFUN(int32x2_t,   int32x2_t,   vmla_lane,  s32)
DECLFUN(uint32x2_t,  uint32x2_t,  vmla_lane,  u32)
DECLFUN(int32x4_t,   int32x2_t,   vmlaq_lane, s32)
DECLFUN(uint32x4_t,  uint32x2_t,  vmlaq_lane, u32)
DECLFUN(float32x2_t, float32x2_t, vmla_lane,  f32)
DECLFUN(float32x4_t, float32x2_t, vmlaq_lane, f32)
#undef DECLFUN

} // namespace neon

} // namespace aura

#endif // AURA_RUNTIME_CORE_NEON_MLALANE_HPP__