#ifndef AURA_RUNTIME_CORE_NEON_MLA_HPP__
#define AURA_RUNTIME_CORE_NEON_MLA_HPP__

#include <arm_neon.h>

namespace aura
{

namespace neon
{

#define DECLFUN(dtype, vtype, prefix, postfix)                        \
    inline vtype vmla(const vtype &v, const vtype &u, const vtype &p) \
    {                                                                 \
        return prefix##_##postfix(v, u, p);                           \
    }

DECLFUN(uint8x8_t,   uint8x8_t,   vmla,  u8)
DECLFUN(uint8x16_t,  uint8x16_t,  vmlaq, u8)
DECLFUN(int8x8_t,    int8x8_t,    vmla,  s8)
DECLFUN(int8x16_t,   int8x16_t,   vmlaq, s8)
DECLFUN(uint16x4_t,  uint16x4_t,  vmla,  u16)
DECLFUN(uint16x8_t,  uint16x8_t,  vmlaq, u16)
DECLFUN(int16x4_t,   int16x4_t,   vmla,  s16)
DECLFUN(int16x8_t,   int16x8_t,   vmlaq, s16)
DECLFUN(uint32x2_t,  uint32x2_t,  vmla,  u32)
DECLFUN(uint32x4_t,  uint32x4_t,  vmlaq, u32)
DECLFUN(int32x2_t,   int32x2_t,   vmla,  s32)
DECLFUN(int32x4_t,   int32x4_t,   vmlaq, s32)
DECLFUN(float32x2_t, float32x2_t, vmla,  f32)
DECLFUN(float32x4_t, float32x4_t, vmlaq, f32)

#  if defined(AURA_ENABLE_NEON_FP16)
inline float16x4_t vmla(const float16x4_t &a, const float16x4_t &b, const float16x4_t &c)
{
    return vadd_f16(a, vmul_f16(b, c));
}

inline float16x8_t vmla(const float16x8_t &a, const float16x8_t &b, const float16x8_t &c)
{
    return vaddq_f16(a, vmulq_f16(b, c));
}
#  endif
#undef DECLFUN

#define DECLFUN(dtype, vtype, stype, prefix, postfix)                  \
    inline dtype vmla(const dtype &v, const vtype &u, const stype &p)  \
    {                                                                  \
        return prefix##_##postfix(v, u, p);                            \
    }

DECLFUN(uint16x4_t,  uint16x4_t,  MI_U16,  vmla_n,  u16)
DECLFUN(uint16x8_t,  uint16x8_t,  MI_U16,  vmlaq_n, u16)
DECLFUN(int16x4_t,   int16x4_t,   MI_S16,  vmla_n,  s16)
DECLFUN(int16x8_t,   int16x8_t,   MI_S16,  vmlaq_n, s16)
DECLFUN(uint32x2_t,  uint32x2_t,  MI_U32,  vmla_n,  u32)
DECLFUN(uint32x4_t,  uint32x4_t,  MI_U32,  vmlaq_n, u32)
DECLFUN(int32x2_t,   int32x2_t,   MI_S32,  vmla_n,  s32)
DECLFUN(int32x4_t,   int32x4_t,   MI_S32,  vmlaq_n, s32)
DECLFUN(float32x2_t, float32x2_t, MI_F32,  vmla_n,  f32)
DECLFUN(float32x4_t, float32x4_t, MI_F32,  vmlaq_n, f32)
#undef DECLFUN

#define DECLFUN(dtype, vtype, prefix, postfix)                         \
    inline dtype vmlal(const dtype &v, const vtype &u, const vtype &p) \
    {                                                                  \
        return prefix##_##postfix(v, u, p);                            \
    }

DECLFUN(uint16x8_t,  uint8x8_t,   vmlal,  u8)
DECLFUN(int16x8_t,   int8x8_t,    vmlal,  s8)
DECLFUN(uint32x4_t,  uint16x4_t,  vmlal,  u16)
DECLFUN(int32x4_t,   int16x4_t,   vmlal,  s16)
DECLFUN(uint64x2_t,  uint32x2_t,  vmlal,  u32)
DECLFUN(int64x2_t,   int32x2_t,   vmlal,  s32)
#undef DECLFUN

#define DECLFUN(dtype, vtype, stype, prefix, postfix)                  \
    inline dtype vmlal(const dtype &v, const vtype &u, const stype &p) \
    {                                                                  \
        return prefix##_##postfix(v, u, p);                            \
    }

DECLFUN(uint32x4_t,  uint16x4_t, MI_U16,  vmlal_n,  u16)
DECLFUN(int32x4_t,   int16x4_t,  MI_S16,  vmlal_n,  s16)
DECLFUN(uint64x2_t,  uint32x2_t, MI_U32,  vmlal_n,  u32)
DECLFUN(int64x2_t,   int32x2_t,  MI_S32,  vmlal_n,  s32)
#undef DECLFUN

} // namespace neon

} // namespace aura

#endif // AURA_RUNTIME_CORE_NEON_MLA_HPP__
