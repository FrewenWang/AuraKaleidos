#ifndef AURA_RUNTIME_CORE_NEON_MLS_HPP__
#define AURA_RUNTIME_CORE_NEON_MLS_HPP__

#include <arm_neon.h>

namespace aura
{

namespace neon
{

#define DECLFUN(dtype, vtype, prefix, postfix)                        \
    inline vtype vmls(const vtype &v, const vtype &u, const vtype &p) \
    {                                                                 \
        return prefix##_##postfix(v, u, p);                           \
    }

DECLFUN(uint8x8_t,   uint8x8_t,   vmls,  u8)
DECLFUN(uint8x16_t,  uint8x16_t,  vmlsq, u8)
DECLFUN(int8x8_t,    int8x8_t,    vmls,  s8)
DECLFUN(int8x16_t,   int8x16_t,   vmlsq, s8)
DECLFUN(uint16x4_t,  uint16x4_t,  vmls,  u16)
DECLFUN(uint16x8_t,  uint16x8_t,  vmlsq, u16)
DECLFUN(int16x4_t,   int16x4_t,   vmls,  s16)
DECLFUN(int16x8_t,   int16x8_t,   vmlsq, s16)
DECLFUN(uint32x2_t,  uint32x2_t,  vmls,  u32)
DECLFUN(uint32x4_t,  uint32x4_t,  vmlsq, u32)
DECLFUN(int32x2_t,   int32x2_t,   vmls,  s32)
DECLFUN(int32x4_t,   int32x4_t,   vmlsq, s32)
DECLFUN(float32x2_t, float32x2_t, vmls,  f32)
DECLFUN(float32x4_t, float32x4_t, vmlsq, f32)
#undef DECLFUN

#define DECLFUN(dtype, vtype, stype, prefix, postfix)                  \
    inline dtype vmls(const dtype &v, const vtype &u, const stype &p)  \
    {                                                                  \
        return prefix##_##postfix(v, u, p);                            \
    }

DECLFUN(uint16x4_t,  uint16x4_t,  MI_U16,  vmls_n,  u16)
DECLFUN(uint16x8_t,  uint16x8_t,  MI_U16,  vmlsq_n, u16)
DECLFUN(int16x4_t,   int16x4_t,   MI_S16,  vmls_n,  s16)
DECLFUN(int16x8_t,   int16x8_t,   MI_S16,  vmlsq_n, s16)
DECLFUN(uint32x2_t,  uint32x2_t,  MI_U32,  vmls_n,  u32)
DECLFUN(uint32x4_t,  uint32x4_t,  MI_U32,  vmlsq_n, u32)
DECLFUN(int32x2_t,   int32x2_t,   MI_S32,  vmls_n,  s32)
DECLFUN(int32x4_t,   int32x4_t,   MI_S32,  vmlsq_n, s32)
DECLFUN(float32x2_t, float32x2_t, MI_F32,  vmls_n,  f32)
DECLFUN(float32x4_t, float32x4_t, MI_F32,  vmlsq_n, f32)
#undef DECLFUN

#define DECLFUN(dtype, vtype, prefix, postfix)                         \
    inline dtype vmlsl(const dtype &v, const vtype &u, const vtype &p) \
    {                                                                  \
        return prefix##_##postfix(v, u, p);                            \
    }

DECLFUN(uint16x8_t,  uint8x8_t,   vmlsl,  u8)
DECLFUN(int16x8_t,   int8x8_t,    vmlsl,  s8)
DECLFUN(uint32x4_t,  uint16x4_t,  vmlsl,  u16)
DECLFUN(int32x4_t,   int16x4_t,   vmlsl,  s16)
DECLFUN(uint64x2_t,  uint32x2_t,  vmlsl,  u32)
DECLFUN(int64x2_t,   int32x2_t,   vmlsl,  s32)
#undef DECLFUN

#define DECLFUN(dtype, vtype, stype, prefix, postfix)                  \
    inline dtype vmlsl(const dtype &v, const vtype &u, const stype &p) \
    {                                                                  \
        return prefix##_##postfix(v, u, p);                            \
    }

DECLFUN(uint32x4_t,  uint16x4_t, MI_U16,  vmlsl_n,  u16)
DECLFUN(int32x4_t,   int16x4_t,  MI_S16,  vmlsl_n,  s16)
DECLFUN(uint64x2_t,  uint32x2_t, MI_U32,  vmlsl_n,  u32)
DECLFUN(int64x2_t,   int32x2_t,  MI_S32,  vmlsl_n,  s32)
#undef DECLFUN

} // namespace neon

} // namespace aura

#endif // AURA_RUNTIME_CORE_NEON_MLS_HPP__
