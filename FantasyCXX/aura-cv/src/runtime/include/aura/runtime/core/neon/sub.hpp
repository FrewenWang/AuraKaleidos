#ifndef AURA_RUNTIME_CORE_NEON_SUB_HPP__
#define AURA_RUNTIME_CORE_NEON_SUB_HPP__

#include <arm_neon.h>

namespace aura
{

namespace neon
{

//vsub
#define DECLFUN(stype, vtype, prefix, postfix)        \
    inline vtype vsub(const vtype &v, const vtype &u) \
    {                                                 \
        return prefix##_##postfix(v, u);              \
    }

DECLFUN(MI_U8,  uint8x8_t,   vsub,  u8)
DECLFUN(MI_U8,  uint8x16_t,  vsubq, u8)
DECLFUN(MI_S8,  int8x8_t,    vsub,  s8)
DECLFUN(MI_S8,  int8x16_t,   vsubq, s8)
DECLFUN(MI_U16, uint16x4_t,  vsub,  u16)
DECLFUN(MI_U16, uint16x8_t,  vsubq, u16)
DECLFUN(MI_S16, int16x4_t,   vsub,  s16)
DECLFUN(MI_S16, int16x8_t,   vsubq, s16)
DECLFUN(MI_U32, uint32x2_t,  vsub,  u32)
DECLFUN(MI_U32, uint32x4_t,  vsubq, u32)
DECLFUN(MI_S32, int32x2_t,   vsub,  s32)
DECLFUN(MI_S32, int32x4_t,   vsubq, s32)
DECLFUN(MI_U64, uint64x1_t,  vsub,  u64)
DECLFUN(MI_U64, uint64x2_t,  vsubq, u64)
DECLFUN(MI_S64, int64x1_t,   vsub,  s64)
DECLFUN(MI_S64, int64x2_t,   vsubq, s64)
DECLFUN(MI_F32, float32x2_t, vsub,  f32)
DECLFUN(MI_F32, float32x4_t, vsubq, f32)

#  if defined(AURA_ENABLE_NEON_FP16)
DECLFUN(float16_t, float16x4_t, vsub, f16)
DECLFUN(float16_t, float16x8_t, vsubq, f16)
#  endif
#undef DECLFUN

// vsubl
#define DECLFUN(dtype, vtype, postfix)                 \
    inline dtype vsubl(const vtype &v, const vtype &u) \
    {                                                  \
        return vsubl_##postfix(v, u);                  \
    }

DECLFUN(uint16x8_t, uint8x8_t,   u8)
DECLFUN(int16x8_t,  int8x8_t,    s8)
DECLFUN(uint32x4_t, uint16x4_t,  u16)
DECLFUN(int32x4_t,  int16x4_t,   s16)
DECLFUN(uint64x2_t, uint32x2_t,  u32)
DECLFUN(int64x2_t,  int32x2_t,   s32)
#undef DECLFUN

// vsubw
#define DECLFUN(dtype, vtype, postfix)                 \
    inline dtype vsubw(const dtype &v, const vtype &u) \
    {                                                  \
        return vsubw_##postfix(v, u);                  \
    }

DECLFUN(uint16x8_t, uint8x8_t,   u8)
DECLFUN(int16x8_t,  int8x8_t,    s8)
DECLFUN(uint32x4_t, uint16x4_t,  u16)
DECLFUN(int32x4_t,  int16x4_t,   s16)
DECLFUN(uint64x2_t, uint32x2_t,  u32)
DECLFUN(int64x2_t,  int32x2_t,   s32)
#undef DECLFUN

// vhsub
#define DECLFUN(dtype, vtype, prefix, postfix)         \
    inline vtype vhsub(const vtype &v, const vtype &u) \
    {                                                  \
        return prefix##_##postfix(v, u);               \
    }

DECLFUN(uint8x8_t,  uint8x8_t,  vhsub,  u8)
DECLFUN(uint8x16_t, uint8x16_t, vhsubq, u8)
DECLFUN(int8x8_t,   int8x8_t,   vhsub,  s8)
DECLFUN(int8x16_t,  int8x16_t,  vhsubq, s8)
DECLFUN(uint16x4_t, uint16x4_t, vhsub,  u16)
DECLFUN(uint16x8_t, uint16x8_t, vhsubq, u16)
DECLFUN(int16x4_t,  int16x4_t,  vhsub,  s16)
DECLFUN(int16x8_t,  int16x8_t,  vhsubq, s16)
DECLFUN(uint32x2_t, uint32x2_t, vhsub,  u32)
DECLFUN(uint32x4_t, uint32x4_t, vhsubq, u32)
DECLFUN(int32x2_t,  int32x2_t,  vhsub,  s32)
DECLFUN(int32x4_t,  int32x4_t,  vhsubq, s32)
#undef DECLFUN

// vqsub
#define DECLFUN(dtype, vtype, prefix, postfix)         \
    inline vtype vqsub(const vtype &v, const vtype &u) \
    {                                                  \
        return prefix##_##postfix(v, u);               \
    }

DECLFUN(uint8x8_t,  uint8x8_t,   vqsub,  u8)
DECLFUN(uint8x16_t, uint8x16_t,  vqsubq, u8)
DECLFUN(int8x8_t,   int8x8_t,    vqsub,  s8)
DECLFUN(int8x16_t,  int8x16_t,   vqsubq, s8)
DECLFUN(uint16x4_t, uint16x4_t,  vqsub,  u16)
DECLFUN(uint16x8_t, uint16x8_t,  vqsubq, u16)
DECLFUN(int16x4_t,  int16x4_t,   vqsub,  s16)
DECLFUN(int16x8_t,  int16x8_t,   vqsubq, s16)
DECLFUN(uint32x2_t, uint32x2_t,  vqsub,  u32)
DECLFUN(uint32x4_t, uint32x4_t,  vqsubq, u32)
DECLFUN(int32x2_t,  int32x2_t,   vqsub,  s32)
DECLFUN(int32x4_t,  int32x4_t,   vqsubq, s32)
DECLFUN(uint64x1_t, uint64x1_t,  vqsub,  u64)
DECLFUN(uint64x2_t, uint64x2_t,  vqsubq, u64)
DECLFUN(int64x1_t,  int64x1_t,   vqsub,  s64)
DECLFUN(int64x2_t,  int64x2_t,   vqsubq, s64)
#undef DECLFUN

// vsubhn
#define DECLFUN(dtype, vtype, postfix)                  \
    inline dtype vsubhn(const vtype &v, const vtype &u) \
    {                                                   \
        return vsubhn_##postfix(v, u);                  \
    }

DECLFUN(uint8x8_t,  uint16x8_t, u16)
DECLFUN(int8x8_t,   int16x8_t,  s16)
DECLFUN(uint16x4_t, uint32x4_t, u32)
DECLFUN(int16x4_t,  int32x4_t,  s32)
DECLFUN(uint32x2_t, uint64x2_t, u64)
DECLFUN(int32x2_t,  int64x2_t,  s64)
#undef DECLFUN

// vrsubhn
#define DECLFUN(dtype, vtype, postfix)                   \
    inline dtype vrsubhn(const vtype &v, const vtype &u) \
    {                                                    \
        return vrsubhn_##postfix(v, u);                  \
    }

DECLFUN(uint8x8_t,  uint16x8_t, u16)
DECLFUN(int8x8_t,   int16x8_t,  s16)
DECLFUN(uint16x4_t, uint32x4_t, u32)
DECLFUN(int16x4_t,  int32x4_t,  s32)
DECLFUN(uint32x2_t, uint64x2_t, u64)
DECLFUN(int32x2_t,  int64x2_t,  s64)
#undef DECLFUN

} // namespace neon

} // namespace aura

#endif // AURA_RUNTIME_CORE_NEON_SUB_HPP__