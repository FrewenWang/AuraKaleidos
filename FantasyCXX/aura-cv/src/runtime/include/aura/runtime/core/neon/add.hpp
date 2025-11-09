#ifndef AURA_RUNTIME_CORE_NEON_ADD_HPP__
#define AURA_RUNTIME_CORE_NEON_ADD_HPP__

#include <arm_neon.h>

namespace aura
{

namespace neon
{

#define DECLFUN(stype, vtype, prefix, postfix)        \
    inline vtype vadd(const vtype &v, const vtype &u) \
    {                                                 \
        return prefix##_##postfix(v, u);              \
    }

DECLFUN(DT_U8,  uint8x8_t,   vadd,  u8)
DECLFUN(DT_U8,  uint8x16_t,  vaddq, u8)
DECLFUN(DT_S8,  int8x8_t,    vadd,  s8)
DECLFUN(DT_S8,  int8x16_t,   vaddq, s8)
DECLFUN(DT_U16, uint16x4_t,  vadd,  u16)
DECLFUN(DT_U16, uint16x8_t,  vaddq, u16)
DECLFUN(DT_S16, int16x4_t,   vadd,  s16)
DECLFUN(DT_S16, int16x8_t,   vaddq, s16)
DECLFUN(DT_U32, uint32x2_t,  vadd,  u32)
DECLFUN(DT_U32, uint32x4_t,  vaddq, u32)
DECLFUN(DT_S32, int32x2_t,   vadd,  s32)
DECLFUN(DT_S32, int32x4_t,   vaddq, s32)
DECLFUN(DT_U64, uint64x1_t,  vadd,  u64)
DECLFUN(DT_U64, uint64x2_t,  vaddq, u64)
DECLFUN(DT_S64, int64x1_t,   vadd,  s64)
DECLFUN(DT_S64, int64x2_t,   vaddq, s64)
DECLFUN(DT_F32, float32x2_t, vadd,  f32)
DECLFUN(DT_F32, float32x4_t, vaddq, f32)

#  if defined(AURA_ENABLE_NEON_FP16)
DECLFUN(float16_t, float16x4_t, vadd, f16)
DECLFUN(float16_t, float16x8_t, vaddq, f16)
#  endif
#undef DECLFUN

// vaddl
#define DECLFUN(dtype, vtype, postfix)                 \
    inline dtype vaddl(const vtype &v, const vtype &u) \
    {                                                  \
        return vaddl_##postfix(v, u);                  \
    }

DECLFUN(uint16x8_t, uint8x8_t,   u8)
DECLFUN(int16x8_t,  int8x8_t,    s8)
DECLFUN(uint32x4_t, uint16x4_t, u16)
DECLFUN(int32x4_t,  int16x4_t,  s16)
DECLFUN(uint64x2_t, uint32x2_t, u32)
DECLFUN(int64x2_t,  int32x2_t,  s32)
#undef DECLFUN

// vaddw
#define DECLFUN(dtype, vtype, postfix)                 \
    inline dtype vaddw(const dtype &v, const vtype &u) \
    {                                                  \
        return vaddw_##postfix(v, u);                  \
    }

DECLFUN(uint16x8_t, uint8x8_t,   u8)
DECLFUN(int16x8_t,  int8x8_t,    s8)
DECLFUN(uint32x4_t, uint16x4_t, u16)
DECLFUN(int32x4_t,  int16x4_t,  s16)
DECLFUN(uint64x2_t, uint32x2_t, u32)
DECLFUN(int64x2_t,  int32x2_t,  s32)
#undef DECLFUN

// vhadd
#define DECLFUN(dtype, vtype, prefix, postfix)         \
    inline vtype vhadd(const vtype &v, const vtype &u) \
    {                                                  \
        return prefix##_##postfix(v, u);               \
    }

DECLFUN(uint8x8_t,  uint8x8_t,  vhadd,  u8)
DECLFUN(uint8x16_t, uint8x16_t, vhaddq, u8)
DECLFUN(int8x8_t,   int8x8_t,   vhadd,  s8)
DECLFUN(int8x16_t,  int8x16_t,  vhaddq, s8)
DECLFUN(uint16x4_t, uint16x4_t, vhadd,  u16)
DECLFUN(uint16x8_t, uint16x8_t, vhaddq, u16)
DECLFUN(int16x4_t,  int16x4_t,  vhadd,  s16)
DECLFUN(int16x8_t,  int16x8_t,  vhaddq, s16)
DECLFUN(uint32x2_t, uint32x2_t, vhadd,  u32)
DECLFUN(uint32x4_t, uint32x4_t, vhaddq, u32)
DECLFUN(int32x2_t,  int32x2_t,  vhadd,  s32)
DECLFUN(int32x4_t,  int32x4_t,  vhaddq, s32)
#undef DECLFUN

// vrhadd
#define DECLFUN(dtype, vtype, prefix, postfix)         \
    inline vtype vrhadd(const vtype &v, const vtype &u) \
    {                                                  \
        return prefix##_##postfix(v, u);               \
    }

DECLFUN(uint8x8_t,  uint8x8_t,  vrhadd,  u8)
DECLFUN(uint8x16_t, uint8x16_t, vrhaddq, u8)
DECLFUN(int8x8_t,   int8x8_t,   vrhadd,  s8)
DECLFUN(int8x16_t,  int8x16_t,  vrhaddq, s8)
DECLFUN(uint16x4_t, uint16x4_t, vrhadd,  u16)
DECLFUN(uint16x8_t, uint16x8_t, vrhaddq, u16)
DECLFUN(int16x4_t,  int16x4_t,  vrhadd,  s16)
DECLFUN(int16x8_t,  int16x8_t,  vrhaddq, s16)
DECLFUN(uint32x2_t, uint32x2_t, vrhadd,  u32)
DECLFUN(uint32x4_t, uint32x4_t, vrhaddq, u32)
DECLFUN(int32x2_t,  int32x2_t,  vrhadd,  s32)
DECLFUN(int32x4_t,  int32x4_t,  vrhaddq, s32)
#undef DECLFUN

// vqadd
#define DECLFUN(dtype, vtype, prefix, postfix)         \
    inline vtype vqadd(const vtype &v, const vtype &u) \
    {                                                  \
        return prefix##_##postfix(v, u);               \
    }

DECLFUN(uint8x8_t,  uint8x8_t,  vqadd,  u8)
DECLFUN(uint8x16_t, uint8x16_t, vqaddq, u8)
DECLFUN(int8x8_t,   int8x8_t,   vqadd,  s8)
DECLFUN(int8x16_t,  int8x16_t,  vqaddq, s8)
DECLFUN(uint16x4_t, uint16x4_t, vqadd,  u16)
DECLFUN(uint16x8_t, uint16x8_t, vqaddq, u16)
DECLFUN(int16x4_t,  int16x4_t,  vqadd,  s16)
DECLFUN(int16x8_t,  int16x8_t,  vqaddq, s16)
DECLFUN(uint32x2_t, uint32x2_t, vqadd,  u32)
DECLFUN(uint32x4_t, uint32x4_t, vqaddq, u32)
DECLFUN(int32x2_t,  int32x2_t,  vqadd,  s32)
DECLFUN(int32x4_t,  int32x4_t,  vqaddq, s32)
DECLFUN(uint64x1_t, uint64x1_t, vqadd,  u64)
DECLFUN(uint64x2_t, uint64x2_t, vqaddq, u64)
DECLFUN(int64x1_t,  int64x1_t,  vqadd,  s64)
DECLFUN(int64x2_t,  int64x2_t,  vqaddq, s64)
#undef DECLFUN

// vaddhn
#define DECLFUN(dtype, vtype, postfix)                  \
    inline dtype vaddhn(const vtype &v, const vtype &u) \
    {                                                   \
        return vaddhn_##postfix(v, u);                  \
    }

DECLFUN(uint8x8_t,  uint16x8_t,   u16)
DECLFUN(int8x8_t,   int16x8_t,    s16)
DECLFUN(uint16x4_t, uint32x4_t,   u32)
DECLFUN(int16x4_t,  int32x4_t,    s32)
DECLFUN(uint32x2_t, uint64x2_t,   u64)
DECLFUN(int32x2_t,  int64x2_t,    s64)
#undef DECLFUN

// vraddhn
#define DECLFUN(dtype, vtype, postfix)                   \
    inline dtype vraddhn(const vtype &v, const vtype &u) \
    {                                                    \
        return vraddhn_##postfix(v, u);                  \
    }

DECLFUN(uint8x8_t,  uint16x8_t,  u16)
DECLFUN(int8x8_t,   int16x8_t,   s16)
DECLFUN(uint16x4_t, uint32x4_t,  u32)
DECLFUN(int16x4_t,  int32x4_t,   s32)
DECLFUN(uint32x2_t, uint64x2_t,  u64)
DECLFUN(int32x2_t,  int64x2_t,   s64)
#undef DECLFUN

} // namespace neon

} // namespace aura

#endif // AURA_RUNTIME_CORE_NEON_ADD_HPP__