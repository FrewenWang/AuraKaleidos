#ifndef AURA_RUNTIME_CORE_NEON_LOAD_HPP__
#define AURA_RUNTIME_CORE_NEON_LOAD_HPP__

#include <arm_neon.h>

namespace aura
{

namespace neon
{

#define DECLFUN(stype, vtype, postfix)      \
    inline vtype vload1(const stype *addr)  \
    {                                       \
        return vld1_##postfix(addr);        \
    }

DECLFUN(DT_U8,  uint8x8_t,   u8)
DECLFUN(DT_S8,  int8x8_t,    s8)
DECLFUN(DT_U16, uint16x4_t,  u16)
DECLFUN(DT_S16, int16x4_t,   s16)
DECLFUN(DT_U32, uint32x2_t,  u32)
DECLFUN(DT_S32, int32x2_t,   s32)
DECLFUN(DT_U64, uint64x1_t,  u64)
DECLFUN(DT_S64, int64x1_t,   s64)
DECLFUN(DT_F32, float32x2_t, f32)

#  if defined(AURA_ENABLE_NEON_FP16)
DECLFUN(float16_t, float16x4_t, f16)
#  endif
#undef DECLFUN

#define DECLFUN(stype, vtype, postfix)       \
    inline vtype vload1q(const stype *addr)  \
    {                                        \
        return vld1q_##postfix(addr);        \
    }

DECLFUN(DT_U8,  uint8x16_t,  u8)
DECLFUN(DT_S8,  int8x16_t,   s8)
DECLFUN(DT_U16, uint16x8_t,  u16)
DECLFUN(DT_S16, int16x8_t,   s16)
DECLFUN(DT_U32, uint32x4_t,  u32)
DECLFUN(DT_S32, int32x4_t,   s32)
#  if defined(__aarch64__)
DECLFUN(DT_U64, uint64x2_t,  u64)
DECLFUN(DT_S64, int64x2_t,   s64)
#  endif // __aarch64__
DECLFUN(DT_F32, float32x4_t, f32)

#  if defined(AURA_ENABLE_NEON_FP16)
DECLFUN(float16_t, float16x8_t, f16)
#  endif
#undef DECLFUN

#define DECLFUN(stype, vtype, postfix)      \
    inline vtype vload2(const stype *addr)  \
    {                                       \
        return vld2_##postfix(addr);        \
    }

DECLFUN(DT_U8,  uint8x8x2_t,   u8)
DECLFUN(DT_S8,  int8x8x2_t,    s8)
DECLFUN(DT_U16, uint16x4x2_t,  u16)
DECLFUN(DT_S16, int16x4x2_t,   s16)
DECLFUN(DT_U32, uint32x2x2_t,  u32)
DECLFUN(DT_S32, int32x2x2_t,   s32)
DECLFUN(DT_U64, uint64x1x2_t,  u64)
DECLFUN(DT_S64, int64x1x2_t,   s64)
DECLFUN(DT_F32, float32x2x2_t, f32)

#  if defined(AURA_ENABLE_NEON_FP16)
DECLFUN(float16_t, float16x4x2_t, f16)
#  endif
#undef DECLFUN

#define DECLFUN(stype, vtype, postfix)       \
    inline vtype vload2q(const stype *addr)  \
    {                                        \
        return vld2q_##postfix(addr);        \
    }

DECLFUN(DT_U8,  uint8x16x2_t,  u8)
DECLFUN(DT_S8,  int8x16x2_t,   s8)
DECLFUN(DT_U16, uint16x8x2_t,  u16)
DECLFUN(DT_S16, int16x8x2_t,   s16)
DECLFUN(DT_U32, uint32x4x2_t,  u32)
DECLFUN(DT_S32, int32x4x2_t,   s32)
#  if defined(__aarch64__)
DECLFUN(DT_U64, uint64x2x2_t,  u64)
DECLFUN(DT_S64, int64x2x2_t,   s64)
#  endif // __aarch64__
DECLFUN(DT_F32, float32x4x2_t, f32)

#  if defined(AURA_ENABLE_NEON_FP16)
DECLFUN(float16_t, float16x8x2_t, f16)
#  endif
#  undef DECLFUN

#define DECLFUN(stype, vtype, postfix)      \
    inline vtype vload3(const stype *addr)  \
    {                                       \
        return vld3_##postfix(addr);        \
    }

DECLFUN(DT_U8,  uint8x8x3_t,   u8)
DECLFUN(DT_S8,  int8x8x3_t,    s8)
DECLFUN(DT_U16, uint16x4x3_t,  u16)
DECLFUN(DT_S16, int16x4x3_t,   s16)
DECLFUN(DT_U32, uint32x2x3_t,  u32)
DECLFUN(DT_S32, int32x2x3_t,   s32)
DECLFUN(DT_U64, uint64x1x3_t,  u64)
DECLFUN(DT_S64, int64x1x3_t,   s64)
DECLFUN(DT_F32, float32x2x3_t, f32)

#  if defined(AURA_ENABLE_NEON_FP16)
DECLFUN(float16_t, float16x4x3_t, f16)
#  endif
#undef DECLFUN

#define DECLFUN(stype, vtype, postfix)       \
    inline vtype vload3q(const stype *addr)  \
    {                                        \
        return vld3q_##postfix(addr);        \
    }

DECLFUN(DT_U8,  uint8x16x3_t,  u8)
DECLFUN(DT_S8,  int8x16x3_t,   s8)
DECLFUN(DT_U16, uint16x8x3_t,  u16)
DECLFUN(DT_S16, int16x8x3_t,   s16)
DECLFUN(DT_U32, uint32x4x3_t,  u32)
DECLFUN(DT_S32, int32x4x3_t,   s32)
#  if defined(__aarch64__)
DECLFUN(DT_U64, uint64x2x3_t,  u64)
DECLFUN(DT_S64, int64x2x3_t,   s64)
#  endif // __aarch64__
DECLFUN(DT_F32, float32x4x3_t, f32)

#  if defined(AURA_ENABLE_NEON_FP16)
DECLFUN(float16_t, float16x8x3_t, f16)
#  endif
#undef DECLFUN

#define DECLFUN(stype, vtype, postfix)      \
    inline vtype vload4(const stype *addr)  \
    {                                       \
        return vld4_##postfix(addr);        \
    }

DECLFUN(DT_U8,  uint8x8x4_t,   u8)
DECLFUN(DT_S8,  int8x8x4_t,    s8)
DECLFUN(DT_U16, uint16x4x4_t,  u16)
DECLFUN(DT_S16, int16x4x4_t,   s16)
DECLFUN(DT_U32, uint32x2x4_t,  u32)
DECLFUN(DT_S32, int32x2x4_t,   s32)
DECLFUN(DT_U64, uint64x1x4_t,  u64)
DECLFUN(DT_S64, int64x1x4_t,   s64)
DECLFUN(DT_F32, float32x2x4_t, f32)

#  if defined(AURA_ENABLE_NEON_FP16)
DECLFUN(float16_t, float16x4x4_t, f16)
#  endif
#undef DECLFUN

#define DECLFUN(stype, vtype, postfix)       \
    inline vtype vload4q(const stype *addr)  \
    {                                        \
        return vld4q_##postfix(addr);        \
    }

DECLFUN(DT_U8,  uint8x16x4_t,  u8)
DECLFUN(DT_S8,  int8x16x4_t,   s8)
DECLFUN(DT_U16, uint16x8x4_t,  u16)
DECLFUN(DT_S16, int16x8x4_t,   s16)
DECLFUN(DT_U32, uint32x4x4_t,  u32)
DECLFUN(DT_S32, int32x4x4_t,   s32)
#  if defined(__aarch64__)
DECLFUN(DT_U64, uint64x2x4_t,  u64)
DECLFUN(DT_S64, int64x2x4_t,   s64)
#  endif // __aarch64__
DECLFUN(DT_F32, float32x4x4_t, f32)

#  if defined(AURA_ENABLE_NEON_FP16)
DECLFUN(float16_t, float16x8x4_t, f16)
#  endif
#undef DECLFUN

#define DECLFUN(stype, vtype, prefix, postfix)         \
    inline DT_VOID vload(const stype *addr, vtype &vu) \
    {                                                  \
        vu = prefix##_##postfix(addr);                 \
    }

DECLFUN(DT_U8,  uint8x8_t,     vld1,  u8)
DECLFUN(DT_S8,  int8x8_t,      vld1,  s8)
DECLFUN(DT_U16, uint16x4_t,    vld1,  u16)
DECLFUN(DT_S16, int16x4_t,     vld1,  s16)
DECLFUN(DT_U32, uint32x2_t,    vld1,  u32)
DECLFUN(DT_S32, int32x2_t,     vld1,  s32)
DECLFUN(DT_U64, uint64x1_t,    vld1,  u64)
DECLFUN(DT_S64, int64x1_t,     vld1,  s64)
DECLFUN(DT_F32, float32x2_t,   vld1,  f32)
#  if defined(AURA_ENABLE_NEON_FP16)
DECLFUN(float16_t, float16x4_t, vld1, f16)
#  endif

DECLFUN(DT_U8,  uint8x16_t,    vld1q, u8)
DECLFUN(DT_S8,  int8x16_t,     vld1q, s8)
DECLFUN(DT_U16, uint16x8_t,    vld1q, u16)
DECLFUN(DT_S16, int16x8_t,     vld1q, s16)
DECLFUN(DT_U32, uint32x4_t,    vld1q, u32)
DECLFUN(DT_S32, int32x4_t,     vld1q, s32)
#  if defined(__aarch64__)
DECLFUN(DT_U64, uint64x2_t,    vld1q, u64)
DECLFUN(DT_S64, int64x2_t,     vld1q, s64)
#  endif // __aarch64__
DECLFUN(DT_F32, float32x4_t,   vld1q, f32)
#  if defined(AURA_ENABLE_NEON_FP16)
DECLFUN(float16_t, float16x8_t, vld1q, f16)
#  endif

DECLFUN(DT_U8,  uint8x8x2_t,   vld2,  u8)
DECLFUN(DT_S8,  int8x8x2_t,    vld2,  s8)
DECLFUN(DT_U16, uint16x4x2_t,  vld2,  u16)
DECLFUN(DT_S16, int16x4x2_t,   vld2,  s16)
DECLFUN(DT_U32, uint32x2x2_t,  vld2,  u32)
DECLFUN(DT_S32, int32x2x2_t,   vld2,  s32)
DECLFUN(DT_U64, uint64x1x2_t,  vld2,  u64)
DECLFUN(DT_S64, int64x1x2_t,   vld2,  s64)
DECLFUN(DT_F32, float32x2x2_t, vld2,  f32)
#  if defined(AURA_ENABLE_NEON_FP16)
DECLFUN(float16_t, float16x4x2_t, vld2, f16)
#  endif

DECLFUN(DT_U8,  uint8x16x2_t,  vld2q, u8)
DECLFUN(DT_S8,  int8x16x2_t,   vld2q, s8)
DECLFUN(DT_U16, uint16x8x2_t,  vld2q, u16)
DECLFUN(DT_S16, int16x8x2_t,   vld2q, s16)
DECLFUN(DT_U32, uint32x4x2_t,  vld2q, u32)
DECLFUN(DT_S32, int32x4x2_t,   vld2q, s32)
#  if defined(__aarch64__)
DECLFUN(DT_U64, uint64x2x2_t,  vld2q, u64)
DECLFUN(DT_S64, int64x2x2_t,   vld2q, s64)
#  endif // __aarch64__
DECLFUN(DT_F32, float32x4x2_t, vld2q, f32)
#  if defined(AURA_ENABLE_NEON_FP16)
DECLFUN(float16_t, float16x8x2_t, vld2q, f16)
#  endif

DECLFUN(DT_U8,  uint8x8x3_t,   vld3,  u8)
DECLFUN(DT_S8,  int8x8x3_t,    vld3,  s8)
DECLFUN(DT_U16, uint16x4x3_t,  vld3,  u16)
DECLFUN(DT_S16, int16x4x3_t,   vld3,  s16)
DECLFUN(DT_U32, uint32x2x3_t,  vld3,  u32)
DECLFUN(DT_S32, int32x2x3_t,   vld3,  s32)
DECLFUN(DT_U64, uint64x1x3_t,  vld3,  u64)
DECLFUN(DT_S64, int64x1x3_t,   vld3,  s64)
DECLFUN(DT_F32, float32x2x3_t, vld3,  f32)
#  if defined(AURA_ENABLE_NEON_FP16)
DECLFUN(float16_t, float16x4x3_t, vld3, f16)
#  endif

DECLFUN(DT_U8,  uint8x16x3_t,  vld3q, u8)
DECLFUN(DT_S8,  int8x16x3_t,   vld3q, s8)
DECLFUN(DT_U16, uint16x8x3_t,  vld3q, u16)
DECLFUN(DT_S16, int16x8x3_t,   vld3q, s16)
DECLFUN(DT_U32, uint32x4x3_t,  vld3q, u32)
DECLFUN(DT_S32, int32x4x3_t,   vld3q, s32)
#  if defined(__aarch64__)
DECLFUN(DT_U64, uint64x2x3_t,  vld3q, u64)
DECLFUN(DT_S64, int64x2x3_t,   vld3q, s64)
#  endif // __aarch64__
DECLFUN(DT_F32, float32x4x3_t, vld3q, f32)
#  if defined(AURA_ENABLE_NEON_FP16)
DECLFUN(float16_t, float16x8x3_t, vld3q, f16)
#  endif

DECLFUN(DT_U8,  uint8x8x4_t,   vld4,  u8)
DECLFUN(DT_S8,  int8x8x4_t,    vld4,  s8)
DECLFUN(DT_U16, uint16x4x4_t,  vld4,  u16)
DECLFUN(DT_S16, int16x4x4_t,   vld4,  s16)
DECLFUN(DT_U32, uint32x2x4_t,  vld4,  u32)
DECLFUN(DT_S32, int32x2x4_t,   vld4,  s32)
DECLFUN(DT_U64, uint64x1x4_t,  vld4,  u64)
DECLFUN(DT_S64, int64x1x4_t,   vld4,  s64)
DECLFUN(DT_F32, float32x2x4_t, vld4,  f32)
#  if defined(AURA_ENABLE_NEON_FP16)
DECLFUN(float16_t, float16x4x4_t, vld4, f16)
#  endif

DECLFUN(DT_U8,  uint8x16x4_t,  vld4q,  u8)
DECLFUN(DT_S8,  int8x16x4_t,   vld4q,  s8)
DECLFUN(DT_U16, uint16x8x4_t,  vld4q,  u16)
DECLFUN(DT_S16, int16x8x4_t,   vld4q,  s16)
DECLFUN(DT_U32, uint32x4x4_t,  vld4q,  u32)
DECLFUN(DT_S32, int32x4x4_t,   vld4q,  s32)
#  if defined(__aarch64__)
DECLFUN(DT_U64, uint64x2x4_t,  vld4q,  u64)
DECLFUN(DT_S64, int64x2x4_t,   vld4q,  s64)
#  endif // __aarch64__
DECLFUN(DT_F32, float32x4x4_t, vld4q,  f32)
#  if defined(AURA_ENABLE_NEON_FP16)
DECLFUN(float16_t, float16x8x4_t, vld4q, f16)
#  endif

#undef DECLFUN

#define DECLFUN(stype, vtype, prefix, postfix)         \
    inline DT_VOID vload(const stype *addr, vtype &vu) \
    {                                                  \
        vu.val[0] = prefix##_##postfix(addr);          \
    }

DECLFUN(DT_U8,  uint8x8x1_t,   vld1,  u8)
DECLFUN(DT_S8,  int8x8x1_t,    vld1,  s8)
DECLFUN(DT_U16, uint16x4x1_t,  vld1,  u16)
DECLFUN(DT_S16, int16x4x1_t,   vld1,  s16)
DECLFUN(DT_U32, uint32x2x1_t,  vld1,  u32)
DECLFUN(DT_S32, int32x2x1_t,   vld1,  s32)
DECLFUN(DT_U64, uint64x1x1_t,  vld1,  u64)
DECLFUN(DT_S64, int64x1x1_t,   vld1,  s64)
DECLFUN(DT_F32, float32x2x1_t, vld1,  f32)
#  if defined(AURA_ENABLE_NEON_FP16)
DECLFUN(float16_t, float16x4x1_t, vld1, f16)
#  endif
#  if defined(__aarch64__)
DECLFUN(DT_F64, float64x1x1_t, vld1,  f64)
#  endif // __aarch64__
DECLFUN(DT_U8,  uint8x16x1_t,  vld1q, u8)
DECLFUN(DT_S8,  int8x16x1_t,   vld1q, s8)
DECLFUN(DT_U16, uint16x8x1_t,  vld1q, u16)
DECLFUN(DT_S16, int16x8x1_t,   vld1q, s16)
DECLFUN(DT_U32, uint32x4x1_t,  vld1q, u32)
DECLFUN(DT_S32, int32x4x1_t,   vld1q, s32)
DECLFUN(DT_U64, uint64x2x1_t,  vld1q, u64)
DECLFUN(DT_S64, int64x2x1_t,   vld1q, s64)
DECLFUN(DT_F32, float32x4x1_t, vld1q, f32)
#  if defined(AURA_ENABLE_NEON_FP16)
DECLFUN(float16_t, float16x8x1_t, vld1q, f16)
#  endif
#  if defined(__aarch64__)
DECLFUN(DT_F64, float64x2x1_t, vld1q, f64)
#  endif // __aarch64__

#undef DECLFUN

} // namespace neon

} // namespace aura

#endif // AURA_RUNTIME_CORE_NEON_LOAD_HPP__
