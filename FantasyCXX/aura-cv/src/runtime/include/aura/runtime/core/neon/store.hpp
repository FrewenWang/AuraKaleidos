#ifndef AURA_RUNTIME_CORE_NEON_STORE_HPP__
#define AURA_RUNTIME_CORE_NEON_STORE_HPP__

#include <arm_neon.h>

namespace aura
{

namespace neon
{

#define DECLFUN(stype, vtype, prefix, postfix)          \
    inline DT_VOID vstore(stype *addr, const vtype &vu) \
    {                                                   \
        prefix##_##postfix(addr, vu);                   \
    }

DECLFUN(DT_U8, uint8x8_t,   vst1, u8)
DECLFUN(DT_U8, uint8x8x2_t, vst2, u8)
DECLFUN(DT_U8, uint8x8x3_t, vst3, u8)
DECLFUN(DT_U8, uint8x8x4_t, vst4, u8)

DECLFUN(DT_S8, int8x8_t,   vst1, s8)
DECLFUN(DT_S8, int8x8x2_t, vst2, s8)
DECLFUN(DT_S8, int8x8x3_t, vst3, s8)
DECLFUN(DT_S8, int8x8x4_t, vst4, s8)

DECLFUN(DT_U16, uint16x4_t,   vst1, u16)
DECLFUN(DT_U16, uint16x4x2_t, vst2, u16)
DECLFUN(DT_U16, uint16x4x3_t, vst3, u16)
DECLFUN(DT_U16, uint16x4x4_t, vst4, u16)

DECLFUN(DT_S16, int16x4_t,   vst1, s16)
DECLFUN(DT_S16, int16x4x2_t, vst2, s16)
DECLFUN(DT_S16, int16x4x3_t, vst3, s16)
DECLFUN(DT_S16, int16x4x4_t, vst4, s16)

DECLFUN(DT_U32, uint32x2_t,   vst1, u32)
DECLFUN(DT_U32, uint32x2x2_t, vst2, u32)
DECLFUN(DT_U32, uint32x2x3_t, vst3, u32)
DECLFUN(DT_U32, uint32x2x4_t, vst4, u32)

DECLFUN(DT_S32, int32x2_t,   vst1, s32)
DECLFUN(DT_S32, int32x2x2_t, vst2, s32)
DECLFUN(DT_S32, int32x2x3_t, vst3, s32)
DECLFUN(DT_S32, int32x2x4_t, vst4, s32)

DECLFUN(DT_U64, uint64x1_t,   vst1, u64)
DECLFUN(DT_U64, uint64x1x2_t, vst2, u64)
DECLFUN(DT_U64, uint64x1x3_t, vst3, u64)
DECLFUN(DT_U64, uint64x1x4_t, vst4, u64)

DECLFUN(DT_S64, int64x1_t,   vst1, s64)
DECLFUN(DT_S64, int64x1x2_t, vst2, s64)
DECLFUN(DT_S64, int64x1x3_t, vst3, s64)
DECLFUN(DT_S64, int64x1x4_t, vst4, s64)

DECLFUN(DT_F32, float32x2_t,   vst1, f32)
DECLFUN(DT_F32, float32x2x2_t, vst2, f32)
DECLFUN(DT_F32, float32x2x3_t, vst3, f32)
DECLFUN(DT_F32, float32x2x4_t, vst4, f32)

#  if defined(AURA_ENABLE_NEON_FP16)
DECLFUN(float16_t, float16x4_t,   vst1, f16)
DECLFUN(float16_t, float16x4x2_t, vst2, f16)
DECLFUN(float16_t, float16x4x3_t, vst3, f16)
DECLFUN(float16_t, float16x4x4_t, vst4, f16)
#  endif

DECLFUN(DT_U8, uint8x16_t,   vst1q, u8)
DECLFUN(DT_U8, uint8x16x2_t, vst2q, u8)
DECLFUN(DT_U8, uint8x16x3_t, vst3q, u8)
DECLFUN(DT_U8, uint8x16x4_t, vst4q, u8)

DECLFUN(DT_S8, int8x16_t,    vst1q, s8)
DECLFUN(DT_S8, int8x16x2_t,  vst2q, s8)
DECLFUN(DT_S8, int8x16x3_t,  vst3q, s8)
DECLFUN(DT_S8, int8x16x4_t,  vst4q, s8)

DECLFUN(DT_U16, uint16x8_t,   vst1q, u16)
DECLFUN(DT_U16, uint16x8x2_t, vst2q, u16)
DECLFUN(DT_U16, uint16x8x3_t, vst3q, u16)
DECLFUN(DT_U16, uint16x8x4_t, vst4q, u16)

DECLFUN(DT_S16, int16x8_t,   vst1q, s16)
DECLFUN(DT_S16, int16x8x2_t, vst2q, s16)
DECLFUN(DT_S16, int16x8x3_t, vst3q, s16)
DECLFUN(DT_S16, int16x8x4_t, vst4q, s16)

DECLFUN(DT_U32, uint32x4_t,   vst1q, u32)
DECLFUN(DT_U32, uint32x4x2_t, vst2q, u32)
DECLFUN(DT_U32, uint32x4x3_t, vst3q, u32)
DECLFUN(DT_U32, uint32x4x4_t, vst4q, u32)

DECLFUN(DT_S32, int32x4_t,   vst1q, s32)
DECLFUN(DT_S32, int32x4x2_t, vst2q, s32)
DECLFUN(DT_S32, int32x4x3_t, vst3q, s32)
DECLFUN(DT_S32, int32x4x4_t, vst4q, s32)

DECLFUN(DT_U64, uint64x2_t,  vst1q, u64)
DECLFUN(DT_S64, int64x2_t,   vst1q, s64)

#  if defined(__aarch64__)
DECLFUN(DT_U64, uint64x2x2_t, vst2q, u64)
DECLFUN(DT_U64, uint64x2x3_t, vst3q, u64)
DECLFUN(DT_U64, uint64x2x4_t, vst4q, u64)

DECLFUN(DT_S64, int64x2x2_t, vst2q, s64)
DECLFUN(DT_S64, int64x2x3_t, vst3q, s64)
DECLFUN(DT_S64, int64x2x4_t, vst4q, s64)
#  endif // __aarch64__
DECLFUN(DT_F32, float32x4_t,   vst1q, f32)
DECLFUN(DT_F32, float32x4x2_t, vst2q, f32)
DECLFUN(DT_F32, float32x4x3_t, vst3q, f32)
DECLFUN(DT_F32, float32x4x4_t, vst4q, f32)

#  if defined(AURA_ENABLE_NEON_FP16)
DECLFUN(float16_t, float16x8_t,   vst1q, f16)
DECLFUN(float16_t, float16x8x2_t, vst2q, f16)
DECLFUN(float16_t, float16x8x3_t, vst3q, f16)
DECLFUN(float16_t, float16x8x4_t, vst4q, f16)
#  endif

#undef DECLFUN

#define DECLFUN(stype, vtype, prefix, postfix)          \
    inline DT_VOID vstore(stype *addr, const vtype &vu) \
    {                                                   \
        prefix##_##postfix(addr, vu.val[0]);            \
    }

DECLFUN(DT_U8,  uint8x8x1_t,   vst1,  u8)
DECLFUN(DT_S8,  int8x8x1_t,    vst1,  s8)
DECLFUN(DT_U16, uint16x4x1_t,  vst1,  u16)
DECLFUN(DT_S16, int16x4x1_t,   vst1,  s16)
DECLFUN(DT_U32, uint32x2x1_t,  vst1,  u32)
DECLFUN(DT_S32, int32x2x1_t,   vst1,  s32)
DECLFUN(DT_U64, uint64x1x1_t,  vst1,  u64)
DECLFUN(DT_S64, int64x1x1_t,   vst1,  s64)
DECLFUN(DT_F32, float32x2x1_t, vst1,  f32)
#  if defined(AURA_ENABLE_NEON_FP16)
DECLFUN(float16_t, float16x4x1_t, vst1,  f16)
#  endif
#  if defined(__aarch64__)
DECLFUN(DT_F64, float64x1x1_t, vst1,  f64)
#  endif // __aarch64__
DECLFUN(DT_U8,  uint8x16x1_t,  vst1q, u8)
DECLFUN(DT_S8,  int8x16x1_t,   vst1q, s8)
DECLFUN(DT_U16, uint16x8x1_t,  vst1q, u16)
DECLFUN(DT_S16, int16x8x1_t,   vst1q, s16)
DECLFUN(DT_U32, uint32x4x1_t,  vst1q, u32)
DECLFUN(DT_S32, int32x4x1_t,   vst1q, s32)
DECLFUN(DT_U64, uint64x2x1_t,  vst1q, u64)
DECLFUN(DT_S64, int64x2x1_t,   vst1q, s64)
DECLFUN(DT_F32, float32x4x1_t, vst1q, f32)
#  if defined(AURA_ENABLE_NEON_FP16)
DECLFUN(float16_t, float16x8x1_t, vst1q, f16)
#  endif
#  if defined(__aarch64__)
DECLFUN(DT_F64, float64x2x1_t, vst1q, f64)
#  endif // __aarch64__

#undef DECLFUN

} // namespace neon

} // namespace aura

#endif // AURA_RUNTIME_CORE_NEON_STORE_HPP__
