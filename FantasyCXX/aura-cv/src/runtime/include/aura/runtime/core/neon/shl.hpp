#ifndef AURA_RUNTIME_CORE_NEON_SHL_HPP__
#define AURA_RUNTIME_CORE_NEON_SHL_HPP__

#include <arm_neon.h>

namespace aura
{

namespace neon
{

#define DECLFUN(stype, vtype, prefix, postfix)        \
    inline vtype vshl(const vtype &v, const stype &u) \
    {                                                 \
        return prefix##_##postfix(v, u);              \
    }

DECLFUN(int8x8_t,    uint8x8_t,   vshl,  u8)
DECLFUN(int8x16_t,   uint8x16_t,  vshlq, u8)
DECLFUN(int8x8_t,    int8x8_t,    vshl,  s8)
DECLFUN(int8x16_t,   int8x16_t,   vshlq, s8)
DECLFUN(int16x4_t,   uint16x4_t,  vshl,  u16)
DECLFUN(int16x8_t,   uint16x8_t,  vshlq, u16)
DECLFUN(int16x4_t,   int16x4_t,   vshl,  s16)
DECLFUN(int16x8_t,   int16x8_t,   vshlq, s16)
DECLFUN(int32x2_t,   uint32x2_t,  vshl,  u32)
DECLFUN(int32x4_t,   uint32x4_t,  vshlq, u32)
DECLFUN(int32x2_t,   int32x2_t,   vshl,  s32)
DECLFUN(int32x4_t,   int32x4_t,   vshlq, s32)
DECLFUN(int64x1_t,   uint64x1_t,  vshl,  u64)
DECLFUN(int64x2_t,   uint64x2_t,  vshlq, u64)
DECLFUN(int64x1_t,   int64x1_t,   vshl,  s64)
DECLFUN(int64x2_t,   int64x2_t,   vshlq, s64)
#undef DECLFUN

#define DECLFUN(stype, vtype, prefix, postfix)         \
    inline vtype vqshl(const vtype &v, const stype &u) \
    {                                                  \
        return prefix##_##postfix(v, u);               \
    }

DECLFUN(int8x8_t,    uint8x8_t,   vqshl,  u8)
DECLFUN(int8x16_t,   uint8x16_t,  vqshlq, u8)
DECLFUN(int8x8_t,    int8x8_t,    vqshl,  s8)
DECLFUN(int8x16_t,   int8x16_t,   vqshlq, s8)
DECLFUN(int16x4_t,   uint16x4_t,  vqshl,  u16)
DECLFUN(int16x8_t,   uint16x8_t,  vqshlq, u16)
DECLFUN(int16x4_t,   int16x4_t,   vqshl,  s16)
DECLFUN(int16x8_t,   int16x8_t,   vqshlq, s16)
DECLFUN(int32x2_t,   uint32x2_t,  vqshl,  u32)
DECLFUN(int32x4_t,   uint32x4_t,  vqshlq, u32)
DECLFUN(int32x2_t,   int32x2_t,   vqshl,  s32)
DECLFUN(int32x4_t,   int32x4_t,   vqshlq, s32)
DECLFUN(int64x1_t,   uint64x1_t,  vqshl,  u64)
DECLFUN(int64x2_t,   uint64x2_t,  vqshlq, u64)
DECLFUN(int64x1_t,   int64x1_t,   vqshl,  s64)
DECLFUN(int64x2_t,   int64x2_t,   vqshlq, s64)
#undef DECLFUN

#define DECLFUN(stype, vtype, prefix, postfix)         \
    inline vtype vrshl(const vtype &v, const stype &u) \
    {                                                  \
        return prefix##_##postfix(v, u);               \
    }

DECLFUN(int8x8_t,    uint8x8_t,   vrshl,  u8)
DECLFUN(int8x16_t,   uint8x16_t,  vrshlq, u8)
DECLFUN(int8x8_t,    int8x8_t,    vrshl,  s8)
DECLFUN(int8x16_t,   int8x16_t,   vrshlq, s8)
DECLFUN(int16x4_t,   uint16x4_t,  vrshl,  u16)
DECLFUN(int16x8_t,   uint16x8_t,  vrshlq, u16)
DECLFUN(int16x4_t,   int16x4_t,   vrshl,  s16)
DECLFUN(int16x8_t,   int16x8_t,   vrshlq, s16)
DECLFUN(int32x2_t,   uint32x2_t,  vrshl,  u32)
DECLFUN(int32x4_t,   uint32x4_t,  vrshlq, u32)
DECLFUN(int32x2_t,   int32x2_t,   vrshl,  s32)
DECLFUN(int32x4_t,   int32x4_t,   vrshlq, s32)
DECLFUN(int64x1_t,   uint64x1_t,  vrshl,  u64)
DECLFUN(int64x2_t,   uint64x2_t,  vrshlq, u64)
DECLFUN(int64x1_t,   int64x1_t,   vrshl,  s64)
DECLFUN(int64x2_t,   int64x2_t,   vrshlq, s64)
#undef DECLFUN

#define DECLFUN(stype, vtype, prefix, postfix)          \
    inline vtype vqrshl(const vtype &v, const stype &u) \
    {                                                   \
        return prefix##_##postfix(v, u);                \
    }

DECLFUN(int8x8_t,    uint8x8_t,   vqrshl,  u8)
DECLFUN(int8x16_t,   uint8x16_t,  vqrshlq, u8)
DECLFUN(int8x8_t,    int8x8_t,    vqrshl,  s8)
DECLFUN(int8x16_t,   int8x16_t,   vqrshlq, s8)
DECLFUN(int16x4_t,   uint16x4_t,  vqrshl,  u16)
DECLFUN(int16x8_t,   uint16x8_t,  vqrshlq, u16)
DECLFUN(int16x4_t,   int16x4_t,   vqrshl,  s16)
DECLFUN(int16x8_t,   int16x8_t,   vqrshlq, s16)
DECLFUN(int32x2_t,   uint32x2_t,  vqrshl,  u32)
DECLFUN(int32x4_t,   uint32x4_t,  vqrshlq, u32)
DECLFUN(int32x2_t,   int32x2_t,   vqrshl,  s32)
DECLFUN(int32x4_t,   int32x4_t,   vqrshlq, s32)
DECLFUN(int64x1_t,   uint64x1_t,  vqrshl,  u64)
DECLFUN(int64x2_t,   uint64x2_t,  vqrshlq, u64)
DECLFUN(int64x1_t,   int64x1_t,   vqrshl,  s64)
DECLFUN(int64x2_t,   int64x2_t,   vqrshlq, s64)
#undef DECLFUN

} // namespace neon

} // namespace aura

#endif // AURA_RUNTIME_CORE_NEON_SHL_HPP__