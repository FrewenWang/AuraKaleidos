#ifndef AURA_RUNTIME_CORE_NEON_REINTERPRET_HPP__
#define AURA_RUNTIME_CORE_NEON_REINTERPRET_HPP__

#include <arm_neon.h>

namespace aura
{

namespace neon
{

#define DECLFUN(ptype, vtype, prefix, postfix1, postfix2) \
    inline ptype vreinterpret(const vtype &a)             \
    {                                                     \
        return prefix##_##postfix1##_##postfix2(a);       \
    }                                                     \

DECLFUN(int8x8_t,   uint8x8_t,  vreinterpret,  s8, u8)
DECLFUN(uint8x8_t,  int8x8_t,   vreinterpret,  u8, s8)
DECLFUN(int8x16_t,  uint8x16_t, vreinterpretq, s8, u8)
DECLFUN(uint8x16_t, int8x16_t,  vreinterpretq, u8, s8)
DECLFUN(int16x4_t,  uint16x4_t, vreinterpret,  s16, u16)
DECLFUN(uint16x4_t, int16x4_t,  vreinterpret,  u16, s16)
DECLFUN(int16x8_t,  uint16x8_t, vreinterpretq, s16, u16)
DECLFUN(uint16x8_t, int16x8_t,  vreinterpretq, u16, s16)
DECLFUN(int32x2_t,  uint32x2_t, vreinterpret,  s32, u32)
DECLFUN(uint32x2_t, int32x2_t,  vreinterpret,  u32, s32)
DECLFUN(int32x4_t,  uint32x4_t, vreinterpretq, s32, u32)
DECLFUN(uint32x4_t, int32x4_t,  vreinterpretq, u32, s32)
#undef DECLFUN

inline float32x4_t vreinterpret(const float32x4_t &v)
{
    return v;
}

inline float32x2_t vreinterpret(const float32x2_t &v)
{
    return v;
}

#if defined(AURA_ENABLE_NEON_FP16)

inline float16x8_t vreinterpret(const float16x8_t &v)
{
    return v;
}

inline float16x4_t vreinterpret(const float16x4_t &v)
{
    return v;
}

#endif

#define DECLFUN(ptype, vtype, prefix, postfix1, postfix2)                                                         \
    template <typename T>                                                                                         \
    inline typename std::enable_if<std::is_same<T, MI_U64>::value || std::is_same<T, MI_S64>::value, ptype>::type \
    vreinterpret_64(const vtype &a)                                                                               \
    {                                                                                                             \
        return prefix##_##postfix1##_##postfix2(a);                                                               \
    }                                                                                                             \

DECLFUN(int64x1_t,  int8x8_t,   vreinterpret,  s64, s8)
DECLFUN(uint64x1_t, uint8x8_t,  vreinterpret,  u64, u8)
DECLFUN(int64x2_t,  int8x16_t,  vreinterpretq, s64, s8)
DECLFUN(uint64x2_t, uint8x16_t, vreinterpretq, u64, u8)
DECLFUN(int64x1_t,  int16x4_t,  vreinterpret,  s64, s16)
DECLFUN(uint64x1_t, uint16x4_t, vreinterpret,  u64, u16)
DECLFUN(int64x2_t,  int16x8_t,  vreinterpretq, s64, s16)
DECLFUN(uint64x2_t, uint16x8_t, vreinterpretq, u64, u16)
DECLFUN(int64x1_t,  int32x2_t,  vreinterpret,  s64, s32)
DECLFUN(uint64x1_t, uint32x2_t, vreinterpret,  u64, u32)
DECLFUN(int64x2_t,  int32x4_t,  vreinterpretq, s64, s32)
DECLFUN(uint64x2_t, uint32x4_t, vreinterpretq, u64, u32)
#undef DECLFUN

#define DECLFUN(ptype, vtype, prefix, postfix1, postfix2)                                                       \
    template <typename T>                                                                                       \
    inline typename std::enable_if<std::is_same<T, MI_U8>::value || std::is_same<T, MI_S8>::value, ptype>::type \
    vreinterpret_64(const vtype &a)                                                                             \
    {                                                                                                           \
        return prefix##_##postfix1##_##postfix2(a);                                                             \
    }                                                                                                           \

DECLFUN(int8x8_t,   int64x1_t,  vreinterpret,  s8,  s64)
DECLFUN(uint8x8_t,  uint64x1_t, vreinterpret,  u8,  u64)
DECLFUN(int8x16_t,  int64x2_t,  vreinterpretq, s8,  s64)
DECLFUN(uint8x16_t, uint64x2_t, vreinterpretq, u8,  u64)
#undef DECLFUN

#define DECLFUN(ptype, vtype, prefix, postfix1, postfix2)                                                         \
    template <typename T>                                                                                         \
    inline typename std::enable_if<std::is_same<T, MI_U16>::value || std::is_same<T, MI_S16>::value, ptype>::type \
    vreinterpret_64(const vtype &a)                                                                               \
    {                                                                                                             \
        return prefix##_##postfix1##_##postfix2(a);                                                               \
    }                                                                                                             \

DECLFUN(int16x4_t,  int64x1_t,  vreinterpret,  s16, s64)
DECLFUN(uint16x4_t, uint64x1_t, vreinterpret,  u16, u64)
DECLFUN(int16x8_t,  int64x2_t,  vreinterpretq, s16, s64)
DECLFUN(uint16x8_t, uint64x2_t, vreinterpretq, u16, u64)
#undef DECLFUN

#define DECLFUN(ptype, vtype, prefix, postfix1, postfix2)                                                         \
    template <typename T>                                                                                         \
    inline typename std::enable_if<std::is_same<T, MI_U32>::value || std::is_same<T, MI_S32>::value, ptype>::type \
    vreinterpret_64(const vtype &a)                                                                               \
    {                                                                                                             \
        return prefix##_##postfix1##_##postfix2(a);                                                               \
    }                                                                                                             \

DECLFUN(int32x2_t,  int64x1_t,  vreinterpret,  s32, s64)
DECLFUN(uint32x2_t, uint64x1_t, vreinterpret,  u32, u64)
DECLFUN(int32x4_t,  int64x2_t,  vreinterpretq, s32, s64)
DECLFUN(uint32x4_t, uint64x2_t, vreinterpretq, u32, u64)
#undef DECLFUN

#define DECLFUN(ptype, vtype, prefix, postfix1, postfix2) \
    inline ptype vreinterpret_##postfix1(const vtype &a)  \
    {                                                     \
        return prefix##_##postfix1##_##postfix2(a);       \
    }

DECLFUN(uint8x8_t, int8x8_t,    vreinterpret, u8, s8)
DECLFUN(uint8x8_t, uint16x4_t,  vreinterpret, u8, u16)
DECLFUN(uint8x8_t, int16x4_t,   vreinterpret, u8, s16)
DECLFUN(uint8x8_t, uint32x2_t,  vreinterpret, u8, u32)
DECLFUN(uint8x8_t, int32x2_t,   vreinterpret, u8, s32)
DECLFUN(uint8x8_t, float32x2_t, vreinterpret, u8, f32)
DECLFUN(uint8x8_t, uint64x1_t,  vreinterpret, u8, u64)
DECLFUN(uint8x8_t, int64x1_t,   vreinterpret, u8, s64)

DECLFUN(int8x8_t, uint8x8_t,   vreinterpret, s8, u8)
DECLFUN(int8x8_t, uint16x4_t,  vreinterpret, s8, u16)
DECLFUN(int8x8_t, int16x4_t,   vreinterpret, s8, s16)
DECLFUN(int8x8_t, uint32x2_t,  vreinterpret, s8, u32)
DECLFUN(int8x8_t, int32x2_t,   vreinterpret, s8, s32)
DECLFUN(int8x8_t, float32x2_t, vreinterpret, s8, f32)
DECLFUN(int8x8_t, uint64x1_t,  vreinterpret, s8, u64)
DECLFUN(int8x8_t, int64x1_t,   vreinterpret, s8, s64)

DECLFUN(uint16x4_t, uint8x8_t,   vreinterpret, u16, u8)
DECLFUN(uint16x4_t, int8x8_t,    vreinterpret, u16, s8)
DECLFUN(uint16x4_t, int16x4_t,   vreinterpret, u16, s16)
DECLFUN(uint16x4_t, uint32x2_t,  vreinterpret, u16, u32)
DECLFUN(uint16x4_t, int32x2_t,   vreinterpret, u16, s32)
DECLFUN(uint16x4_t, float32x2_t, vreinterpret, u16, f32)
DECLFUN(uint16x4_t, uint64x1_t,  vreinterpret, u16, u64)
DECLFUN(uint16x4_t, int64x1_t,   vreinterpret, u16, s64)

DECLFUN(int16x4_t, uint8x8_t,   vreinterpret, s16, u8)
DECLFUN(int16x4_t, int8x8_t,    vreinterpret, s16, s8)
DECLFUN(int16x4_t, uint16x4_t,  vreinterpret, s16, u16)
DECLFUN(int16x4_t, uint32x2_t,  vreinterpret, s16, u32)
DECLFUN(int16x4_t, int32x2_t,   vreinterpret, s16, s32)
DECLFUN(int16x4_t, float32x2_t, vreinterpret, s16, f32)
DECLFUN(int16x4_t, uint64x1_t,  vreinterpret, s16, u64)
DECLFUN(int16x4_t, int64x1_t,   vreinterpret, s16, s64)

DECLFUN(uint32x2_t, uint8x8_t,   vreinterpret, u32, u8)
DECLFUN(uint32x2_t, int8x8_t,    vreinterpret, u32, s8)
DECLFUN(uint32x2_t, uint16x4_t,  vreinterpret, u32, u16)
DECLFUN(uint32x2_t, int16x4_t,   vreinterpret, u32, s16)
DECLFUN(uint32x2_t, int32x2_t,   vreinterpret, u32, s32)
DECLFUN(uint32x2_t, float32x2_t, vreinterpret, u32, f32)
DECLFUN(uint32x2_t, uint64x1_t,  vreinterpret, u32, u64)
DECLFUN(uint32x2_t, int64x1_t,   vreinterpret, u32, s64)

DECLFUN(int32x2_t, uint8x8_t,   vreinterpret, s32, u8)
DECLFUN(int32x2_t, int8x8_t,    vreinterpret, s32, s8)
DECLFUN(int32x2_t, uint16x4_t,  vreinterpret, s32, u16)
DECLFUN(int32x2_t, int16x4_t,   vreinterpret, s32, s16)
DECLFUN(int32x2_t, uint32x2_t,  vreinterpret, s32, u32)
DECLFUN(int32x2_t, float32x2_t, vreinterpret, s32, f32)
DECLFUN(int32x2_t, uint64x1_t,  vreinterpret, s32, u64)
DECLFUN(int32x2_t, int64x1_t,   vreinterpret, s32, s64)

DECLFUN(float32x2_t, uint8x8_t,   vreinterpret, f32, u8)
DECLFUN(float32x2_t, int8x8_t,    vreinterpret, f32, s8)
DECLFUN(float32x2_t, uint16x4_t,  vreinterpret, f32, u16)
DECLFUN(float32x2_t, int16x4_t,   vreinterpret, f32, s16)
DECLFUN(float32x2_t, uint32x2_t,  vreinterpret, f32, u32)
DECLFUN(float32x2_t, int32x2_t,   vreinterpret, f32, s32)
DECLFUN(float32x2_t, uint64x1_t,  vreinterpret, f32, u64)
DECLFUN(float32x2_t, int64x1_t,   vreinterpret, f32, s64)

DECLFUN(uint64x1_t, uint8x8_t,  vreinterpret,  u64, u8)
DECLFUN(uint64x1_t, int8x8_t,   vreinterpret,  u64, s8)
DECLFUN(uint64x1_t, uint16x4_t, vreinterpret,  u64, u16)
DECLFUN(uint64x1_t, int16x4_t,  vreinterpret,  u64, s16)
DECLFUN(uint64x1_t, uint32x2_t, vreinterpret,  u64, u32)
DECLFUN(uint64x1_t, int32x2_t,  vreinterpret,  u64, s32)

#if defined(AURA_ENABLE_NEON_FP16)

DECLFUN(float16x4_t, uint8x8_t,   vreinterpret, f16, u8)
DECLFUN(float16x4_t, int8x8_t,    vreinterpret, f16, s8)
DECLFUN(float16x4_t, uint16x4_t,  vreinterpret, f16, u16)
DECLFUN(float16x4_t, int16x4_t,   vreinterpret, f16, s16)
DECLFUN(float16x4_t, uint32x2_t,  vreinterpret, f16, u32)
DECLFUN(float16x4_t, int32x2_t,   vreinterpret, f16, s32)
DECLFUN(float16x4_t, float32x2_t, vreinterpret, f16, f32)

DECLFUN(uint8x8_t,   float16x4_t, vreinterpret, u8,  f16)
DECLFUN(int8x8_t,    float16x4_t, vreinterpret, s8,  f16)
DECLFUN(uint16x4_t,  float16x4_t, vreinterpret, u16, f16)
DECLFUN(int16x4_t,   float16x4_t, vreinterpret, s16, f16)
DECLFUN(uint32x2_t,  float16x4_t, vreinterpret, u32, f16)
DECLFUN(int32x2_t,   float16x4_t, vreinterpret, s32, f16)
DECLFUN(float32x2_t, float16x4_t, vreinterpret, f32, f16)

#endif

DECLFUN(uint8x16_t, int8x16_t,   vreinterpretq, u8, s8)
DECLFUN(uint8x16_t, uint16x8_t,  vreinterpretq, u8, u16)
DECLFUN(uint8x16_t, int16x8_t,   vreinterpretq, u8, s16)
DECLFUN(uint8x16_t, uint32x4_t,  vreinterpretq, u8, u32)
DECLFUN(uint8x16_t, int32x4_t,   vreinterpretq, u8, s32)
DECLFUN(uint8x16_t, float32x4_t, vreinterpretq, u8, f32)
DECLFUN(uint8x16_t, uint64x2_t,  vreinterpretq, u8, u64)
DECLFUN(uint8x16_t, int64x2_t,   vreinterpretq, u8, s64)

DECLFUN(int8x16_t, uint8x16_t,  vreinterpretq, s8, u8)
DECLFUN(int8x16_t, uint16x8_t,  vreinterpretq, s8, u16)
DECLFUN(int8x16_t, int16x8_t,   vreinterpretq, s8, s16)
DECLFUN(int8x16_t, uint32x4_t,  vreinterpretq, s8, u32)
DECLFUN(int8x16_t, int32x4_t,   vreinterpretq, s8, s32)
DECLFUN(int8x16_t, float32x4_t, vreinterpretq, s8, f32)
DECLFUN(int8x16_t, uint64x2_t,  vreinterpretq, s8, u64)
DECLFUN(int8x16_t, int64x2_t,   vreinterpretq, s8, s64)

DECLFUN(uint16x8_t, uint8x16_t,  vreinterpretq, u16, u8)
DECLFUN(uint16x8_t, int8x16_t,   vreinterpretq, u16, s8)
DECLFUN(uint16x8_t, int16x8_t,   vreinterpretq, u16, s16)
DECLFUN(uint16x8_t, uint32x4_t,  vreinterpretq, u16, u32)
DECLFUN(uint16x8_t, int32x4_t,   vreinterpretq, u16, s32)
DECLFUN(uint16x8_t, float32x4_t, vreinterpretq, u16, f32)
DECLFUN(uint16x8_t, uint64x2_t,  vreinterpretq, u16, u64)
DECLFUN(uint16x8_t, int64x2_t,   vreinterpretq, u16, s64)

DECLFUN(int16x8_t, uint8x16_t,  vreinterpretq, s16, u8)
DECLFUN(int16x8_t, int8x16_t,   vreinterpretq, s16, s8)
DECLFUN(int16x8_t, uint16x8_t,  vreinterpretq, s16, u16)
DECLFUN(int16x8_t, uint32x4_t,  vreinterpretq, s16, u32)
DECLFUN(int16x8_t, int32x4_t,   vreinterpretq, s16, s32)
DECLFUN(int16x8_t, float32x4_t, vreinterpretq, s16, f32)
DECLFUN(int16x8_t, uint64x2_t,  vreinterpretq, s16, u64)
DECLFUN(int16x8_t, int64x2_t,   vreinterpretq, s16, s64)

DECLFUN(uint32x4_t, uint8x16_t,  vreinterpretq, u32, u8)
DECLFUN(uint32x4_t, int8x16_t,   vreinterpretq, u32, s8)
DECLFUN(uint32x4_t, uint16x8_t,  vreinterpretq, u32, u16)
DECLFUN(uint32x4_t, int16x8_t,   vreinterpretq, u32, s16)
DECLFUN(uint32x4_t, int32x4_t,   vreinterpretq, u32, s32)
DECLFUN(uint32x4_t, float32x4_t, vreinterpretq, u32, f32)
DECLFUN(uint32x4_t, uint64x2_t,  vreinterpretq, u32, u64)
DECLFUN(uint32x4_t, int64x2_t,   vreinterpretq, u32, s64)

DECLFUN(int32x4_t, uint8x16_t,  vreinterpretq, s32, u8)
DECLFUN(int32x4_t, int8x16_t,   vreinterpretq, s32, s8)
DECLFUN(int32x4_t, uint16x8_t,  vreinterpretq, s32, u16)
DECLFUN(int32x4_t, int16x8_t,   vreinterpretq, s32, s16)
DECLFUN(int32x4_t, uint32x4_t,  vreinterpretq, s32, u32)
DECLFUN(int32x4_t, float32x4_t, vreinterpretq, s32, f32)
DECLFUN(int32x4_t, uint64x2_t,  vreinterpretq, s32, u64)
DECLFUN(int32x4_t, int64x2_t,   vreinterpretq, s32, s64)

DECLFUN(float32x4_t, uint8x16_t,  vreinterpretq, f32, u8)
DECLFUN(float32x4_t, int8x16_t,   vreinterpretq, f32, s8)
DECLFUN(float32x4_t, uint16x8_t,  vreinterpretq, f32, u16)
DECLFUN(float32x4_t, int16x8_t,   vreinterpretq, f32, s16)
DECLFUN(float32x4_t, uint32x4_t,  vreinterpretq, f32, u32)
DECLFUN(float32x4_t, int32x4_t,   vreinterpretq, f32, s32)
DECLFUN(float32x4_t, uint64x2_t,  vreinterpretq, f32, u64)
DECLFUN(float32x4_t, int64x2_t,   vreinterpretq, f32, s64)

DECLFUN(uint64x2_t, uint8x16_t, vreinterpretq, u64, u8)
DECLFUN(uint64x2_t, int8x16_t,  vreinterpretq, u64, s8)
DECLFUN(uint64x2_t, uint16x8_t, vreinterpretq, u64, u16)
DECLFUN(uint64x2_t, int16x8_t,  vreinterpretq, u64, s16)
DECLFUN(uint64x2_t, uint32x4_t, vreinterpretq, u64, u32)
DECLFUN(uint64x2_t, int32x4_t,  vreinterpretq, u64, s32)

#if defined(AURA_ENABLE_NEON_FP16)

DECLFUN(float16x8_t, uint8x16_t,  vreinterpretq, f16, u8)
DECLFUN(float16x8_t, int8x16_t,   vreinterpretq, f16, s8)
DECLFUN(float16x8_t, uint16x8_t,  vreinterpretq, f16, u16)
DECLFUN(float16x8_t, int16x8_t,   vreinterpretq, f16, s16)
DECLFUN(float16x8_t, uint32x4_t,  vreinterpretq, f16, u32)
DECLFUN(float16x8_t, int32x4_t,   vreinterpretq, f16, s32)
DECLFUN(float16x8_t, float32x4_t, vreinterpretq, f16, f32)

DECLFUN(uint8x16_t,  float16x8_t, vreinterpretq, u8,  f16)
DECLFUN(int8x16_t,   float16x8_t, vreinterpretq, s8,  f16)
DECLFUN(uint16x8_t,  float16x8_t, vreinterpretq, u16, f16)
DECLFUN(int16x8_t,   float16x8_t, vreinterpretq, s16, f16)
DECLFUN(uint32x4_t,  float16x8_t, vreinterpretq, u32, f16)
DECLFUN(int32x4_t,   float16x8_t, vreinterpretq, s32, f16)
DECLFUN(float32x4_t, float16x8_t, vreinterpretq, f32, f16)

#endif

#undef DECLFUN

#define DECLFUN(ptype, vtype, prefix, postfix1, postfix2) \
    inline ptype vreinterpret_##postfix1(const vtype &a)  \
    {                                                     \
        return (a);       \
    }

DECLFUN(uint8x8_t,   uint8x8_t,   vreinterpret, u8,  u8)
DECLFUN(int8x8_t,    int8x8_t,    vreinterpret, s8,  s8)
DECLFUN(uint16x4_t,  uint16x4_t,  vreinterpret, u16, u16)
DECLFUN(int16x4_t,   int16x4_t,   vreinterpret, s16, s16)
DECLFUN(uint32x2_t,  uint32x2_t,  vreinterpret, u32, u32)
DECLFUN(int32x2_t,   int32x2_t,   vreinterpret, s32, s32)
DECLFUN(float32x2_t, float32x2_t, vreinterpret, f32, f32)

DECLFUN(uint8x16_t,   uint8x16_t, vreinterpretq, u8,  u8)
DECLFUN(int8x16_t,    int8x16_t,  vreinterpretq, s8,  s8)
DECLFUN(uint16x8_t,  uint16x8_t,  vreinterpretq, u16, u16)
DECLFUN(int16x8_t,   int16x8_t,   vreinterpretq, s16, s16)
DECLFUN(uint32x4_t,  uint32x4_t,  vreinterpretq, u32, u32)
DECLFUN(int32x4_t,   int32x4_t,   vreinterpretq, s32, s32)
DECLFUN(float32x4_t, float32x4_t, vreinterpretq, f32, f32)

#if defined(AURA_ENABLE_NEON_FP16)

DECLFUN(float16x4_t, float16x4_t, vreinterpret,  f16, f16)
DECLFUN(float16x8_t, float16x8_t, vreinterpretq, f16, f16)

#endif

#undef DECLFUN

} // namespace neon

} // namespace aura

#endif // AURA_RUNTIME_CORE_NEON_REINTERPRET_HPP__