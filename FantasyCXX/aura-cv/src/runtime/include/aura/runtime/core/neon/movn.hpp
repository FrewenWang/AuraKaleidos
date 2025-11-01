#ifndef AURA_RUNTIME_CORE_NEON_MOVN_HPP__
#define AURA_RUNTIME_CORE_NEON_MOVN_HPP__

#include <arm_neon.h>

namespace aura
{

namespace neon
{

#define DECLFUN(dtype, vtype, prefix, postfix) \
    inline dtype vmovn(const vtype &v)         \
    {                                          \
        return prefix##_##postfix(v);          \
    }

DECLFUN(uint8x8_t,   uint16x8_t,   vmovn,  u16)
DECLFUN(int8x8_t,    int16x8_t,    vmovn,  s16)
DECLFUN(uint16x4_t,   uint32x4_t,  vmovn,  u32)
DECLFUN(int16x4_t,    int32x4_t,   vmovn,  s32)
DECLFUN(uint32x2_t,   uint64x2_t,  vmovn,  u64)
DECLFUN(int32x2_t,    int64x2_t,   vmovn,  s64)
#undef DECLFUN

#define DECLFUN(dtype, vtype, prefix, postfix) \
    inline dtype vqmovn(const vtype &v)        \
    {                                          \
        return prefix##_##postfix(v);          \
    }

DECLFUN(uint8x8_t,   uint16x8_t,  vqmovn,  u16)
DECLFUN(int8x8_t,    int16x8_t,   vqmovn,  s16)
DECLFUN(uint16x4_t,  uint32x4_t,  vqmovn,  u32)
DECLFUN(int16x4_t,   int32x4_t,   vqmovn,  s32)
DECLFUN(uint32x2_t,  uint64x2_t,  vqmovn,  u64)
DECLFUN(int32x2_t,   int64x2_t,   vqmovn,  s64)
#undef DECLFUN

#define DECLFUN(dtype, vtype, prefix, postfix) \
    inline dtype vqmovun(const vtype &v)       \
    {                                          \
        return prefix##_##postfix(v);          \
    }

DECLFUN(uint8x8_t,   int16x8_t,  vqmovun,  s16)
DECLFUN(uint16x4_t,  int32x4_t,  vqmovun,  s32)
DECLFUN(uint32x2_t,  int64x2_t,  vqmovun,  s64)
#undef DECLFUN

} // namespace neon

} // namespace aura

#endif // AURA_RUNTIME_CORE_NEON_MOVN_HPP__