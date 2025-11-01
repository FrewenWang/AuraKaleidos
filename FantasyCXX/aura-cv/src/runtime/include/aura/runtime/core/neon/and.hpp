#ifndef AURA_RUNTIME_CORE_NEON_AND_HPP__
#define AURA_RUNTIME_CORE_NEON_AND_HPP__

#include <arm_neon.h>

namespace aura
{

namespace neon
{

#define DECLFUN(dtype, vtype, prefix, postfix)        \
    inline vtype vand(const vtype &v, const vtype &u) \
    {                                                 \
        return prefix##_##postfix(v, u);              \
    }

DECLFUN(uint8x8_t,   uint8x8_t,   vand,  u8)
DECLFUN(uint8x16_t,  uint8x16_t,  vandq, u8)
DECLFUN(int8x8_t,    int8x8_t,    vand,  s8)
DECLFUN(int8x16_t,   int8x16_t,   vandq, s8)
DECLFUN(uint16x4_t,  uint16x4_t,  vand,  u16)
DECLFUN(uint16x8_t,  uint16x8_t,  vandq, u16)
DECLFUN(int16x4_t,   int16x4_t,   vand,  s16)
DECLFUN(int16x8_t,   int16x8_t,   vandq, s16)
DECLFUN(uint32x2_t,  uint32x2_t,  vand,  u32)
DECLFUN(uint32x4_t,  uint32x4_t,  vandq, u32)
DECLFUN(int32x2_t,   int32x2_t,   vand,  s32)
DECLFUN(int32x4_t,   int32x4_t,   vandq, s32)
DECLFUN(uint64x1_t,  uint64x1_t,  vand,  u64)
DECLFUN(uint64x2_t,  uint64x2_t,  vandq, u64)
DECLFUN(int64x1_t,   int64x1_t,   vand,  s64)
DECLFUN(int64x2_t,   int64x2_t,   vandq, s64)
#undef DECLFUN

} // namespace neon

} // namespace aura

#endif // AURA_RUNTIME_CORE_NEON_AND_HPP__