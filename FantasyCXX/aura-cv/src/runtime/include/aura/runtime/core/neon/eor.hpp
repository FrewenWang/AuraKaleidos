#ifndef AURA_RUNTIME_CORE_NEON_EOR_HPP__
#define AURA_RUNTIME_CORE_NEON_EOR_HPP__

#include <arm_neon.h>

namespace aura
{

namespace neon
{

#define DECLFUN(dtype, vtype, prefix, postfix)        \
    inline vtype veor(const vtype &v, const vtype &u) \
    {                                                 \
        return prefix##_##postfix(v, u);              \
    }

DECLFUN(uint8x8_t,   uint8x8_t,   veor,  u8)
DECLFUN(uint8x16_t,  uint8x16_t,  veorq, u8)
DECLFUN(int8x8_t,    int8x8_t,    veor,  s8)
DECLFUN(int8x16_t,   int8x16_t,   veorq, s8)
DECLFUN(uint16x4_t,  uint16x4_t,  veor,  u16)
DECLFUN(uint16x8_t,  uint16x8_t,  veorq, u16)
DECLFUN(int16x4_t,   int16x4_t,   veor,  s16)
DECLFUN(int16x8_t,   int16x8_t,   veorq, s16)
DECLFUN(uint32x2_t,  uint32x2_t,  veor,  u32)
DECLFUN(uint32x4_t,  uint32x4_t,  veorq, u32)
DECLFUN(int32x2_t,   int32x2_t,   veor,  s32)
DECLFUN(int32x4_t,   int32x4_t,   veorq, s32)
DECLFUN(uint64x1_t,  uint64x1_t,  veor,  u64)
DECLFUN(uint64x2_t,  uint64x2_t,  veorq, u64)
DECLFUN(int64x1_t,   int64x1_t,   veor,  s64)
DECLFUN(int64x2_t,   int64x2_t,   veorq, s64)
#undef DECLFUN

} // namespace neon

} // namespace aura

#endif // AURA_RUNTIME_CORE_NEON_EOR_HPP__