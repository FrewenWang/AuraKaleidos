#ifndef AURA_RUNTIME_CORE_NEON_ABA_HPP__
#define AURA_RUNTIME_CORE_NEON_ABA_HPP__

#include <arm_neon.h>

namespace aura
{

namespace neon
{

#define DECLFUN(dtype, vtype, prefix, postfix)                        \
    inline vtype vaba(const vtype &v, const vtype &u, const vtype &p) \
    {                                                                 \
        return prefix##_##postfix(v, u, p);                           \
    }

DECLFUN(uint8x8_t,   uint8x8_t,   vaba,  u8)
DECLFUN(int8x8_t,    int8x8_t,    vaba,  s8)
DECLFUN(uint8x16_t,  uint8x16_t,  vabaq, u8)
DECLFUN(int8x16_t,   int8x16_t,   vabaq, s8)
DECLFUN(uint16x4_t,  uint16x4_t,  vaba,  u16)
DECLFUN(int16x4_t,   int16x4_t,   vaba,  s16)
DECLFUN(uint16x8_t,  uint16x8_t,  vabaq, u16)
DECLFUN(int16x8_t,   int16x8_t,   vabaq, s16)
DECLFUN(uint32x2_t,  uint32x2_t,  vaba,  u32)
DECLFUN(int32x2_t,   int32x2_t,   vaba,  s32)
DECLFUN(uint32x4_t,  uint32x4_t,  vabaq, u32)
DECLFUN(int32x4_t,   int32x4_t,   vabaq, s32)
#undef DECLFUN

#define DECLFUN(dtype, vtype, prefix, postfix)                        \
    inline dtype vaba(const dtype &v, const vtype &u, const vtype &p) \
    {                                                                 \
        return prefix##_##postfix(v, u, p);                           \
    }
DECLFUN(uint16x8_t,  uint8x8_t,  vabal, u8)
DECLFUN(int16x8_t,   int8x8_t,   vabal, s8)
DECLFUN(uint32x4_t,  uint16x4_t, vabal, u16)
DECLFUN(int32x4_t,   int16x4_t,  vabal, s16)
DECLFUN(uint64x2_t,  uint32x2_t, vabal, u32)
DECLFUN(int64x2_t,   int32x2_t,  vabal, s32)
#undef DECLFUN

} // namespace neon

} // namespace aura

#endif // AURA_RUNTIME_CORE_NEON_ABA_HPP__