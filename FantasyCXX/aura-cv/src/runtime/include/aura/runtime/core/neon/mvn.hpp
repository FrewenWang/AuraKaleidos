#ifndef AURA_RUNTIME_CORE_NEON_MVN_HPP__
#define AURA_RUNTIME_CORE_NEON_MVN_HPP__

#include <arm_neon.h>

namespace aura
{

namespace neon
{

#define DECLFUN(dtype, vtype, prefix, postfix) \
    inline vtype vmvn(const vtype &v)          \
    {                                          \
        return prefix##_##postfix(v);          \
    }

DECLFUN(uint8x8_t,   uint8x8_t,   vmvn,  u8)
DECLFUN(uint8x16_t,  uint8x16_t,  vmvnq, u8)
DECLFUN(int8x8_t,    int8x8_t,    vmvn,  s8)
DECLFUN(int8x16_t,   int8x16_t,   vmvnq, s8)
DECLFUN(uint16x4_t,  uint16x4_t,  vmvn,  u16)
DECLFUN(uint16x8_t,  uint16x8_t,  vmvnq, u16)
DECLFUN(int16x4_t,   int16x4_t,   vmvn,  s16)
DECLFUN(int16x8_t,   int16x8_t,   vmvnq, s16)
DECLFUN(uint32x2_t,  uint32x2_t,  vmvn,  u32)
DECLFUN(uint32x4_t,  uint32x4_t,  vmvnq, u32)
DECLFUN(int32x2_t,   int32x2_t,   vmvn,  s32)
DECLFUN(int32x4_t,   int32x4_t,   vmvnq, s32)
#undef DECLFUN

} // namespace neon

} // namespace aura

#endif // AURA_RUNTIME_CORE_NEON_MVN_HPP__