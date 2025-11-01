#ifndef AURA_RUNTIME_CORE_NEON_CLZ_HPP__
#define AURA_RUNTIME_CORE_NEON_CLZ_HPP__

#include <arm_neon.h>

namespace aura
{

namespace neon
{

#define DECLFUN(dtype, vtype, prefix, postfix) \
    inline vtype vclz(const vtype &v)          \
    {                                          \
        return prefix##_##postfix(v);          \
    }

DECLFUN(uint8x8_t,   uint8x8_t,   vclz,  u8)
DECLFUN(uint8x16_t,  uint8x16_t,  vclzq, u8)
DECLFUN(int8x8_t,    int8x8_t,    vclz,  s8)
DECLFUN(int8x16_t,   int8x16_t,   vclzq, s8)
DECLFUN(uint16x4_t,  uint16x4_t,  vclz,  u16)
DECLFUN(uint16x8_t,  uint16x8_t,  vclzq, u16)
DECLFUN(int16x4_t,   int16x4_t,   vclz,  s16)
DECLFUN(int16x8_t,   int16x8_t,   vclzq, s16)
DECLFUN(uint32x2_t,  uint32x2_t,  vclz,  u32)
DECLFUN(uint32x4_t,  uint32x4_t,  vclzq, u32)
DECLFUN(int32x2_t,   int32x2_t,   vclz,  s32)
DECLFUN(int32x4_t,   int32x4_t,   vclzq, s32)
#undef DECLFUN

} // namespace neon

} // namespace aura

#endif // AURA_RUNTIME_CORE_NEON_CLZ_HPP__