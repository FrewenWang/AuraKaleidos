#ifndef AURA_RUNTIME_CORE_NEON_TST_HPP__
#define AURA_RUNTIME_CORE_NEON_TST_HPP__

#include <arm_neon.h>

namespace aura
{

namespace neon
{

#define DECLFUN(dtype, vtype, prefix, postfix)        \
    inline vtype vtst(const vtype &v, const vtype &u) \
    {                                                 \
        return prefix##_##postfix(v, u);              \
    }

DECLFUN(uint8x8_t,   uint8x8_t,   vtst,  u8 )
DECLFUN(uint8x16_t,  uint8x16_t,  vtstq, u8 )
DECLFUN(uint8x8_t,   int8x8_t,    vtst,  s8 )
DECLFUN(uint8x16_t,  int8x16_t,   vtstq, s8 )
DECLFUN(uint16x4_t,  uint16x4_t,  vtst,  u16)
DECLFUN(uint16x8_t,  uint16x8_t,  vtstq, u16)
DECLFUN(uint16x4_t,  int16x4_t,   vtst,  s16)
DECLFUN(uint16x8_t,  int16x8_t,   vtstq, s16)
DECLFUN(uint32x2_t,  uint32x2_t,  vtst,  u32)
DECLFUN(uint32x4_t,  uint32x4_t,  vtstq, u32)
DECLFUN(uint32x2_t,  int32x2_t,   vtst,  s32)
DECLFUN(uint32x4_t,  int32x4_t,   vtstq, s32)
#undef DECLFUN

} // namespace neon

} // namespace aura

#endif // AURA_RUNTIME_CORE_NEON_TST_HPP__