#ifndef AURA_RUNTIME_CORE_NEON_RSRAN_HPP__
#define AURA_RUNTIME_CORE_NEON_RSRAN_HPP__

#include <arm_neon.h>

namespace aura
{

namespace neon
{

#define DECLFUN(vtype, prefix, postfix)                     \
    template <int n>                                        \
    inline vtype vrsra_n(const vtype &v, const vtype &u)    \
    {                                                       \
        return prefix##_##postfix(v, u, n);                 \
    }

DECLFUN(uint8x8_t,   vrsra_n,  u8)
DECLFUN(uint8x16_t,  vrsraq_n, u8)
DECLFUN(int8x8_t,    vrsra_n,  s8)
DECLFUN(int8x16_t,   vrsraq_n, s8)
DECLFUN(uint16x4_t,  vrsra_n,  u16)
DECLFUN(uint16x8_t,  vrsraq_n, u16)
DECLFUN(int16x4_t,   vrsra_n,  s16)
DECLFUN(int16x8_t,   vrsraq_n, s16)
DECLFUN(uint32x2_t,  vrsra_n,  u32)
DECLFUN(uint32x4_t,  vrsraq_n, u32)
DECLFUN(int32x2_t,   vrsra_n,  s32)
DECLFUN(int32x4_t,   vrsraq_n, s32)
#undef DECLFUN

} // namespace neon

} // namespace aura

#endif // AURA_RUNTIME_CORE_NEON_RSRAN_HPP__