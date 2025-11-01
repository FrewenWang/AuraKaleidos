#ifndef AURA_RUNTIME_CORE_NEON_SRAN_HPP__
#define AURA_RUNTIME_CORE_NEON_SRAN_HPP__

#include <arm_neon.h>

namespace aura
{

namespace neon
{

#define DECLFUN(vtype, prefix, postfix)                 \
    template <int n>                                    \
    inline vtype vsran(const vtype &v, const vtype &u)  \
    {                                                   \
        return prefix##_##postfix(v, u, n);             \
    }

DECLFUN(uint8x8_t,  vsra_n,  u8 )
DECLFUN(uint8x16_t, vsraq_n, u8 )
DECLFUN(int8x8_t,   vsra_n,  s8 )
DECLFUN(int8x16_t,  vsraq_n, s8 )
DECLFUN(uint16x4_t, vsra_n,  u16)
DECLFUN(uint16x8_t, vsraq_n, u16)
DECLFUN(int16x4_t,  vsra_n,  s16)
DECLFUN(int16x8_t,  vsraq_n, s16)
DECLFUN(uint32x2_t, vsra_n,  u32)
DECLFUN(uint32x4_t, vsraq_n, u32)
DECLFUN(int32x2_t,  vsra_n,  s32)
DECLFUN(int32x4_t,  vsraq_n, s32)
#undef DECLFUN

} // namespace neon

} // namespace aura

#endif // AURA_RUNTIME_CORE_NEON_SRAN_HPP__