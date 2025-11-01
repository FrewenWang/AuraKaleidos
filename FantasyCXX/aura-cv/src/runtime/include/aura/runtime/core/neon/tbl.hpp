#ifndef AURA_RUNTIME_CORE_NEON_TBL_HPP__
#define AURA_RUNTIME_CORE_NEON_TBL_HPP__

#include <arm_neon.h>

namespace aura
{

namespace neon
{

#define DECLFUN(stype, vtype, prefix, postfix)        \
    inline vtype vtbl(const stype &v, const vtype &u) \
    {                                                 \
        return prefix##_##postfix(v, u);              \
    }

DECLFUN(uint8x8_t,   uint8x8_t, vtbl1, u8)
DECLFUN(int8x8_t,    int8x8_t,  vtbl1, s8)
DECLFUN(uint8x8x2_t, uint8x8_t, vtbl2, u8)
DECLFUN(int8x8x2_t,  int8x8_t,  vtbl2, s8)
DECLFUN(uint8x8x3_t, uint8x8_t, vtbl3, u8)
DECLFUN(int8x8x3_t,  int8x8_t,  vtbl3, s8)
DECLFUN(uint8x8x4_t, uint8x8_t, vtbl4, u8)
DECLFUN(int8x8x4_t,  int8x8_t,  vtbl4, s8)
#undef DECLFUN

} // namespace neon

} // namespace aura

#endif // AURA_RUNTIME_CORE_NEON_TBL_HPP__