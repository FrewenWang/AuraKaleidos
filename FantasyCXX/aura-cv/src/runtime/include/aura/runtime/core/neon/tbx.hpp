#ifndef AURA_RUNTIME_CORE_NEON_TBX_HPP__
#define AURA_RUNTIME_CORE_NEON_TBX_HPP__

#include <arm_neon.h>

namespace aura
{

namespace neon
{

#define DECLFUN(stype, vtype, prefix, postfix)                        \
    inline vtype vtbx(const vtype &v, const stype &u, const vtype &p) \
    {                                                                 \
        return prefix##_##postfix(v, u, p);                           \
    }

DECLFUN(uint8x8_t,   uint8x8_t, vtbx1, u8)
DECLFUN(int8x8_t,    int8x8_t,  vtbx1, s8)
DECLFUN(uint8x8x2_t, uint8x8_t, vtbx2, u8)
DECLFUN(int8x8x2_t,  int8x8_t,  vtbx2, s8)
DECLFUN(uint8x8x3_t, uint8x8_t, vtbx3, u8)
DECLFUN(int8x8x3_t,  int8x8_t,  vtbx3, s8)
DECLFUN(uint8x8x4_t, uint8x8_t, vtbx4, u8)
DECLFUN(int8x8x4_t,  int8x8_t,  vtbx4, s8)
#undef DECLFUN

} // namespace neon

} // namespace aura

#endif // AURA_RUNTIME_CORE_NEON_TBX_HPP__