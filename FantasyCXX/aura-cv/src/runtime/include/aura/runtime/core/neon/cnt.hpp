#ifndef AURA_RUNTIME_CORE_NEON_CNT_HPP__
#define AURA_RUNTIME_CORE_NEON_CNT_HPP__

#include <arm_neon.h>

namespace aura
{

namespace neon
{

#define DECLFUN(dtype, vtype, prefix, postfix) \
    inline vtype vcnt(const vtype &v)          \
    {                                          \
        return prefix##_##postfix(v);          \
    }

DECLFUN(uint8x8_t,   uint8x8_t,   vcnt,  u8)
DECLFUN(uint8x16_t,  uint8x16_t,  vcntq, u8)
DECLFUN(int8x8_t,    int8x8_t,    vcnt,  s8)
DECLFUN(int8x16_t,   int8x16_t,   vcntq, s8)
#undef DECLFUN

} // namespace neon

} // namespace aura

#endif // AURA_RUNTIME_CORE_NEON_CNT_HPP__