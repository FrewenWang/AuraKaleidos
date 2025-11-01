#ifndef AURA_RUNTIME_CORE_NEON_MOVL_HPP__
#define AURA_RUNTIME_CORE_NEON_MOVL_HPP__

#include <arm_neon.h>

namespace aura
{

namespace neon
{

#define DECLFUN(dtype, vtype, prefix, postfix) \
    inline dtype vmovl(const vtype &v)         \
    {                                          \
        return prefix##_##postfix(v);          \
    }

DECLFUN(uint16x8_t,   uint8x8_t,   vmovl,  u8)
DECLFUN(int16x8_t,    int8x8_t,    vmovl,  s8)
DECLFUN(uint32x4_t,   uint16x4_t,  vmovl,  u16)
DECLFUN(int32x4_t,    int16x4_t,   vmovl,  s16)
DECLFUN(uint64x2_t,   uint32x2_t,  vmovl,  u32)
DECLFUN(int64x2_t,    int32x2_t,   vmovl,  s32)
#undef DECLFUN

} // namespace neon

} // namespace aura

#endif // AURA_RUNTIME_CORE_NEON_MOVL_HPP__