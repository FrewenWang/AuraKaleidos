#ifndef AURA_RUNTIME_CORE_NEON_QDMLSLN_HPP__
#define AURA_RUNTIME_CORE_NEON_QDMLSLN_HPP__

#include <arm_neon.h>

namespace aura
{

namespace neon
{

#define DECLFUN(dtype, vtype, stype, postfix)                               \
    inline dtype vqdmlsl_n(const dtype &v, const vtype &u, const stype &p)  \
    {                                                                       \
        return vqdmlsl_n_##postfix(v, u, p);                                \
    }

DECLFUN(int32x4_t,  int16x4_t, DT_S16, s16)
DECLFUN(int64x2_t,  int32x2_t, DT_S32, s32)
#undef DECLFUN

} // namespace neon

} // namespace aura

#endif // AURA_RUNTIME_CORE_NEON_QDMLSLN_HPP__