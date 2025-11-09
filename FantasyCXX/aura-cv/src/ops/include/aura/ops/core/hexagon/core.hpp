#ifndef AURA_OPS_CORE_HEXAGON_CORE_HPP__
#define AURA_OPS_CORE_HEXAGON_CORE_HPP__

#include "aura/runtime/mat.h"

/**
 * @defgroup ops Operators
 * @{
 *      @defgroup core_hexagon Core Hexagon
 * @}
*/

namespace aura
{
/**
 * @addtogroup core_hexagon
 * @{
*/

/**
 * @brief External declaration of vrdelta_reverse_d8 with alignment attribute.
 */
extern const DT_U8 vrdelta_reverse_d8[]  __attribute__((aligned(128)));

/**
 * @brief External declaration of vrdelta_reverse_d16 with alignment attribute.
 */
extern const DT_U8 vrdelta_reverse_d16[] __attribute__((aligned(128)));

/**
 * @brief External declaration of vrdelta_reverse_d32 with alignment attribute.
 */
extern const DT_U8 vrdelta_reverse_d32[] __attribute__((aligned(128)));

/**
 * @brief External declaration of vrdelta_replicate_last_d32 with alignment attribute.
 */
extern const DT_U32 vrdelta_replicate_last_d32[] __attribute__((aligned(128)));

/**
 * @brief Get a border vector for the specified type, border type, and border area.
 * 
 * The supported border types are REPLICATE, REFLECT_101, CONSTANT. And each type has a corresponding specialized implementation.
 *
 * @tparam St The scalar type.
 * @tparam BORDER_TYPE The border type.
 * @tparam BORDER_AREA The border area.
 *
 * @param v_src The source vector.
 * @param replicate The replicate value.
 * @param constant The constant value.
 *
 * @return The resulting border vector.
 */
template <typename St, BorderType BORDER_TYPE, BorderArea BORDER_AREA, typename std::enable_if<BorderType::CONSTANT == BORDER_TYPE>::type* = DT_NULL>
AURA_INLINE HVX_Vector GetBorderVector(const HVX_Vector &v_src, St replicate, St constant)
{
    AURA_UNUSED(replicate);
    AURA_UNUSED(v_src);
    HVX_Vector v_result = vsplat(constant);

    return v_result;
}

template <typename St, BorderType BORDER_TYPE, BorderArea BORDER_AREA, typename std::enable_if<BorderType::REPLICATE == BORDER_TYPE>::type* = DT_NULL>
AURA_INLINE HVX_Vector GetBorderVector(const HVX_Vector &v_src, St replicate, St constant)
{
    AURA_UNUSED(constant);
    AURA_UNUSED(v_src);
    HVX_Vector v_result = vsplat(replicate);

    return v_result;
}

template <typename St, BorderType BORDER_TYPE, BorderArea BORDER_AREA, typename std::enable_if<BorderType::REFLECT_101 == BORDER_TYPE>::type* = DT_NULL>
AURA_INLINE HVX_Vector GetBorderVector(const HVX_Vector &v_src, St replicate, St constant)
{
    AURA_UNUSED(replicate);
    AURA_UNUSED(constant);

    HVX_Vector v_reverse;
    if (std::is_same<St, DT_U8>::value || std::is_same<St, DT_S8>::value)
    {
        v_reverse = Q6_V_vdelta_VV(v_src, vmemu(vrdelta_reverse_d8));
    }
    else if (std::is_same<St, DT_U16>::value || std::is_same<St, DT_S16>::value)
    {
        v_reverse = Q6_V_vdelta_VV(v_src, vmemu(vrdelta_reverse_d16));
    }
    else
    {
        v_reverse = Q6_V_vdelta_VV(v_src, vmemu(vrdelta_reverse_d32));
    }

    HVX_Vector v_result;
    if (BorderArea::LEFT == BORDER_AREA)
    {
        v_result = Q6_V_vror_VR(v_reverse, AURA_HVLEN - sizeof(St));
    }
    else
    {
        v_result = Q6_V_vror_VR(v_reverse, sizeof(St));
    }

    return v_result;
}

/**
 * @}
 */
} // namespace aura

#endif // AURA_OPS_CORE_HEXAGON_CORE_HPP__