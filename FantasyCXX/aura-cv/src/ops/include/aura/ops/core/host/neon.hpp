#ifndef AURA_OPS_CORE_HOST_NEON_HPP__
#define AURA_OPS_CORE_HOST_NEON_HPP__

#include "aura/runtime/array.h"
#include "aura/runtime/logger.h"
#include "aura/runtime/mat.h"
#include "aura/runtime/core.h"

/**
 * @defgroup ops Operators
 * @{
 *      @defgroup core_host Core Host
 * @}
*/

namespace aura
{
/**
 * @addtogroup core_host
 * @{
*/

/**
 * @brief Check if the width is compatible with Neon vector length.
 * 检查宽度是否与 Neon 矢量长度兼容。
 * This function checks if the width of the input array is compatible with the Neon vector length.
 *
 * @param array The input array.
 * 
 * @return Status::OK if successful; otherwise, an appropriate error status.
 */
AURA_INLINE Status CheckNeonWidth(const Array &array)
{
    // 如果数据的宽度小于48，则不允许使用NEON。 TODO 为什么？
    DT_S32 width = array.GetSizes().m_width;
    if (width < 48)
    {
        return Status::ERROR;
    }

    return Status::OK;
}

/**
 * @brief Get the border vector for a specified border type.
 * 
 * The supported border types are REPLICATE, REFLECT_101, CONSTANT. And each type has a corresponding specialized implementation.
 *
 * @tparam BORDER_TYPE The type of border handling.
 * @tparam BORDER_AREA The area of the border.
 * @tparam VType The vector type.
 * @tparam SType The scalar type.
 *
 * @param reflect_101 The vector for REFLECT_101 border handling.
 * @param replicate The scalar for REPLICATE border handling.
 * @param constant The scalar for CONSTANT border handling.
 *
 * @return The resulting border vector.
 */
template <BorderType BORDER_TYPE, BorderArea BORDER_AREA, typename VType, typename SType = typename neon::Scalar<VType>::SType,
          typename std::enable_if<BorderType::CONSTANT == BORDER_TYPE, SType>::type* = DT_NULL>
AURA_INLINE VType GetBorderVector(VType reflect_101, SType replicate, SType constant)
{
    AURA_UNUSED(reflect_101);
    AURA_UNUSED(replicate);

    VType result;
    neon::vdup(result, constant);
    return result;
}

template <BorderType BORDER_TYPE, BorderArea BORDER_AREA, typename VType, typename SType = typename neon::Scalar<VType>::SType,
          typename std::enable_if<BorderType::REPLICATE == BORDER_TYPE, SType>::type* = DT_NULL>
AURA_INLINE VType GetBorderVector(VType reflect_101, SType replicate, SType constant)
{
    AURA_UNUSED(reflect_101);
    AURA_UNUSED(constant);

    VType result;
    neon::vdup(result, replicate);
    return result;
}

template <BorderType BORDER_TYPE, BorderArea BORDER_AREA, typename VType, typename SType = typename neon::Scalar<VType>::SType,
          typename std::enable_if<BorderType::REFLECT_101 == BORDER_TYPE, SType>::type* = DT_NULL>
AURA_INLINE VType GetBorderVector(VType reflect_101, SType replicate, SType constant)
{
    AURA_UNUSED(replicate);
    AURA_UNUSED(constant);

    constexpr DT_S32 elem_counts = static_cast<DT_S32>(sizeof(VType) / sizeof(SType));
    constexpr DT_S32 number      = static_cast<DT_S32>(2 - sizeof(SType) / 2);
    constexpr DT_S32 left        = 1 + number * (number + 1);
    constexpr DT_S32 shift       = BorderArea::LEFT == BORDER_AREA ? left : elem_counts - left;

    VType v_rev = neon::vrev64(reflect_101);
    return neon::vext<shift>(v_rev, v_rev);
}

/**
 * @}
 */
} // namespace aura

#endif // AURA_OPS_CORE_HOST_NEON_HPP__
