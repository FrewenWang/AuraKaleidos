#ifndef AURA_RUNTIME_CORE_LIMITS_HPP__
#define AURA_RUNTIME_CORE_LIMITS_HPP__

#include "aura/runtime/core/types/fp16.hpp"

/**
 * @defgroup runtime Runtime
 * @{
 *      @defgroup core Runtime Core
 *      @{
 *           @defgroup fp16 Runtime Core Fp16
 *      @}
 * @}
*/

namespace std
{
/**
 * @addtogroup fp16
 * @{
*/

#if defined(AURA_ENABLE_NEON_FP16)
/**
 * @brief Specialization of `std::numeric_limits` for the MI_F16 type.
 * 
 * @ingroup core
 */
template<> struct numeric_limits<MI_F16> : std::numeric_limits<half_float::half> {};
#endif

/**
 * @}
*/
} // namespace std

#endif // AURA_RUNTIME_CORE_LIMITS_HPP__