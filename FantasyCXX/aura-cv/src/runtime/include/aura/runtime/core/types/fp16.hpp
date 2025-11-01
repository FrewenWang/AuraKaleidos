#ifndef AURA_RUNTIME_CORE_TYPES_FP16_HPP__
#define AURA_RUNTIME_CORE_TYPES_FP16_HPP__

#include "aura/config.h"

#define HALF_ARITHMETIC_TYPE float

#include "aura/runtime/core/types/fp16/fp16_impl.hpp"

/**
 * @defgroup runtime Runtime
 * @{
 *      @defgroup core Runtime Core
 *      @{
 *           @defgroup fp16 Runtime Core Fp16
 *      @}
 * @}
*/

/**
 * @addtogroup fp16
 * @{
 */

#if !defined(MI_F16_DEFINED)
#define MI_F16_DEFINED

/**
 * @brief Define the type MI_F16 based on platform support for half-precision floating point.
 *
 * The type MI_F16 is defined as __fp16 if AURA_ENABLE_NEON_FP16(ARM) is defined, otherwise it is defined as half_float::half.
 * For ARM platforms, __fp16 typically represents a half-precision floating-point type.
 */

# if defined(AURA_ENABLE_NEON_FP16)
   typedef __fp16           MI_F16;
#  else
   typedef half_float::half MI_F16;
#  endif
#endif // MI_F16_DEFINED

/**
 * @}
 */

#endif // AURA_RUNTIME_CORE_TYPES_FP16_HPP__