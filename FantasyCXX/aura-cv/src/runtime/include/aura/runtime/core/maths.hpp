#ifndef AURA_RUNTIME_CORE_MATHS_HPP__
#define AURA_RUNTIME_CORE_MATHS_HPP__

#include "aura/runtime/core/defs.hpp"
#include "aura/runtime/core/types/built-in.hpp"

#include <cmath>

#if defined(AURA_BUILD_HOST)
#  if defined(AURA_ENABLE_NEON)
#    include <arm_neon.h>
#  endif // AURA_ENABLE_NEON
#endif // AURA_BUILD_HOST

#define AURA_PI                    (3.1415926535897932384626433832795)

/**
 * @defgroup runtime Runtime
 * @{
 *      @defgroup core Runtime Core
 *      @{
 *           @defgroup math Runtime Core Math
 *      @}
 * @}
*/

namespace aura
{
/**
 * @addtogroup math
 * @{
*/

/**
 * @brief Computes the absolute value of a given value.
 * 
 * @tparam Tp The type of the input value.
 * 
 * @param x The input value.
 * 
 * @return The absolute value of the input.
 */
template<typename Tp>
AURA_INLINE Tp Abs(Tp x)
{
    return x > 0 ? x : (-x);
}

/**
 * @brief Swaps the values of two variables.
 * 
 * @tparam Tp The type of the variables.
 * 
 * @param x The first variable.
 * @param y The second variable.
 */
template<typename Tp>
AURA_INLINE DT_VOID Swap(Tp &x, Tp &y)
{
    Tp t = x;
    x = y;
    y = t;
}

/**
 * @brief Computes the maximum of two values.
 * 
 * @tparam Tp The type of the values.
 * 
 * @param x The first value.
 * @param y The second value.
 * 
 * @return The maximum value.
 */
template<typename Tp>
AURA_INLINE Tp Max(const Tp &x, const Tp &y)
{
    return x > y ? x : y;
}

/**
 * @brief Computes the minimum of two values.
 * 
 * @tparam Tp The type of the values.
 * 
 * @param x The first value.
 * @param y The second value.
 * 
 * @return The minimum value.
 */
template<typename Tp>
AURA_INLINE Tp Min(const Tp &x, const Tp &y)
{
    return x < y ? x : y;
}

/**
 * @brief Updates two variables to be in ascending order.
 * 
 * @tparam Tp The type of the variables.
 * 
 * @param x The first variable.
 * @param y The second variable.
 */
template<typename Tp>
AURA_INLINE DT_VOID MinMax(Tp &x, Tp &y)
{
    if (x > y)
    {
        Swap(x, y);
    }
}

/**
 * @brief Updates two variables to be in descending order.
 * 
 * @tparam Tp The type of the variables.
 * 
 * @param x The first variable.
 * @param y The second variable.
 */
template<typename Tp>
AURA_INLINE DT_VOID MaxMin(Tp &x, Tp &y)
{
    if (x < y)
    {
        Swap(x, y);
    }
}

/**
 * @brief Clamps a value within a specified range.
 * 
 * @tparam Tp The type of the value.
 * 
 * @param x The value to be clamped.
 * @param min The minimum bound.
 * @param max The maximum bound.
 * 
 * @return The clamped value.
 */
template<typename Tp>
AURA_INLINE Tp Clamp(const Tp &x, const Tp &min, const Tp &max)
{
    return Max(Min(x, (max)), (min));
}

/**
 * @brief Rounds a float value to the nearest integer using floor.
 * 
 * @param value The input float value.
 * 
 * @return The rounded integer value.
 */
AURA_INLINE DT_S32 Floor(DT_F32 value)
{
    DT_S32 temp = (DT_S32)value;
    DT_F32 diff = (DT_F32)(value - temp);
    return temp - (diff < 0);
}

/**
 * @brief Rounds a float value to the nearest integer using ceiling.
 * 
 * @param value The input float value.
 * 
 * @return The rounded integer value.
 */
AURA_INLINE DT_S32 Ceil(DT_F32 value)
{
    DT_S32 temp = (DT_S32)value;
    DT_F32 diff = (DT_F32)(value - temp);
    return temp + (diff > 0);
}

/**
 * @brief Rounds a float value to the nearest integer(or nearest even). Round to an nearest even number only when the decimal part is exactly 0.5.
 * 
 * e.g. 1) round to the even nearest integer: 1.5 -> 2, 2.5 -> 2; 2) round to the nearest integer: 1.3 -> 1, 1.7 -> 2.
 * 
 * @param value The input float value.
 * 
 * @return The rounded integer value.
 */
AURA_INLINE DT_S32 Round(DT_F32 value)
{
    #if defined(AURA_ENABLE_NEON)
        #  if defined(__aarch64__)
            return vcvtns_s32_f32(value);
        #  else // __aarch64__
            DT_S32 result;
            DT_F32 temp;
            (void)temp;
            __asm__("vcvtr.s32.f32 %[temp], %[value]\n vmov %[result], %[temp]"
                : [result] "=r" (result), [temp] "=w" (temp) : [value] "w" (value));
            return result;
        #  endif // __aarch64__
    #else
        DT_F32 intpart, fractpart;
        fractpart = modff(value, &intpart);
        if ((fabsf(fractpart) != 0.5) || ((((DT_S32)intpart) & 1) != 0))
        {
            return (DT_S32)(value + (value >= 0 ? 0.5 : -0.5));
        }
        else
        {
            return (DT_S32)intpart;
        }
    #endif
}

/**
 * @brief Rounds a double value to the nearest integer using floor.
 * 
 * @param value The input double value.
 * 
 * @return The rounded integer value.
 */
AURA_INLINE DT_S64 Floor(DT_F64 value)
{
    DT_S64 temp = (DT_S64)value;
    DT_F64 diff = (DT_F64)(value - temp);
    return temp - (diff < 0);
}

/**
 * @brief Rounds a double value to the nearest integer using ceiling.
 * 
 * @param value The input double value.
 * 
 * @return The rounded integer value.
 */
AURA_INLINE DT_S64 Ceil(DT_F64 value)
{
    DT_S64 temp = (DT_S64)value;
    DT_F64 diff = (DT_F64)(value - temp);
    return temp + (diff > 0);
}

/**
 * @brief Rounds a double value to the nearest integer(or nearest even). Round to an nearest even number only when the decimal part is exactly 0.5.
 * 
 * e.g. 1) round to the even nearest integer: 1.5 -> 2, 2.5 -> 2; 2) round to the nearest integer: 1.3 -> 1, 1.7 -> 2.
 * 
 * @param value The input double value.
 * 
 * @return The rounded integer value.
 * 
 * @note The input value should be in the range of int32: [−2147483648, 2147483647]. Otherwise, the undefined behaviors may occur.
 */
AURA_INLINE DT_S32 Round(DT_F64 value)
{
    DT_F64 intpart, fractpart;
    fractpart = modf(value, &intpart);
    if ((fabs(fractpart) != 0.5) || ((((DT_S32)intpart) & 1) != 0))
    {
        return (DT_S32)(value + (value >= 0 ? 0.5 : -0.5));
    }
    else
    {
        return (DT_S32)intpart;
    }
}

AURA_INLINE DT_F32 Sqrt(DT_F32 value)
{
    return sqrtf(value);
}

AURA_INLINE DT_F64 Sqrt(DT_F64 value)
{
    return sqrt(value);
}

AURA_INLINE DT_F32 Div(DT_F32 n, DT_F32 d)
{
    return n / d;
}

AURA_INLINE DT_F32 Exp(DT_F32 value)
{
    return expf(value);
}

AURA_INLINE DT_F64 Exp(DT_F64 value)
{
    return exp(value);
}

AURA_INLINE DT_F32 Exp2(DT_F32 value)
{
    return exp2f(value);
}

AURA_INLINE DT_F64 Exp2(DT_F64 value)
{
    return exp2(value);
}

AURA_INLINE DT_F32 Log(DT_F32 value)
{
    return logf(value);
}

AURA_INLINE DT_F64 Log(DT_F64 value)
{
    return log(value);
}

AURA_INLINE DT_F32 Log2(DT_F32 value)
{
    return log2f(value);
}

AURA_INLINE DT_F64 Log2(DT_F64 value)
{
    return log2(value);
}

AURA_INLINE DT_F32 Log10(DT_F32 value)
{
    return log10f(value);
}

AURA_INLINE DT_F64 Log10(DT_F64 value)
{
    return log10(value);
}

AURA_INLINE DT_F32 Pow(DT_F32 x, DT_F32 y)
{
    return powf(x, y);
}

AURA_INLINE DT_F64 Pow(DT_F64 x, DT_F64 y)
{
    return pow(x, y);
}

AURA_INLINE DT_F32 Sin(DT_F32 value)
{
    return sinf(value);
}

AURA_INLINE DT_F64 Sin(DT_F64 value)
{
    return sin(value);
}

AURA_INLINE DT_F32 Cos(DT_F32 value)
{
    return cosf(value);
}

AURA_INLINE DT_F64 Cos(DT_F64 value)
{
    return cos(value);
}

AURA_INLINE DT_F32 Tan(DT_F32 value)
{
    return tanf(value);
}

AURA_INLINE DT_F64 Tan(DT_F64 value)
{
    return tan(value);
}

AURA_INLINE DT_F32 Asin(DT_F32 value)
{
    return asinf(value);
}

AURA_INLINE DT_F64 Asin(DT_F64 value)
{
    return asin(value);
}

AURA_INLINE DT_F32 Acos(DT_F32 value)
{
    return acosf(value);
}

AURA_INLINE DT_F64 Acos(DT_F64 value)
{
    return acos(value);
}

AURA_INLINE DT_F32 Atan(DT_F32 value)
{
    return atanf(value);
}

AURA_INLINE DT_F64 Atan(DT_F64 value)
{
    return atan(value);
}

AURA_INLINE DT_F32 Atan2(DT_F32 x, DT_F32 y)
{
    return atan2f(x, y);
}

AURA_INLINE DT_F64 Atan2(DT_F64 x, DT_F64 y)
{
    return atan2(x, y);
}

/**
 * @brief Checks if two single-precision floating-point values are nearly equal.
 * 
 * @param x The first value.
 * @param y The second value.
 * 
 * @return True if the values are nearly equal, false otherwise.
 */
AURA_INLINE DT_BOOL NearlyEqual(DT_F32 x, DT_F32 y)
{
    return (((x) + 1e-5) > (y)) && (((x) - 1e-5) < (y));
}

/**
 * @brief Checks if two double-precision floating-point values are nearly equal.
 * 
 * @param x The first value.
 * @param y The second value.
 * 
 * @return True if the values are nearly equal, false otherwise.
 */
AURA_INLINE DT_BOOL NearlyEqual(DT_F64 x, DT_F64 y)
{
    return (((x) + 1e-5) > (y)) && (((x) - 1e-5) < (y));
}

/**
 * @}
*/
} // namespace aura

#endif // AURA_RUNTIME_CORE_MATHS_HPP__