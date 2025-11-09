#ifndef AURA_RUNTIME_CORE_XTENSA_MATHS_HPP__
#define AURA_RUNTIME_CORE_XTENSA_MATHS_HPP__

#include "aura/runtime/core/defs.hpp"
#include "aura/runtime/core/types/built-in.hpp"

#define AURA_PI                    (3.1415926535897932384626433832795)

/**
 * @defgroup runtime Runtime
 * @{
 *      @defgroup core Runtime Core
 *      @{
 *           @defgroup maths Runtime Core Xtensa Maths
 *      @}
 * @}
*/

namespace aura
{
namespace xtensa
{
/**
 * @addtogroup maths
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
    return Max(Min(x, max), min);
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
 * @brief Function to compute the fractional and integral parts of a floating-point value.
 *
 * @param value The input floating-point value.
 * @param iptr Pointer to store the integral part of the value.
 *
 * @return The fractional part of the value.
 */
DT_F32 Modff(DT_F32 value, DT_F32* iptr);

/**
 * @brief Function to compute the fractional and integral parts of a double precision floating-point value.
 *
 * @param value The input double value.
 * @param iptr Pointer to store the integral part of the value.
 *
 * @return The fractional part of the value.
 */
DT_F64 Modf(DT_F64 value, DT_F64* iptr);

/**
 * @brief Function to compute the absolute value of a floating-point value.
 *
 * @param value The input floating-point value.
 *
 * @return The absolute value of the value.
 */
DT_F32 Fabsf(DT_F32 value);

/**
 * @brief Function to compute the absolute value of a double precision floating-point value.
 *
 * @param value The input double value.
 *
 * @return The absolute value of the value.
 */
DT_F64 Fabs(DT_F64 value);

/**
 * @brief Function to compute the square root of a floating-point value.
 *
 * @param value The input floating-point value.
 *
 * @return The square root of the value.
 */
DT_F32 Sqrtf(DT_F32 value);

/**
 * @brief Function to compute the square root of a double precision floating-point value.
 *
 * @param value The input double value.
 *
 * @return The square root of the value.
 */
DT_F64 Sqrt(DT_F64 value);

/**
 * @brief Function to compute the base-e exponential of a floating-point value.
 *
 * @param value The input floating-point value.
 *
 * @return The base-e exponential of the value.
 */
DT_F32 Expf(DT_F32 value);

/**
 * @brief Function to compute the base-e exponential of a double precision floating-point value.
 *
 * @param value The input double value.
 *
 * @return The base-e exponential of the value.
 */
DT_F64 Exp(DT_F64 value);

/**
 * @brief Function to compute the base-2 exponential of a floating-point value.
 *
 * @param value The input floating-point value.
 *
 * @return The base-2 exponential of the value.
 */
DT_F32 Exp2f(DT_F32 value);

/**
 * @brief Function to compute the base-2 exponential of a double precision floating-point value.
 *
 * @param value The input double value.
 *
 * @return The base-2 exponential of the value.
 */
DT_F64 Exp2(DT_F64 value);

/**
 * @brief Function to compute the natural logarithm of a floating-point value.
 *
 * @param value The input floating-point value.
 *
 * @return The natural logarithm of the value.
 */
DT_F32 Logf(DT_F32 value);

/**
 * @brief Function to compute the natural logarithm of a double precision floating-point value.
 *
 * @param value The input double value.
 *
 * @return The natural logarithm of the value.
 */
DT_F64 Log(DT_F64 value);

/**
 * @brief Function to compute the base-2 logarithm of a floating-point value.
 *
 * @param value The input floating-point value.
 *
 * @return The base-2 logarithm of the value.
 */
DT_F32 Log2f(DT_F32 value);

/**
 * @brief Function to compute the base-2 logarithm of a double precision floating-point value.
 *
 * @param value The input double value.
 *
 * @return The base-2 logarithm of the value.
 */
DT_F64 Log2(DT_F64 value);

/**
 * @brief Function to compute the base-10 logarithm of a floating-point value.
 *
 * @param value The input floating-point value.
 *
 * @return The base-10 logarithm of the value.
 */
DT_F32 Log10f(DT_F32 value);

/**
 * @brief Function to compute the base-10 logarithm of a double precision floating-point value.
 *
 * @param value The input double value.
 *
 * @return The base-10 logarithm of the value.
 */
DT_F64 Log10(DT_F64 value);

/**
 * @brief Function to compute the power of a floating-point value.
 *
 * @param base The base of the power.
 * @param exponent The exponent of the power.
 *
 * @return The base raised to the power of the exponent.
 */
DT_F32 Powf(DT_F32 base, DT_F32 exponent);

/**
 * @brief Function to compute the power of a double precision floating-point value.
 *
 * @param base The base of the power.
 * @param exponent The exponent of the power.
 *
 * @return The base raised to the power of the exponent.
 */
DT_F64 Pow(DT_F64 base, DT_F64 exponent);

/**
 * @brief Function to compute the sine of a floating-point value.
 *
 * @param value The input floating-point value.
 *
 * @return The sine of the value.
 */
DT_F32 Sinf(DT_F32 value);

/**
 * @brief Function to compute the sine of a double precision floating-point value.
 *
 * @param value The input double value.
 *
 * @return The sine of the value.
 */
DT_F64 Sin(DT_F64 value);

/**
 * @brief Function to compute the cosine of a floating-point value.
 *
 * @param value The input floating-point value.
 *
 * @return The cosine of the value.
 */
DT_F32 Cosf(DT_F32 value);

/**
 * @brief Function to compute the cosine of a double precision floating-point value.
 *
 * @param value The input double value.
 *
 * @return The cosine of the value.
 */
DT_F64 Cos(DT_F64 value);

/**
 * @brief Function to compute the tangent of a floating-point value.
 *
 * @param value The input floating-point value.
 *
 * @return The tangent of the value.
 */
DT_F32 Tanf(DT_F32 value);

/**
 * @brief Function to compute the tangent of a double precision floating-point value.
 *
 * @param value The input double value.
 *
 * @return The tangent of the value.
 */
DT_F64 Tan(DT_F64 value);

/**
 * @brief Function to compute the arc sine of a floating-point value.
 *
 * @param value The input floating-point value.
 *
 * @return The arc sine of the value.
 */
DT_F32 Asinf(DT_F32 value);

/**
 * @brief Function to compute the arc sine of a double precision floating-point value.
 *
 * @param value The input double value.
 *
 * @return The arc sine of the value.
 */
DT_F64 Asin(DT_F64 value);

/**
 * @brief Function to compute the arc cosine of a floating-point value.
 *
 * @param value The input floating-point value.
 *
 * @return The arc cosine of the value.
 */
DT_F32 Acosf(DT_F32 value);

/**
 * @brief Function to compute the arc cosine of a double precision floating-point value.
 *
 * @param value The input double value.
 *
 * @return The arc cosine of the value.
 */
DT_F64 Acos(DT_F64 value);

/**
 * @brief Function to compute the arc tangent of a floating-point value.
 *
 * @param value The input floating-point value.
 *
 * @return The arc tangent of the value.
 */
DT_F32 Atanf(DT_F32 value);

/**
 * @brief Function to compute the arc tangent of a double precision floating-point value.
 *
 * @param value The input double value.
 *
 * @return The arc tangent of the value.
 */
DT_F64 Atan(DT_F64 value);

/**
 * @brief Function to compute the arc tangent of two floating-point numbers.
 *
 * @param y The y-coordinate.
 * @param x The x-coordinate.
 *
 * @return The arc tangent of the quotient of the two numbers.
 */
DT_F32 Atan2f(DT_F32 y, DT_F32 x);

/**
 * @brief Function to compute the arc tangent of two double precision floating-point value.
 *
 * @param y The y-coordinate.
 * @param x The x-coordinate.
 *
 * @return The arc tangent of the quotient of the two numbers.
 */
DT_F64 Atan2(DT_F64 y, DT_F64 x);

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
    DT_F32 intpart, fractpart;
    fractpart = Modff(value, &intpart);
    if ((Fabsf(fractpart) != 0.5) || ((((DT_S32)intpart) & 1) != 0))
    {
        return (DT_S32)(value + (value >= 0 ? 0.5 : -0.5));
    }
    else
    {
        return (DT_S32)intpart;
    }
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
    fractpart = Modf(value, &intpart);
    if ((Fabs(fractpart) != 0.5) || ((((DT_S32)intpart) & 1) != 0))
    {
        return (DT_S32)(value + (value >= 0 ? 0.5 : -0.5));
    }
    else
    {
        return (DT_S32)intpart;
    }
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
} // namespace xtensa
} // namespace aura

#endif // AURA_RUNTIME_CORE_XTENSA_MATHS_HPP__
