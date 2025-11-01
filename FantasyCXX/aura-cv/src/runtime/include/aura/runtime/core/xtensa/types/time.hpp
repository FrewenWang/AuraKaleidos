#ifndef AURA_RUNTIME_CORE_XTENSA_TYPES_TIME_HPP__
#define AURA_RUNTIME_CORE_XTENSA_TYPES_TIME_HPP__

#include "aura/runtime/core/xtensa/types/built-in.hpp"

/**
 * @defgroup runtime Runtime
 * @{
 *      @defgroup core Runtime Core
 *      @{
 *           @defgroup time Runtime Core Xtensa Time
 *      @}
 * @}
*/

namespace aura
{
namespace xtensa
{
/**
 * @addtogroup time
 * @{
*/

/**
 * @brief Structure representing time with second, millisecond, and microsecond precision.
*/
struct AURA_EXPORTS Time
{
    /**
    * @brief Default constructor with optional parameters.
    *
    * @param second The second component of the time (default is 0).
    * @param millisec The millisecond component of the time (default is 0).
    * @param microsec The microsecond component of the time (default is 0).
    */
    Time(MI_S64 second = 0, MI_S32 millisec = 0, MI_S32 microsec = 0)
    {
        MI_S32 ms_carry  = 0;
        MI_S32 sec_carry = 0;

        ms_carry = microsec / 1000;
        us = microsec % 1000;

        sec_carry = (millisec + ms_carry) / 1000;
        ms = (millisec + ms_carry) % 1000;

        sec = second + sec_carry;
    }

    /**
     * @brief Get the time in seconds.
     *
     * @return The time in seconds as a floating-point number.
     */
    MI_F64 AsSec()
    {
        return sec + ms / 1000.0 + us / 1000000.0;
    }

    /**
     * @brief Get the time in milliseconds.
     *
     * @return The time in milliseconds as a floating-point number.
     */
    MI_F64 AsMilliSec()
    {
        return sec * 1000 + ms + us / 1000.0;
    }

    /**
     * @brief Get the time in microseconds.
     *
     * @return The time in microseconds as a 64-bit integer.
     */
    MI_S64 AsMicroSec()
    {
        return sec * 1000000 + ms * 1000 + us;
    }

    MI_S64 sec;  /*!< The second component of the time. */
    MI_S32 ms;   /*!< The millisecond component of the time. */
    MI_S32 us;   /*!< The microsecond component of the time. */
};

/**
 * @brief Equality comparison operator for Time objects.
 *
 * @param t0 The first Time object.
 * @param t1 The second Time object.
 *
 * @return `true` if the Time objects are equal, `false` otherwise.
 */
AURA_INLINE MI_BOOL operator==(const Time &t0, const Time &t1)
{
    if (t0.sec == t1.sec && t0.ms == t1.ms && t0.us == t1.us)
    {
        return MI_TRUE;
    }

    return MI_FALSE;
}

/**
 * @brief Inequality comparison operator for Time objects.
 *
 * @param t0 The first Time object.
 * @param t1 The second Time object.
 *
 * @return `true` if the Time objects are not equal, `false` otherwise.
 */
AURA_INLINE MI_BOOL operator!=(const Time &t0, const Time &t1)
{
    return !(t0 == t1);
}

/**
 * @brief Greater than comparison operator for Time objects.
 *
 * @param t0 The first Time object.
 * @param t1 The second Time object.
 *
 * @return `true` if t0 is greater than t1, `false` otherwise.
 */
AURA_INLINE MI_BOOL operator>(const Time &t0, const Time &t1)
{
    if (t0.sec != t1.sec)
    {
        return (t0.sec > t1.sec) ? MI_TRUE : MI_FALSE;
    }

    if (t0.ms != t1.ms)
    {
        return (t0.ms > t1.ms) ? MI_TRUE : MI_FALSE;
    }

    return (t0.us > t1.us) ? MI_TRUE : MI_FALSE;
}

/**
 * @brief Greater than or equal to comparison operator for Time objects.
 *
 * @param t0 The first Time object.
 * @param t1 The second Time object.
 *
 * @return `true` if t0 is greater than or equal to t1, `false` otherwise.
 */
AURA_INLINE MI_BOOL operator>=(const Time &t0, const Time &t1)
{
    return (t0 > t1 || t0 == t1);
}

/**
 * @brief Less than comparison operator for Time objects.
 *
 * @param t0 The first Time object.
 * @param t1 The second Time object.
 *
 * @return `true` if t0 is less than t1, `false` otherwise.
 */
AURA_INLINE MI_BOOL operator<(const Time &t0, const Time &t1)
{
    return !(t0 >= t1);
}

/**
 * @brief Less than or equal to comparison operator for Time objects.
 *
 * @param t0 The first Time object.
 * @param t1 The second Time object.
 *
 * @return `true` if t0 is less than or equal to t1, `false` otherwise.
 */
AURA_INLINE MI_BOOL operator<=(const Time &t0, const Time &t1)
{
    return !(t0 > t1);
}

/**
 * @brief Subtraction operator for Time objects.
 *
 * @param t0 The minuend Time object.
 * @param t1 The subtrahend Time object.
 *
 * @return The difference between t0 and t1 as a Time object.
 */
AURA_INLINE Time operator-(const Time &t0, const Time &t1)
{
    Time start, end;
    if (t0 > t1)
    {
        start = t1;
        end   = t0;
    }
    else
    {
        start = t0;
        end   = t1;
    }

    Time diff;

    // us
    if (end.us >= start.us)
    {
        diff.us = end.us - start.us;
    }
    else
    {
        diff.us = end.us + 1000 - start.us;
        end.ms--;
    }

    // ms
    if (end.ms >= start.ms)
    {
        diff.ms = end.ms - start.ms;
    }
    else
    {
        diff.ms = end.ms + 1000 - start.ms;
        end.sec--;
    }

    // sec
    diff.sec = end.sec - start.sec;

    return diff;
}

/**
 * @brief Addition operator for Time objects.
 *
 * @param t0 The first Time object.
 * @param t1 The second Time object.
 *
 * @return The sum of t0 and t1 as a Time object.
 */
AURA_INLINE Time operator+(const Time &t0, const Time &t1)
{
    Time sum;

    MI_S32 ms_carry  = 0;
    MI_S32 sec_carry = 0;

    // us
    MI_S64 sum_us = t0.us + t1.us;
    if (sum_us >= 1000)
    {
        sum.us = sum_us - 1000;
        ms_carry = 1;
    }
    else
    {
        sum.us = sum_us;
    }

    // ms
    MI_S64 sum_ms = t0.ms + t1.ms + ms_carry;
    if (sum_ms >= 1000)
    {
        sum.ms = sum_ms - 1000;
        sec_carry = 1;
    }
    else
    {
        sum.ms = sum_ms;
    }

    // sec
    sum.sec = t0.sec + t1.sec + sec_carry;

    return sum;
}

/**
 * @}
*/
} // namespace xtensa
} // namespace aura

#endif // AURA_RUNTIME_CORE_XTENSA_TYPES_TIME_HPP__