#ifndef AURA_RUNTIME_CORE_TIME_HPP__
#define AURA_RUNTIME_CORE_TIME_HPP__

#include "aura/runtime/core/types/built-in.hpp"

#include <sstream>
#include <iostream>
#include <iomanip>
#include <stdarg.h>
#include <chrono>
#include <ctime>

/**
 * @defgroup runtime Runtime
 * @{
 *      @defgroup core Runtime Core
 *      @{
 *           @defgroup time Runtime Core Time
 *      @}
 * @}
*/

namespace aura
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
    Time(DT_S64 second = 0, DT_S32 millisec = 0, DT_S32 microsec = 0)
    {
        DT_S32 ms_carry  = 0;
        DT_S32 sec_carry = 0;

        ms_carry = microsec / 1000;
        us = microsec % 1000;

        sec_carry = (millisec + ms_carry) / 1000;
        ms = (millisec + ms_carry) % 1000;

        sec = second + sec_carry;
    }

    /**
     * @brief Static function to get the current time.
     *
     * @return A Time object representing the current time.
     */
    static Time Now()
    {
        Time time;
        time.Update();
        return time;
    }

    /**
     * @brief Update the Time object with the current time.
     */
    DT_VOID Update()
    {
        auto now = std::chrono::steady_clock::now().time_since_epoch();
        std::chrono::seconds seconds = std::chrono::duration_cast<std::chrono::seconds>(now);
        sec = static_cast<DT_S64>(seconds.count());
        DT_S32 microseconds = static_cast<DT_S32>(std::chrono::duration_cast<std::chrono::microseconds>(now - seconds).count());
        ms = microseconds / 1000;
        us = microseconds - ms * 1000;
    }

    /**
     * @brief Get the time in seconds.
     *
     * @return The time in seconds as a floating-point number.
     */
    DT_F64 AsSec()
    {
        return sec + ms / 1000.0 + us / 1000000.0;
    }

    /**
     * @brief Get the time in milliseconds.
     *
     * @return The time in milliseconds as a floating-point number.
     */
    DT_F64 AsMilliSec()
    {
        return sec * 1000 + ms + us / 1000.0;
    }

    /**
     * @brief Get the time in microseconds.
     *
     * @return The time in microseconds as a 64-bit integer.
     */
    DT_S64 AsMicroSec()
    {
        return sec * 1000000 + ms * 1000 + us;
    }

    /**
     * @brief Overloaded stream insertion operator for Time structure.
     *
     * @param os The output stream.
     * @param time The Time object.
     *
     * @return The output stream with Time information.
     */
    AURA_EXPORTS friend std::ostream& operator<<(std::ostream &os, const Time &time)
    {
        if (time.sec)
        {
            os << time.sec << "." << std::setfill('0') << std::setw(3) << time.ms << "s";
        }
        else
        {
            os << time.ms  << "." << std::setfill('0') << std::setw(3) << time.us << "ms";
        }
        return os;
    }

    /**
     * @brief Convert the Time object to a string representation.
     *
     * @return The string representation of the Time object.
     */
    std::string ToString() const
    {
        std::stringstream ss;
        ss << (*this);
        return ss.str();
    }

    static std::string ToDate()
    {
        auto now = std::chrono::system_clock::now();
        auto cur_time = std::chrono::system_clock::to_time_t(now);
        std::tm *local_time = std::localtime(&cur_time);
        auto microseconds = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() % 1000;

        std::ostringstream ss;
        ss << std::put_time(local_time, "%Y-%m-%d-%H:%M:%S") << ":" << std::setfill('0') << std::setw(3) << microseconds;
        return ss.str();
    }

    DT_S64 sec;  /*!< The second component of the time. */
    DT_S32 ms;   /*!< The millisecond component of the time. */
    DT_S32 us;   /*!< The microsecond component of the time. */
};

/**
 * @brief Equality comparison operator for Time objects.
 *
 * @param t0 The first Time object.
 * @param t1 The second Time object.
 *
 * @return `true` if the Time objects are equal, `false` otherwise.
 */
AURA_INLINE DT_BOOL operator==(const Time &t0, const Time &t1)
{
    if (t0.sec == t1.sec && t0.ms == t1.ms && t0.us == t1.us)
    {
        return DT_TRUE;
    }

    return DT_FALSE;
}

/**
 * @brief Inequality comparison operator for Time objects.
 *
 * @param t0 The first Time object.
 * @param t1 The second Time object.
 *
 * @return `true` if the Time objects are not equal, `false` otherwise.
 */
AURA_INLINE DT_BOOL operator!=(const Time &t0, const Time &t1)
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
AURA_INLINE DT_BOOL operator>(const Time &t0, const Time &t1)
{
    if (t0.sec != t1.sec)
    {
        return (t0.sec > t1.sec) ? DT_TRUE : DT_FALSE;
    }

    if (t0.ms != t1.ms)
    {
        return (t0.ms > t1.ms) ? DT_TRUE : DT_FALSE;
    }

    return (t0.us > t1.us) ? DT_TRUE : DT_FALSE;
}

/**
 * @brief Greater than or equal to comparison operator for Time objects.
 *
 * @param t0 The first Time object.
 * @param t1 The second Time object.
 *
 * @return `true` if t0 is greater than or equal to t1, `false` otherwise.
 */
AURA_INLINE DT_BOOL operator>=(const Time &t0, const Time &t1)
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
AURA_INLINE DT_BOOL operator<(const Time &t0, const Time &t1)
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
AURA_INLINE DT_BOOL operator<=(const Time &t0, const Time &t1)
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

    DT_S32 ms_carry  = 0;
    DT_S32 sec_carry = 0;

    // us
    DT_S64 sum_us = t0.us + t1.us;
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
    DT_S64 sum_ms = t0.ms + t1.ms + ms_carry;
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
 * @brief Class representing a timestamp.
 */
class AURA_EXPORTS TimeStamp
{
public:
    /**
     * @brief Default constructor.
     *
     * @param mon The month (1 ~ 12).
     * @param day The day (1 ~ 31).
     * @param hour The hour (0 ~ 23).
     * @param min The minute (0 ~ 59).
     * @param sec The second (0 ~ 59).
     * @param ms The millisecond (0 ~ 999).
     * @param us The microsecond (0 ~ 999).
     */
    TimeStamp( DT_S8 mon = 0, DT_S8 day = 0, DT_S8 hour = 0, DT_S8 min = 0, DT_S8 sec = 0, DT_S16 ms = 0, DT_S16 us = 0)
             : m_mon(mon), m_day(day), m_hour(hour), m_min(min), m_sec(sec), m_ms(ms), m_us(us)
    {}

    /**
     * @brief Update the timestamp to the current time.
     *
     * This function updates the timestamp based on the current system time.
     */
    DT_VOID Update()
    {
#if defined(AURA_BUILD_HEXAGON)
        // TODO: add cdsp time stamp
#elif defined(AURA_BUILD_HOST)
        time_t t = time(NULL);
        struct tm *tm = localtime(&t);

        m_mon  = static_cast<DT_S8>(tm->tm_mon + 1);
        m_day  = static_cast<DT_S8>(tm->tm_mday);
        m_hour = static_cast<DT_S8>(tm->tm_hour);
        m_min  = static_cast<DT_S8>(tm->tm_min);
        m_sec  = static_cast<DT_S8>(tm->tm_sec);
#endif

        Time now = Time::Now();
        m_ms = now.ms;
        m_us = now.us;
    }

    /**
     * @brief Get the current timestamp.
     *
     * @return A TimeStamp object representing the current timestamp.
     */
    static TimeStamp Now()
    {
        TimeStamp ts;
        ts.Update();
        return ts;
    }

    /**
     * @brief Output the timestamp in the format: "MM-DD HH:mm:ss.SSS".
     *
     * @param os The output stream.
     * @param time_stamp The timestamp to be output.
     *
     * @return The output stream.
     */
    AURA_EXPORTS friend std::ostream& operator<<(std::ostream& os, const TimeStamp &time_stamp)
    {
        os        << std::setfill('0') << std::setw(2) << (DT_S32)(time_stamp.m_mon);
        os << "-" << std::setfill('0') << std::setw(2) << (DT_S32)(time_stamp.m_day);
        os << " " << std::setfill('0') << std::setw(2) << (DT_S32)(time_stamp.m_hour);
        os << ":" << std::setfill('0') << std::setw(2) << (DT_S32)(time_stamp.m_min);
        os << ":" << std::setfill('0') << std::setw(2) << (DT_S32)(time_stamp.m_sec);
        os << "." << std::setfill('0') << std::setw(3) << (DT_S32)(time_stamp.m_ms);

        return os;
    }

    /**
     * @brief Convert the timestamp to a string representation.
     *
     * @return A string representing the timestamp.
     */
    std::string ToString() const
    {
        std::stringstream ss;
        ss << (*this);
        return ss.str();
    }

private:
    DT_S8  m_mon;  /*!< Month (1 ~ 12). */
    DT_S8  m_day;  /*!< Day (1 ~ 31). */
    DT_S8  m_hour; /*!< Hour (0 ~ 23). */
    DT_S8  m_min;  /*!< Minute (0 ~ 59). */
    DT_S8  m_sec;  /*!< Second (0 ~ 59). */
    DT_S16 m_ms;   /*!< Millisecond (0 ~ 999). */
    DT_S16 m_us;   /*!< Microsecond (0 ~ 999). */
};

/**
 * @}
*/
} // namespace aura

#endif // AURA_RUNTIME_CORE_TIME_HPP__
