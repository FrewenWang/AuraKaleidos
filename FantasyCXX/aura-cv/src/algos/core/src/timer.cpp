#include "aura/algos/core/timer.hpp"

namespace aura
{

Timer::Timer() : m_end_time(INT64_MAX)
{}

#if defined(AURA_BUILD_HEXAGON)
Timer::Timer(const Time &end_time, const Time &host_base_time)
             : m_end_time(end_time), m_host_base_time(host_base_time),
               m_hexagon_base_time(Time::Now())
{}
#endif // AURA_BUILD_HEXAGON

Time Timer::Now() const
{
#if defined(AURA_BUILD_HOST)
    return Time::Now();
#elif defined(AURA_BUILD_HEXAGON)
    Time diff = Time::Now() - m_hexagon_base_time;
    return diff + m_host_base_time;
#endif
}

#if defined(AURA_BUILD_HOST)
DT_VOID Timer::SetTimeout(DT_S32 timeout_ms)
{
    m_end_time = Time::Now() + Time(0, timeout_ms);
}
#endif // AURA_BUILD_HOST

DT_BOOL Timer::IsTimedOut() const
{
    return Now() >= m_end_time;
}

} // namespace aura