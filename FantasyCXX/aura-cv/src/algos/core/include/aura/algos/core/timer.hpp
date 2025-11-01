
#ifndef AURA_ALGOS_CORE_TIMER_HPP__
#define AURA_ALGOS_CORE_TIMER_HPP__

#include "aura/runtime/core.h"

namespace aura
{

class AURA_EXPORTS Timer
{
public:
    Timer();
#if defined(AURA_BUILD_HEXAGON)
    Timer(const Time &end_time, const Time &host_base_time);
#endif // AURA_BUILD_HEXAGON

    Time Now() const;
    MI_BOOL IsTimedOut() const;
#if defined(AURA_BUILD_HOST)
    AURA_VOID SetTimeout(MI_S32 timeout_ms);
#endif // AURA_BUILD_HOST

    Time m_end_time;
#if defined(AURA_BUILD_HEXAGON)
    Time m_host_base_time;
    Time m_hexagon_base_time;
#endif // AURA_BUILD_HEXAGON
};

} // namespace aura

#endif // AURA_ALGOS_CORE_TIMER_HPP__