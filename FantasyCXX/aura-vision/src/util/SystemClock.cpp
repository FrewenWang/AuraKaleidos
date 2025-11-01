#include "SystemClock.h"

#include <iomanip>
#include <sstream>

namespace aura::vision {

    using namespace std::chrono;

    std::int64_t SystemClock::nowMillis() {
        auto ts = system_clock::now().time_since_epoch();
        return duration_cast<milliseconds>(ts).count();
    }

    std::int64_t SystemClock::nowMicroseconds() {
        auto ts = system_clock::now().time_since_epoch();
        return duration_cast<microseconds>(ts).count();
    }

    std::string SystemClock::currentTimeStr() {
        auto now = system_clock::to_time_t(system_clock::now());

        std::stringstream ss;
        ss << std::put_time(std::localtime(&now), "%F %T");
        ss << "." << std::setfill('0') << std::setw(3) << static_cast<int>(nowMillis() % 1000);
        return ss.str();
    }

    time_point<system_clock, nanoseconds> SystemClock::now() {
        return system_clock::now();
    }

    std::chrono::time_point<std::chrono::high_resolution_clock> SystemClock::high_now() {
        return std::chrono::high_resolution_clock::now();
    }

    int64_t SystemClock::duraMillis(time_point<system_clock, nanoseconds> t2,
                                    time_point<system_clock, nanoseconds> t1) {
        return duration_cast<milliseconds>(t2 - t1).count();
    }
} // namespace aura::vision
