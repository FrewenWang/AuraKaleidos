#ifndef VISION_SYSTEM_CLOCK_H
#define VISION_SYSTEM_CLOCK_H

#include <cstdint>
#include <string>
#include <chrono>

namespace aura::vision {
    class SystemClock {
    public:
        /**
         * 获取当前系统时间戳
         * @return 当前时间戳
         */
        static int64_t nowMillis();

        static std::int64_t nowMicroseconds();

        static std::string currentTimeStr();

        static std::chrono::time_point<std::chrono::system_clock, std::chrono::nanoseconds> now();


        static std::chrono::time_point<std::chrono::high_resolution_clock> high_now();

        static int64_t duraMillis(std::chrono::time_point<std::chrono::system_clock, std::chrono::nanoseconds> t2,
                                  std::chrono::time_point<std::chrono::system_clock, std::chrono::nanoseconds> t1);
    };
} // namespace aura::vision

#endif //VISION_SYSTEM_CLOCK_H
