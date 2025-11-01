#ifndef VISION_AUTO_PERF_H
#define VISION_AUTO_PERF_H

#include <chrono>
#include <string>

namespace xperf {

#define TIME_PERF(duration) SpeedPerfUtil perf(duration)

class SpeedPerfUtil {
public:
    explicit SpeedPerfUtil(long& duration)
        : _duration(&duration) {
        _start = std::chrono::high_resolution_clock::now();
    }

    ~SpeedPerfUtil() {
        auto end = std::chrono::high_resolution_clock::now();
        *_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - _start).count();
    }

private:
    long* _duration;
    std::chrono::high_resolution_clock::time_point _start;
};

} // namespace xperf

#endif //VISION_AUTO_PERF_H
