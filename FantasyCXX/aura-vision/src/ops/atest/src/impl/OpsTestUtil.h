#pragma once

#include <chrono>
#include <string>

namespace aura::vision {
namespace op {

#define TIME_PERF(duration) OpsPerfUtil perf(&duration);

class OpsPerfUtil {
public:
    explicit OpsPerfUtil(long *duration)
        : _duration(duration) {
        _start = std::chrono::high_resolution_clock::now();
    }

    ~OpsPerfUtil() {
        auto end = std::chrono::high_resolution_clock::now();
        *_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - _start).count();
    }

private:
    long *_duration;
    std::chrono::high_resolution_clock::time_point _start;
};

} // namespace op
} // namespace aura::vision