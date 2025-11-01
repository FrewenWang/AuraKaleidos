

#ifndef VISION_NATIVE_COMMON_H
#define VISION_NATIVE_COMMON_H

#include <chrono>
#include <iostream>
#include <sstream>
#include <thread>

namespace aura::vision {
    /**
        * clock time
        * @param faceInfo
        * @param max_cols
        * @param min_cols
        */
    using Time = decltype(std::chrono::high_resolution_clock::now());

    inline Time time() {
        return std::chrono::high_resolution_clock::now();
    }

    inline double time_diff(Time t1, Time t2) {
        // 定义一个毫秒类型
        typedef std::chrono::microseconds ms;
        auto diff = t2 - t1;
        ms counter = std::chrono::duration_cast<ms>(diff);
        return counter.count() / 1000.0;
    }

    inline std::string get_tid() {
        // 获取线程 id
        std::stringstream tmp;
        tmp << std::this_thread::get_id();
        return tmp.str();
    }

} // namespace aura::vision
#endif //VISION_NATIVE_COMMON_H
