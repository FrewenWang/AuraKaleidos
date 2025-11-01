#ifndef VISION_SPEED_PROFILER_H
#define VISION_SPEED_PROFILER_H

#include <functional>
#include <vector>

namespace xperf {

class SpeedProfiler {
public:
    using TestFunc = std::function<void()>;
    using TestFuncInfo = std::pair<TestFunc, std::string>;
    using TestFuncList = std::vector<TestFuncInfo>;
    using SpeedResult = std::pair<long, std::string>;
    static void profile(TestFuncList& func_list, const TestFunc& setup_func, const TestFunc& clean_func, std::vector<SpeedResult>& speed_profile);

private:
    static void print_results(std::vector<SpeedResult>& result);
    static void save_results(std::string tag, std::vector<long>& records);
    static int _k_test_times;
};

} // namespace xperf

#endif //VISION_SPEED_PROFILER_H
