#include "speed_profiler.h"
#include <algorithm>
#include <iomanip>
#include <iostream>
#include "util/speed_perf_util.h"

namespace xperf {

int SpeedProfiler::_k_test_times = 300;
static int k_log_batch_size = 50;
static const char* TAG = "SpeedProfiler";

void SpeedProfiler::profile(TestFuncList& func_list, const TestFunc& setup_func, const TestFunc& clean_func,
        std::vector<SpeedResult>& speed_profile) {
    if (func_list.empty()) {
        return;
    }

    std::cout << std::setw(30) << std::setfill('=') << " " << std::endl;
    std::cout << "Xperf Speed Profiler Starts..." << std::endl;
    std::cout << "Test function number: " << func_list.size() << std::endl;
    std::cout << "Test times: " << _k_test_times << std::endl;
    std::cout << std::setw(30) << std::setfill('=') << " " << std::endl;

    auto func_num = func_list.size();
    speed_profile.resize(func_num);
    std::vector<std::vector<long>> test_record(func_num);
    std::vector<long> total_durations(func_num);

    for (int i = 0; i < _k_test_times; ++i) {
        if (setup_func) {
            setup_func();
        }

        int index = 0;
        for (const auto& func : func_list) {
            long duration;
            {
                TIME_PERF(duration);
                func.first();
            }

            total_durations[index] += duration;
            test_record[index].emplace_back(duration);
            index++;
        }

        if ((i + 1) % k_log_batch_size == 0) {
            std::cout << std::endl;
            for (int idx = 0; idx < func_num; ++idx) {
                std::cout << "[" << TAG << "] func=" << func_list[idx].second
                          << ", batch=" << (i + 1) / k_log_batch_size
                          << ", avg_duration=" << total_durations[idx] / (i + 1) << " ms"
                          << std::endl;
            }
        }

        if (clean_func) {
            clean_func();
        }
    }

    int index = 0;
    for (const auto& dur : total_durations) {
        long avg_dur = dur / _k_test_times;
        auto func_tag = func_list[index].second;
        speed_profile[index++] = {avg_dur, func_tag};
        save_results(func_tag, test_record[index]);
    }

    print_results(speed_profile);
}

void SpeedProfiler::print_results(std::vector<SpeedResult>& speed_profile) {
    std::cout << "\nXperf Speed profile results:" << std::endl;
    std::cout << std::setw(80) << std::setfill('=') << " " << std::endl;
    for (auto& profile : speed_profile) {
        std::cout << std::left << std::setw(40) << std::setfill(' ') << profile.second
                  << "time: " << std::setw(3) << std::setfill(' ') << profile.first << " ms\t"
                  << "fps: " << std::setw(3) << 1000.f/profile.first
                  << std::endl;
    }
    std::cout << std::right << std::setw(80) << std::setfill('=') << " " << std::endl << std::endl;
}

void SpeedProfiler::save_results(std::string tag, std::vector<long>& records) {
    // todo
}

} // namespace xperf