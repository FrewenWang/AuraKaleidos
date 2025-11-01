//
// Created by LiWendong on 2023-03-15.
//
#pragma once

#include <functional>
#include <iostream>
#include <string>
#include <vector>

namespace aura::vision {
namespace op {

class OpProfiler {

public:
    using TestFunc = std::function<std::vector<double>()>;
    using TestFuncInfo = std::pair<TestFunc, std::string>;
    using TestFuncList = std::vector<TestFuncInfo>;
    using SpeedResult = std::pair<std::vector<double>, std::string>;
    using OutputResult = std::pair<std::vector<double>, std::string>;
    static void profile(TestFuncList &func_list,
                        const TestFunc &setup_func,
                        const TestFunc &clean_func,
                        std::vector<SpeedResult> &speed_profile,
                        std::vector<OutputResult> &output_profile
    );

private:
    static void print_results(std::vector<SpeedResult> &speed_result,
                              std::vector<OutputResult> &output_result);
    static void save_results(std::string tag, std::vector<long> &records);
    static int _k_test_times;

};

} // namespace op
} // namespace aura::vision