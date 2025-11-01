//
// Created by Frewen.Wang on 2022/8/7.
//
#pragma once

#include <functional>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

namespace aura::aura_cv {

class AuraCVProfile {
public:
    // TODO 这个地方的function需要学习，C++14才有的功能
    using TestFunc = std::function<std::vector<double>()>;
    using TestFuncInfo = std::pair<TestFunc, std::string>;
    using TestFuncList = std::vector<TestFuncInfo>;
    using SpeedResult = std::pair<std::vector<double>, std::string>;
    using OutputResult = std::pair<std::vector<double>, std::string>;

    static void profile(TestFuncList &funcList,
                        const TestFunc &setupFunc,
                        const TestFunc &cleanFunc,
                        std::vector<SpeedResult> &speedProfile,
                        std::vector<OutputResult> &outputProfile);

private:
    /**
     * @brief 打印测试结果
     * @param speedResult  性能测试结果
     * @param outputResult  输出测试结果
     */
    static void printResults(std::vector<SpeedResult> &speedResult,
                             std::vector<OutputResult> &outputResult);

    /**
     * @brief 保存测试结果
     * @param tag
     * @param records
     */
    static void saveResults(std::string tag, std::vector<long> &records);

    static int _kTestTimes;
};

} // namespace aura::aura_cv
