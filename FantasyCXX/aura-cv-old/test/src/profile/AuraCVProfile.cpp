//
// Created by Frewen.Wang on 2022/8/7.
//

#include "AuraCVProfile.h"
#include <cassert>
#include <iomanip>
#include <iostream>

namespace aura::aura_cv {

int AuraCVProfile::_kTestTimes = 10;

void AuraCVProfile::profile(AuraCVProfile::TestFuncList &funcList, const AuraCVProfile::TestFunc &setupFunc,
                            const AuraCVProfile::TestFunc &cleanFunc, std::vector<SpeedResult> &speedProfile,
                            std::vector<OutputResult> &outputProfile) {
    if (funcList.empty()) {
        return;
    }
    // std::setw使用: https://www.runoob.com/w3cnote/cpp-func-setw.html
    // n 表示宽度，用数字表示。setw() 函数只对紧接着的输出产生作用。
    // 当后面紧跟着地输出字段长度小于 n 的时候，在该字段前面用空格补齐，当输出字段长度大于 n 时，全部整体输出。
    std::cout << std::setw(30) << std::setfill('=') << " " << std::endl;
    std::cout << "AuraCV Profiler Starts..." << std::endl;
    std::cout << "Test function number: " << funcList.size() << std::endl;
    std::cout << "Test times: " << _kTestTimes << std::endl;
    std::cout << std::setw(30) << std::setfill('=') << " " << std::endl;
}

void AuraCVProfile::printResults(vector<SpeedResult> &speedResult, vector<OutputResult> &outputResult) {
    // TODO
}

void AuraCVProfile::saveResults(std::string tag, std::vector<long> &records) {
    // TODO
}


}
