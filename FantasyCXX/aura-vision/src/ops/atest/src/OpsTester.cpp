#include <iostream>

#include "vision/util/log.h"
#include "profile/OpProfiler.h"
#include "impl/TestMemory.h"

using namespace aura::vision;
using namespace op;
using namespace std;

int main(int argc, char **argv) {

    std::vector<OpProfiler::SpeedResult> speed_result;
    std::vector<OpProfiler::OutputResult> output_result;
    OpProfiler::TestFuncList sFunctionList{
        {TestMemory::testMemcpy, "testMemcpy"},
    };

    OpProfiler::profile(sFunctionList, nullptr, nullptr, speed_result, output_result);
    return 0;
}