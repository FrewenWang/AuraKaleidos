//
// Created by Frewen.Wang on 2022/8/7.
//
#include "aura/aura_utils/utils/AuraLog.h"
#include "impl/TestCrop.h"
#include "profile/AuraCVProfile.h"
#include "gtest/gtest.h"
#include <iostream>

using namespace std;
using namespace aura::aura_cv;

int main(int argc, char **argv) {
    ALOGD("TAG", "[AuraCV] Hello AuraCVUnitTester");

    std::vector<AuraCVProfile::SpeedResult> speedResult;
    std::vector<AuraCVProfile::OutputResult> outputResult;

    // AuraCVProfile::TestFuncList funcList{
    //         {TestCrop::test_crop_hwc_5x5, "test_crop_hwc_5x5"},
    //
    // };

    testing::InitGoogleTest(&argc, argv);
    if (RUN_ALL_TESTS() == 0) {
        return 0;
    }
    return 0;
}