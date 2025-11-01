//
// Created by Frewen.Wang on 2022/8/7.
//
#include "aura/utils/core.h"
#include "profile/AuraCVProfile.h"
#include "gtest/gtest.h"
#include <iostream>

using namespace std;
using namespace aura::aura_cv;

int main(int argc, char **argv) {
    ALOGD("TAG", "[AuraCV] Hello AuraCVUnitTester");


    testing::InitGoogleTest(&argc, argv);
    if (RUN_ALL_TESTS() == 0) {
        return 0;
    }
    return 0;
}