//
// Created by frewen on 22-10-10.
//
#include "aura/utils/core.h"
#include "aura/utils/logger.h"
#include "gtest/gtest.h"

#define TAG "TestAllocator"

using namespace aura::utils;

class TestAllocator : public testing::Test {
public:
    static void SetUpTestSuite() {
        ALOGD(TAG, "SetUpTestSuite");
    }

    static void TearDownTestSuite() {
        ALOGD(TAG, "TearDownTestSuite");
    }
};

TEST_F(TestAllocator, testBGR2Yuv444Planer_UV) {

}

TEST_F(TestAllocator, testBRG2nv21) {

}

TEST_F(TestAllocator, testbgr2Yuv444SemiPlanar) {

}
