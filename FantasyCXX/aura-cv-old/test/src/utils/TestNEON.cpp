//
// Created by frewen on 22-10-10.
//
#include "aura/utils/core.h"
#include "gtest/gtest.h"

#define TAG "TestNEON"

using namespace aura::utils;

class TestNEON : public testing::Test {
public:
    static void SetUpTestSuite() {
        ALOGD(TAG, "TestNEON SetUpTestSuite");
    }
    
    static void TearDownTestSuite() {
        ALOGD(TAG, "TestNEON TearDownTestSuite");
    }
};

TEST_F(TestNEON, testBGR2Yuv444Planer_UV) {

}

TEST_F(TestNEON, testBRG2nv21) {

}

TEST_F(TestNEON, testbgr2Yuv444SemiPlanar) {

}
