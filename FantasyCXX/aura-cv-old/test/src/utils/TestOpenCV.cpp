
//
// Created by frewen on 22-10-10.
//
#include "aura/utils/core.h"
#include "gtest/gtest.h"
#include "opencv2/core/matx.hpp"

#define TAG "TestNEON"

using namespace aura::utils;

class TestOpenCV : public testing::Test {
public:
    static void SetUpTestSuite() {
        ALOGD(TAG, "TestNEON SetUpTestSuite");
    }
    
    static void TearDownTestSuite() {
        ALOGD(TAG, "TestNEON TearDownTestSuite");
    }
};

TEST_F(TestOpenCV, testCVVec3d) {
    cv::Vec3d vec(3, 1, 2); // 创建Vec3d向量
    // 对Vec3d向量进行排序
    std::sort(vec.val, vec.val + 3);
    // 打印排序后的Vec3d向量
    for (int i = 0; i < 3; i++) {
        std::cout << vec[i] << " ";
    }
}
