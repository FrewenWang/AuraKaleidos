//
// Created by Frewen.Wang on 2022/11/20.
//
#include "aura/aura_utils/utils/AuraLog.h"
#include "gtest/gtest.h"
#include <unistd.h>
#include <vector>
#include <chrono>   //计算时间

#include <random>
#include <Eigen/Dense>       // Eigen 矩阵库
#include <ceres/ceres.h>     // Ceres 优化库

const static char *TAG = "TestCubicCurveFitting";
using namespace std;
using namespace std::chrono;
/**
 *
 */
class TestMemoryCache : public testing::Test {
public:
  static void SetUpTestSuite() {
    ALOGD(TAG, "SetUpTestSuite");
  }

  static void TearDownTestSuite() {
    ALOGD(TAG, "TearDownTestSuite");
  }
};


TEST_F(TestMemoryCache, testMemoryCache) {
  ALOGD(TAG, "============== testMemoryCache ==============");
  constexpr int N = 1000;
  // 这个为什么报错？
  int num[N][N] = {0};

  // 形式一
  auto starttime = system_clock::now();
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      num[i][j] = 0;
    }
  }
  auto cost = std::chrono::duration_cast<std::chrono::microseconds>(system_clock::now() - starttime).count();
  ALOGD(TAG, "二维数组[i][j]计算耗时：%ld", cost);

  // 形式二
  starttime = system_clock::now();
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      num[j][i] = 0;
    }
  }
  cost = std::chrono::duration_cast<std::chrono::microseconds>(system_clock::now() - starttime).count();
  ALOGD(TAG, "二维数组[j][i]计算耗时：%ld", cost);
}

TEST_F(TestMemoryCache, testInstructionCache) {
  ALOGD(TAG, "============== testInstructionCache ==============");
  constexpr int N = 1000;
  int nums[N];

  for (int &num: nums) {
    num = rand() % 100;
  }

  // 操作一：数组遍历
  for (int i = 0; i < N; i++) {
    if (nums[i] < 50);
  }
  // 操作二： 针对数组进行排序
  sort(nums, nums + N);
}
