//
// Created by Frewen.Wang on 2024/10/24.
//
#include "aura/aura_utils/utils/AuraLog.h"
#include "aura/aad/kalman_filter/KalmanCVFilter.h"
#include "gtest/gtest.h"
#include <Eigen/Dense>

const static char *TAG = "TestKalmanCVFilter";

// 使用Eigen命名空间
using namespace Eigen;
using namespace std;

class TestKalmanCVFilter : public testing::Test {
public:
	static void SetUpTestSuite() {
		ALOGE(TAG, "SetUpTestSuite");
	}
	
	static void TearDownTestSuite() {
		ALOGE(TAG, "TearDownTestSuite");
	}
};


TEST_F(TestKalmanCVFilter, testKalmanCVFilterInit) {
	ALOGD(TAG, "============== testKalmanCVFilterInit ==============");
	aura::aad::KalmanCVFilter kalman_cv_filter;
	
	// 使用CV模型的KalmanFilter需要输入的观测状态：state[x, y, vx, vy]
	/// 第一步：我们来进行初始化默认的全局观测矩阵：
	Eigen::Matrix4d global_uncertainty = Eigen::Matrix4d::Identity();
	global_uncertainty(0, 0) = 300;
	global_uncertainty(1, 1) = kalman_cv_filter.GetUncertainty()(1, 1);
	global_uncertainty(2, 2) = 30;
	global_uncertainty(3, 3) = 5;
	
	auto ret = kalman_cv_filter.Init(kalman_cv_filter.GetStates(), global_uncertainty);
	ALOGE(TAG, "testKalmanCVFilterInit ret: %d", ret);
	
}