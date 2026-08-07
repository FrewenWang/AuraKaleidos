//
// Created by Frewen.Wang on 2024/10/24.
//
#include "aura/aura_utils/utils/AuraLog.h"
#include "aura/aura_utils/utils/FileUtil.h"
#include "aura/aura_utils/utils/StringUtil.h"
#include "gtest/gtest.h"
#include <Eigen/Dense>

const static char *TAG = "TestHPCBasic";

// 使用Eigen命名空间
using namespace Eigen;
using namespace std;
using namespace aura::utils;

class TestHPCBasic : public testing::Test {
public:
	
	static void SetUpTestSuite() {
		ALOGE(TAG, "SetUpTestSuite");

	}
	
	static void TearDownTestSuite() {
		ALOGE(TAG, "TearDownTestSuite");
	}
};


TEST_F(TestHPCBasic, TestHPCBasicHelloWorld) {

}
