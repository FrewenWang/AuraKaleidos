//
// Created by Frewen.Wang on 2022/11/20.
//
#include <iostream>
#include <vector>
#include <algorithm>
#include <functional> // 包含 std::greater
#include "aura/aura_utils/utils/AuraLog.h"
#include "gtest/gtest.h"

const static char *TAG = "TestCXXVector";

using namespace std;

/**
 * 文章参考：
 */
class TestCXXVector : public testing::Test {
public:
	static void SetUpTestSuite() {
		ALOGD(TAG, "SetUpTestSuite");
	}
	
	static void TearDownTestSuite() {
		ALOGD(TAG, "TearDownTestSuite");
	}
};

TEST_F(TestCXXVector, testVectorWorld) {
	ALOGD(TAG, "============== testVectorWorld ==============");
	std::vector<std::string> vec;
	std::string s = "hello";
	vec.push_back(s);       // 明确：拷贝 s 到容器
	vec.emplace_back("world"); // 明确：直接构造字符串
	vec.emplace_back(s); // 明确：直接构造字符串


}

