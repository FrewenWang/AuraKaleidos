//
// Created by Frewen.Wang on 2022/11/20.
//
#include <iostream>
#include <vector>
#include <algorithm>
#include <functional> // 包含 std::greater
#include "aura/aura_utils/utils/AuraLog.h"
#include "gtest/gtest.h"

const static char *TAG = "TestC++Const";

using namespace std;

/**
 * 文章参考：
 * https://zhaoxuhui.top/blog/2019/08/21/eigen-note-1.html#1%E5%BA%93%E7%9A%84%E5%AE%89%E8%A3%85
 */
class TestCXXConst : public testing::Test {
public:
	static void SetUpTestSuite() {
		ALOGD(TAG, "SetUpTestSuite");
	}
	
	static void TearDownTestSuite() {
		ALOGD(TAG, "TearDownTestSuite");
	}
};

TEST_F(TestCXXConst, testStdGreater) {
	ALOGD(TAG, "============== testStdGreater ==============");
	std::vector<int> numbers = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5};
	// 使用 std::greater 进行降序排序
	std::sort(numbers.begin(), numbers.end(), std::greater<int>());
	
	// 输出排序后的结果
	for (int number: numbers) {
		std::cout << number << " ";
	}
}

