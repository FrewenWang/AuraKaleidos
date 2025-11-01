//
// Created by Frewen.Wang on 2022/11/20.
//
#include <iostream>
#include <vector>
#include <algorithm>
#include <functional> // 包含 std::greater
#include "aura/aura_utils/utils/AuraLog.h"
#include "gtest/gtest.h"
#include <iostream>
#include <iomanip>
#include <ctime>
#include <cmath>

const static char *TAG = "TestC++Basic";

using namespace std;

/**
 * 文章参考：
 */
class TestFormateTime : public testing::Test {
public:
	static void SetUpTestSuite() {
		ALOGD(TAG, "SetUpTestSuite");
	}
	
	static void TearDownTestSuite() {
		ALOGD(TAG, "TearDownTestSuite");
	}
};

TEST_F(TestFormateTime, testStrftime) {
	ALOGD(TAG, "============== testStrftime ==============");
	 // 假设您有一个时间戳（以秒为单位，带小数）
    double timestamp = 1730348415.982902; // 示例时间戳

    // 将时间戳的整数部分转换为std::time_t
    std::time_t int_part = static_cast<std::time_t>(std::floor(timestamp));

    // 将时间戳的小数部分提取出来
    double fractional_part = timestamp - int_part;

    // 将时间戳转换为tm结构
    std::tm* tm_struct = std::localtime(&int_part);

    // 创建一个缓冲区来存储格式化的时间字符串
    char buffer[80];

    // 格式化tm结构为日期和时间字符串
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", tm_struct);

    // 输出格式化的时间字符串，并添加小数秒部分
    std::cout << "Formatted Date and Time: " << buffer
              << "." << std::setw(3) << std::setfill('0') << static_cast<int>(fractional_part * 1000)
              << std::endl;
}

