//
// Created by Frewen.Wang on 2022/11/20.
//
#include <iostream>
#include <vector>
#include <algorithm>
#include <functional> // 包含 std::greater
#include "aura/aura_utils/utils/AuraLog.h"
#include "gtest/gtest.h"

const static char *TAG = "TestCXXMemAlign";

using namespace std;

#include<stdio.h>
struct{
	int x;
	char y;
}s;

/**
 * 测试内存对齐的相关逻辑
 */
class TestCXXMemAlign : public testing::Test {
public:
	static void SetUpTestSuite() {
		ALOGD(TAG, "SetUpTestSuite");
	}
	
	static void TearDownTestSuite() {
		ALOGD(TAG, "TearDownTestSuite");
	}
};

TEST_F(TestCXXMemAlign, testHelloWorld) {
	// 上面的结构体应该占4+1=5byte
	// 但是实际上，通过运行程序得到的结果是8 byte
	printf("%lu\n", sizeof(s)); // 输出8
	printf("%lu\n", sizeof(s.x)); // 输出4
	printf("%lu\n", sizeof(s.y)); // 输出1
}

