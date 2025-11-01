//
// Created by Frewen.Wang on 2022/11/20.
//
#include <iostream>
#include "aura/aura_utils/utils/AuraLog.h"
#include "gtest/gtest.h"
#include <stdio.h>
#include <string.h>

const static char *TAG = "TestCStrdup";

using namespace std;

/**
 * 文章参考：https://blog.csdn.net/yzy1103203312/article/details/77651278
 * strdup() 和 strndup() 函数的原型分别为：
 * #include <string.h>
 * char *strdup(const char *s);
 * char *strndup(const char *s, size_t n);
 * strdup() 函数将参数 s 指向的字符串复制到一个字符串指针上去，这个字符串指针事先可以没被初始化。
 * 在复制时，strdup() 会给这个指针分配空间，使用 malloc() 函数进行分配，如果不再使用这个指针，相应的用 free() 来释放掉这部分空间。
 * strndup() 函数只复制前面 n 个字符。
 */
class TestCStrdup : public testing::Test {
public:
  static void SetUpTestSuite() {
    ALOGD(TAG, "SetUpTestSuite");
  }

  static void TearDownTestSuite() {
    ALOGD(TAG, "TearDownTestSuite");
  }
};

TEST_F(TestCStrdup, testCStrdup) {
  char *s = "hello world";
  char *p;

  p = strdup(s);
  printf("%s\n", p);
}
