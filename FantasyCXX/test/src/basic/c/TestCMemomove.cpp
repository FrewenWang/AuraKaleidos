//
// Created by Frewen.Wang on 2022/11/20.
//
#include <iostream>
#include <vector>
#include <algorithm>
#include <functional> // 包含 std::greater
#include "aura/aura_utils/utils/AuraLog.h"
#include "gtest/gtest.h"
#include <stdio.h>
#include <string.h>

const static char *TAG = "TestCMemomove";

using namespace std;

/**
 * C 库函数 void *memmove(void *str1, const void *str2, size_t n) 从 str2 复制 n 个字符到 str1，
 * 但是在重叠内存块这方面，memmove() 是比 memcpy() 更安全的方法。
 * 如果目标区域和源区域有重叠的话，memmove() 能够保证源串在被覆盖之前将重叠区域的字节拷贝到目标区域中，复制后源区域的内容会被更改。如果目标区域与源区域没有重叠，则和 memcpy() 函数功能相同。
 * 文章参考：https://www.runoob.com/cprogramming/c-function-memmove.html
 *
 *
 */
class TestCMemomove : public testing::Test {
public:
  static void SetUpTestSuite() {
    ALOGD(TAG, "SetUpTestSuite");
  }

  static void TearDownTestSuite() {
    ALOGD(TAG, "TearDownTestSuite");
  }
};

TEST_F(TestCMemomove, testCMemomove) {
  const char dest[] = "oldstring";
  const char src[] = "newstring";

  printf("Before memmove dest = %s, src = %s\n", dest, src);
  memmove((void *)dest, src, 9);
  printf("After memmove dest = %s, src = %s\n", dest, src);


  const char dest2[] = "oldstring";
  const char src2[] = "newstring";

  printf("Before memmove dest = %s, src = %s\n", dest2, src2);
  memmove((void *)dest2, src2, 5);
  printf("After memmove dest = %s, src = %s\n", dest2, src2);
}
