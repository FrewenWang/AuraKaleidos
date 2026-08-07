//
// Created by Frewen.Wang on 2022/11/20.
//
#include <iostream>
#include <vector>
#include <algorithm>
#include <functional> // 包含 std::greater
#include "aura/aura_utils/utils/AuraLog.h"
#include "gtest/gtest.h"

const static char *TAG = "TestLambda";

using namespace std;

/**
 * 文章参考：
 * https://blog.csdn.net/gongjianbo1992/article/details/105128849
 */
class TestLambda : public testing::Test {
public:
  static void SetUpTestSuite() {
    ALOGD(TAG, "SetUpTestSuite");
  }

  static void TearDownTestSuite() {
    ALOGD(TAG, "TearDownTestSuite");
  }
};

TEST_F(TestLambda, testLambda) {
  ALOGD(TAG, "============== testLambda ==============");

  // 定义一个lambda表达式
  auto f = [](auto a, int b = 10) {
    std::cout << a << " " << b << std::endl;
  };
  // 当以 auto 为形参类型时，该 lambda 为泛型 lambda(C++14 起)
  // 调用一个lambda时给定的实参被用来初始化lambda的形参。
  // 所以我们传入实参1.5的时候，这个时候auto a其实就会被初始化成为 float a
  // 所以输出结果:1.5, 2
  f(1.5, 2);
  // 所以我们传入实参true的时候，这个时候auto a其实就会被初始化成为 bool a
  // 而且后面的int b没有传入实参，所以我们按照默认值进行初始化
  // 所以输出结果:1, 10
  f(true);
}

TEST_F(TestLambda, testLambdaValueCatch) {
  ALOGD(TAG, "============== testLambdaValueCatch ==============");
  int i = 10, j = 10;
  //加上mutable才可以在lambda函数中改变捕获的变量值
  // 不加mutable，不能在lambda函数体中修改i的值，但是无论加不加mutable, 都可以修改j的值，而且可以直接改变外部的j的值
  auto f = [i, &j]() mutable {
    i = 100, j = 100;
  };
  i = 0, j = 0;
  f();
  //输出:0 100
  std::cout << i << " " << j << std::endl;
}

TEST_F(TestLambda, testLambdaAll) {
  ALOGD(TAG, "============== testLambdaAll ==============");
  std::vector<int> c = {1, 2, 3, 4, 5, 6, 7};
  int x = 5;
  // remove_if的参数是迭代器，前两个参数表示迭代的起始位置和这个起始位置所对应的停止位置。
  // 最后一个参数：传入一个回调函数(可以传入lanme打标识)，如果回调函数返回为真，则将当前所指向的参数移到尾部。
  c.erase(std::remove_if(c.begin(), c.end(), [x](int n) { return n < x; }), c.end());

  std::cout << "c: ";
  std::for_each(c.begin(), c.end(), [](int i) { std::cout << i << ' '; });
  std::cout << '\n';

  // 闭包的类型不能被指名，但可用 auto 提及
  // C++14 起，lambda 能拥有自身的默认实参
  auto func1 = [](int i = 6) { return i + 4; };
  std::cout << "func1: " << func1() << '\n';

  // 与所有可调用对象相同，闭包能可以被捕获到 std::function 之中
  // （这可能带来不必要的开销）
  std::function<int(int)> func2 = [](int i) { return i + 4; };
  std::cout << "func2: " << func2(6) << '\n';
}
