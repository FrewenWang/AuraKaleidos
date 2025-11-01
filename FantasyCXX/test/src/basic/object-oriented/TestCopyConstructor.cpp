//
// Created by Frewen.Wang on 2022/11/20.
//
#include <iostream>
#include <vector>
#include <algorithm>
#include <functional> // 包含 std::greater
#include "aura/aura_utils/utils/AuraLog.h"
#include "gtest/gtest.h"
#include "Base.h"

const static char *TAG = "TestCopyConstructor";

using namespace std;

using namespace std;


class TestCopyConstructor : public testing::Test {
public:
  static void SetUpTestSuite() {
    ALOGD(TAG, "SetUpTestSuite");
  }

  static void TearDownTestSuite() {
    ALOGD(TAG, "TearDownTestSuite");
  }
};

TEST_F(TestCopyConstructor, testCopyConstructor) {
  ALOGD(TAG, "============== testCopyConstructor ==============");

  cout << "===================普通对象实例化=================" << endl;
  Base base1(10010, "张三", 18, 100);
  base1.input_data();
  base1.show_data();
  base1.display();
  // 直接实例化的对象使用完毕就会进行回收。调用类的析构函数
  // 而使用指针进行动态内存申请的对象。如果不调用delete 则对应不会回收

  cout << "===================指针方式类的实例化=================" << endl;

  Base *basePtr = new Base();
  basePtr->input_data();
  basePtr->show_data();
  // 使用指针进行动态内存申请的对象。如果不调用delete 则对应不会回收
  // delete指针对象。可以让对象调用析构函数，就加快资源释放。
  // delete stuPtr;


  cout << "===================指针方式类的含参数构造函数实例化=================" << endl;
  Base *baseParamPtr = new Base(1001, "李四", 20, 596);
  baseParamPtr->input_data();
  baseParamPtr->show_data();

  cout << "===================通过类的拷贝构造函数类进行实例化=================" << endl;
  Base student4 = *baseParamPtr;
  student4.display();
  // 使用拷贝构造函数实例化的对象使用完毕就会进行回收。调用类的析构函数
  // 而使用指针进行动态内存申请的对象。如果不调用delete 则对应不会回收
}
