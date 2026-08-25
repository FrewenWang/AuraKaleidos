//
// Created by Frewen.Wang on 2022/11/20.
//
#include <iostream>
#include <vector>
#include <algorithm>
#include <functional> // 包含 std::greater
#include "aura/aura_utils/utils/AuraLog.h"
#include "gtest/gtest.h"
#include <string>
#include <sys/time.h>
#include <sstream>
#include <stdexcept>
#include <stdio.h>

#define OUT_IN_REPEATE_NUM 2
#define IN_REPEATE_NUM 1000

const static char *TAG = "TestStringAppend";

using namespace std;


string s1="abcedfg";
string s2="hijklmn";
string s3="opqrst";
void  plusTest(string& ret)
{
  for(int i=0; i<IN_REPEATE_NUM; i++)
  {
    ret += s1;
    ret += s2;
    ret += s3;
  }
}
void  appendTest(string& ret)
{
  for(int i=0; i<IN_REPEATE_NUM; i++)
  {
    ret.append(s1);
    ret.append(s2);
    ret.append(s3);
  }
}

void sprintfTest(string &ret) {
  const size_t itemLength = s1.length() + s2.length() + s3.length();
  vector<char> buffer(itemLength * IN_REPEATE_NUM + 1, '\0');
  size_t offset = 0;
  for (int i = 0; i < IN_REPEATE_NUM; i++) {
    const int written = snprintf(buffer.data() + offset, buffer.size() - offset,
                                 "%s%s%s", s1.c_str(), s2.c_str(), s3.c_str());
    if (written < 0 || static_cast<size_t>(written) != itemLength) {
      throw runtime_error("snprintf failed while building the test string");
    }
    offset += static_cast<size_t>(written);
  }
  ret.assign(buffer.data(), offset);
}

void  ssTest(string& ret)
{
  stringstream ss;
  for(int i=0; i<IN_REPEATE_NUM; i++)
  {
    ss<<s1;
    ss<<s2;
    ss<<s3;
  }
  ret = ss.str();
}

class TestStringAppend : public testing::Test {
public:
  static void SetUpTestSuite() {
    ALOGD(TAG, "SetUpTestSuite");
  }

  static void TearDownTestSuite() {
    ALOGD(TAG, "TearDownTestSuite");
  }
};

TEST_F(TestStringAppend, TestStringAppendBasic) {
  ALOGD(TAG, "============== TestStringAppend ==============");
  string ss, plus, append, sprintf;
  struct timeval sTime{}, eTime{};

  /// 计算string的+=元素运算符的耗时事件
  gettimeofday(&sTime, nullptr);
  for(int i=0; i<OUT_IN_REPEATE_NUM; i++)
  {
    plus="";
    plusTest(plus);
  }
  gettimeofday(&eTime, nullptr);
  long PlusTime = (eTime.tv_sec-sTime.tv_sec)*1000000+(eTime.tv_usec-sTime.tv_usec); //exeTime 单位是微秒


  /// 计算sprintf的耗时事件
  gettimeofday(&sTime, nullptr);
  for(int i=0; i<OUT_IN_REPEATE_NUM; i++)
  {
    sprintf="";
    sprintfTest(sprintf);
  }
  gettimeofday(&eTime, nullptr);
  long SprintfTime = (eTime.tv_sec-sTime.tv_sec)*1000000+(eTime.tv_usec-sTime.tv_usec); //exeTime 单位是微秒

  ///// 测试append
  gettimeofday(&sTime, nullptr);
  for(int i=0; i<OUT_IN_REPEATE_NUM; i++)
  {
    append="";
    appendTest(append);
  }
  gettimeofday(&eTime, nullptr);
  long AppendTime = (eTime.tv_sec-sTime.tv_sec)*1000000+(eTime.tv_usec-sTime.tv_usec); //exeTime 单位是微秒

  gettimeofday(&sTime, nullptr);
  for(int i=0; i<OUT_IN_REPEATE_NUM; i++)
  {
    ss="";
    ssTest(ss);
  }
  gettimeofday(&eTime, nullptr);
  long SsTime = (eTime.tv_sec-sTime.tv_sec)*1000000+(eTime.tv_usec-sTime.tv_usec); //exeTime 单位是微秒

  cout<<"PlusTime is :   "<<PlusTime<<endl;
  cout<<"AppendTime is : "<<AppendTime<<endl;
  cout<<"SsTime is :     "<<SsTime<<endl;
  cout<<"SprintfTime is :"<<SprintfTime<<endl;
  if(ss==sprintf && append==plus && ss==plus)
  {
    cout<<"They are same"<<endl;
  }
  else
  {
    cout<<"Different!"<<endl;
    cout<<"Sprintf: "<<sprintf<<endl;
    cout<<"ss:        "<<ss<<endl;
    cout<<"Plus:     "<<plus<<endl;
    cout<<"Append:"<<append<<endl;
  }

}
