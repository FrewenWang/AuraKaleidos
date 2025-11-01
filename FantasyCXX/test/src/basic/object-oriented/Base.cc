//
// Created by frewen on 25-2-11.
//

#include "Base.h"
#include <iostream>
#include <cstdlib>

using namespace std;
/**
 * 类在使用前必须要进行声明。我们一般情况将类的声明放在.H文件里面
 * @param id
 * @param english
 * @param math
 * @param article
 */
Base::Base(int id, string name, int age, float score) {
  cout << ">>>>>调用类的含参构造函数" << endl;
  baseID = id;
  baseAge = age;
  baseName = name;
  totalScore = score;
}

/**
 * 可以通过冒号运算符直接对变量进行赋值
 * @param id
 * @param name
 * @param age
 */
Base::Base(int id, string name, int age) : baseID(id), baseAge(age), baseName(name) {
  cout << ">>>>>调用类的直接进行变量赋值含参构造函数" << endl;
}

/**
 * 拷贝构造函数的定义。
 * @param student 入参是对应本身的引用
 */
Base::Base(const Base &base) {
  cout << ">>>>>调用类的拷贝构造函数" << endl;
  // 我们使用this指针来访问当前对应的属性
  this->baseID = base.baseID;
  this->baseAge = base.baseAge;
  this->baseName = base.baseName;
  // 如果类的某个属性，没有进行初始化，那么这个属性就是随机值
  // 所以我们要进行针对下面totalScore进行拷贝初始化
  this->totalScore = base.totalScore;
}


// 如果是在类外面编写成员函数，只要在外部定义时函数名称前面加上类名称与范围解析运算符（::）即可。
// 范围解析运算符的主要作用就是指出成员函数所属的类。
void Base::input_data() {
  cout << "请输入您的成绩：";
  // cin >> totalScore;
}

void Base::show_data() {  //实现show_data函数
  cout << "成绩是：" << totalScore << endl;
}

void Base::display() {   //实现display函数的定义
  cout << baseName << "的年龄是:" << baseAge << "，成绩是:" << totalScore << endl;
}