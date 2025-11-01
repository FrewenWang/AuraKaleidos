//
// Created by frewen on 25-2-11.
//
#pragma once
#include<iostream>

using namespace std;

class Base {
private:
  std::string baseName;
  int baseID;
  int baseAge;
  float totalScore;
protected:
  float englishScore, mathScore, articleScore;

public:
  /**
   * 无参构造函数
   */
  Base() {
  }

  /**
   * 普通含参构造函数的声明
   * @param name
   * @param age
   * @param score
   */
  Base(int id, std::string name, int age, float score);  //普通构造函数

  /**
   * 普通含参构造函数
   */
  Base(int id, std::string name, int age);  //普通构造函数

  /**
   * 拷贝构造函数（声明）
   * @param base
   */
  Base(const Base &base);

  /**
   * 拷贝构造函数（声明）
   * @param base
   */
  Base(const Base &&base);

  /**
   * 析构函数的声明
   * .H文件里面是不是可以把类的定义也写在里面
   */
  ~Base() {
    std::cout << "=====调用类的析构函数==========" << std::endl;
  }

  void input_data(); //声明成员函数的原型

  void show_data();   //声明成员函数的原型

  void display();   //声明成员函数的原型
};

