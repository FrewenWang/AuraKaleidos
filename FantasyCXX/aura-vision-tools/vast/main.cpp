//
// Created by Frewen.Wong on 2022/4/5.
//
#include "iostream"
#include "string"
#include "json/json.h"

int main(int argc, char **argv) {
    std::cout << "开始执行程序识别====" << std::endl;
    //该字符串为上述json对象
    const std::string rawJson = R"({"Age": 20, "Name": "colin"})";

    Json::Reader reader;
    Json::Value value;

    //strValue字符串转value的json对象
    if (reader.parse(rawJson, value)) {
        std::string subCommand = value["Age"].asString();//var的json对象转换为string类型字符串
        std::cout << subCommand << std::endl;//显示src对象内容
    } else {
        std::cout << "开始执行程序识别====" << rawJson << std::endl;
    }
}