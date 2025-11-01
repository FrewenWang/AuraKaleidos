//
// Created by Frewen.Wong on 2022/4/23.
//
#pragma once

#include <memory>
#include <string>
#include <vector>

namespace aura::light_buffer {

class PlainCompiler {
public:
    /**
     * 增加代码编译器
     * @param codeGen
     */
    void addCompiler(const std::shared_ptr<int> &codeGen);
    /**
     * 代码编译方法
     * @param argc
     * @param argv
     * @return
     */
    int compile(int argc, char **argv);
    /**
     * 获取使用帮助指南
     * @return
     */
    std::string getUsageHelp();

private:
    int parseArguments(int argc, char **argv,
                       std::string &protoDir,
                       std::vector<std::string> &genOutDir,
                       std::string &protoFileName);
    // std::vector<std::shared_ptr<>>
};

} // namespace aura::light_buffer
