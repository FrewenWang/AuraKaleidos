//
// Created by Frewen.Wong on 2022/4/23.
//
#pragma once

#include <memory>
#include <string>

namespace aura::light_buffer {

class Schema;

class CodeGenerator {
public:
    /**
     * 代码生成器的构造函数
     */
    CodeGenerator() = default;

    /**
     * 默认的代码生成器的析构函数
     */
    virtual ~CodeGenerator() = default;

    /**
     * 代码生成的纯虚函数
     * @return
     */
    virtual bool generate(const std::string &fileName,
                          const std::string &outDir,
                          std::shared_ptr<Schema> &schema) = 0;

    std::string option;
    std::string help;
};

} // namespace aura::light_buffer