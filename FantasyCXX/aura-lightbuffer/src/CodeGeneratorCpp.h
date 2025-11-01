//
// Created by Frewen.Wong on 2022/4/23.
//
#pragma once

#include "CodeGenerator.h"

namespace aura::light_buffer {
/**
 * C++版本的代码生成器
 */
class CodeGeneratorCpp : public CodeGenerator {
public:
    CodeGeneratorCpp();

    /**
     * C++版本的代码生成器的generate必须声明纯虚函数
     * @param fileName
     * @param outDir
     * @param schema
     * @return
     */
    bool generate(const std::string &fileName,
                  const std::string &outDir,
                  std::shared_ptr<Schema> &schema) override;

private:
    /**
     * 生成头文件
     * @param fileName
     * @param outDir
     * @param schema
     * @return
     */
    bool generateHeader(const std::string &fileName,
                        const std::string &outDir,
                        std::shared_ptr<Schema> &schema);

    /**
     * 生成C++源码
     * @param fileName
     * @param outDir
     * @param schema
     * @return
     */
    bool generateSource(const std::string &fileName,
                        const std::string &outDir,
                        std::shared_ptr<Schema> &schema);
};

}// namespace lightbuffer
