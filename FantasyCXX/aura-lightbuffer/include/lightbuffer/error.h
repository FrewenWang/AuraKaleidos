//
// Created by frewen on 22-11-6.
//
#pragma once

#include <string>

namespace aura ::light_buffer {

class LineInfo;

enum class ErrCode {
    OK,                    // 正常
    SYNTAX_ERROR,          // 语法错误
    MISSING_SEMICOLON,     // 缺少分号
    PARENTHESES_NOT_MATCH, // 大括号不匹配
    UNKNOWN_QUALIFIER,     // 未知的限定符
    UNDEFINED_TYPE,        // 未定义类型
    VALUE_NAME_NOT_FOUND,  // 变量名未找到
    OFFSET_CONFLICT,       // 变量序号冲突
    PREPROCESS_ERROR,      // 预处理错误
    VERSION_ERROR,         // 版本错误
};

class Error {
public:
    static void log(ErrCode code, const std::string &errmsg, LineInfo *li = nullptr);
    static ErrCode getLastError();
    static void clearError();

private:
    static ErrCode errCode;
};


} // namespace aura::light_buffer