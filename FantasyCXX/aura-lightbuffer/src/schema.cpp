#include <algorithm>
#include <sstream>
#include "lightbuffer/schema.h"

namespace aura::light_buffer {

std::vector<std::string>Keyword::wordSet{"syntax", "package", "message", "string", "int64", "int32", "int16", "int8",
                                         "float", "double", "bool", "bytes", "optional", "required", "default",
                                         "repeated"};

std::vector<std::string> Keyword::qualifierSet{"optional", "required", "repeated", "unknown"};

std::vector<std::string> Keyword::primaryTypeList{"string", "int64", "int32", "int16", "int8", "float", "double",
                                                  "bool", "bytes"};

std::unordered_map<std::string, std::string> Keyword::cppPrimaryTypeMap{{"string", "std::string"},
                                                                        {"int64",  "int64_t"},
                                                                        {"int32",  "int"},
                                                                        {"int16",  "short"},
                                                                        {"int8",   "char"},
                                                                        {"float",  "float"},
                                                                        {"double", "double"},
                                                                        {"bool",   "bool"},
                                                                        {"bytes",  "const char*"}};

std::unordered_map<std::string, std::string> Keyword::pythonPrimaryTypeMap{
// TODO 待实现
};

std::unordered_map<std::string, std::string> Keyword::javaPrimaryTypeMap{
// TODO 待实现
};

bool Keyword::checkKeyword(const std::string &str) {
    return std::find(wordSet.begin(), wordSet.end(), str) != wordSet.end();
}

std::string Keyword::castCppType(const std::string &type_str) {
    if (cppPrimaryTypeMap.find(type_str) == cppPrimaryTypeMap.end()) {
        return std::string{};
    }
    return cppPrimaryTypeMap[type_str];
}

std::string Keyword::castPythonType(const std::string &type_str) {
    if (pythonPrimaryTypeMap.find(type_str) == pythonPrimaryTypeMap.end()) {
        return std::string{};
    }
    return pythonPrimaryTypeMap[type_str];
}

std::string Keyword::castJavaType(const std::string &type_str) {
    if (javaPrimaryTypeMap.find(type_str) == javaPrimaryTypeMap.end()) {
        return std::string{};
    }
    return javaPrimaryTypeMap[type_str];
}

std::string Message::toString() const {
    std::stringstream ss;
    for (const auto value: value_list) {
        ss << std::endl;
        ss << "value type: " << value.type_name_str << std::endl;
        ss << "default_value: " << value.default_value_str << std::endl;
        ss << "offset: " << value.offset << std::endl;
        ss << "qualifier: " << Keyword::qualifier_set[static_cast<int>(value.qualifier)] << std::endl;
    }
    return ss.str();
}

std::string Schema::toString() const {
    std::stringstream ss;
    ss << std::endl;
    ss << "syntax: " << syntax << std::endl;
    ss << "package: " << package_name << std::endl;

    for (const auto &msg: message_list) {
        ss << std::endl;
        ss << "==== message: " << msg.first << " ====";
        ss << msg.second.to_string();
    }
    return ss.str();
}


Qualifier EnumHelper::qualifier_from_str(const std::string &str) {
    if (str == "optional") {
        return Qualifier::OPTIONAL;
    } else if (str == "required") {
        return Qualifier::REQUIRED;
    } else if (str == "repeated") {
        return Qualifier::REPEATED;
    }
    return Qualifier::UNKNOWN;
}

PrimaryType EnumHelper::primary_type_from_str(const std::string &str) {
    for (auto i = 0; i < Keyword::primary_type_list.size(); ++i) {
        if (Keyword::primary_type_list[i] == str) {
            return static_cast<PrimaryType>(i);
        }
    }
    return PrimaryType::UNKNOWN;
}

ValueType EnumHelper::value_type_from_str(const std::string &str) {
    return ValueType::PRIMARY_TYPE;
}

bool EnumHelper::is_primary_type(const std::string &str) {
    for (auto &s: Keyword::primary_type_list) {
        if (s == str) {
            return true;
        }
    }
    return false;
}

} // namespace plainbuffer

