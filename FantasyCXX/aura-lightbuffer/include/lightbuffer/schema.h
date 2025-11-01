//
// Created by Frewen.Wong on 2022/4/23.
//
#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace aura::light_buffer {

class Keyword {
public:
    static std::vector<std::string> wordSet;
    static std::vector<std::string> qualifierSet;
    static std::vector<std::string> primaryTypeList;
    static std::unordered_map<std::string, std::string> cppPrimaryTypeMap;
    static std::unordered_map<std::string, std::string> pythonPrimaryTypeMap;
    static std::unordered_map<std::string, std::string> javaPrimaryTypeMap;

    static bool checkKeyword(const std::string &str);

    static std::string castCppType(const std::string &type_str);

    static std::string castPythonType(const std::string &type_str);

    static std::string castJavaType(const std::string &type_str);
};

enum class Qualifier {
    OPTIONAL = 0, REQUIRED = 1, REPEATED = 2, UNKNOWN
};

enum class ValueType {
    PRIMARY_TYPE = 0, DEFINED_TYPE = 1, UNKNOWN
};

enum class PrimaryType {
    STRING = 0,
    INT64 = 1,
    INT32 = 2,
    INT16 = 3,
    INT8 = 4,
    FLOAT = 5,
    DOUBLE = 6,
    BOOL = 7,
    UNKNOWN = 8,
    PRIMARY_TYPE_CNT
};

class EnumHelper {
public:
    static Qualifier qualifierFromStr(const std::string &str);

    static PrimaryType primaryTypeFromStr(const std::string &str);

    static ValueType valueTypeFromStr(const std::string &str);

    static bool isPrimaryType(const std::string &str);
};

struct ValueItem {
    std::string typeNameStr;
    std::string valueNameStr;
    std::string defaultValueStr;
    int offset;
    bool hasDefault;
    Qualifier qualifier;
    ValueType valueType;
};

struct Message {
    std::vector<ValueItem> valueList;
    std::string typeName;

    std::string toString() const;
};

struct Schema {
    std::string syntax;
    std::string packageName;
    std::string fileName;
    std::vector<std::pair<std::string, Message>> messageList;

    std::string toString() const;
};

} // namespace aura::light_buffer