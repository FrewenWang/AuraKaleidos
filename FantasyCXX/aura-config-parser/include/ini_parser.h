//
// Created by Frewen.Wang on 25-4-29.
// 轻量级 INI 配置文件解析器（头文件声明 + 模板化取值；实现见 src/ini_parser.cpp）。
//

#ifndef INI_PARSER_H
#define INI_PARSER_H

#include <fstream>
#include <sstream>
#include <unordered_map>
#include <algorithm>
#include <stdexcept>
#include <string>

class IniParser {
public:
    IniParser();

    explicit IniParser(const std::string &file_path);

    void parse(const std::string &file_path);

    void Reload();

    void Save();

    bool HasSection(const std::string &section) const;

    bool HasKey(const std::string &section, const std::string &key) const;

    // 读取指定 section/key 的值；缺失时返回 defaultValue。
    template<typename T>
    T GetValue(const std::string &section, const std::string &key, T defaultValue = T()) const {
        auto it = m_data.find(section);
        if (it == m_data.end()) return defaultValue;
        auto kit = it->second.find(key);
        if (kit == it->second.end()) return defaultValue;
        T val{};
        std::istringstream iss(kit->second);
        if (iss >> val) return val;
        return defaultValue;
    }

    void SetValue(const std::string &section, const std::string &key, const std::string &value);

private:
    using SectionMap = std::unordered_map<std::string, std::string>;
    using IniData = std::unordered_map<std::string, SectionMap>;

    void ParseLine(const std::string &line);
    static std::string Trim(const std::string &str);

    IniData m_data;
    std::string m_filePath;
    std::string m_curSection;
    bool m_modified = false;
};

// std::string 特化：原样返回字符串（不做数值转换）
template<>
inline std::string IniParser::GetValue<std::string>(const std::string &section,
                                                    const std::string &key,
                                                    std::string defaultValue) const {
    auto it = m_data.find(section);
    if (it == m_data.end()) return defaultValue;
    auto kit = it->second.find(key);
    if (kit == it->second.end()) return defaultValue;
    return kit->second;
}

#endif //INI_PARSER_H
