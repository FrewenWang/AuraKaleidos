//
// Created by Frewen.Wang on 25-4-29.
//

#ifndef INI_PARSER_H
#define INI_PARSER_H

#include <fstream>
#include <sstream>
#include <unordered_map>
#include <algorithm>
#include <stdexcept>

namespace aura
{

namespace parser
{

}


}

class IniParser {
public:
    IniParser();

    explicit IniParser(const std::string &file_path);


    void parse(const std::string &file_path);


    template<typename T>
    T GetValue(const std::string &section, const std::string &key,
               T defaultValue = T()) const;

    void SetValue(const std::string &section, const std::string &key,
                  const std::string &value);

    void Reload();

    void Save();

    bool HasSection(const std::string &section) const;

    bool HasKey(const std::string &section, const std::string &key) const;

private:
    using SectionMap = std::unordered_map<std::string, std::string>;
    using IniData = std::unordered_map<std::string, SectionMap>;

    void ParseLine(const std::string &line);

    static std::string Trim(const std::string &str);

    IniData m_data;
    std::string m_filePath;
    bool m_modified = false;
};

#endif //INI_PARSER_H
