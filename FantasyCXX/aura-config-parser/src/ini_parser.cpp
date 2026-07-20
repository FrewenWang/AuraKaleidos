//
// IniParser 实现（与 include/ini_parser.h 配套）。
//

#include "ini_parser.h"

#include <fstream>
#include <sstream>
#include <algorithm>

std::string IniParser::Trim(const std::string &str) {
    const char *ws = " \t\r\n";
    size_t b = str.find_first_not_of(ws);
    if (b == std::string::npos) return "";
    size_t e = str.find_last_not_of(ws);
    return str.substr(b, e - b + 1);
}

IniParser::IniParser() : m_modified(false) {}

IniParser::IniParser(const std::string &file_path) : m_modified(false) {
    parse(file_path);
}

void IniParser::ParseLine(const std::string &line) {
    std::string s = Trim(line);
    if (s.empty()) return;
    if (s[0] == '#' || s[0] == ';') return;

    // [section]
    if (s[0] == '[' && s.back() == ']') {
        m_curSection = Trim(s.substr(1, s.size() - 2));
        return;
    }

    // key = value
    size_t eq = s.find('=');
    if (eq == std::string::npos) return;
    std::string key = Trim(s.substr(0, eq));
    std::string value = Trim(s.substr(eq + 1));
    if (key.empty()) return;
    m_data[m_curSection][key] = value;
}

void IniParser::parse(const std::string &file_path) {
    m_filePath = file_path;
    m_data.clear();
    m_curSection.clear();
    std::ifstream ifs(file_path);
    if (!ifs) return;
    std::string line;
    while (std::getline(ifs, line)) {
        ParseLine(line);
    }
}

void IniParser::Reload() {
    parse(m_filePath);
}

void IniParser::Save() {
    if (m_filePath.empty()) return;
    std::ofstream ofs(m_filePath, std::ios::trunc);
    if (!ofs) return;
    for (const auto &sec : m_data) {
        ofs << "[" << sec.first << "]" << "\n";
        for (const auto &kv : sec.second) {
            ofs << kv.first << " = " << kv.second << "\n";
        }
        ofs << "\n";
    }
    m_modified = false;
}

void IniParser::SetValue(const std::string &section, const std::string &key, const std::string &value) {
    m_data[section][key] = value;
    m_modified = true;
}

bool IniParser::HasSection(const std::string &section) const {
    return m_data.find(section) != m_data.end();
}

bool IniParser::HasKey(const std::string &section, const std::string &key) const {
    auto it = m_data.find(section);
    if (it == m_data.end()) return false;
    return it->second.find(key) != it->second.end();
}
