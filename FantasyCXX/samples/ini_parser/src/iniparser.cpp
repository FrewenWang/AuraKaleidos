//
// Created by wangzhijiang on 25-4-30.
//
#include "xiaomi/parser/iniparser.h"

#include <algorithm>
#include <cstdio>
#include <cctype>
#include <climits>
#include <cstring>
#include <iostream>

namespace xiaomi
{
namespace parser
{
IniParser::Value::Value(const char *v) : type(ValueType::STR)
{
    s = Strdup(v);
}

IniParser::Value::~Value()
{
    // 确保只有字符串类型需要释放
    if (type == ValueType::STR && s != nullptr)
    {
        delete[] s;
    }
}

IniParser::Value::Value(Value &&other) noexcept
{
    type = other.type;
    switch (type)
    {
        case ValueType::INT:
            i = other.i;
            break;
        case ValueType::FLOAT:
            f = other.f;
            break;
        case ValueType::STR:
            s = other.s;
            break;
    }
    other.type = ValueType::INT; // 避免重复释放
    other.i = 0;
    other.s = nullptr;
}

IniParser::Value &IniParser::Value::operator=(Value &&other) noexcept
{
    if (this != &other)
    {
        this->~Value();
        new(this) Value(std::move(other));
    }
    return *this;
}

IniParser::IniParser()
{
    m_section_capacity = 8;
    sections = new Section[m_section_capacity];

    memset(m_ini_file, 0, sizeof(m_ini_file));
}

IniParser::~IniParser()
{
    for (int i = 0; i < m_section_count; ++i)
    {
        delete[] sections[i].name;
        for (int j = 0; j < sections[i].count; ++j)
        {
            delete[] sections[i].entries[j].key;
            // delete[] sections[i].entries[j].value.s;
        }
        delete[] sections[i].entries;
    }
    delete[] sections;
}

bool IniParser::Parse(const char *filename)
{
    FILE *fp = fopen(filename, "r");
    if (!fp)
    {
        return false;
    }

    char line[MAX_FILE_LINE_LEN];
    Section *cur_section = AddSection("");

    while (fgets(line, sizeof(line), fp))
    {
        Trim(line);
        if (*line == '\0' || *line == ';') continue;

        if (*line == '[')
        {
            char *end = strchr(line, ']');
            if (!end)
            {
                continue;
            }
            *end = '\0';
            cur_section = AddSection(line + 1);
            continue;
        }

        char *eq = strchr(line, '=');
        if (!eq)
        {
            continue;
        }
        *eq = '\0';

        char *key = line;
        char *value = eq + 1;
        Trim(key);
        Trim(value);

        // 解析值类型
        Value val;
        int int_val;
        float float_val;

        if (TryParseInt(value, int_val))
        {
            val = Value(int_val);
        } else if (TryParseFloat(value, float_val))
        {
            val = Value(float_val);
        } else
        {
            // 处理带引号字符串
            if (*value == '"')
            {
                char *end_quote = strchr(value + 1, '"');
                if (end_quote)
                {
                    *end_quote = '\0';
                    value++;
                }
            }
            val = Value(value);
        }

        SetValue(cur_section, key, std::move(val));
    }

    fclose(fp);
    return true;
}

bool IniParser::Save(const char *filename) const
{
    const char *save_file = filename ? filename : m_ini_file;
    FILE *fp = fopen(save_file, "w");
    if (!fp)
    {
        return false;
    }
    for (int i = 0; i < m_section_count; ++i)
    {
        if (*sections[i].name)
        {
            fprintf(fp, "[%s]\n", sections[i].name);
        }

        for (int j = 0; j < sections[i].count; ++j)
        {
            const auto &entry = sections[i].entries[j];
            switch (entry.value.type)
            {
                case ValueType::INT:
                    fprintf(fp, "%s = %d\n", entry.key, entry.value.i);
                    break;
                case ValueType::FLOAT:
                    fprintf(fp, "%s = %.6g\n", entry.key, entry.value.f);
                    break;
                case ValueType::STR:
                    fprintf(fp, "%s = %s\n", entry.key, entry.value.s);
                    break;
            }
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
    return true;
}

int IniParser::GetInt(const char *section, const char *key, int default_val) const
{
    Section *sec = FindSection(section);
    if (!sec)
    {
        return default_val;
    }
    Entry *entry = FindEntry(sec, key);
    if (!entry)
    {
        return default_val;
    }
    switch (entry->value.type)
    {
        case ValueType::INT:
        {
            return entry->value.i;
        }
        case ValueType::FLOAT:
        {
            return static_cast<int>(entry->value.f);
        }
        case ValueType::STR:
        {
            int val;
            return TryParseInt(entry->value.s, val) ? val : default_val;
        }
        default:
        {
            return default_val;
        }
    }
}

float IniParser::GetFloat(const char *section, const char *key, float default_val) const
{
    Section *sec = FindSection(section);
    if (!sec)
    {
        return default_val;
    }
    Entry *entry = FindEntry(sec, key);
    if (!entry)
    {
        return default_val;
    }
    switch (entry->value.type)
    {
        case ValueType::FLOAT:
        {
            return entry->value.f;
        }
        case ValueType::INT:
        {
            return static_cast<float>(entry->value.i);
        }
        case ValueType::STR:
        {
            float val;
            return TryParseFloat(entry->value.s, val) ? val : default_val;
        }
        default:
        {
            return default_val;
        }
    }
}

const char *IniParser::GetStr(const char *section, const char *key, const char *default_val) const
{
    thread_local char buffer[MAX_VAL_LEN];
    Section *sec = FindSection(section);
    if (!sec)
    {
        return default_val;
    }
    Entry *entry = FindEntry(sec, key);
    if (!entry)
    {
        return default_val;
    }
    switch (entry->value.type)
    {
        case ValueType::STR:
        {
            return entry->value.s;
        }
        case ValueType::INT:
        {
            snprintf(buffer, sizeof(buffer), "%d", entry->value.i);
            return buffer;
        }
        case ValueType::FLOAT:
        {
            snprintf(buffer, sizeof(buffer), "%.6g", entry->value.f);
            return buffer;
        }
        default:
        {
            return default_val;
        }
    }
}

bool IniParser::SetInt(const char *section, const char *key, int val)
{
    Section *sec = FindSection(section);
    if (!sec)
    {
        sec = AddSection(section);
    }
    return SetValue(sec, key, Value(val));
}

bool IniParser::SetFloat(const char *section, const char *key, float val)
{
    Section *sec = FindSection(section);
    if (!sec)
    {
        sec = AddSection(section);
    }
    SetValue(sec, key, Value(val));
}

bool IniParser::SetStr(const char *section, const char *key, const char *val)
{
    Section *sec = FindSection(section);
    if (!sec)
    {
        sec = AddSection(section);
    }
    SetValue(sec, key, Value(val));
}

IniParser::Entry *IniParser::FindEntry(Section *sec, const char *key) const
{
    for (int i = 0; i < sec->count; ++i)
    {
        if (strcmp(sec->entries[i].key, key) == 0)
        {
            return &sec->entries[i];
        }
    }
    return nullptr;
}

IniParser::Section *IniParser::AddSection(const char *name)
{
    // Expand array if needed
    if (m_section_count >= m_section_capacity)
    {
        m_section_capacity *= 2;
        auto *newSections = new Section[m_section_capacity];
        memcpy(newSections, sections, m_section_count * sizeof(Section));
        delete[] sections;
        sections = newSections;
    }

    sections[m_section_count].name = strdup(name);
    sections[m_section_count].entries = nullptr;
    sections[m_section_count].count = 0;
    return &sections[m_section_count++];
}

bool IniParser::SetValue(Section *sec, const char *key, Value &&value) const
{
    Entry *entry = FindEntry(sec, key);
    if (entry)
    {
        entry->value = std::move(value);
        return true;
    }
    // 扩容条目数组
    auto *newEntries = new Entry[sec->count + 1];
    for (int i = 0; i < sec->count; ++i)
    {
        newEntries[i].key = sec->entries[i].key;
        newEntries[i].value = std::move(sec->entries[i].value);
    }
    delete[] sec->entries;
    sec->entries = newEntries;

    sec->entries[sec->count].key = strdup(key);
    sec->entries[sec->count].value = std::move(value);
    sec->count++;
    return true;
}

bool IniParser::TryParseInt(const char *s, int &out)
{
    char *end;
    const long val = strtol(s, &end, 10);
    if (*end == '\0' && val >= INT_MIN && val <= INT_MAX)
    {
        out = static_cast<int>(val);
        return true;
    }
    return false;
}

bool IniParser::TryParseFloat(const char *s, float &out)
{
    char *end;
    out = strtof(s, &end);
    return *end == '\0';
}


IniParser::Section *IniParser::FindSection(const char *name) const
{
    for (int i = 0; i < m_section_count; ++i)
    {
        if (strcmp(sections[i].name, name) == 0)
        {
            return &sections[i];
        }
    }
    return nullptr;
}

bool IniParser::hasSection(const char *section) const
{
    return FindSection(section) != nullptr;
}

bool IniParser::hasKey(const char *section, const char *key) const
{
    Section *sec = FindSection(section);
    return sec && FindEntry(sec, key);
}

char *IniParser::Strdup(const char *s)
{
    if (!s)
    {
        return nullptr;
    }
    size_t len = strlen(s) + 1;
    char *newStr = new char[len];
    memcpy(newStr, s, len);
    return newStr;
}

void IniParser::Trim(char *str)
{
    if (!str || !*str)
    {
        return;
    }
    // Trim leading spaces
    const char *start = str;
    while (*start && isspace(*start))
    {
        ++start;
    }
    // Trim trailing spaces
    char *end = str + strlen(str) - 1;
    while (end > str && isspace(*end))
    {
        end--;
    }
    *(end + 1) = '\0';
    if (start != str)
    {
        memmove(str, start, end - start + 2);
    }
}
}
}
