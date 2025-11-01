//
// Created by wangzhijiang on 25-4-30.
//

// 所有头文件都需要使用 #define 来防止头文件被多重包含
// 为保证唯一性，宏命名格式： MAGE_<一级目录>_<二级目录>_<file>__
#ifndef XIAOMI_PARSER_INIPARSER_H__
#define XIAOMI_PARSER_INIPARSER_H__

#define MAX_FILE_NAME_LEN (256)
#define MAX_FILE_LINE_LEN (1024)
#define MAX_VAL_LEN (64)

namespace xiaomi
{
namespace parser
{
class IniParser
{
public:
    IniParser();

    ~IniParser();

    /**
     * @brief
     * @param filename
     * @return
     */
    bool Parse(const char *filename);

    /**
     * @brief
     * @param filename
     * @return
     */
    bool Save(const char *filename = nullptr) const;

    /**
     * @brief
     * @param section
     * @param key
     * @param default_val
     * @return
     */
    int GetInt(const char *section, const char *key, int default_val) const;

    /**
     * @brief
     * @param section
     * @param key
     * @param default_val
     * @return
     */
    float GetFloat(const char *section, const char *key, float default_val) const;

    /**
     * @brief
     * @param section
     * @param key
     * @param default_val
     * @return
     */
    const char *GetStr(const char *section, const char *key, const char *default_val) const;

    /**
     * @brief
     * @param section
     * @param key
     * @param val
     */
    bool SetInt(const char *section, const char *key, int val);

    /**
     * @brief
     * @param section
     * @param key
     * @param val
     */
    bool SetFloat(const char *section, const char *key, float val);

    /**
     * @brief
     * @param section
     * @param key
     * @param val
     */
    bool SetStr(const char *section, const char *key, const char *val);

    /**
     * @brief
     * @param section
     * @return true or false
     */
    bool hasSection(const char *section) const;

    /**
     *
     * @param section
     * @param key
     * @return
     */
    bool hasKey(const char *section, const char *key) const;

private:
    enum class ValueType { INT, FLOAT, STR };

    struct Value
    {
        ValueType type;

        union
        {
            int i;
            float f;
            char *s;
        };

        Value() : type(ValueType::INT), i(0)
        {
        }

        explicit Value(int v) : type(ValueType::INT), i(v)
        {
        }

        explicit Value(float v) : type(ValueType::FLOAT), f(v)
        {
        }

        explicit Value(const char *v);

        ~Value();

        Value(const Value &) = delete;

        Value &operator=(const Value &) = delete;

        Value(Value &&other) noexcept;

        Value &operator=(Value &&other) noexcept;
    };

    struct Entry
    {
        char *key;
        Value value;
    };

    struct Section
    {
        char *name;
        Entry *entries;
        int count;
    };

    Section *sections = nullptr;
    int m_section_count = 0;
    int m_section_capacity = 8;
    char m_ini_file[MAX_FILE_NAME_LEN]{};

    IniParser(const IniParser &) = delete;

    void operator=(const IniParser &) = delete;

    Section *FindSection(const char *name) const;

    Entry *FindEntry(Section *sec, const char *key) const;

    Section *AddSection(const char *name);

    bool SetValue(Section *sec, const char *key, Value &&value) const;

    static bool TryParseInt(const char *s, int &out);

    static bool TryParseFloat(const char *s, float &out);

    static char *Strdup(const char *s);

    /**
     * trim line spaces
     * @param str
     */
    static void Trim(char *str);
};
}
}

#endif //XIAOMI_PARSER_INIPARSER_H__
