#ifndef AURA_RUNTIME_CORE_XTENSA_TYPES_STRING_HPP__
#define AURA_RUNTIME_CORE_XTENSA_TYPES_STRING_HPP__

#include "aura/runtime/core/xtensa/comm.hpp"

/**
 * @defgroup runtime Runtime
 * @{
 *      @defgroup core Runtime Core
 *      @{
 *           @defgroup string Runtime Core Xtensa String
 *      @}
 * @}
*/

#define AURA_STRING_DEFAULT_MAX_SIZE    (256)
namespace aura
{
namespace xtensa
{
/**
 * @addtogroup string
 * @{
*/

/**
 * @brief string class.
 *
 * This class provides basic functionality for handling strings, including construction,
 * assignment, comparison, and concatenation operations.
 *
 * @tparam MAX_SIZE  The maximum size of the string.
 */
template <MI_S32 MAX_SIZE = AURA_STRING_DEFAULT_MAX_SIZE>
class string_
{
public:
    /**
     * @brief Constructs a string object.
     */
    string_() : m_size(0)
    {
        Memset(m_data, '\0', MAX_SIZE);
    }

    /**
     * @brief Constructs a string object from a char array.
     *
     * @param s The char array to initialize the string object with.
     */
    string_(const MI_CHAR *s)
    {
        Memset(m_data, '\0', MAX_SIZE);
        assign(s);
    }

    /**
     * @brief Constructs a string object from a string.
     *
     * @param s The string to initialize the string object with.
     */
    string_(const string_ &s)
    {
        Memset(m_data, '\0', MAX_SIZE);
        assign(s);
    }

    /**
     * @brief Returns the length of the string.
     *
     * @return The length of the string.
     */
    MI_S32 size() const
    {
        return m_size;
    }

    /**
     * @brief Checks if the string is empty.
     *
     * @return True if the string is empty, otherwise False.
     */
    MI_BOOL empty() const
    {
        return 0 == m_size;
    }

    /**
     * @brief Returns the maximum capacity of the string.
     *
     * @return The maximum capacity of the string.
     */
    MI_S32 capacity() const
    {
        return MAX_SIZE;
    }

    /**
     * @brief Returns a pointer to the string.
     *
     * @return A pointer to the string.
     */
    const MI_CHAR* c_str() const
    {
        return m_data;
    }

    /**
     * @brief Compares the string stored with the input string.
     *
     * @param s The input char array to compare with the string.
     *
     * @return An integer less than, equal to, or greater than zero if string is found,
     *         respectively, to be less than, to match, or be greater than string.
     */
    MI_S32 compare(const MI_CHAR *s) const
    {
        return Strcmp(m_data, s);
    }

    /**
     * @brief Compares the string stored with the input string.
     *
     * @param s The input string to compare with this string.
     *
     * @return An integer less than, equal to, or greater than zero if string is found,
     *         respectively, to be less than, to match, or be greater than string.
     */
    MI_S32 compare(const string_ &s) const
    {
        return Strcmp(m_data, s.c_str());
    }

    MI_BOOL operator==(const string_ &other) const
    {
        return compare(other);
    }

    /**
     * @brief Appends a char array to the end of the string object.
     *
     * @param s The char array to append to the string object.
     *
     * @return MI_TRUE if the append operation was successful, MI_FALSE if the combined length exceeds the maximum size.
     */
    MI_BOOL append(const MI_CHAR *s)
    {
        MI_S32 len = Strlen(s);
        if (m_size + len >= MAX_SIZE)
        {
            AURA_XTENSA_LOG("input str len + m_size need less than MAX_SIZE!\n");
            return MI_FALSE;
        }

        Strcpy(m_data + m_size, s);
        m_size += len;
        m_data[m_size] = '\0';

        return MI_TRUE;
    }

    /**
     * @brief Appends a string to the end of the string object.
     *
     * @param s The string object which need to append.
     *
     * @return MI_TRUE if the append operation was successful, MI_FALSE if the combined length exceeds the maximum size.
     */
    MI_BOOL append(const string_ &s)
    {
        return append(s.c_str());
    }

    /**
     * @brief Assigns a char array to the string object.
     *
     * @param s The char array which assign to the string object.
     *
     * @return MI_TRUE if the assignment operation was successful, MI_FALSE if the input string length exceeds the maximum size.
     */
    MI_BOOL assign(const MI_CHAR *s)
    {
        MI_S32 len = Strlen(s);
        if (len >= MAX_SIZE)
        {
            AURA_XTENSA_LOG("input str len need less than MAX_SIZE!\n");
            return MI_FALSE;
        }

        Strcpy(m_data, s);
        m_size = len;
        m_data[m_size] = '\0';

        return MI_TRUE;
    }

    /**
     * @brief Assigns a string to this string object.
     *
     * @param s The string object which assign to this string object.
     *
     * @return MI_TRUE if the assignment operation was successful, MI_FALSE if the input string length exceeds the maximum size.
     */
    MI_BOOL assign(const string_ &s)
    {
        return assign(s.c_str());
    }

    /**
     * @brief Clears the content of the string object.
     */
    void clear()
    {
        m_size = 0;
        Memset(m_data, '\0', MAX_SIZE);
    }

    /**
     * @brief Gets the character at the specified index.The user needs to ensure that the index does not go out of bounds.
     *
     * @param index The index of the character to retrieve.
     * @param out The character retrieved will be stored here.
     *
     * @return The character which is retrieved.
     */
    MI_CHAR at(MI_S32 index) const
    {
        return m_data[index];
    }

    /**
     * @brief Assigns a char array to the string object.The user needs to ensure that the index does not go out of bounds.
     *
     * @param s The char array to assign to the string object.
     *
     * @return The reference of string object
     */
    string_& operator=(const MI_CHAR *s)
    {
        assign(s);
        return *this;
    }

    /**
     * @brief Assigns a string to this string object.The user needs to ensure that the index does not go out of bounds.
     *
     * @param s The string to assign to this string object.
     *
     * @return The reference of string object
     */
    string_& operator=(const string_ &s)
    {
        assign(s);
        return *this;
    }

    /**
     * @brief Appends a char array to the end of the string object.The user needs to ensure that the index does not go out of bounds.
     *
     * @param s The string to append to the string object.
     *
     * @return The reference of string object
     */
    string_& operator+(const MI_CHAR *s)
    {
        append(s);
        return *this;
    }

    /**
     * @brief Appends a string to the end of the string object.The user needs to ensure that the index does not go out of bounds.
     *
     * @param s The string to append to the string object.
     *
     * @return The reference of string object
     */
    string_& operator+(const string_ &s)
    {
        append(s);
        return *this;
    }

    /**
     * @brief Appends a char array to the end of the string object.The user needs to ensure that the index does not go out of bounds.
     *
     * @param s The char array to append to the string object.
     *
     * @return The reference of string object
     */
    string_& operator+=(const MI_CHAR *s)
    {
        append(s);
        return *this;
    }

    /**
     * @brief Appends a string to the end of the string object.The user needs to ensure that the index does not go out of bounds.
     *
     * @param s The string to append to the string object.
     *
     * @return The reference of string object
     */
    string_& operator+=(const string_ &s)
    {
        append(s);
        return *this;
    }

    /**
    * @brief Accesses the element at the specified index in the string.The user needs to ensure that the index does not go out of bounds.
    *
    * @param index The index of the element to access.
    *
    * @return The character which is retrieved.
    */
    MI_CHAR operator[](int index)
    {
        return m_data[index];
    }

    /**
     * @brief Inserts a char array at the specified index.
     *
     * @param index The position index where the string will be inserted.
     * @param s The char array to insert.
     *
     * @return MI_TRUE if the insert operation was successful, MI_FALSE if the combined length exceeds the maximum size.
     */
    MI_BOOL insert(MI_S32 index, const MI_CHAR *s)
    {
        if (index < 0 || index > m_size)
        {
            AURA_XTENSA_LOG("invalid input index!\n");
            return MI_FALSE;
        }

        MI_S32 len = Strlen(s);
        if (m_size + len >= MAX_SIZE)
        {
            AURA_XTENSA_LOG("input str len + m_size need less than MAX_SIZE!\n");
            return MI_FALSE;
        }

        Memmove(m_data + index + len, m_data + index, m_size - index);
        Memcpy(m_data + index, s, len);
        m_size += len;
        m_data[m_size] = '\0';

        return MI_TRUE;
    }

    /**
     * @brief Inserts a string at the specified index.
     *
     * @param index The position index where the string will be inserted.
     * @param s The string to insert.
     *
     * @return MI_TRUE if the insert operation was successful, MI_FALSE if the combined length exceeds the maximum size.
     */
    MI_BOOL insert(MI_S32 index, const string_ &s)
    {
        return insert(index, s.c_str());
    }

    /**
     * @brief Removes characters from the specified index.
     *
     * @param index The starting index of the characters to remove.
     * @param count The number of characters to remove, default is 1.
     *
     * @return MI_TRUE if the removal was successful, MI_FALSE if the index is invalid or the resulting string is empty.
     */
    MI_BOOL erase(MI_S32 index, MI_S32 count = 1)
    {
        if (index < 0 || index >= m_size)
        {
            AURA_XTENSA_LOG("invalid input index!\n");
            return MI_FALSE;
        }

        if (index + count > m_size)
        {
            count = m_size - index;
        }

        Memmove(m_data + index, m_data + index + count, m_size - index - count);
        m_size -= count;

        Memset(m_data + m_size, '\0', count);

        return MI_TRUE;
    }

    /**
     * @brief Finds the first occurrence of a substring within the string object.
     *
     * @param s The substring to search for.
     *
     * @return The index of the first occurrence of the substring, or -1 if not found.
     */
    MI_S32 find(const MI_CHAR *s) const
    {
        const char* result = Strstr(m_data, s);
        if (MI_NULL == result)
        {
            return -1;
        }
        return static_cast<MI_S32>(result - m_data);
    }

    /**
     * @brief Finds the first occurrence of a string within this string object.
     *
     * @param s The string to search for.
     *
     * @return The index of the first occurrence of the string, or -1 if not found.
     */
    MI_S32 find(const string_ &s) const
    {
        return find(s.c_str());
    }

    /**
     * @brief Extracts a substring starting from the specified index with the specified length.
     *
     * @param start The starting index of the substring.
     * @param length The length of the substring.
     * @param str The extracted substring.
     *
     * @return MI_TRUE if the substring was extracted successfully, MI_FALSE if the start or length is invalid.
     */
    MI_BOOL substr(MI_S32 start, MI_S32 length, string_ &str)
    {
        if ((start < 0) || (start >= m_size))
        {
            AURA_XTENSA_LOG("invalid input index!\n");
            return MI_FALSE;
        }

        MI_CHAR data[MAX_SIZE];
        Memset(data, '\0', MAX_SIZE);
        MI_S32 len = length > m_size - start ? m_size - start : length;
        Memcpy(data, m_data + start, len);

        str.assign(data);

        return MI_TRUE;
    }

    /**
     * @brief Replaces a portion of the string with another string.
     *
     * @param pos The starting position of the substring to replace.
     * @param len The number of characters to replace.
     * @param str The string to replace with.
     */
    MI_BOOL replace(MI_S32 pos, MI_S32 len, const string_ &str)
    {
        if (pos > m_size)
        {
            AURA_XTENSA_LOG("invalid input index!\n");
            return MI_FALSE;
        }

        if (pos + len > m_size)
        {
            len = m_size - pos;
        }

        MI_S32 str_len = str.size();
        MI_S32 new_len = m_size - len + str_len;

        Memmove(m_data + pos + str_len, m_data + pos + len, m_size - pos - len + 1);
        Memcpy(m_data + pos, str.c_str(), str_len);

        m_size = new_len;
        m_data[m_size] = '\0';

        return MI_TRUE;
    }

private:
    MI_CHAR m_data[MAX_SIZE];
    MI_S32  m_size;
};

using string = string_<AURA_STRING_DEFAULT_MAX_SIZE>;

/**
 * @}
*/
} //namespace xtensa
} //namespace aura

#endif //AURA_RUNTIME_CORE_XTENSA_TYPES_STRING_HPP__
