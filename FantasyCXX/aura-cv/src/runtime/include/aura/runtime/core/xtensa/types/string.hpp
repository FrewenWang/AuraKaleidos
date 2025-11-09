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
template <DT_S32 MAX_SIZE = AURA_STRING_DEFAULT_MAX_SIZE>
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
    string_(const DT_CHAR *s)
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
    DT_S32 size() const
    {
        return m_size;
    }

    /**
     * @brief Checks if the string is empty.
     *
     * @return True if the string is empty, otherwise False.
     */
    DT_BOOL empty() const
    {
        return 0 == m_size;
    }

    /**
     * @brief Returns the maximum capacity of the string.
     *
     * @return The maximum capacity of the string.
     */
    DT_S32 capacity() const
    {
        return MAX_SIZE;
    }

    /**
     * @brief Returns a pointer to the string.
     *
     * @return A pointer to the string.
     */
    const DT_CHAR* c_str() const
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
    DT_S32 compare(const DT_CHAR *s) const
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
    DT_S32 compare(const string_ &s) const
    {
        return Strcmp(m_data, s.c_str());
    }

    DT_BOOL operator==(const string_ &other) const
    {
        return compare(other);
    }

    /**
     * @brief Appends a char array to the end of the string object.
     *
     * @param s The char array to append to the string object.
     *
     * @return DT_TRUE if the append operation was successful, DT_FALSE if the combined length exceeds the maximum size.
     */
    DT_BOOL append(const DT_CHAR *s)
    {
        DT_S32 len = Strlen(s);
        if (m_size + len >= MAX_SIZE)
        {
            AURA_XTENSA_LOG("input str len + m_size need less than MAX_SIZE!\n");
            return DT_FALSE;
        }

        Strcpy(m_data + m_size, s);
        m_size += len;
        m_data[m_size] = '\0';

        return DT_TRUE;
    }

    /**
     * @brief Appends a string to the end of the string object.
     *
     * @param s The string object which need to append.
     *
     * @return DT_TRUE if the append operation was successful, DT_FALSE if the combined length exceeds the maximum size.
     */
    DT_BOOL append(const string_ &s)
    {
        return append(s.c_str());
    }

    /**
     * @brief Assigns a char array to the string object.
     *
     * @param s The char array which assign to the string object.
     *
     * @return DT_TRUE if the assignment operation was successful, DT_FALSE if the input string length exceeds the maximum size.
     */
    DT_BOOL assign(const DT_CHAR *s)
    {
        DT_S32 len = Strlen(s);
        if (len >= MAX_SIZE)
        {
            AURA_XTENSA_LOG("input str len need less than MAX_SIZE!\n");
            return DT_FALSE;
        }

        Strcpy(m_data, s);
        m_size = len;
        m_data[m_size] = '\0';

        return DT_TRUE;
    }

    /**
     * @brief Assigns a string to this string object.
     *
     * @param s The string object which assign to this string object.
     *
     * @return DT_TRUE if the assignment operation was successful, DT_FALSE if the input string length exceeds the maximum size.
     */
    DT_BOOL assign(const string_ &s)
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
    DT_CHAR at(DT_S32 index) const
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
    string_& operator=(const DT_CHAR *s)
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
    string_& operator+(const DT_CHAR *s)
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
    string_& operator+=(const DT_CHAR *s)
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
    DT_CHAR operator[](int index)
    {
        return m_data[index];
    }

    /**
     * @brief Inserts a char array at the specified index.
     *
     * @param index The position index where the string will be inserted.
     * @param s The char array to insert.
     *
     * @return DT_TRUE if the insert operation was successful, DT_FALSE if the combined length exceeds the maximum size.
     */
    DT_BOOL insert(DT_S32 index, const DT_CHAR *s)
    {
        if (index < 0 || index > m_size)
        {
            AURA_XTENSA_LOG("invalid input index!\n");
            return DT_FALSE;
        }

        DT_S32 len = Strlen(s);
        if (m_size + len >= MAX_SIZE)
        {
            AURA_XTENSA_LOG("input str len + m_size need less than MAX_SIZE!\n");
            return DT_FALSE;
        }

        Memmove(m_data + index + len, m_data + index, m_size - index);
        Memcpy(m_data + index, s, len);
        m_size += len;
        m_data[m_size] = '\0';

        return DT_TRUE;
    }

    /**
     * @brief Inserts a string at the specified index.
     *
     * @param index The position index where the string will be inserted.
     * @param s The string to insert.
     *
     * @return DT_TRUE if the insert operation was successful, DT_FALSE if the combined length exceeds the maximum size.
     */
    DT_BOOL insert(DT_S32 index, const string_ &s)
    {
        return insert(index, s.c_str());
    }

    /**
     * @brief Removes characters from the specified index.
     *
     * @param index The starting index of the characters to remove.
     * @param count The number of characters to remove, default is 1.
     *
     * @return DT_TRUE if the removal was successful, DT_FALSE if the index is invalid or the resulting string is empty.
     */
    DT_BOOL erase(DT_S32 index, DT_S32 count = 1)
    {
        if (index < 0 || index >= m_size)
        {
            AURA_XTENSA_LOG("invalid input index!\n");
            return DT_FALSE;
        }

        if (index + count > m_size)
        {
            count = m_size - index;
        }

        Memmove(m_data + index, m_data + index + count, m_size - index - count);
        m_size -= count;

        Memset(m_data + m_size, '\0', count);

        return DT_TRUE;
    }

    /**
     * @brief Finds the first occurrence of a substring within the string object.
     *
     * @param s The substring to search for.
     *
     * @return The index of the first occurrence of the substring, or -1 if not found.
     */
    DT_S32 find(const DT_CHAR *s) const
    {
        const char* result = Strstr(m_data, s);
        if (DT_NULL == result)
        {
            return -1;
        }
        return static_cast<DT_S32>(result - m_data);
    }

    /**
     * @brief Finds the first occurrence of a string within this string object.
     *
     * @param s The string to search for.
     *
     * @return The index of the first occurrence of the string, or -1 if not found.
     */
    DT_S32 find(const string_ &s) const
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
     * @return DT_TRUE if the substring was extracted successfully, DT_FALSE if the start or length is invalid.
     */
    DT_BOOL substr(DT_S32 start, DT_S32 length, string_ &str)
    {
        if ((start < 0) || (start >= m_size))
        {
            AURA_XTENSA_LOG("invalid input index!\n");
            return DT_FALSE;
        }

        DT_CHAR data[MAX_SIZE];
        Memset(data, '\0', MAX_SIZE);
        DT_S32 len = length > m_size - start ? m_size - start : length;
        Memcpy(data, m_data + start, len);

        str.assign(data);

        return DT_TRUE;
    }

    /**
     * @brief Replaces a portion of the string with another string.
     *
     * @param pos The starting position of the substring to replace.
     * @param len The number of characters to replace.
     * @param str The string to replace with.
     */
    DT_BOOL replace(DT_S32 pos, DT_S32 len, const string_ &str)
    {
        if (pos > m_size)
        {
            AURA_XTENSA_LOG("invalid input index!\n");
            return DT_FALSE;
        }

        if (pos + len > m_size)
        {
            len = m_size - pos;
        }

        DT_S32 str_len = str.size();
        DT_S32 new_len = m_size - len + str_len;

        Memmove(m_data + pos + str_len, m_data + pos + len, m_size - pos - len + 1);
        Memcpy(m_data + pos, str.c_str(), str_len);

        m_size = new_len;
        m_data[m_size] = '\0';

        return DT_TRUE;
    }

private:
    DT_CHAR m_data[MAX_SIZE];
    DT_S32  m_size;
};

using string = string_<AURA_STRING_DEFAULT_MAX_SIZE>;

/**
 * @}
*/
} //namespace xtensa
} //namespace aura

#endif //AURA_RUNTIME_CORE_XTENSA_TYPES_STRING_HPP__
