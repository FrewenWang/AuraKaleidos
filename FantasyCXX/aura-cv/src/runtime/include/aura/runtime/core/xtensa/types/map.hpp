#ifndef AURA_RUNTIME_CORE_XTENSA_TYPES_MAP_HPP__
#define AURA_RUNTIME_CORE_XTENSA_TYPES_MAP_HPP__

#include "aura/runtime/core/xtensa/types/string.hpp"
#include "aura/runtime/core/xtensa/types/iterator.hpp"

#include <utility>
#include <initializer_list>

/**
 * @defgroup runtime Runtime
 * @{
 *      @defgroup core Runtime Core
 *      @{
 *           @defgroup map Runtime Core Xtensa Map
 *      @}
 * @}
*/

#define AURA_DEFAULT_MAP_MAX_SIZE       (16)
namespace aura
{
namespace xtensa
{
/**
 * @addtogroup map
 * @{
*/

/**
 * @brief A key-value mapping container.
 * 
 * This class provides an associative container that stores elements formed by a combination of a key 
 * value and a mapped value, following a specific order.
 * 
 * Keys are strings of maximum length KEY_MAX_SIZE.
 * Values are of type Tp.
 * 
 * @tparam Tp Type of the mapped value objects.
 * @tparam KEY_MAX_SIZE Maximum size of the key string.
 * @tparam MAP_MAX_SIZE Maximum size of the map.
 */
template <typename Tp, DT_S32 KEY_MAX_SIZE = AURA_STRING_DEFAULT_MAX_SIZE, DT_S32 MAP_MAX_SIZE = AURA_DEFAULT_MAP_MAX_SIZE>
class map
{
public:
    /**
     * @brief Default constructor for map.
     *
     * Constructs an empty map with size initialized to 0.
     */
    map() : m_size(0)
    {}

    /**
     * @brief Constructor for map with initializer list.
     *
     * Constructs a map with key-value pairs from the initializer list.
     *
     * @param list The initializer list containing key-value pairs.
     */
    map(std::initializer_list<std::pair<string_<KEY_MAX_SIZE>, Tp>> lists) : m_size(0)
    {
        DT_BOOL ret = DT_FALSE;
        for (const auto &list : lists)
        {
            ret = insert(list.first.c_str(), list.second);
            if (!ret)
            {
                AURA_XTENSA_LOG("insert failed!\n");
            }
        }
    }

    /**
     * @brief Iterator type for map.
     *
     * Iterator type for iterating over elements in the map.
     */
    using iterator = Iterator<std::pair<string_<KEY_MAX_SIZE>, Tp>>;

    /**
     * @brief Iterator type for const map.
     *
     * Const iterator type for iterating over elements in the map.
     */
    using const_iterator = Iterator<const std::pair<string_<KEY_MAX_SIZE>, Tp>>;

    /**
     * @brief Iterator type for reverse map.
     *
     * Iterator type for iterating over elements in the map in reverse order.
     */
    using reverse_iterator = ReverseIterator<std::pair<string_<KEY_MAX_SIZE>, Tp>>;

    /**
     * @brief Iterator type for const reverse map.
     *
     * Const iterator type for iterating over elements in the map in reverse order.
     */
    using const_reverse_iterator = ReverseIterator<const std::pair<string_<KEY_MAX_SIZE>, Tp>>;

    /**
     * @brief Returns an iterator pointing to the first element of the map.
     *
     * @return Iterator pointing to the first element of the map.
     */
    iterator begin()
    {
        return iterator(m_data, 0);
    }

    /**
     * @brief Returns an iterator referring to the past-the-end element of the map.
     *
     * @return Iterator referring to the past-the-end element of the map.
     */
    iterator end()
    {
        return iterator(m_data, m_size);
    }

    /**
     * @brief Returns an iterator pointing to the first element of the map.
     *
     * @return Const iterator pointing to the first element of the map.
     */
    const_iterator begin() const
    {
        return const_iterator(m_data, 0);
    }

    /**
     * @brief Returns an iterator referring to the past-the-end element of the map.
     *
     * @return Const iterator referring to the past-the-end element of the map.
     */
    const_iterator end() const
    {
        return const_iterator(m_data, m_size);
    }

    /**
     * @brief Returns a reverse iterator pointing to the first element of the map in reverse order.
     *
     * @return Reverse iterator pointing to the first element of the map in reverse order.
     */
    reverse_iterator rbegin()
    {
        return reverse_iterator(m_data, m_size - 1);
    }

    /**
     * @brief Returns a reverse iterator referring to the past-the-end element of the map in reverse order.
     *
     * @return Reverse iterator referring to the past-the-end element of the map in reverse order.
     */
    reverse_iterator rend()
    {
        return reverse_iterator(m_data, -1);
    }

    /**
     * @brief Returns a const reverse iterator pointing to the first element of the map in reverse order.
     *
     * @return Const reverse iterator pointing to the first element of the map in reverse order.
     */
    const_reverse_iterator rbegin() const
    {
        return const_reverse_iterator(m_data, m_size - 1);
    }

    /**
     * @brief Returns a const reverse iterator referring to the past-the-end element of the map in reverse order.
     *
     * @return Const reverse iterator referring to the past-the-end element of the map in reverse order.
     */
    const_reverse_iterator rend() const
    {
        return const_reverse_iterator(m_data, -1);
    }

    /**
     * @brief Overloaded subscript operator for accessing or inserting elements by key.
     *
     * Returns a reference to the value associated with the specified key.
     * If the key is not found, inserts a new element with the provided key and return the last value.
     *
     * @param k The key to access or insert.
     * @return Reference to the value associated with the key.
     */
    Tp& operator[](const  DT_CHAR *k)
    {
        iterator it = find(k);
        if (it != end())
        {
            return it->second;
        }

        DT_BOOL ret = insert(k, Tp{});
        if (!ret)
        {
            AURA_XTENSA_LOG("insert(%s) failed!\n", k);
        }

        return m_data[m_size - 1].second;
    }

    /**
     * @brief Overloaded subscript operator for accessing or inserting const elements by key.
     *
     * Returns a reference to the const value associated with the specified key.
     * If the key is not found, return the last value.
     *
     * @param k The key(char pointer) to access or insert.
     * @return Const reference to the const value associated with the key.
     */
    const Tp& operator[](const  DT_CHAR *k) const
    {
        const_iterator it = find(k);
        if (it != end())
        {
            return it->second;
        }

        AURA_XTENSA_LOG("[%s] failed! can not found the key in map!\n", k);

        return m_data[m_size - 1].second;
    }

    /**
     * @brief Overloaded subscript operator for accessing or inserting elements by key.
     *
     * @param k The key(string) to access or insert.
     * @return Reference to the value associated with the key.
     */
    Tp& operator[](const string_<KEY_MAX_SIZE> &k)
    {
        return operator[](k.c_str());
    }

    /**
     * @brief Inserts a key-value pair into the map.
     *
     * Inserts the specified key-value pair into the map if the size limit is not exceeded.
     *
     * @param k The key to insert.
     * @param v The value to insert.
     * @return True if the insertion was successful, otherwise False.
     */
    DT_BOOL insert(const DT_CHAR *k, const Tp &v)
    {
        if (m_size >= MAP_MAX_SIZE)
        {
            AURA_XTENSA_LOG("The map size reach the max size!\n");
            return DT_FALSE;
        }

        for (DT_S32 i = 0; i < m_size; ++i)
        {
            if (0 == Strcmp(m_data[i].first.c_str(), k))
            {
                m_data[i].second = v;
                return DT_TRUE;
            }
        }

        m_data[m_size++] = { k, v };

        return DT_TRUE;
    }

    /**
     * @brief Inserts a key-value pair into the map.
     *
     * Inserts the specified key-value pair into the map if the size limit is not exceeded.
     *
     * @param k The key to insert.
     * @param v The value to insert.
     * @return True if the insertion was successful, otherwise False.
     */
    DT_BOOL insert(const string_<KEY_MAX_SIZE> &k, const Tp &v)
    {
        return insert(k.c_str(), v);
    }

    /**
     * @brief Erases the element with the specified key from the map.
     *
     * @param k The key of the element to erase.
     * @return True if the erase was successful (element found and removed), otherwise False.
     */
    DT_BOOL erase(const string_<KEY_MAX_SIZE> &k)
    {
        for (DT_S32 i = 0; i < m_size; ++i)
        {
            if (0 == Strcmp(m_data[i].first.c_str(), k.c_str()))
            {
                for (DT_S32 j = i; j < m_size - 1; ++j)
                {
                    m_data[j].first = m_data[j + 1].first;
                    m_data[j].second = m_data[j + 1].second;
                }
                --m_size;

                return DT_TRUE;
            }
        }

        return DT_FALSE;
    }

    /**
     * @brief Erases the element with the specified key from the map.
     *
     * @param k The key of the element to erase.
     * @return True if the erase was successful (element found and removed), otherwise False.
     */
    DT_BOOL erase(const DT_CHAR *k)
    {
        string_<KEY_MAX_SIZE> str(k);
        return erase(str);
    }

    /**
     * @brief Accesses the value associated with the specified key.
     *
     * @param k The key(char pointer) to access.
     * @return Reference to the value associated with the key.
     */
    Tp& at(const DT_CHAR *k)
    {
        for (DT_S32 i = 0; i < m_size; ++i)
        {
            if (0 == Strcmp(m_data[i].first.c_str(), k))
            {
                return m_data[i].second;
            }
        }

        AURA_XTENSA_LOG("At(%s) failed! can not found the key in map!\n", k);

        return m_data[m_size].second;
    }

    /**
     * @brief Accesses the const value associated with the specified key.
     *
     * @param k The key(char pointer) to access.
     * @return Const reference to the const value associated with the key.
     */
    const Tp& at(const DT_CHAR *k) const
    {
        for (DT_S32 i = 0; i < m_size; ++i)
        {
            if (0 == Strcmp(m_data[i].first.c_str(), k))
            {
                return m_data[i].second;
            }
        }

        AURA_XTENSA_LOG("At(%s) failed! can not found the key in map!\n", k);

        return m_data[m_size].second;
    }

    /**
     * @brief Accesses the value associated with the specified key.
     *
     * @param k The key(string) to access.
     * @return Reference to the value associated with the key.
     */
    Tp& at(const string_<KEY_MAX_SIZE> &k)
    {
        return at(k.c_str());
    }

    /**
     * @brief Accesses the const value associated with the specified key.
     *
     * @param k The key(string) to access.
     * @return Const reference to the value associated with the key.
     */
    const Tp& at(const string_<KEY_MAX_SIZE> &k) const
    {
        return at(k.c_str());
    }

    /**
     * @brief Finds the iterator pointing to the element with the specified key.
     *
     * @param k The key(char pointer) to search for.
     * @return Iterator pointing to the element if found, otherwise end() iterator.
     */
    iterator find(const DT_CHAR *k)
    {
        for (DT_S32 i = 0; i < m_size; ++i)
        {
            if (0 == Strcmp(m_data[i].first.c_str(), k))
            {
                return iterator(m_data, i);
            }
        }

        return end();
    }

    /**
     * @brief Finds the const_iterator pointing to the element with the specified key.
     *
     * @param k The key(char pointer) to search for.
     * @return Const_iterator pointing to the element if found, otherwise end() const_iterator.
     */
    const_iterator find(const DT_CHAR *k) const
    {
        for (DT_S32 i = 0; i < m_size; ++i)
        {
            if (0 == Strcmp(m_data[i].first.c_str(), k))
            {
                return const_iterator(m_data, i);
            }
        }

        return end();
    }

    /**
     * @brief Finds the iterator pointing to the element with the specified key.
     *
     * @param k The key(string) to search for.
     * @return Iterator pointing to the element if found, otherwise end() iterator.
     */
    iterator find(const string_<KEY_MAX_SIZE> &k)
    {
        return find(k.c_str());
    }

    /**
     * @brief Finds the const_iterator pointing to the element with the specified key.
     *
     * @param k The key(string) to search for.
     * @return Const_iterator pointing to the element if found, otherwise end() const_iterator.
     */
    const_iterator find(const string_<KEY_MAX_SIZE> &k) const
    {
        return find(k.c_str());
    }

    /**
     * @brief Count the occurrences of a specified key within the container.
     *
     * @param k The key (char pointer) to search for.
     * @return The number of occurrences of the specified key within the container.
     */
    DT_S32 count(const DT_CHAR *k)
    {
        DT_S32 cnt = 0;
        for (DT_S32 i = 0; i < m_size; ++i)
        {
            if (0 == Strcmp(m_data[i].first.c_str(), k))
            {
                cnt++;
            }
        }

        return cnt;
    }

    /**
     * @brief Count the occurrences of a specified key within the container.
     *
     * @param k The key (string) to search for.
     * @return The number of occurrences of the specified key within the container.
     */
    DT_S32 count(const string_<KEY_MAX_SIZE> &k)
    {
        return count(k.c_str());
    }

    /**
     * @brief Gets the number of elements in the map.
     *
     * @return The number of elements in the map.
     */
    DT_S32 size() const
    {
        return m_size;
    }

    /**
     * @brief Checks if the map is empty.
     *
     * @return True if the map is empty, otherwise False.
     */
    DT_BOOL empty()
    {
        if (0 == m_size)
        {
            return DT_TRUE;
        }
        return DT_FALSE;
    }

    /**
     * @brief Clears all elements from the map.
     *
     * Removes all elements from the map, leaving it with a size of 0.
     */
    void clear()
    {
        m_size = 0;
    }

private:
    std::pair<string_<KEY_MAX_SIZE>, Tp> m_data[MAP_MAX_SIZE]; /*!< Array of key-value pairs. */
    DT_S32                               m_size;               /*!< Current size of the map. */
};

/**
 * @}
*/
} //namespace xtensa
} //namespace aura

#endif //AURA_RUNTIME_CORE_XTENSA_TYPES_MAP_HPP__