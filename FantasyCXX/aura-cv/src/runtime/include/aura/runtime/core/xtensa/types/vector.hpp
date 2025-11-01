#ifndef AURA_RUNTIME_CORE_XTENSA_TYPES_VECTOR_HPP__
#define AURA_RUNTIME_CORE_XTENSA_TYPES_VECTOR_HPP__

#include "aura/runtime/core/xtensa/types/iterator.hpp"

#include <initializer_list>

/**
 * @defgroup runtime Runtime
 * @{
 *      @defgroup core Runtime Core
 *      @{
 *           @defgroup vector Runtime Core Xtensa Vector
 *      @}
 * @}
*/

#define AURA_VECTOR_DEFAULT_MAX_SIZE    (16)

namespace aura
{
namespace xtensa
{
/**
 * @addtogroup vector
 * @{
*/

/**
 * @brief vector class.
 *
 * The vector class is a template class that represents a dynamic array of elements of type Tp.
 *
 * @tparam Tp The type of elements in the vector.
 * @tparam MAX_SIZE The maximum size of the vector.
 */
template <typename Tp, MI_S32 MAX_SIZE = AURA_VECTOR_DEFAULT_MAX_SIZE>
class vector
{
public:
    /**
     * @brief Default constructor for creating an empty vector.
     */
    vector() : m_size(0)
    {}

    /**
     * @brief Initializes the vector with an initializer list of initial elements.
     *
     * @param init_list The initializer list of initial elements.
     */
    vector(std::initializer_list<Tp> init_list)
    {
        if (init_list.size() > MAX_SIZE)
        {
            AURA_XTENSA_LOG("the input list size > MAX_SIZE!\n");
        }

        m_size = init_list.size() > MAX_SIZE ? MAX_SIZE : init_list.size();
        MI_S32 i = 0;
        for (const auto& value : init_list)
        {
            if (i >= m_size)
            {
                break;
            }

            m_data[i] = value;
            ++i;
        }
    }

    /**
     * @brief Initializes the vector with a specified size and default value.
     *
     * @param size The size of the vector.
     * @param default_value The default value.
     */
    explicit vector(MI_S32 size, Tp default_value = Tp())
    {
        if (size > MAX_SIZE)
        {
            AURA_XTENSA_LOG("the input size > MAX_SIZE!\n");
        }

        m_size = size > MAX_SIZE ? MAX_SIZE : size;
        for (MI_S32 i = 0; i < m_size; ++i)
        {
            m_data[i] = default_value;
        }
    }

    /**
     * @brief Iterator type for vector.
     *
     * Iterator type for iterating over elements in the vector.
     */
    using iterator = Iterator<Tp>;

    /**
     * @brief Iterator type for const vector.
     *
     * Const iterator type for iterating over elements in the vector.
     */
    using const_iterator = Iterator<const Tp>;

    /**
     * @brief Iterator type for reverse vector.
     *
     * Iterator type for iterating over elements in the vector in reverse order.
     */
    using reverse_iterator = ReverseIterator<Tp>;

    /**
     * @brief Iterator type for const reverse vector.
     *
     * Const iterator type for iterating over elements in the vector in reverse order.
     */
    using const_reverse_iterator = ReverseIterator<const Tp>;

    /**
     * @brief Initializes the vector with elements from the iterator range.
     *
     * @tparam InputIt The input iterator type.
     * @param first The starting iterator.
     * @param last The ending iterator.
     */
    vector(iterator first, iterator last)
    {
        m_size = 0;
        while (first != last && m_size < MAX_SIZE)
        {
            m_data[m_size] = *first;
            ++m_size;
            ++first;
        }

        if (first != last)
        {
            AURA_XTENSA_LOG("the input list size > MAX_SIZE!\n");
        }
    }

    /**
     * @brief Copy constructor.
     * Copies the contents of another vector.
     *
     * @param other The vector to copy.
     */
    vector(const vector &other)
    {
        m_size = other.m_size;
        if (m_size > MAX_SIZE)
        {
            AURA_XTENSA_LOG("the input vector size > MAX_SIZE!\n");
        }

        for (MI_S32 i = 0; i < m_size; ++i)
        {
            m_data[i] = other.m_data[i];
        }
    }

    /**
     * @brief Copy assignment operator.
     * Assigns the contents of another vector to this one.
     *
     * @param other The vector to copy.
     * @return Reference to this vector after assignment.
     */
    vector& operator=(vector &&other)
    {
        if (this != &other)
        {
            m_size = other.m_size;
            for (MI_S32 i = 0; i < m_size; ++i)
            {
                m_data[i] = std::move(other.m_data[i]);
            }
            other.m_size = 0;
        }

        return *this;
    }

    /**
     * @brief Returns an iterator pointing to the first element in the vector.
     *
     * @return An iterator pointing to the beginning of the vector.
     */
    iterator begin()
    {
        return iterator(m_data, 0);
    }

    /**
     * @brief Returns an iterator pointing to the past-the-end element in the vector.
     *
     * @return An iterator pointing to the element following the last element of the vector.
     */
    iterator end()
    {
        return iterator(m_data, m_size);
    }

    /**
     * @brief Returns an const iterator pointing to the first element in the vector.
     *
     * @return An const iterator pointing to the beginning of the vector.
     */
    const_iterator begin() const
    {
        return const_iterator(m_data, 0);
    }

    /**
     * @brief Returns an const iterator pointing to the past-the-end element in the vector.
     *
     * @return An const iterator pointing to the element following the last element of the vector.
     */
    const_iterator end() const
    {
        return const_iterator(m_data, m_size);
    }

    /**
     * @brief Returns a reverse iterator pointing to the first element of the vector in reverse order.
     *
     * @return Reverse iterator pointing to the first element of the vector in reverse order.
     */
    reverse_iterator rbegin()
    {
        return reverse_iterator(m_data, m_size - 1);
    }

    /**
     * @brief Returns a reverse iterator referring to the past-the-end element of the vector in reverse order.
     *
     * @return Reverse iterator referring to the past-the-end element of the vector in reverse order.
     */
    reverse_iterator rend()
    {
        return reverse_iterator(m_data, -1);
    }

    /**
     * @brief Returns a const reverse iterator pointing to the first element of the vector in reverse order.
     *
     * @return Const reverse iterator pointing to the first element of the vector in reverse order.
     */
    const_reverse_iterator rbegin() const
    {
        return const_reverse_iterator(m_data, m_size - 1);
    }

    /**
     * @brief Returns a const reverse iterator referring to the past-the-end element of the vector in reverse order.
     *
     * @return Const reverse iterator referring to the past-the-end element of the vector in reverse order.
     */
    const_reverse_iterator rend() const
    {
        return const_reverse_iterator(m_data, -1);
    }

    /**
     * @brief Check if the vector is empty.
     *
     * @return True if the vector is empty, otherwise False.
     */
    MI_BOOL empty() const
    {
        return 0 == m_size;
    }

    /**
     * @brief Get the size of the vector.
     *
     * @return The number of elements in the vector.
     */
    MI_S32 size() const
    {
        return m_size;
    }

    /**
     * @brief Resize the VdspVector to contain a specific number of elements.
     *
     * @param size The new size of the vector.
     * @param default_value The value to initialize new elements with (if resizing larger).
     */
    AURA_VOID resize(MI_S32 size, Tp default_value = Tp())
    {
        if (size <= m_size)
        {
            m_size = size;
        }
        else
        {
            MI_S32 i = m_size;
            for (; i < size && i < MAX_SIZE; ++i)
            {
                m_data[i] = default_value;
            }
            m_size = i;
        }
    }

    /**
     * @brief Returns a pointer to the underlying array serving as element storage.
     *
     * @return A pointer to the underlying element storage.
     */
    const Tp* data() const
    {
        return m_data;
    }

    /**
     * @brief Returns a pointer to the underlying array serving as element storage.
     *
     * @return A pointer to the underlying element storage.
     */
    Tp* data()
    {
        return m_data;
    }

    /**
     * @brief Get the capacity of the vector.
     *
     * @return The maximum number of elements the vector can hold.
     */
    MI_S32 capacity() const
    {
        return MAX_SIZE;
    }

    /**
     * @brief Pushes an item to the back of the vector.
     *
     * @param value The value to push to the back of the vector.
     *
     * @return True if the push was successful, otherwise False (if the vector is full).
     */
    MI_BOOL push_back(const Tp &value)
    {
        if (m_size >= MAX_SIZE)
        {
            AURA_XTENSA_LOG("The vector size reach the max size!\n");
            return MI_FALSE;
        }
        m_data[m_size++] = value;

        return MI_TRUE;
    }

    /**
     * @brief Removes the last element from the vector.
     *
     * @return True if the pop was successful (vector is not empty), otherwise False.
     */
    MI_BOOL pop_back()
    {
        if (0 == m_size)
        {
            return MI_FALSE;
        }
        --m_size;

        return MI_TRUE;
    }

    /**
     * @brief Accesses the element at the specified index in the vector.
     *
     * @param index The index of the element to access.
     *
     * @return Reference to the element at the specified index.
     */
    Tp& at(MI_S32 index)
    {
        if (index >= m_size)
        {
            AURA_XTENSA_LOG("call at() failed! index = %d m_size:%d\n", index, m_size);
            return m_data[m_size - 1];
        }

        return m_data[index];
    }

    /**
     * @brief Accesses the const element at the specified index in the vector.
     *
     * @param index The index of the const element to access.
     *
     * @return Const reference to the const element at the specified index.
     */
    const Tp& at(MI_S32 index) const
    {
        if (index >= m_size)
        {
            AURA_XTENSA_LOG("call at() failed! index = %d\n", index);
            return m_data[m_size - 1];
        }

        return m_data[index];
    }

    /**
     * @brief Accesses the element at the specified index in the vector.
     *
     * This function provides array-like access to elements of the vector.
     *
     * @param index The index of the element to access.
     *
     * @return Reference to the element at the specified index.
     */
    Tp& operator[](MI_S32 index)
    {
        return at(index);
    }

    /**
     * @brief Accesses the const element at the specified index in the vector.
     *
     * This function provides array-like access to elements of the vector.
     *
     * @param index The index of the const element to access.
     *
     * @return Const reference to the const element at the specified index.
     */
    const Tp& operator[](MI_S32 index) const
    {
        return at(index);
    }

    /**
     * @brief Accesses the first element of the vector.
     *
     * @return Reference to the first element of the vector.
     */
    Tp& front()
    {
        return m_data[0];
    }

    /**
     * @brief Accesses the last element of the vector.
     *
     * @return Reference to the last element of the vector.
     */
    Tp& back()
    {
        return m_data[m_size - 1];
    }

    /**
     * @brief Accesses the first element of the const vector.
     *
     * @return Const reference to the first element of the vector.
     */
    const Tp& front() const
    {
        return m_data[0];
    }

    /**
     * @brief Accesses the last element of the const vector.
     *
     * @return Const reference to the last element of the vector.
     */
    const Tp& back() const
    {
        return m_data[m_size - 1];
    }

    /**
     * @brief Clears all elements from the vector.
     */
    void clear()
    {
        m_size = 0;
    }

    /**
     * @brief Inserts an element at a specified position in the vector.
     *
     * @param position The position at which to insert the element.
     * @param value The value of the element to insert.
     *
     * @return True if the insertion was successful, otherwise False (if the vector is full or position is invalid).
     */
    MI_BOOL insert(MI_S32 position, const Tp &value)
    {
        if (m_size >= MAX_SIZE)
        {
            AURA_XTENSA_LOG("The vector size reach the max size!\n");
            return MI_FALSE;
        }

        if ((position < 0) || (position > m_size))
        {
            AURA_XTENSA_LOG("Invalid position!\n");
            return MI_FALSE;
        }

        for (int i = m_size; i > position; --i)
        {
            m_data[i] = m_data[i - 1];
        }

        m_data[position] = value;
        ++m_size;

        return MI_TRUE;
    }

    /**
     * @brief erases an element at the specified index from the vector.
     *
     * @param index The index of the element to erase.
     *
     * @return True if the erase was successful (index is valid), otherwise False.
     */
    MI_BOOL erase(MI_S32 index)
    {
        if ((index < 0) || (index >= m_size))
        {
            AURA_XTENSA_LOG("Invalid index!\n");
            return MI_FALSE;
        }

        for (MI_S32 i = index; i < m_size - 1; ++i)
        {
            m_data[i] = m_data[i + 1];
        }
        --m_size;

        return MI_TRUE;
    }

public:
    Tp      m_data[MAX_SIZE];
    MI_S32  m_size;
};

/**
 * @}
*/
} //namespace xtensa
} //namespace aura

#endif //AURA_RUNTIME_CORE_XTENSA_TYPES_VECTOR_HPP__