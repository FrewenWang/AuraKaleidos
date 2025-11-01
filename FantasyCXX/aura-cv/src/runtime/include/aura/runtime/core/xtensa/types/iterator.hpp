#ifndef AURA_RUNTIME_CORE_XTENSA_TYPES_ITERATOR_HPP__
#define AURA_RUNTIME_CORE_XTENSA_TYPES_ITERATOR_HPP__

#include "aura/runtime/core/xtensa/comm.hpp"

/**
 * @defgroup runtime Runtime
 * @{
 *      @defgroup core Runtime Core
 *      @{
 *           @defgroup iterator Runtime Core Xtensa Iterator
 *      @}
 * @}
*/

namespace aura
{
namespace xtensa
{
/**
 * @addtogroup iterator
 * @{
*/

/**
 * @brief A generic iterator class for iterating over a collection of key-value pairs.
 *
 * This class provides functionality for iterating over a collection of key-value pairs.
 * It can be used with various data structures to traverse their elements.
 *
 * @tparam Tp The type of the iterator itself.
 */
template <typename Tp>
class Iterator
{
public:
    /**
     * @brief Constructs a generic iterator object with the given key-value pair array pointer and index.
     *
     * @param data Pointer to the array of key-value pairs.
     * @param index Index of the current element in the array.
     */
    Iterator(Tp *data, MI_S32 index) : m_data(data), m_index(index)
    {}

    /**
     * @brief Dereferences the iterator, returning a reference to the key-value pair.
     *
     * @return Reference to the key-value pair.
     */
    Tp& operator*() const
    {
        return m_data[m_index];
    }

    /**
     * @brief Dereferences the iterator, returning a pointer to the key-value pair.
     *
     * @return Pointer to the key-value pair.
     */
    Tp* operator->() const
    {
        return &(m_data[m_index]);
    }

    /**
     * @brief Equality comparison operator for iterators.
     *
     * @param other The iterator to compare with.
     * @return True if the iterators are equal, otherwise False.
     */
    MI_BOOL operator==(const Iterator &other)
    {
        return m_index == other.m_index;
    }
    /**
     * @brief Inequality comparison operator for iterators.
     *
     * @param other The iterator to compare with.
     * @return True if the iterators are not equal, otherwise False.
     */
    MI_BOOL operator!=(const Iterator &other)
    {
        return m_index != other.m_index;
    }

    /**
     * @brief Prefix increment operator for iterators.
     *
     * @return Reference to the incremented iterator.
     */
    Iterator& operator++()
    {
        m_index++;
        return *this;
    }

    /**
     * @brief Postfix increment operator for iterators.
     *
     * @return Copy of the iterator before incrementing.
     */
    Iterator operator++(int)
    {
        Iterator it = *this;
        m_index++;
        return it;
    }

    /**
     * @brief Prefix decrement operator for iterators.
     *
     * @return Reference to the decremented iterator.
     */
    Iterator& operator--()
    {
        m_index--;
        return *this;
    }

    /**
     * @brief Postfix decrement operator for iterators.
     *
     * @return Copy of the iterator before decrementing.
     */
    Iterator operator--(int)
    {
        Iterator it = *this;
        m_index--;
        return it;
    }

private:
    Tp     *m_data;  /*!< Pointer to the array of key-value pairs. */
    MI_S32 m_index;  /*!< Index of the current element in the array. */
};

template <typename Tp>
class ReverseIterator
{
public:
    /**
     * @brief Constructs a generic iterator object with the given key-value pair array pointer and index.
     *
     * @param data Pointer to the array of key-value pairs.
     * @param index Index of the current element in the array.
     */
    ReverseIterator(Tp *data, MI_S32 index) : m_data(data), m_index(index)
    {}

    /**
     * @brief Dereferences the iterator, returning a reference to the key-value pair.
     *
     * @return Reference to the key-value pair.
     */
    Tp& operator*() const
    {
        return m_data[m_index];
    }

    /**
     * @brief Dereferences the iterator, returning a pointer to the key-value pair.
     *
     * @return Pointer to the key-value pair.
     */
    Tp* operator->() const
    {
        return &(m_data[m_index]);
    }

    /**
     * @brief Equality comparison operator for iterators.
     *
     * @param other The iterator to compare with.
     * @return True if the iterators are equal, otherwise False.
     */
    MI_BOOL operator==(const ReverseIterator &other)
    {
        return m_index == other.m_index;
    }
    /**
     * @brief Inequality comparison operator for iterators.
     *
     * @param other The iterator to compare with.
     * @return True if the iterators are not equal, otherwise False.
     */
    MI_BOOL operator!=(const ReverseIterator &other)
    {
        return m_index != other.m_index;
    }

    /**
     * @brief Prefix increment operator for iterators.
     *
     * @return Reference to the incremented iterator.
     */
    ReverseIterator& operator++()
    {
        m_index--;
        return *this;
    }

    /**
     * @brief Postfix increment operator for iterators.
     *
     * @return Copy of the iterator before incrementing.
     */
    ReverseIterator operator++(int)
    {
        ReverseIterator it = *this;
        m_index--;
        return it;
    }

    /**
     * @brief Prefix decrement operator for iterators.
     *
     * @return Reference to the decremented iterator.
     */
    ReverseIterator& operator--()
    {
        m_index++;
        return *this;
    }

    /**
     * @brief Postfix decrement operator for iterators.
     *
     * @return Copy of the iterator before decrementing.
     */
    ReverseIterator operator--(int)
    {
        ReverseIterator it = *this;
        m_index++;
        return it;
    }

private:
    Tp     *m_data;  /*!< Pointer to the array of key-value pairs. */
    MI_S32 m_index;  /*!< Index of the current element in the array. */
};

/**
 * @}
*/
} //namespace xtensa
} //namespace aura

#endif //AURA_RUNTIME_CORE_XTENSA_TYPES_ITERATOR_HPP__