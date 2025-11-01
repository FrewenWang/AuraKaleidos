#ifndef AURA_RUNTIME_CORE_XTENSA_TYPES_BUFFER_HPP__
#define AURA_RUNTIME_CORE_XTENSA_TYPES_BUFFER_HPP__

#include "aura/runtime/core/xtensa/comm.hpp"

/**
 * @defgroup runtime Runtime
 * @{
 *      @defgroup core Runtime Core
 *      @{
 *           @defgroup buffer Runtime Core Xtensa Buffer
 *      @}
 * @}
*/

namespace aura
{
namespace xtensa
{
/**
 * @addtogroup buffer
 * @{
*/

/**
 * @brief Represents a buffer used for storing data.
 * 
 * This class provides functionalities to manage a buffer with detailed attributes
 * such as type, capacity, size, data pointer, origin pointer, and additional properties.
 */
class Buffer
{
public:
    /**
     * @brief Default constructor for Buffer.
     */
    Buffer() : m_type(AURA_XTENSA_MEM_INVALID), m_capacity(0), m_size(0),
               m_data(MI_NULL), m_origin(MI_NULL), m_property(0)
    {}

    /**
     * @brief Parameterized constructor for Buffer.
     *
     * @param type The type of the buffer.
     * @param capacity The capacity of the buffer.
     * @param size The size of the buffer.
     * @param data A pointer to the data of the buffer.
     * @param origin A pointer to the origin of the buffer.
     * @param property The property of the buffer.
     */
    Buffer(MI_S32 type, MI_S64 capacity, MI_S64 size,
           AURA_VOID *data, AURA_VOID *origin, MI_S32 property)
           : m_type(type), m_capacity(capacity), m_size(size),
             m_data(data), m_origin(origin), m_property(property)
    {}

    /**
     * @brief Copy constructor for Buffer.
     *
     * @param buffer The Buffer instance to copy.
     */
    Buffer(const Buffer &buffer)
    {
        m_type     = buffer.m_type;
        m_capacity = buffer.m_capacity;
        m_size     = buffer.m_size;
        m_data     = buffer.m_data;
        m_origin   = buffer.m_origin;
        m_property = buffer.m_property;
    }

    /**
     * @brief Assignment operator for Buffer.
     *
     * @param buffer The Buffer instance to assign.
     * 
     * @return Reference to the assigned Buffer.
     */
    Buffer& operator=(const Buffer &buffer)
    {
        m_type     = buffer.m_type;
        m_capacity = buffer.m_capacity;
        m_size     = buffer.m_size;
        m_data     = buffer.m_data;
        m_origin   = buffer.m_origin;
        m_property = buffer.m_property;

        return *this;
    }

    /**
     * @brief Equality operator for Buffer.
     *
     * @param buffer0 The first Buffer to compare.
     * @param buffer1 The second Buffer to compare.
     * 
     * @return `MI_TRUE` if the Buffers are equal, `MI_FALSE` otherwise.
     */
    friend MI_BOOL operator==(const Buffer &buffer0, const Buffer &buffer1)
    {
        if (buffer0.m_type     == buffer1.m_type &&
            buffer0.m_capacity == buffer1.m_capacity &&
            buffer0.m_size     == buffer1.m_size &&
            buffer0.m_data     == buffer1.m_data &&
            buffer0.m_origin   == buffer1.m_origin &&
            buffer0.m_property == buffer1.m_property)
        {
            return MI_TRUE;
        }
        return MI_FALSE;
    }

    /**
     * @brief Get the offset of the buffer data from the origin.
     *
     * @return The offset.
     */
    MI_S32 GetOffset() const
    {
        MI_S32 offset = reinterpret_cast<MI_UPTR_T>(m_data) - reinterpret_cast<MI_UPTR_T>(m_origin);
        return offset;
    }

    /**
     * @brief Resize the buffer.
     *
     * @param size The new size of the buffer.
     * @param offset The offset for resizing (default is 0).
     * 
     * @return MI_S32::OK if successful; otherwise, an appropriate error status.
     */
    MI_S32 Resize(MI_S32 size, MI_S32 offset = 0)
    {
        m_data = reinterpret_cast<MI_U8*>(m_data) + offset;
        m_size = size;
        return AURA_XTENSA_OK;
    }

    /**
     * @brief Check if the buffer is valid.
     *
     * @return `MI_TRUE` if the buffer is valid, `MI_FALSE` otherwise.
     */
    MI_BOOL IsValid() const
    {
        return ((m_type != AURA_XTENSA_MEM_INVALID) && (m_origin != MI_NULL)
                && (m_data != MI_NULL) && (m_capacity >= m_size) && (m_size > 0));
    }

    /**
     * @brief Clear the buffer.
     */
    AURA_VOID Clear()
    {
        m_type     = AURA_XTENSA_MEM_INVALID;
        m_capacity = 0;
        m_size     = 0;
        m_data     = MI_NULL;
        m_origin   = MI_NULL;
        m_property = 0;
    }

    /**
     * @brief Get data from the buffer.
     *
     * @tparam Tp The type of data to get.
     *
     * @param offset The offset for getting data (default is 0).
     * 
     * @return The data.
     */
    template<typename Tp>
    Tp GetData(MI_S32 offset = 0)
    {
        if (std::is_pointer<Tp>::value)
        {
            return reinterpret_cast<Tp>(reinterpret_cast<MI_U8*>(m_data) + offset);
        }
        else
        {
            return reinterpret_cast<Tp*>(reinterpret_cast<MI_U8*>(m_data) + offset)[0];
        }
    }

    /**
     * @brief Get constant data from the buffer.
     *
     * @tparam Tp The type of constant data to get.
     *
     * @param offset The offset for getting constant data (default is 0).
     * 
     * @return The constant data.
     */
    template<typename Tp>
    const Tp GetData(MI_S32 offset = 0) const
    {
        if (std::is_pointer<Tp>::value)
        {
            return reinterpret_cast<Tp>(reinterpret_cast<MI_U8*>(m_data) + offset);
        }
        else
        {
            return reinterpret_cast<Tp*>(reinterpret_cast<MI_U8*>(m_data) + offset)[0];
        }
    }

    MI_S32 m_type;      /*!< The type of the buffer. */
    MI_S64 m_capacity;  /*!< The capacity of the buffer. */
    MI_S64 m_size;      /*!< The size of the buffer. */

    AURA_VOID *m_data;    /*!< A pointer to the data of the buffer. */
    AURA_VOID *m_origin;  /*!< A pointer to the origin of the buffer. */
    MI_S32  m_property; /*!< The property of the buffer. */
};

/**
 * @}
*/
} // namespace xtensa
} // namespace aura

#endif // AURA_RUNTIME_CORE_XTENSA_TYPES_BUFFER_HPP__
