#ifndef AURA_RUNTIME_MEMORY_ALLOCATOR_HPP__
#define AURA_RUNTIME_MEMORY_ALLOCATOR_HPP__

#include "aura/runtime/core.h"

#include <iostream>
#include <sstream>
#include <string>

/**
 * @defgroup runtime Runtime
 * @{
 *    @defgroup memory Memory
 * @}
 */

/**
 * @addtogroup memory
 * @{
 */

/**
 * @brief Invalid memory type.
*/
#define AURA_MEM_INVALID            (0)

/**
 * @brief Heap memory type.  普通的堆内存
*/
#define AURA_MEM_HEAP               (1)

/**
 * @brief DMA buffer heap memory type.   DMA缓存区堆内存
*/
#define AURA_MEM_DMA_BUF_HEAP       (2)

/**
 * @brief SVM (Shared Virtual Memory) memory type.  共享虚拟内存
*/
#define AURA_MEM_SVM                (3)

/**
 * @brief VTCM (Vector Tightly Coupled Memory) memory type.  HVX中的向量紧耦合内存
*/
#define AURA_MEM_VTCM               (4)

/**
 * @brief Reserved memory type.         预留的内存类型
*/
#define AURA_MEM_RESERVE            (255)

/**
 * 下面的定义是指的在编译的ANDROID系统中，默认使用DMA buffer
 * 其他的平台下默认使用的普通堆内存
 * @brief Default memory type used.
 */
#if defined(AURA_BUILD_ANDROID)
#  define AURA_MEM_DEFAULT                  AURA_MEM_DMA_BUF_HEAP
#else
#  define AURA_MEM_DEFAULT                  AURA_MEM_HEAP
#endif

/**
 * @}
 */

namespace aura
{
/**
 * @addtogroup memory
 * @{
 */

AURA_INLINE std::string MemTypeToString(const DT_S32 mem_type)
{
    if (mem_type == AURA_MEM_INVALID)
    {
        return "Invalid Buffer";
    }
    if (mem_type == AURA_MEM_HEAP)
    {
        return "Heap Buffer";
    }
    if (mem_type == AURA_MEM_DMA_BUF_HEAP)
    {
        return "DMA Buffer";
    }
    if (mem_type == AURA_MEM_SVM)
    {
        return "SVM";
    }
    if (mem_type == AURA_MEM_VTCM)
    {
        return "VTCM";
    }

    if (mem_type <= AURA_MEM_RESERVE)
    {
        return "User Defined Buffer";
    }

    return "Illegal Buffer Type";
}

/**
 * @brief Memory Buffer Class.
 *
 * The `Buffer` class encapsulates the concept of a memory buffer, providing comprehensive information about its type,
 * capacity, size, data, origin, and property. Buffers are fundamental in managing memory, particularly in scenarios
 * involving memory-intensive operations such as iaura processing or data manipulation. This class facilitates the
 * construction, destruction, comparison, resizing, and extraction of key information related to memory buffers.
 */
class AURA_EXPORTS Buffer
{
public:
    /**
     * @brief Default constructor for Buffer.
     */
    Buffer() : m_type(AURA_MEM_INVALID), m_capacity(0), m_size(0),
               m_data(DT_NULL), m_origin(DT_NULL), m_property(0)
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
    Buffer(DT_S32 type, DT_S64 capacity, DT_S64 size,
           DT_VOID *data, DT_VOID *origin, DT_S32 property)
           : m_type(type), m_capacity(capacity), m_size(size),
             m_data(data), m_origin(origin), m_property(property)
    {}

    /**
     * @brief Destructor for Buffer.
     */
    ~Buffer()
    {
        Clear();
    }

    AURA_EXPORTS friend std::ostream& operator<<(std::ostream &os, const Buffer &buffer)
    {
         os << "Buffer:" << std::endl
            << "  |- type           : " << MemTypeToString(buffer.m_type)  << std::endl
            << "  |- capacity       : " << buffer.m_capacity               << std::endl
            << "  |- size           : " << buffer.m_size                   << std::endl
            << "  |- data           : " << buffer.m_data                   << std::endl
            << "  |- origin         : " << buffer.m_origin                 << std::endl
            << "  |- property       : " << buffer.m_property               << std::endl;

        return os;
    }

    /**
     * @brief Converts the Buffer to a string representation.
     *
     * @return The string representation of the Buffer.
     */
    std::string ToString()
    {
        std::ostringstream ss;
        ss << (*this);
        return ss.str();
    }

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
     * @return `DT_TRUE` if the Buffers are equal, `DT_FALSE` otherwise.
     */
    AURA_EXPORTS friend DT_BOOL operator==(const Buffer &buffer0, const Buffer &buffer1)
    {
        if (buffer0.m_type     == buffer1.m_type &&
            buffer0.m_capacity == buffer1.m_capacity &&
            buffer0.m_size     == buffer1.m_size &&
            buffer0.m_data     == buffer1.m_data &&
            buffer0.m_origin   == buffer1.m_origin &&
            buffer0.m_property == buffer1.m_property)
        {
            return DT_TRUE;
        }
        return DT_FALSE;
    }

    /**
     * @brief Get the offset of the buffer data from the origin.
     *
     * @return The offset.
     */
    DT_S32 GetOffset() const
    {
        DT_S32 offset = reinterpret_cast<DT_UPTR_T>(m_data) - reinterpret_cast<DT_UPTR_T>(m_origin);
        return offset;
    }

    /**
     * @brief Resize the buffer.
     *
     * @param size The shrinked size for resizing.
     * @param relative_offset The relative offset to current data pointer(m_data), not origin pointer(m_origin) (default is 0).
     *
     * @return Status::OK if successful; otherwise, an appropriate error status.
     */
    Status Resize(DT_S32 size, DT_S32 relative_offset = 0)
    {
        m_data = reinterpret_cast<DT_U8*>(m_data) + relative_offset;
        m_size = size;
        return Status::OK;
    }

    /**
     * @brief Check if the buffer is valid.
     *
     * @return `DT_TRUE` if the buffer is valid, `DT_FALSE` otherwise.
     */
    DT_BOOL IsValid() const
    {
        return ((m_type != AURA_MEM_INVALID) && (m_origin != DT_NULL)
                && (m_data != DT_NULL) && (m_capacity >= m_size) && (m_size > 0));
    }

    /**
     * @brief Clear the buffer.
     */
    DT_VOID Clear()
    {
        m_type     = AURA_MEM_INVALID;
        m_capacity = 0;
        m_size     = 0;
        m_data     = DT_NULL;
        m_origin   = DT_NULL;
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
    Tp GetData(DT_S32 offset = 0)
    {
        if (std::is_pointer<Tp>::value)
        {
            return reinterpret_cast<Tp>(reinterpret_cast<DT_U8*>(m_data) + offset);
        }
        else
        {
            return reinterpret_cast<Tp*>(reinterpret_cast<DT_U8*>(m_data) + offset)[0];
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
    const Tp GetData(DT_S32 offset = 0) const
    {
        if (std::is_pointer<Tp>::value)
        {
            return reinterpret_cast<Tp>(reinterpret_cast<DT_U8*>(m_data) + offset);
        }
        else
        {
            return reinterpret_cast<Tp*>(reinterpret_cast<DT_U8*>(m_data) + offset)[0];
        }
    }

    DT_S32 m_type;      /*!< The type of the buffer. */
    DT_S64 m_capacity;  /*!< The capacity of the buffer. */
    DT_S64 m_size;      /*!< The size of the buffer. */

    DT_VOID *m_data;    /*!< A pointer to the data of the buffer. */
    DT_VOID *m_origin;  /*!< A pointer to the origin of the buffer. */
    DT_S32  m_property; /*!< The property of the buffer. */
};

/**
 * @brief Allocator Class.
 *
 * The `Allocator` class serves as a fundamental component in the memory management framework, offering a versatile
 * interface for memory allocation, deallocation, mapping, and unmapping. This class is designed to be a generic base
 * for implementing various memory allocation strategies, providing users with the flexibility to choose memory
 * management policies that align with specific requirements.
 *
 * Key Responsibilities:
 * - Allocate and free memory buffers.
 * - Map and unmap memory for direct access.
 *
 * @note This class is intended to be subclassed for concrete implementations of different memory allocation strategies.
 */
class AURA_EXPORTS Allocator
{
public:
    /**
     * @brief Constructor for Allocator.
     *
     * @param type The type of the allocator.
     * @param name The name of the allocator.
     */
    Allocator(DT_S32 type, const std::string &name) : m_type(type), m_name(name)
    {
        AURA_UNUSED(m_type);
    }

    /**
     * @brief Destructor for Allocator.
     */
    virtual ~Allocator(DT_VOID)
    {}

    /**
     * @brief Get the name of the allocator.
     *
     * @return The name of the allocator.
     */
    std::string GetName()
    {
        return m_name;
    }

    /**
     * @brief Deleted copy constructor and assignment operator.
     */
    AURA_DISABLE_COPY_AND_ASSIGN(Allocator);

    /**
     * @brief Allocate a buffer using the allocator.
     *
     * @param size The size of the buffer to allocate.
     * @param align The alignment of the buffer (default is 0).
     *
     * @return The allocated buffer.
     */
    virtual Buffer Allocate(DT_S64 size, DT_S32 align = 0) = 0;

    /**
     * @brief Free a buffer allocated by the allocator.
     *
     * @param buffer The buffer to free.
     */
    virtual DT_VOID Free(Buffer &buffer) = 0;

    /**
     * @brief Map a buffer for access.
     *
     * @param buffer The buffer to map.
     *
     * @return Status::OK if successful; otherwise, an appropriate error status.
     */
    virtual Status Map(const Buffer &buffer) = 0;

    /**
     * @brief Unmap a buffer.
     *
     * @param buffer The buffer to unmap.
     *
     * @return Status::OK if successful; otherwise, an appropriate error status.
     */
    virtual Status Unmap(const Buffer &buffer) = 0;

private:
    DT_S32 m_type;          /*!< The type of the allocator. */
    std::string m_name;     /*!< The name of the allocator. */
};

/**
 * @}
 */
} //namespace aura

#endif // AURA_RUNTIME_MEMORY_ALLOCATOR_HPP__
