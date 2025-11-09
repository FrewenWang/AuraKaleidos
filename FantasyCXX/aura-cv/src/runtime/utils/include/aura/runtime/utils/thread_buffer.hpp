#ifndef AURA_RUNTIME_UTILS_HOST_THREAD_BUFFER_HPP__
#define AURA_RUNTIME_UTILS_HOST_THREAD_BUFFER_HPP__

#include "aura/runtime/context.h"
#include "aura/runtime/memory.h"
#include "aura/runtime/logger.h"

#if !defined(AURA_BUILD_XPLORER)
#  include <thread>
#  include <mutex>
#endif // AURA_BUILD_XPLORER

/**
 * @defgroup runtime Runtime
 * @{
 *      @defgroup utils Utils
 *      @{
 *          @defgroup thread_buffer Thread Buffer
 *      @}
 * @}
*/

#if defined(AURA_BUILD_HEXAGON)
#  define THREAD_ID                 qurt_thread_t
#  define THIS_THREAD_ID            qurt_thread_get_id()
#elif defined(AURA_BUILD_XPLORER)
#  define  THREAD_ID                DT_S32
#  define  THIS_THREAD_ID           0
#else
#  define  THREAD_ID                std::thread::id
#  define  THIS_THREAD_ID           std::this_thread::get_id()
#endif // AURA_BUILD_HEXAGON

namespace aura
{
/**
 * @addtogroup thread_buffer
 * @{
*/

/**
 * @brief Class for managing thread buffers.
 *
 * This class allows the reservation, retrieval, and release of thread buffers.
 */
class AURA_EXPORTS ThreadBuffer
{
public:
    /**
    * @brief Constructs a ThreadBuffer with a specified buffer size and memory type.
    * 
    * @param ctx Pointer to the Context object.
    * @param size Size of the buffer in bytes.
    * @param mem_type Type of memory to be allocated (default is AURA_MEM_DEFAULT).
    */
    ThreadBuffer(Context *ctx, DT_S32 size, DT_S32 mem_type = AURA_MEM_DEFAULT)
                 : m_ctx(ctx), m_buffer_size(size), m_mem_type(mem_type), m_last_node(&m_first_node)
    {}

    /**
    * @brief Constructs a ThreadBuffer for specific threads with a specified buffer size and memory type.
    * 
    * @param ctx Pointer to the Context object.
    * @param thread_ids Vector of thread IDs for which the buffer is allocated.
    * @param size Size of the buffer in bytes.
    * @param mem_type Type of memory to be allocated (default is AURA_MEM_DEFAULT).
    */
    ThreadBuffer(Context *ctx, const std::vector<THREAD_ID> &thread_ids, DT_S32 size, DT_S32 mem_type = AURA_MEM_DEFAULT)
                 : m_ctx(ctx), m_buffer_size(size), m_mem_type(mem_type), m_last_node(&m_first_node)
    {
        for (auto &id : thread_ids)
        {
            Buffer buffer = GetThreadBuffer(id);

            if (!buffer.IsValid())
            {
                AURA_ADD_ERROR_STRING(m_ctx, "created thread buffer invalid ");
            }
        }
    }

    ~ThreadBuffer()
    {
        Clear();
    }

    AURA_DISABLE_COPY_AND_ASSIGN(ThreadBuffer);

    DT_VOID Clear()
    {
        Node *current_node = m_first_node.next;
        while (current_node != DT_NULL)
        {
            Node *next = current_node->next;

            AURA_FREE(m_ctx, current_node->buffer.m_origin);
            AURA_FREE(m_ctx, current_node);

            current_node = next;
        }
    }

    /**
    * @brief Retrieves the buffer associated with the specified thread ID.
    *
    * This function looks up the buffer associated with a given thread ID from
    * an internal map. If the buffer for the specified thread ID does not exist,
    * it will attempt to allocate a new buffer, initialize it, and store it in the
    * map. The function also handles memory allocation failures and invalid buffer 
    * scenarios, returning an empty `Buffer` object in such cases.
    *
    * @param id The thread ID for which to retrieve the buffer. If not provided,
    *           the current thread's ID is used as the default.
    * 
    * @return A `Buffer` object associated with the provided thread ID. If the
    *         buffer allocation or initialization fails, an empty `Buffer` is returned.
    */
    Buffer GetThreadBuffer(const THREAD_ID id = THIS_THREAD_ID)
    {
        // 1. check if the buffer is already in the list
        Node *current_node = m_first_node.next;
        while (current_node != DT_NULL)
        {
            if (id == current_node->thread_id)
            {
                return current_node->buffer;
            }

            current_node = current_node->next;
        }

        // 2. insert the buffer into the list
        {
#if !defined(AURA_BUILD_XPLORER)
            std::unique_lock<std::mutex> lock(m_mutex);
#endif // AURA_BUILD_XPLORER

            DT_VOID *buffer_data = AURA_ALLOC_PARAM(m_ctx, m_mem_type, m_buffer_size, 0);
            if (DT_NULL == buffer_data)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "ThreadBuffer allocate memory failed.");
                return Buffer();
            }

            Buffer buffer(m_mem_type, m_buffer_size, m_buffer_size, buffer_data, buffer_data, 0);

            if (!buffer.IsValid())
            {
                AURA_ADD_ERROR_STRING(m_ctx, "buffer invalid.");
                AURA_FREE(m_ctx, buffer_data);
                return Buffer();
            }

            Node *new_node = static_cast<Node*>(AURA_ALLOC(m_ctx, sizeof(Node)));
            if (DT_NULL == new_node)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "Node create failed.");
                AURA_FREE(m_ctx, buffer_data);
                return Buffer();
            }
            else
            {
                new_node->thread_id = id;
                new_node->buffer    = buffer;
                new_node->next      = DT_NULL;
            }

            m_last_node->next = new_node;
            m_last_node       = new_node;

            return buffer;
        }

        return Buffer();
    }

    /**
    * @brief Retrieves the thread-specific data of type `Tp`.
    *
    * This template function retrieves the buffer associated with a specific thread ID 
    * and interprets the buffer's data as a pointer to the specified type `Tp`. If the 
    * buffer is invalid, an error message is added to the context and a `nullptr` is returned.
    *
    * @tparam Tp The type to which the buffer data should be cast.
    * 
    * @param id The thread ID for which to retrieve the data. If not provided, the current thread's ID 
    *           is used as the default.
    * 
    * @return A pointer to the data of type `Tp` associated with the thread, or `nullptr` 
    *         if the buffer is invalid.
    */
    template <typename Tp>
    Tp* GetThreadData(const THREAD_ID id = THIS_THREAD_ID)
    {
        Buffer buffer = GetThreadBuffer(id);

        if (!buffer.IsValid())
        {
            AURA_ADD_ERROR_STRING(m_ctx, "buffer invalid.");
            return DT_NULL;
        }

        return reinterpret_cast<Tp*>(buffer.m_data);
    }

    /**
     * @brief Returns the size of the buffer.
     *
     * @return The size of the buffer in bytes.
     */
    DT_S32 GetBufferSize() const
    {
        return m_buffer_size;
    }

    /**
     * @brief Returns the memory type of the buffer.
     *
     * @return The memory type of the buffer.
     */
    DT_S32 GetMemType() const
    {
        return m_mem_type;
    }

private:
    struct Node
    {
        Node() : next(DT_NULL)
        {}

        Node(const THREAD_ID id, const Buffer &buffer, Node *next = DT_NULL)
             : thread_id(id), buffer(buffer), next(next)
        {}

        THREAD_ID thread_id;
        Buffer    buffer;
        Node      *next;
    };

    Context    *m_ctx;         // The associated context.
    DT_S32     m_buffer_size; // The size of the each buffer.
    DT_S32     m_mem_type;    // The memory type.
#if !defined(AURA_BUILD_XPLORER)
    std::mutex m_mutex;       // The mutex for m_last_node.
#endif // AURA_BUILD_XPLORER
    Node       m_first_node;  // Sentinel node for the Node linked list.
    Node       *m_last_node;   // Pointer to the last node in the Node linked list.
};

}
#endif // AURA_RUNTIME_UTILS_HOST_THREAD_BUFFER_HPP__