#ifndef AURA_RUNTIME_UTILS_THREAD_OBJECT_HPP__
#define AURA_RUNTIME_UTILS_THREAD_OBJECT_HPP__

#include "aura/runtime/memory.h"

#if !defined(AURA_BUILD_XPLORER)
#  include <thread>
#  include <mutex>
#endif // AURA_BUILD_XPLORER

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

/**
 * @defgroup runtime Runtime
 * @{
 *      @defgroup utils Utils
 *      @{
 *          @defgroup thread_object Thread Object
 *      @}
 * @}
*/

namespace aura
{
/**
 * @addtogroup thread_object
 * @{
*/

/**
 * @brief Class template for managing thread objects of type Tp within a given context.
 *
 * This class allows the creation, retrieval, and release of thread objects of type Tp.
 *
 * @tparam Tp The type of object managed by ThreadObject.
 */
template <typename Tp>
class ThreadObject
{
public:
    /**
     * @brief Constructor for ThreadObject.
     *
     * @param ctx The context associated with ThreadObject.
     */
    ThreadObject(Context *ctx) : m_ctx(ctx), m_last_node(&m_first_node)
    {}

    /**
     * @brief Constructor for ThreadObject.
     *
     * @tparam ArgsType Parameter pack for forwarding arguments to the object creation.
     * 
     * @param ctx  The context associated with ThreadObject.
     * @param num  The number of objects to create and manage.
     * @param args The arguments forwarded to the object creation.
     */
    template <typename ...ArgsType>
    ThreadObject(Context *ctx, std::vector<THREAD_ID> &thread_ids, ArgsType &&...args) : m_ctx(ctx), m_last_node(&m_first_node)
    {
        for (auto &id : thread_ids)
        {
            Tp *new_object = DT_NULL;

            new_object = GetObject(id, std::forward<ArgsType>(args)...);

            if (DT_NULL == new_object)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "Failed to get object.");
            }
        }
    }

    /**
     * @brief Destructor for ThreadObject.
     *
     * Deletes all managed objects upon destruction.
     */
    ~ThreadObject()
    {
        Clear();
    }

    /**
     * @brief Disable copy and assignment constructor operations.
     */
    AURA_DISABLE_COPY_AND_ASSIGN(ThreadObject);

    /**
    * @brief Retrieves an object of type `Tp` associated with a specific thread.
    * 
    * This function searches through a linked list of `Node` nodes to find 
    * an object associated with the specified thread ID. If the object is found, a pointer 
    * to it is returned; otherwise, `DT_NULL` is returned.
    *
    * @param id The thread identifier. If not provided, defaults to `THIS_THREAD_ID`.
    * 
    * @return Tp* A pointer to the object of type `Tp` associated with the specified thread, 
    *             or `DT_NULL` if the object is not found.
    */
    Tp* GetObject(const THREAD_ID id = THIS_THREAD_ID)
    {
        // only find
        Node *cur_node = m_first_node.next;
        while (cur_node != DT_NULL)
        {
            if (id == cur_node->thread_id)
            {
                return cur_node->object;
            }

            cur_node = cur_node->next;
        }

        return DT_NULL;
    }

    /**
    * @brief Retrieves or creates an object of type `Tp` associated with a specific thread.
    * 
    * This function returns a pointer to an object of type `Tp` associated with the 
    * thread specified by the given thread ID. If the object does not exist, it will 
    * be created using the provided arguments.
    *
    * @tparam ArgsType The types of the arguments used to create the object.
    * 
    * @param id The thread identifier. If not provided, defaults to `THIS_THREAD_ID`.
    * @param args Arguments forwarded to the constructor of `Tp` if the object needs to be created.
    * 
    * @return Tp* A pointer to the object of type `Tp` associated with the specified thread.
    */
    template <typename ...ArgsType>
    Tp* GetObject(const THREAD_ID id, ArgsType &&...args)
    {
        // 1. find
        Tp *ret_object = GetObject(id);
        if (ret_object != DT_NULL)
        {
            return ret_object;
        }

        // 2. insert
        {
#if !defined(AURA_BUILD_XPLORER)
            std::unique_lock<std::mutex> lock(m_mutex);
#endif // AURA_BUILD_XPLORER

            Tp *new_object = Create<Tp>(m_ctx, std::forward<ArgsType>(args)...);

            if (DT_NULL == new_object)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "Failed to create object.");
                return DT_NULL;
            }

            Node *new_node = static_cast<Node*>(AURA_ALLOC(m_ctx, sizeof(Node)));
            if (DT_NULL == new_node)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "Node create failed.");
                Delete<Tp>(m_ctx, &new_object);
                return DT_NULL;
            }
            else
            {
                new_node->thread_id = id;
                new_node->object    = new_object;
                new_node->next      = DT_NULL;
            }

            m_last_node->next = new_node;
            m_last_node       = new_node;

            return new_object;
        }

        return DT_NULL;
    }

    /**
     * @brief Releases all Tp objects.
     */
    DT_VOID Clear()
    {
        Node *cur_node = m_first_node.next;

        while (cur_node != DT_NULL)
        {
            Node *next = cur_node->next;

            Delete<Tp>(m_ctx, &(cur_node->object));
            AURA_FREE(m_ctx, cur_node);

            cur_node = next;
        }
    }

private:
    struct Node
    {
        Node() : object(DT_NULL), next(DT_NULL)
        {}

        Node(const THREAD_ID id, Tp *data, Node *next_obj = DT_NULL)
             : thread_id(id), object(data), next(next_obj)
        {}

        THREAD_ID thread_id;
        Tp        *object;
        Node      *next;
    };

    Context    *m_ctx;        // The associated context.
#if !defined(AURA_BUILD_XPLORER)
    std::mutex m_mutex;      // The mutex for m_last_node.
#endif // AURA_BUILD_XPLORER
    Node       m_first_node; // Sentinel node for the Node linked list.
    Node       *m_last_node;  // Pointer to the last node in the Node linked list.
};

/**
 * @}
*/
} // namespace aura
#endif // AURA_RUNTIME_UTILS_THREAD_OBJECT_HPP__
