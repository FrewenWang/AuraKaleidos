#ifndef AURA_RUNTIME_MEMORY_MEM_POOL_HPP__
#define AURA_RUNTIME_MEMORY_MEM_POOL_HPP__

#include "aura/runtime/memory/allocator.hpp"
#include "aura/runtime/context.h"

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
 * @brief Macro to free memory using the AURA memory pool.
 *
 * This macro is a convenient way to free memory using the AURA memory pool.
 * It delegates to the internal FreeInternal function, providing the associated context and the pointer to free.
 *
 * @param ctx The pointer to the Context object.
 * @param ptr A pointer to the memory to free.
 */
#define AURA_FREE(ctx, ptr)                         aura::FreeInternal(ctx, ptr)

/**
 * @brief Macro to allocate memory with specified parameters using the AURA memory pool.
 *
 * This macro simplifies memory allocation with the AURA memory pool by invoking the internal AllocateInternal function.
 * It takes into account the memory type, size, alignment, and includes information about the source file, function, and line number for tracing purposes.
 *
 * @param ctx The pointer to the Context object.
 * @param type The type of memory allocation.
 * @param size The size of the memory to allocate.
 * @param align The alignment of the memory.
 * 
 * @return A pointer to the allocated memory.
 */
#define AURA_ALLOC_PARAM(ctx, type, size, align)    aura::AllocateInternal(ctx, type, size, align, __FILE__, __FUNCTION__, __LINE__)

/**
 * @brief Macro to allocate memory with default parameters using the AURA memory pool.
 *
 * This macro provides a convenient way to allocate memory using the AURA memory pool with default parameters.
 * It internally uses the AURA_ALLOC_PARAM macro, setting the memory type to AURA_MEM_DEFAULT and alignment to 0.
 *
 * @param ctx The pointer to the Context object.
 * @param size The size of the memory to allocate.
 * 
 * @return A pointer to the allocated memory.
 */
#define AURA_ALLOC(ctx, size)    AURA_ALLOC_PARAM(ctx, AURA_MEM_DEFAULT, size, 0)

/**
 * @}
 */

namespace aura
{
/**
 * @addtogroup memory
 * @{
 */

/**
 * @brief Memory Pool Class.
 *
 * The `MemPool` class provides a flexible memory management system, encompassing efficient allocation and
 * deallocation operations. It supports memory tracing and accommodates various allocators to handle different
 * memory types effectively.
 */
class AURA_EXPORTS MemPool
{
public:
    /**
     * @brief Constructor for MemPool.
     *
     * @param ctx The pointer to the Context object.
     */
    MemPool(Context *ctx);

    /**
     * @brief Destructor for MemPool.
     */
    ~MemPool();

    /**
     * @brief Deleted copy constructor and assignment operator.
     */
    AURA_DISABLE_COPY_AND_ASSIGN(MemPool);

    /**
     * @brief Allocate memory from the memory pool.
     *
     * @param type The type of memory allocation.
     * @param size The size of the memory to allocate.
     * @param align The alignment of the memory.
     * @param file The file where the allocation is requested (for tracing).
     * @param func The function where the allocation is requested (for tracing).
     * @param line The line number where the allocation is requested (for tracing).
     * 
     * @return A pointer to the allocated memory.
     */
    AURA_VOID* Allocate(MI_S32 type, MI_S64 size, MI_S32 align,
                      const MI_CHAR *file, const MI_CHAR *func, MI_S32 line);

    /**
     * @brief Free memory allocated by the memory pool.
     *
     * @param ptr A pointer to the memory to free.
     * 
     * @return Status::OK if successful; otherwise, an appropriate error status.
     */
    Status Free(AURA_VOID *ptr);

    /**
     * @brief Map a buffer for access.
     *
     * @param buffer The buffer to map.
     * 
     * @return Status::OK if successful; otherwise, an appropriate error status.
     */
    Status Map(const Buffer &buffer);

    /**
     * @brief Unmap a buffer.
     *
     * @param buffer The buffer to unmap.
     * 
     * @return Status::OK if successful; otherwise, an appropriate error status.
     */
    Status Unmap(const Buffer &buffer);

    /**
     * @brief Get a Buffer instance associated with a memory pointer.
     *
     * @param ptr The memory pointer.
     * 
     * @return The Buffer instance.
     */
    Buffer GetBuffer(AURA_VOID *ptr);

    /**
     * @brief Register an allocator into memory pool.
     *
     * @param type The type of the allocator.
     * @param allocator A pointer to the allocator.
     * 
     * @return Status::OK if successful; otherwise, an appropriate error status.
     */
    Status RegisterAllocator(MI_S32 type, Allocator *allocator);

    /**
     * @brief Unregister an allocator from the memory pool.
     *
     * @param type The type of the allocator.
     * 
     * @return Status::OK if successful; otherwise, an appropriate error status.
     */
    Status UnregisterAllocator(MI_S32 type);

    /**
     * @brief Get the allocator associated with a type.
     *
     * @param type The type of the allocator.
     * 
     * @return A pointer to the allocator.
     */
    Allocator* GetAllocator(MI_S32 type);

    /**
     * @brief Enable or disable memory tracing.
     *
     * @param enable `MI_TRUE` to enable memory tracing, `MI_FALSE` to disable.
     */
    AURA_VOID MemTraceSet(MI_BOOL enable);

    /**
     * @brief Begin a memory trace with a specific tag.
     *
     * @param tag The tag for the memory trace.
     */
    AURA_VOID MemTraceBegin(const std::string &tag);

    /**
     * @brief End a memory trace with a specific tag.
     *
     * @param tag The tag for the memory trace.
     */
    AURA_VOID MemTraceEnd(const std::string &tag);

    /**
     * @brief Clear all memory traces.
     */
    AURA_VOID MemTraceClear();

    /**
     * @brief Generate a memory trace report.
     *
     * @return The memory trace report as a string.
     */
    std::string MemTraceReport();

private:
    class Impl;     /*!< Forward declaration of the implementation class. */
    Impl  *m_impl;  /*!< Pointer to the implementation class. */
};

/**
 * @brief Internal function to allocate memory using the associated context's memory pool.
 *
 * @param ctx The pointer to the Context object.
 * @param type The type of memory allocation.
 * @param size The size of the memory to allocate.
 * @param align The alignment of the memory.
 * @param file The file where the allocation is requested (for tracing).
 * @param func The function where the allocation is requested (for tracing).
 * @param line The line number where the allocation is requested (for tracing).
 * 
 * @return A pointer to the allocated memory.
 */
AURA_INLINE AURA_VOID* AllocateInternal(Context *ctx, MI_S32 type, MI_S64 size, MI_S32 align,
                                      const MI_CHAR *file, const MI_CHAR *func, MI_S32 line)
{
    if (ctx && ctx->GetMemPool())
    {
        return (ctx)->GetMemPool()->Allocate(type, size, align, file, func, line);
    }
    return MI_NULL;
}

/**
 * @brief Internal function to free memory allocated using the associated context's memory pool.
 *
 * @param ctx The pointer to the Context object.
 * @param ptr A pointer to the memory to free.
 * 
 * @return Status::OK if successful; otherwise, an appropriate error status.
 */
AURA_INLINE Status FreeInternal(Context *ctx, AURA_VOID *ptr)
{
    if (MI_NULL == ptr)
    {
        return Status::OK;
    }

    if (ctx && ctx->GetMemPool())
    {
        return (ctx)->GetMemPool()->Free(ptr);
    }
    return Status::ERROR;
}

/**
 * @brief Create a new instance of type Tp using the associated context's memory pool.
 *
 * @tparam Tp The type to create.
 * @tparam ArgsType The types of the arguments to pass to the constructor.
 *
 * @param ctx The pointer to the Context object.
 * @param args The arguments to pass to the constructor.
 * 
 * @return A pointer to the created instance.
 */
template <typename Tp, typename ...ArgsType>
AURA_INLINE Tp* Create(Context *ctx, ArgsType &&...args)
{
    Tp *ptr = MI_NULL;
    if (ctx)
    {
        AURA_VOID *buffer = AURA_ALLOC_PARAM(ctx, AURA_MEM_HEAP, sizeof(Tp), 0);
        if (buffer != MI_NULL)
        {
            ptr = new(buffer) Tp(ctx, std::forward<ArgsType>(args)...);
        }
    }

    return ptr;
}

/**
 * @brief Delete an instance of type Tp and free its memory using the associated context's memory pool.
 *
 * @tparam Tp The type to delete.
 *
 * @param ctx The pointer to the Context object.
 * @param ptr A pointer to the instance to delete.
 */
template <typename Tp>
AURA_INLINE AURA_VOID Delete(Context *ctx, Tp **ptr)
{
    if (ctx != MI_NULL && ptr != MI_NULL && *ptr != MI_NULL)
    {
        (*ptr)->~Tp();
        AURA_FREE(ctx, *ptr);
        *ptr = MI_NULL;
    }
}

/**
 * @}
 */
} // namespace aura

#endif // AURA_RUNTIME_MEMORY_MEM_POOL_HPP__