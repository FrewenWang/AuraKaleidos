#ifndef AURA_RUNTIME_UTILS_SYSTRACE_HPP__
#define AURA_RUNTIME_UTILS_SYSTRACE_HPP__

#include "aura/runtime/core.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <mutex>

/**
 * @defgroup runtime Runtime
 * @{
 *      @defgroup utils Utils
 *      @{
 *          @defgroup systrace Systrace
 *      @}
 * @}
*/

/**
 * @addtogroup systrace
 * @{
*/

/**
 * @brief Macro to begin a system trace event.
 *
 * @param ctx The context associated with the system trace.
 * @param tag The tag for the system trace event.
 */
#define AURA_SYSTRACE_BEGIN(ctx, tag)                                                                  \
    do {                                                                                               \
        if (ctx && ctx->GetSystrace())                                                                 \
        {                                                                                              \
            ctx->GetSystrace()->Begin(tag);                                                            \
        }                                                                                              \
    } while (0)

/**
 * @brief Macro to end a system trace event.
 *
 * @param ctx The context associated with the system trace.
 * @param tag The tag for the system trace event.
 */
#define AURA_SYSTRACE_END(ctx, tag)                                                                   \
    do {                                                                                              \
        if (ctx && ctx->GetSystrace())                                                                \
        {                                                                                             \
            ctx->GetSystrace()->End(tag);                                                             \
        }                                                                                             \
    } while (0)

/**
 * @brief Macro to begin an asynchronous system trace event.
 *
 * @param ctx The context associated with the system trace.
 * @param tag The tag for the asynchronous system trace event.
 * @param cookie A unique identifier for the asynchronous event.
 */
#define AURA_SYSTRACE_ASYNC_BEGIN(ctx, tag, cookie)                                                   \
    do {                                                                                              \
        if (ctx && ctx->GetSystrace())                                                                \
        {                                                                                             \
            ctx->GetSystrace()->AsyncBegin(tag, cookie);                                              \
        }                                                                                             \
    } while (0)

/**
 * @brief Macro to end an asynchronous system trace event.
 *
 * @param ctx The context associated with the system trace.
 * @param tag The tag for the asynchronous system trace event.
 * @param cookie A unique identifier for the asynchronous event.
 */
#define AURA_SYSTRACE_ASYNC_END(ctx, tag, cookie)                                                     \
    do {                                                                                              \
        if (ctx && ctx->GetSystrace())                                                                \
        {                                                                                             \
            ctx->GetSystrace()->AsyncEnd(tag, cookie);                                                \
        }                                                                                             \
    } while (0)

namespace aura
{

/**
 * @brief Class for handling system trace events.
 *
 * This class is for managing system trace events' beginning and ending.
 */
class AURA_EXPORTS Systrace
{
public:
    /**
     * @brief Constructor for Systrace class.
     *
     * @param enable_trace Flag indicating whether system tracing is enabled.
     */
    Systrace(MI_BOOL enable_trace);

    /**
     * @brief Destructor for Systrace.
     */
    ~Systrace();

    /**
     * @brief Begins a synchronous system trace event.
     *
     * @param tag The tag for the system trace event.
     */
    AURA_VOID Begin(const MI_CHAR *tag);

    /**
     * @brief Ends a synchronous system trace event.
     *
     * @param tag The tag for the system trace event.
     */
    AURA_VOID End(const MI_CHAR *tag);

    /**
     * @brief Begins an asynchronous system trace event.
     *
     * @param tag The tag for the asynchronous system trace event.
     * @param cookie A unique identifier for the asynchronous event.
     */
    AURA_VOID AsyncBegin(const MI_CHAR *tag, MI_S32 cookie);

    /**
     * @brief Ends an asynchronous system trace event.
     *
     * @param tag The tag for the asynchronous system trace event.
     * @param cookie A unique identifier for the asynchronous event.
     */
    AURA_VOID AsyncEnd(const MI_CHAR *tag, MI_S32 cookie);

    /**
     * @brief Disable copy and assignment constructor operations.
     */
    AURA_DISABLE_COPY_AND_ASSIGN(Systrace);

private:
    std::mutex m_handle_lock;   /*!< Mutex for thread safety. */
    MI_S32 m_handle;            /*!< Handle for system trace events. */
};

} // namespace aura

/**
 * @}
*/

#endif // AURA_RUNTIME_UTILS_SYSTRACE_HPP__