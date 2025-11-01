#ifndef AURA_UTILS_RUNTIME_SYSTRACE_HPP__
#define AURA_UTILS_RUNTIME_SYSTRACE_HPP__

#include "aura/utils/core.h"
#include "aura/utils/core/types/built-in.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <mutex>

namespace aura::utils
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
    Systrace(AURA_BOOL enable_trace);

    /**
     * @brief Destructor for Systrace.
     */
    ~Systrace();

    /**
     * @brief Begins a synchronous system trace event.
     *
     * @param tag The tag for the system trace event.
     */
    AURA_VOID Begin(const AURA_CHAR *tag);

    /**
     * @brief Ends a synchronous system trace event.
     *
     * @param tag The tag for the system trace event.
     */
    AURA_VOID End(const AURA_CHAR *tag);

    /**
     * @brief Begins an asynchronous system trace event.
     *
     * @param tag The tag for the asynchronous system trace event.
     * @param cookie A unique identifier for the asynchronous event.
     */
    AURA_VOID AsyncBegin(const AURA_CHAR *tag, AURA_S32 cookie);

    /**
     * @brief Ends an asynchronous system trace event.
     *
     * @param tag The tag for the asynchronous system trace event.
     * @param cookie A unique identifier for the asynchronous event.
     */
    AURA_VOID AsyncEnd(const AURA_CHAR *tag, AURA_S32 cookie);

    /**
     * @brief Disable copy and assignment constructor operations.
     */
    AURA_DISABLE_COPY_ASSIGN(Systrace);

private:
    std::mutex m_handle_lock; /*!< Mutex for thread safety. */
    AURA_S32 m_handle; /*!< Handle for system trace events. */
};

} // namespace aura::utils

/**
 * @}
 */

#endif // AURA_UTILS_RUNTIME_SYSTRACE_HPP__
