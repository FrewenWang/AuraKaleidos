#ifndef AURA_RUNTIME_CORE_LOG_HPP__
#define AURA_RUNTIME_CORE_LOG_HPP__

/**
 * @defgroup runtime Runtime
 * @{
 *      @defgroup core Runtime Core
 *      @{
 *           @defgroup types Runtime Core Types
 *      @}
 * @}
*/

namespace aura
{
/**
 * @addtogroup types
 * @{
*/

/**
 * @brief Enumeration representing different log output options.
 */
enum class LogOutput
{
    STDOUT = 0, /*!< Standard output */
    FILE,       /*!< Log to a file */
    LOGCAT,     /*!< Android logcat */
    FARF,       /*!< Qualcomm DSP's FARF logging */
};

/**
 * @brief Enumeration representing different log levels.
 */
enum class LogLevel
{
    ERROR    = 0,   /*!< Log level for errors */
    INFO,           /*!< Log level for informational messages */
    DEBUG,          /*!< Log level for debugging messages */
};

/**
 * @}
*/
} // namespace aura

#endif // AURA_RUNTIME_CORE_LOG_HPP__