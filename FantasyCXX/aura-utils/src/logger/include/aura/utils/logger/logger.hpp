#ifndef AURA_UTILS_LOG_HPP__
#define AURA_UTILS_LOG_HPP__

#include "aura/utils/core.h"
#include "log.hpp"

#include <memory>
#include <string>

/**
 * @brief Log a message without context, tag, log level, and format.
 *
 * @param tag    The tag associated with the log message.
 * @param level  The log level (e.g. ERROR, INFO, DEBUG).
 * @param format The format string for the log message.
 * @param ...    Additional arguments for formatting the log message.
 *
 * @note  If on Android, the log print can only support 1023 characters maximum per line.
 *        For long message, consider using '\n' to split into multiple lines or use multiple log prints.
 *
 * @code
 * AURA_PRINT(TAG, LogLevel::DEBUG, "%s \n %s \n %s \n", str1, str2, str3);
 * @endcode
 */
#define AURA_PRINT(tag, level, format, ...)                                                                            \
    do                                                                                                                 \
    {                                                                                                                  \
        aura::utils::Print(tag, level, format, ##__VA_ARGS__);                                                         \
    } while (0)

/**
 * @brief Log an error message with file, function, and line information without context.
 */
#define AURA_PRINTE(tag, format, ...)                                                                                  \
    AURA_PRINT(tag, aura::utils::LogLevel::ERROR, "%s[%s:%d] " format, __FUNCTION__, __FILE__, __LINE__, ##__VA_ARGS__)

/**
 * @brief Log an informational message without context.
 */
#define AURA_PRINTI(tag, format, ...) AURA_PRINT(tag, aura::utils::LogLevel::INFO, format, ##__VA_ARGS__)

/**
 * @brief Log a debug message without context.
 */
#define AURA_PRINTD(tag, format, ...) AURA_PRINT(tag, aura::utils::LogLevel::DEBUG, format, ##__VA_ARGS__)

/**
 * @brief Logs a message with a specified context, tag, log level, and format.
 *
 * @param ctx    The pointer to the Context object.
 * @param tag    The tag associated with the log message.
 * @param level  The log level (e.g. ERROR, INFO, DEBUG).
 * @param format The format string for the log message.
 * @param ...    Additional arguments for formatting the log message.
 *
 * @note  If on Android, the log print can only support 1023 characters maximum per line.
 *        For long message, consider using '\n' to split into multiple lines or use multiple log prints.
 *
 * @code
 * AURA_LOG(ctx, TAG, LogLevel::DEBUG, "%s \n %s \n %s \n", str1, str2, str3);
 * @endcode
 */
#define AURA_LOG(ctx, tag, level, format, ...)                                                                         \
    do                                                                                                                 \
    {                                                                                                                  \
        if ((ctx) && (ctx)->GetLogger())                                                                               \
            (ctx)->GetLogger()->Log(tag, level, format, ##__VA_ARGS__);                                                \
        else                                                                                                           \
            AURA_PRINT(tag, level, format, ##__VA_ARGS__);                                                             \
    } while (0)


/**
 * @brief Logs an error message with file, function, and line information.
 */
#define AURA_LOGE(ctx, tag, format, ...)                                                                               \
    AURA_LOG(ctx, tag, aura::utils::LogLevel::ERROR, "%s[%s:%d] " format, __FUNCTION__, __FILE__, __LINE__,            \
             ##__VA_ARGS__)

/**
 * @brief Logs an informational message.
 */
#define AURA_LOGI(ctx, tag, format, ...) AURA_LOG(ctx, tag, aura::utils::LogLevel::INFO, format, ##__VA_ARGS__)

/**
 * @brief Logs a debug message.
 */
#define AURA_LOGD(ctx, tag, format, ...) AURA_LOG(ctx, tag, aura::utils::LogLevel::DEBUG, format, ##__VA_ARGS__)

/**
 * @brief Adds an error string to the logger, including file, function, and line information.
 * 如果返回的是错误信息，则我们在错误信息里面添加日志信息。并且打印出来具体的行数
 * @param ctx  The pointer to the Context object.
 * @param info The error information to be added.
 */
#define AURA_ADD_ERROR_STRING(ctx, info)                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        if ((ctx) && (ctx)->GetLogger())                                                                               \
            (ctx)->GetLogger()->AddErrorString(__FILE__, __FUNCTION__, __LINE__, info);                                \
    } while (0)

/**
 * @brief Returns with error handling, adding an error string to the logger if the status is ERROR.
 * 返回错误处理，如果状态为 ERROR，则向 Logger 添加错误字符串。
 * @param ctx The pointer to the Context object.
 * @param ret The status to be checked.
 */
#define AURA_RETURN(ctx, ret)                                                                                          \
    do                                                                                                                 \
    {                                                                                                                  \
        if (aura::utils::Status::ERROR == ret)                                                                         \
        {                                                                                                              \
            AURA_ADD_ERROR_STRING(ctx, "fail");                                                                        \
        }                                                                                                              \
        return ret;                                                                                                    \
    } while (0)

namespace aura::utils
{

AURA_EXPORTS AURA_VOID Print(const AURA_CHAR *tag, LogLevel level, const AURA_CHAR *format, ...);

AURA_EXPORTS AURA_VOID StdoutPrint(const AURA_CHAR *tag, LogLevel level, const AURA_CHAR *format, ...);

AURA_EXPORTS AURA_VOID FilePrint(FILE *fp, const AURA_CHAR *tag, const AURA_CHAR *format, ...);

AURA_EXPORTS AURA_VOID LogcatPrint(const AURA_CHAR *tag, LogLevel level, const AURA_CHAR *format, ...);

/**
 * @brief Logger class for handling logging and error string management.
 *
 * This class provides functionality for logging messages, adding error strings, and retrieving error information.
 */
class AURA_EXPORTS Logger
{
public:
    /**
     * @brief Constructor for the Logger class.
     *
     * @param output Log output option.
     * @param level Log level.
     * @param file The file associated with the logger.
     */
    Logger(LogOutput output, LogLevel level, const std::string &file);

    /**
     * @brief Destructor for the Logger class.
     */
    ~Logger();

    /**
     * @brief Logs a formatted message with the specified tag, log level, and format.
     *
     * @param tag The tag associated with the log message.
     * @param level The log level for the message.
     * @param format The format string for the log message.
     * @param ... Additional arguments for formatting the log message.
     */
    AURA_VOID Log(const AURA_CHAR *tag, LogLevel level, const AURA_CHAR *format, ...);

    /**
     * @brief Adds an error string to the logger.
     *
     * @param file The file associated with the error.
     * @param func The function name associated with the error.
     * @param line The line number associated with the error.
     * @param info Additional information about the error.
     */
    AURA_VOID AddErrorString(const AURA_CHAR *file, const AURA_CHAR *func, AURA_S32 line, const AURA_CHAR *info);

    /**
     * @brief Get the accumulated error string.
     *
     * @return The accumulated error string.
     */
    std::string GetErrorString();

    /**
     * @brief Disable copy and assignment constructor.
     */
    AURA_DISABLE_COPY_ASSIGN(Logger);

private:
    class Impl; /*!< Forward declaration of the logger implementation class. */
    std::shared_ptr<Impl> m_impl; /*!< Pointer to the logger implementation class. */
};

} // namespace aura::utils

#endif // AURA_UTILS_LOG_HPP__
