#ifndef AURA_UTILS_LOGGER_HPP__
#define AURA_UTILS_LOGGER_HPP__

namespace aura::utils
{
/**
 * @brief Enumeration representing different log output options.
 */
enum class LogOutput
{
    STDOUT = 0, /*!< Standard output */
    FILE   = 1, /*!< Log to a file */
    LOGCAT = 2, /*!< Android logcat */
};

/**
 * @brief Enumeration representing different log levels.
 */
enum class LogLevel
{
    NONE    = 0, // 不输出日志
    VERBOSE = 1, // 包含更细节的调试信息，包括文件、函数和行号，发版时应当禁止输出
    DEBUG   = 2, // 输出调试相关的信息
    INFO    = 3, // 输出执行流中的信息
    WARN    = 4, // 一个未预期的结果，但不会引起执行流终止
    ERROR   = 5, // 当前执行流因为错误被停止，当前应用不受影响
    FATAL   = 6, // 当前应用完全无法执行，需要退出
};


} // namespace aura::utils

#endif //
