#ifndef VISION_UTIL_LOG_H
#define VISION_UTIL_LOG_H

#define LOG_TAG "VisionNative"

#define FILENAME(name) strrchr(name, '/') ? strrchr(name, '/') + 1 : name

#include <cstdint>
#include <cstdio>
#include <stdarg.h>
#include <string>
#include <vector>

#if defined(BUILD_ANDROID) and !defined(LOG_TO_CONSOLE)
#include <android/log.h>
#define WRAP_TAG(tag) (std::string("VaNative-") + tag).c_str()
#define VLOGI(tag, ...) __android_log_print(ANDROID_LOG_INFO, WRAP_TAG(tag), __VA_ARGS__)
#define VLOGW(tag, ...) __android_log_print(ANDROID_LOG_WARN, WRAP_TAG(tag), __VA_ARGS__)
#define VLOGD(tag, ...) __android_log_print(ANDROID_LOG_DEBUG, WRAP_TAG(tag), __VA_ARGS__)
#define VLOGE(tag, ...) __android_log_print(ANDROID_LOG_ERROR, WRAP_TAG(tag), __VA_ARGS__)
#define VLOGF(tag, ...) __android_log_print(ANDROID_LOG_ERROR, WRAP_TAG(tag), __VA_ARGS__)
#define VLOGV(tag, ...)

#elif defined(BUILD_QNX) and defined(BUILD_SLOG)
#include "LogHelper.h"
#define WRAP_TAG(tag) (std::string("VaNative-") + tag).c_str()
#define VLOGV(tag, format, ...)                                                                                        \
    do {                                                                                                               \
        if (vision::Logger::getLogLevel() < vision::LogLevel::VERBOSE) {                                               \
            break;                                                                                                     \
        }                                                                                                              \
        if (vision::Logger::getLogDestValue() & static_cast<std::uint8_t>(0b100)) {                                    \
            logger_print_debug("[D] %s %s %s:%s:%d " #format, LOG_TAG, tag, FILENAME(__FILE__), __FUNCTION__,       \
                               __LINE__, ##__VA_ARGS__);                                                               \
        }                                                                                                              \
        if (vision::Logger::getLogDestValue() & static_cast<std::uint8_t>(0b011)) {                                    \
            vision::Logger::debug(tag, format, ##__VA_ARGS__);                                                        \
        }                                                                                                              \
    } while (0);
#define VLOGD(tag, format, ...)                                                                                        \
    do {                                                                                                               \
        if (vision::Logger::getLogLevel() < vision::LogLevel::DEBUGGER) {                                              \
            break;                                                                                                     \
        }                                                                                                              \
        if (vision::Logger::getLogDestValue() & static_cast<std::uint8_t>(0b100)) {                                    \
            logger_print_debug("[D] %s %s %s:%s:%d " #format, LOG_TAG, tag, FILENAME(__FILE__), __FUNCTION__,       \
                               __LINE__, ##__VA_ARGS__);                                                               \
        }                                                                                                              \
        if (vision::Logger::getLogDestValue() & static_cast<std::uint8_t>(0b011)) {                                    \
            vision::Logger::debug(tag, format, ##__VA_ARGS__);                                                         \
        }                                                                                                              \
    } while (0);
#define VLOGI(tag, format, ...)                                                                                        \
    do {                                                                                                               \
        if (vision::Logger::getLogLevel() < vision::LogLevel::INFO) {                                                  \
            break;                                                                                                     \
        }                                                                                                              \
        if (vision::Logger::getLogDestValue() & static_cast<std::uint8_t>(0b100)) {                                    \
            logger_print_debug("[I->D] %s %s %s:%s:%d " #format, LOG_TAG, tag, FILENAME(__FILE__), __FUNCTION__,       \
                              __LINE__, ##__VA_ARGS__);                                                                \
        }                                                                                                              \
        if (vision::Logger::getLogDestValue() & static_cast<std::uint8_t>(0b011)) {                                    \
            vision::Logger::info(tag, format, ##__VA_ARGS__);                                                          \
        }                                                                                                              \
    } while (0);
#define VLOGW(tag, format, ...)                                                                                        \
    do {                                                                                                               \
        if (vision::Logger::getLogLevel() < vision::LogLevel::WARN) {                                                  \
            break;                                                                                                     \
        }                                                                                                              \
        if (vision::Logger::getLogDestValue() & static_cast<std::uint8_t>(0b100)) {                                    \
            logger_print_warning("[W] %s %s %s:%s:%d " #format, LOG_TAG, tag, FILENAME(__FILE__), __FUNCTION__,     \
                                 __LINE__, ##__VA_ARGS__);                                                             \
        }                                                                                                              \
        if (vision::Logger::getLogDestValue() & static_cast<std::uint8_t>(0b011)) {                                    \
            vision::Logger::warn(tag, format, ##__VA_ARGS__);                                                          \
        }                                                                                                              \
    } while (0);
#define VLOGE(tag, format, ...)                                                                                        \
    do {                                                                                                               \
        if (vision::Logger::getLogLevel() < vision::LogLevel::ERROR) {                                                 \
            break;                                                                                                     \
        }                                                                                                              \
        if (vision::Logger::getLogDestValue() & static_cast<std::uint8_t>(0b100)) {                                    \
            logger_print_error("[E] %s %s %s:%s:%d " #format, LOG_TAG, tag, FILENAME(__FILE__), __FUNCTION__,       \
                               __LINE__, ##__VA_ARGS__);                                                               \
        }                                                                                                              \
        if (vision::Logger::getLogDestValue() & static_cast<std::uint8_t>(0b011)) {                                    \
            vision::Logger::error(tag, format, ##__VA_ARGS__);                                                         \
        }                                                                                                              \
    } while (0);
#define VLOGF(tag, format, ...)                                                                                        \
    do {                                                                                                               \
        if (vision::Logger::getLogLevel() < vision::LogLevel::FATAL) {                                                 \
            break;                                                                                                     \
        }                                                                                                              \
        if (vision::Logger::getLogDestValue() & static_cast<std::uint8_t>(0b100)) {                                    \
            logger_print_critical("[F] %s %s %s:%s:%d " #format, LOG_TAG, tag, FILENAME(__FILE__), __FUNCTION__,    \
                                  __LINE__, ##__VA_ARGS__);                                                            \
        }                                                                                                              \
        if (vision::Logger::getLogDestValue() & static_cast<std::uint8_t>(0b011)) {                                    \
            vision::Logger::fatal(tag, format, ##__VA_ARGS__);                                                         \
        }                                                                                                              \
    } while (0);
#else
#define VLOGI(tag, format, ...) vision::Logger::info(tag, format, ##__VA_ARGS__)
#define VLOGW(tag, format, ...) vision::Logger::warn(tag, format, ##__VA_ARGS__)
#define VLOGD(tag, format, ...) vision::Logger::debug(tag, format, ##__VA_ARGS__)
#define VLOGE(tag, format, ...) vision::Logger::error(tag, format, ##__VA_ARGS__)
#define VLOGF(tag, format, ...) vision::Logger::fatal(tag, format, ##__VA_ARGS__)
#define VLOGV(tag, format, ...)                                                                                        \
    do {                                                                                                               \
        if (vision::Logger::getLogLevel() < vision::LogLevel::VERBOSE) {                                               \
            break;                                                                                                     \
        }                                                                                                              \
        vision::Logger::logHead(vision::LogLevel::VERBOSE);                                                            \
        vision::Logger::logPrint("in file: %s, function: %s, line: %d:\t", __FILE__, __FUNCTION__, __LINE__);          \
        vision::Logger::logPrint(format, ##__VA_ARGS__);                                                               \
        vision::Logger::logTail();                                                                                     \
    } while (0);
#endif

#define VLOG_START() vision::Logger::logStart()
#define VLOG_END() vision::Logger::logEnd()
#define VLOG_SET_FILE_NAME(file) vision::Logger::setLogFileName(file)

namespace aura::vision {
/**
 * @brief 日志等级
 */
enum class LogLevel {
    NONE = 0, // 不输出日志
    FATAL,    // 当前应用完全无法执行，需要退出
    ERROR,    // 当前执行流因为错误被停止，当前应用不受影响
    WARN,     // 一个未预期的结果，但不会引起执行流终止
    INFO,     // 输出执行流中的信息
    DEBUGGER, // 输出调试相关的信息
    VERBOSE   // 包含更细节的调试信息，包括文件、函数和行号，发版时应当禁止输出
};

/**
 * @brief 日志输出目的
 */
enum class LogDest {
    CONSOLE = 0b001, // 控制台输出
    LOGFILE = 0b010, // 文件输出
    SLOG = 0b100,    // slog输出
    ALL = 0b111      // 控制台、文件、slog都输出
};

/**
 * 简单的日志实现
 */
class Logger {
  public:
    static void fatal(const char *tag, const char *format, ...);
    static void error(const char *tag, const char *format, ...);
    static void warn(const char *tag, const char *format, ...);
    static void info(const char *tag, const char *format, ...);
    static void debug(const char *tag, const char *format, ...);

    /**
     * @brief 初始化Logger
     */
    static bool init();
    static void setLogLevel(const LogLevel &logLevel);
    static LogLevel getLogLevel() { return level; }
    static void setLogDest(const LogDest &logDest);
    static LogDest getLogDest() { return dest; }
    static std::uint8_t getLogDestValue() { return destValue; }
    static void setAppName(const std::string &app_name);

    /**
     * 设置能力层日志文件存储目录
     * @param fileName
     */
    static void setLogFileDir(const std::string &fileDir);

    /** 获取日志文件存储目录 */
    static const std::string getLogFileDir() { return logFileDir; }

    /**  获取当前日志文件的名称   */
    static const std::string getLogFileName() { return logFileName; }

    static void logStart();
    static void logEnd();

    static void log(const LogLevel &logLevel, const char *tag, const char *format, va_list args);
    static void logBody(const char *format, va_list args);
    static void logHead(const LogLevel &logLevel, const std::string &tag = "");
    static void logTail();
    static void logPrint(const char *format, ...);

  private:
    static std::string logLevelToStr(const LogLevel &logLevel);
    static LogLevel level;
    static LogDest dest;
    static std::uint8_t destValue;
    static std::string appName;
    static std::vector<std::string> levelStrings;
    /** 日志文件名称 */
    static std::string logFileName;
    /** 日志文件所在目录 */
    static std::string logFileDir;
    static FILE *logFile;
    static bool hasInit;
    static bool ensureInit();

    // 暂不对外提供设置日志文件名称的方法，日志文件由时间戳自动生成
    static void setLogFileName(const std::string &fileName);

    static bool openLogFile(const std::string &fileDir, const std::string &fileName);
};

} // namespace vision

#endif // VISION_UTIL_LOG_H
