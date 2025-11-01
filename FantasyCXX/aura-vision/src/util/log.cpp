#include "vision/util/log.h"

#include <cstdarg>
#include <cstdio>
#include <thread>
#include <vector>

#include "util/SystemClock.h"

#ifdef BUILD_SLOG
#include "LogHelper.h"
// slog日志级别定义
#define B_SLOG2_SHUTDOWN  0   /* Shut down the system NOW. eg: for OEM use */
#define B_SLOG2_CRITICAL  1   /* Unexpected unrecoverable error. eg: hard disk error */
#define B_SLOG2_ERROR     2   /* Unexpected recoverable error. eg: needed to reset a hw controller */
#define B_SLOG2_WARNING   3   /* Expected error. eg: parity error on a serial port */
#define B_SLOG2_NOTICE    4   /* Warnings. eg: Out of paper */
#define B_SLOG2_INFO      5   /* Information. eg: Printing page 3 */
#define B_SLOG2_DEBUG1    6   /* Debug messages eg: Normal detail */
#define B_SLOG2_DEBUG2    7   /* Debug messages eg: Fine detail */
#endif

namespace aura::vision {

LogLevel Logger::level = LogLevel::VERBOSE;
LogDest Logger::dest = LogDest::CONSOLE;
std::uint8_t Logger::destValue = static_cast<std::uint8_t>(dest);
std::string Logger::appName = "VisionNative"; // NOLINT
std::vector<std::string> Logger::levelStrings = {"NONE ", "FATAL", "ERROR", "WARN ", "INFO ", "DEBUG", "VERBO"}; // NOLINT
std::string Logger::logFileDir = "/data/vision/log/";
std::string Logger::logFileName = "vision_log.log";
FILE *Logger::logFile = nullptr;   // NOLINT
static const char *TAG = "Logger"; // NOLINT
bool Logger::hasInit = false;

void Logger::log(const LogLevel &logLevel, const char* tag, const char* format, va_list args) {
    if (level < logLevel) {
        return;
    }

    // 只有console、logfile两种方式可以执行到此处
    logHead(logLevel, tag);
    logBody(format, args);
    logTail();
}

void Logger::logBody(const char* format, va_list args) {
#ifdef VISION_DEBUG
    if (destValue & static_cast<std::uint8_t>(0b010)) {
        va_list args2;
        va_copy(args2, args);
        if (logFile) {
            vfprintf(logFile, format, args2);
            fflush(logFile);
        }
        va_end(args2);
    }

    if (destValue & static_cast<std::uint8_t>(0b001)) {
        vprintf(format, args);
    }
#endif
}

void Logger::logHead(const LogLevel &logLevel, const std::string &tag) {
    auto time = SystemClock::currentTimeStr();
    if (!tag.empty()) {
        logPrint("%s %p %s [%s] %s: ", time.c_str(), std::this_thread::get_id(), appName.c_str(),
                 logLevelToStr(logLevel).c_str(), tag.c_str());
    } else {
        logPrint("%s %s [%s]: ", time.c_str(), appName.c_str(), logLevelToStr(logLevel).c_str());
    }
}

void Logger::logTail() {
    logPrint("\n");
}

void Logger::logPrint(const char* format, ...) {
#ifdef VISION_DEBUG
    va_list args1, args2;
    va_start(args1, format);

    if (destValue & static_cast<std::uint8_t>(0b010)) {
        va_copy(args2, args1);
        if (logFile) {
            vfprintf(logFile, format, args2);
            fflush(logFile);
        }
    }

    if (destValue & static_cast<std::uint8_t>(0b001)) {
        vprintf(format, args1);
    }

    va_end(args1);
    va_end(args2);
#endif
}

void Logger::fatal(const char *tag, const char *format, ...) {
    va_list args;
    va_start(args, format);
    log(LogLevel::FATAL, tag, format, args);
    va_end(args);
}

void Logger::error(const char *tag, const char *format, ...) {
    va_list args;
    va_start(args, format);
    log(LogLevel::ERROR, tag, format, args);
    va_end(args);
}

void Logger::warn(const char *tag, const char *format, ...) {
    va_list args;
    va_start(args, format);
    log(LogLevel::WARN, tag, format, args);
    va_end(args);
}

void Logger::info(const char *tag, const char *format, ...) {
    va_list args;
    va_start(args, format);
    log(LogLevel::INFO, tag, format, args);
    va_end(args);
}

void Logger::debug(const char *tag, const char *format, ...) {
    va_list args;
    va_start(args, format);
    log(LogLevel::DEBUGGER, tag, format, args);
    va_end(args);
}

bool Logger::init() {
    hasInit = true;
#ifdef BUILD_SLOG
    // todo 暂时去掉slog2初始化逻辑，由上层保证初始化。后续考虑提供打印日志的抽象接口，由上层实现
//    logger_init(appName);
#endif
    // 初始化设置默认带时间戳的日志文件，默认文件当前可执行程序的路径
    setLogFileName("vision_" + SystemClock::currentTimeStr() + ".log");
    return true;
}

void Logger::setLogLevel(const LogLevel &logLevel) {
    ensureInit();
    level = logLevel;
#ifdef BUILD_SLOG
    #ifdef USE_LIGHT_SLOG
        // 使用主线轻量级SLOG2暂不做操作
    #else
        switch (level) {
            case vision::LogLevel::FATAL:
                logger_set_level(B_SLOG2_CRITICAL);
                break;
            case vision::LogLevel::ERROR:
                logger_set_level(B_SLOG2_ERROR);
                break;
            case vision::LogLevel::WARN:
                logger_set_level(B_SLOG2_WARNING);
                break;
            case vision::LogLevel::INFO:
                logger_set_level(B_SLOG2_INFO);
                break;
            case vision::LogLevel::DEBUGGER:
                logger_set_level(B_SLOG2_DEBUG1);
                break;
            default:
                break;
        }
    #endif
#endif
}

void Logger::setLogDest(const LogDest &logDest) {
    ensureInit();
    dest = logDest;
    destValue = static_cast<std::uint8_t>(dest);
    // 设置更新日志输出形式的时候，如果日志输出到文件中，可以重新尝试打开日志文件
    openLogFile(logFileDir, logFileName);
}

void Logger::setAppName(const std::string &app_name) {
    if (app_name.empty()) {
        return;
    }
    ensureInit();
    appName = std::string(app_name);
}

void Logger::setLogFileDir(const std::string &fileDir) {
    if (fileDir.empty()) {
        return;
    }
    ensureInit();
    logFileDir = std::string(fileDir);
    openLogFile(logFileDir, logFileName);
}

void Logger::setLogFileName(const std::string &fileName) {
    if (fileName.empty()) {
        return;
    }
    ensureInit();
    logFileName = std::string(fileName);
    openLogFile(logFileDir, logFileName);
}

bool Logger::openLogFile(const std::string &fileDir, const std::string &fileName) {
    ensureInit();
    if (dest != LogDest::ALL && dest != LogDest::LOGFILE) {
        VLOGI(TAG, "no need to set log file name cause of dest:%d", dest);
        return false;
    }
    if (logFile) {
        fflush(logFile);
        fclose(logFile);
    }
    logFile = fopen((logFileDir + logFileName).c_str(), "w");
    if (!logFile) {
        VLOGW(TAG, "logFile open failed!");
        return false;
    } else {
        VLOGI(TAG, "logFile open success!");
        return true;
    }
}


std::string Logger::logLevelToStr(const LogLevel &logLevel) {
    auto index = static_cast<int>(logLevel);
    if (index < 0 or index >= static_cast<int>(levelStrings.size())) {
        return "";
    }
    return levelStrings[index];
}

void Logger::logStart() {
    if (dest == LogDest::LOGFILE || dest == LogDest::ALL) {
        if (!logFile) {
            logFile = fopen(logFileName.c_str(), "w");
            if (!logFile) {
                VLOGW(TAG, "logFile open failed!");
            }
        } else {
            VLOGD(TAG, "log file has opened........");
        }
    }
}

void Logger::logEnd() {
    if (logFile) {
        fflush(logFile);
        fclose(logFile);
        VLOGD(TAG, "close log file------------");
    }
}

bool Logger::ensureInit() {
    if (!hasInit) {
        return init();
    }
    return true;
}

} // namespace aura::vision
