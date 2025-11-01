

#include "aura/aura_utils/utils/AuraLog.h"
#include "aura/aura_utils/utils/SystemClock.h"
#include <cstdarg>
#include <cstdio>
#include <vector>

namespace aura{
namespace aura_utils{

LogLevel Logger::level = LogLevel::VERBOSE;
LogDest Logger::dest = LogDest::CONSOLE;
std::uint8_t Logger::destValue = static_cast<std::uint8_t>(dest);
std::string Logger::appName = "AuraLib";
std::vector<std::string> Logger::levelStr = {"NONE ", "FATAL", "ERROR", "WARN ", "INFO ", "DEBUG", "VERBOSE"};
const char *Logger::logFileName = "vision_log.log";
FILE *Logger::logFile = nullptr;
/**
 * 输出到日志中的互斥量
 */
static std::mutex mMutex;
static const char *gTAG = "Logger";


void Logger::log(LogLevel log_level, const char *tag, const char *format, va_list args) {
  if (level < log_level) {
    return;
  }
  // 只有console、logfile两种方式可以执行到此处
  logHead(log_level, tag);
  logBody(format, args);
  logTail();
}

void Logger::logBody(const char *format, va_list args) {
  if (destValue & static_cast<std::uint8_t>(0b010)) {
    va_list args2;
    va_copy(args2, args);
    // 如果输出目录是日志文件，或者输出ALL
    if (dest == LogDest::LOG_FILE || dest == LogDest::ALL) {
      if (logFile) {
        // 防止输出乱行，此处加锁
        std::unique_lock<std::mutex> lck(mMutex);
        // C 库函数 int vfprintf(FILE *stream, const char *format, va_list arg) 使用参数列表发送格式化输出到流stream中。
        // 下面是 vfprintf() 函数的声明。
        // stream -- 这是指向 FILE 对象的指针，该 FILE 对象标识了流。
        // format -- 这是 C 字符串，包含了要被写入到流 stream 中的文本。
        // 它可以包含嵌入的 format 标签，format 标签可被随后的附加参数中指定的值替换，并按需求进行格式化。
        // format 标签属性是 %[flags][width][.precision][length]specifier，具体讲解如下：
        vfprintf(logFile, format, args2);
        fflush(logFile);
      }
    }
    va_end(args2);
  }
  
  if (destValue & static_cast<std::uint8_t>(0b001)) {
    vprintf(format, args);
  }
}

void Logger::logHead(LogLevel log_level, const std::string &tag) {
  auto time = SystemClock::currentTimeStr();
  if (!tag.empty()) {
    logPrint("%s %s [%s] %s: ", time.c_str(), appName.c_str(), logLevelToStr(log_level).c_str(), tag.c_str());
  } else {
    logPrint("%s %s [%s]: ", time.c_str(), appName.c_str(), logLevelToStr(log_level).c_str());
  }
}

void Logger::logTail() {
  logPrint("\n");
}

void Logger::logPrint(const char *format, ...) {
  va_list args1, args2;
  va_start(args1, format);
  
  if (destValue & static_cast<std::uint8_t>(0b010)) {
    va_copy(args2, args1);
    if (logFile) {
      // 防止输出乱行，此处加锁
      std::unique_lock<std::mutex> lck(mMutex);
      vfprintf(logFile, format, args2);
      fflush(logFile);
    }
  }
  
  if (destValue & static_cast<std::uint8_t>(0b001)) {
    vprintf(format, args1);
  }
  
  va_end(args1);
  va_end(args2);
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

void Logger::init(const std::string &app_name) {
  appName = std::string(app_name);
}

void Logger::setLogLevel(LogLevel log_level) {
  level = log_level;
}

void Logger::setLogDest(LogDest log_dest) {
  dest = log_dest;
  destValue = static_cast<std::uint8_t>(dest);
}

void Logger::setAppName(const std::string &app_name) {
  appName = std::string(app_name);
}

void Logger::setLogFileName(const char *file_name) {
  logFileName = file_name;
  if (dest == LogDest::LOG_FILE || dest == LogDest::ALL) {
    // 防止输出乱行，此处加锁
    std::unique_lock<std::mutex> lck(mMutex);
    if (logFile) {
      fflush(logFile);
      fclose(logFile);
    }
    logFile = fopen(logFileName, "w");
    if (!logFile) {
      ALOGW(gTAG, "logFile open failed!");
    }
  }
}

std::string Logger::logLevelToStr(LogLevel log_level) {
  auto index = static_cast<int>(log_level);
  if (index < 0 or index >= static_cast<int>(levelStr.size())) {
    return "";
  }
  return levelStr[index];
}

void Logger::logStart() {
  if (dest == LogDest::LOG_FILE || dest == LogDest::ALL) {
    // 防止输出乱行，此处加锁
    std::unique_lock<std::mutex> lck(mMutex);
    if (!logFile) {
      logFile = fopen(logFileName, "w");
      if (!logFile) {
        ALOGW(gTAG, "logFile open failed!");
      }
    } else {
      ALOGD(gTAG, "log file has opened........");
    }
  }
}

void Logger::logEnd() {
  if (dest == LogDest::LOG_FILE || dest == LogDest::ALL) {
    // 防止输出乱行，此处加锁
    std::unique_lock<std::mutex> lck(mMutex);
    if (logFile) {
      fflush(logFile);
      fclose(logFile);
      ALOGD(gTAG, "close log file------------");
    }
  }
}

void Logger::strAppend(std::stringstream &ss, std::string msg, ...) {
  // 定位参数list
  // va_list arg_ptr;
  // va_start(var, source);
  // int src = va_arg(arg_ptr, int);
  // va_start(arg_ptr, cnt);
  // while (src >= VisConstant::SOURCE_1 && src <= VisConstant::SOURCE_3) {
  //     auto it = std::find(sources->begin(), sources->end(), src);
  //     if (it == sources->end()) {
  //         sources->push_back(src);
  //     }
  //     src = va_arg(var, int);
  // }
  // va_end(var);
}

}
} // namespace aura::aura_lib
