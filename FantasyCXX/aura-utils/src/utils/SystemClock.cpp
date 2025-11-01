#include "aura/aura_utils/utils/SystemClock.h"

#include <chrono>
#include <iomanip>
#include <sstream>
#include <ctime>

using namespace aura::utils;
using namespace std::chrono;

std::int64_t SystemClock::nowMillis() {
  auto ts = system_clock::now().time_since_epoch();
  return duration_cast<milliseconds>(ts).count();
}

/**
 * std::chrono::system_clock::now().time_since_epoch()
 * 获取到的也是到时间元年(1970-01-01)的时间间隔，
 * 可以用std::chrono::duration_cast<>函数可以方便的改变获取到时间的精度。
 * @return
 */
std::int64_t SystemClock::uptimeMillis() {
  auto ts = system_clock::now().time_since_epoch();
  return duration_cast<milliseconds>(ts).count();
}

std::int64_t SystemClock::uptimeMillisStartup() {
  struct timespec start;
  clock_gettime(CLOCK_MONOTONIC, &start);
  uint64_t delta_ms = start.tv_sec * 1000 + start.tv_nsec / 1000000;
  return delta_ms;
}

std::string SystemClock::currentTimeStr() {
  auto now = system_clock::to_time_t(system_clock::now());

  std::stringstream ss;
  struct tm *ptm = localtime(&now);
  char date[60] = {0};
  // 注释掉如下方法。sprintf已经过时，容易出现缓冲区溢出。编译器建议使用更安全的 snprintf 替代。
  // sprintf(date, "%d-%02d-%02d-%02d:%02d:%02d", (int) ptm->tm_year + 1900, (int) ptm->tm_mon + 1, (int) ptm->tm_mday,
  //         (int) ptm->tm_hour, (int) ptm->tm_min, (int) ptm->tm_sec);
  snprintf(date, sizeof(date), "%d-%02d-%02d-%02d:%02d:%02d", (int)ptm->tm_year + 1900, (int)ptm->tm_mon + 1,
           (int)ptm->tm_mday,
           (int)ptm->tm_hour, (int)ptm->tm_min, (int)ptm->tm_sec);
  ss << std::string(date);
  ss << "." << std::setfill('0') << std::setw(3) << static_cast<int>(uptimeMillis() % 1000);
  return ss.str();
}
