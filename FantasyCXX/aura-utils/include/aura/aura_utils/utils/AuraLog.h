//
// AuraLog - 极简日志宏
// 兼容旧代码中对 aura/aura_utils/utils/AuraLog.h 的引用（如 auto-driving 单测）。
// 仅依赖标准库，不引入额外依赖。
//
#ifndef AURA_AURA_UTILS_UTILS_AURALOG_H
#define AURA_AURA_UTILS_UTILS_AURALOG_H

#include <cstdio>

// ALOGD / ALOGI 输出到 stdout，ALOGW / ALOGE 输出到 stderr。
// 用法：ALOGD("TAG", "value=%d", 42);
#define ALOGD(tag, fmt, ...) ::fprintf(stdout, "[D][%s] " fmt "\n", tag, ##__VA_ARGS__)
#define ALOGI(tag, fmt, ...) ::fprintf(stdout, "[I][%s] " fmt "\n", tag, ##__VA_ARGS__)
#define ALOGW(tag, fmt, ...) ::fprintf(stderr, "[W][%s] " fmt "\n", tag, ##__VA_ARGS__)
#define ALOGE(tag, fmt, ...) ::fprintf(stderr, "[E][%s] " fmt "\n", tag, ##__VA_ARGS__)

#endif  // AURA_AURA_UTILS_UTILS_AURALOG_H
