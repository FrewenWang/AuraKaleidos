//
// Created by Frewen.Wang on 2022/8/7.
//
#pragma once

#include <climits>
#include <cmath>

/// API visibility
#if defined _WIN32 || defined __CYGWIN__
#ifdef BUILDING_DLL
#ifdef __GNUC__
#define AURA_PUBLIC __attribute__((dllexport))
#else
#define AURA_PUBLIC __declspec(dllexport)
#endif
#else
#ifdef __GNUC__
#define AURA_PUBLIC __attribute__((dllimport))
#else
#define AURA_PUBLIC __declspec(dllimport)
#endif
#endif
#define AURA_LOCAL
#define AURA_DEPRECATED __declspec(deprecated)
#else
#if __GNUC__ >= 4
#define AURA_PUBLIC __attribute__((visibility("default")))
#define AURA_LOCAL __attribute__((visibility("hidden")))
#define AURA_DEPRECATED __attribute__ ((deprecated))
#else
#define AURA_PUBLIC
#define AURA_LOCAL
#endif
#endif

#ifndef MIN
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#endif

#ifndef MAX
#define MAX(x, y) ((x) > (y) ? (x) : (y))
#endif

#ifndef CLAMP
#define CLAMP(x, min, max) MIN(MAX(x, min), max)
#endif


/// commonly-used macros
/// 返回int类型的错误码
#define A_RET(err) return static_cast<int>(err)

#define A_CHECK_RET_VOID(cond) if (cond != 0) return

#define A_CHECK_CONT(cond) if (cond) continue

#define A_CHECK_CONT_MSG(cond, tag, msg) \
do {                         \
    if (cond) {             \
        ALOGI(tag, msg); \
        continue;            \
    }                        \
} while (0)

#define A_CHECK_EXIT_ERR_MSG(cond,tag, msg) \
do {                                        \
    if (cond) {                             \
        ALOGE(tag, msg);                    \
        exit(EXIT_FAILURE);                 \
    }                                       \
} while (0)


/// CONVERT
#define TO_INT(value) static_cast<int>(value)
#define TO_SHORT(value) static_cast<short>(value)
#define TO_FLOAT(value) static_cast<float>(value)
#define FLOAT_TO_BOOL(value) (value > 1e-6)
#define FLOAT_EQUAL_ZERO(value) (fabs(value) < 1e-6)
#define INT_TO_BOOL(value) (value == 1)


#define exitOnError(errCode, errMsg)        \
    do {                                    \
        errno = errCode;                    \
        perror(errMsg);                     \
        exit(EXIT_FAILURE);                 \
    } while (0)


#define ERR_UNSUPPORTED  (-3)
#define ERR_INVALID_ARG  (-2)
#define A_OK  (0)
