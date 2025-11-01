#ifndef VISION_MACRO_H
#define VISION_MACRO_H

#include <climits>
#include "vision/util/log.h"
#include <cmath>

/// API visibility
#if defined _WIN32 || defined __CYGWIN__
#ifdef BUILDING_DLL
#ifdef __GNUC__
#define VA_PUBLIC __attribute__((dllexport))
#else
#define VA_PUBLIC __declspec(dllexport)
#endif
#else
#ifdef __GNUC__
#define VA_PUBLIC __attribute__((dllimport))
#else
#define VA_PUBLIC __declspec(dllimport)
#endif
#endif
#define VA_LOCAL
#define VA_DEPRECATED __declspec(deprecated)
#else
#if __GNUC__ >= 4
#define VA_PUBLIC __attribute__((visibility("default")))
#define VA_LOCAL __attribute__((visibility("hidden")))
#define VA_DEPRECATED __attribute__ ((deprecated))
#else
#define VA_PUBLIC
#define VA_LOCAL
#endif
#endif

/// commonly-used macros
#define V_RET(err) return static_cast<int>(err)

#define V_CHECK_RET_VOID(cond) if (cond != 0) return

#define V_CHECK_CONT(cond) if (cond) continue

#define V_CHECK_CONT_MSG(cond, msg) \
do {                         \
    if (cond) {             \
        VLOGI(LOG_TAG, msg); \
        continue;            \
    }                        \
} while (0)

#define V_CHECK(cond)      \
do {                        \
    auto ret = cond;        \
    if (ret != 0) {         \
        V_RET(ret);        \
    }                       \
} while (0)

#define V_CHECK_RET(cond, ret)     \
do {                                \
    if (cond != 0) {                \
        V_RET(ret);                \
    }                               \
} while (0)

#define V_CHECK_MSG(cond, msg)         \
do {                                    \
    auto ret = cond;                    \
    if (ret != 0) {                     \
        VLOGI(LOG_TAG, msg);            \
        V_RET(ret);                    \
    }                                   \
} while (0)

#define VA_CHECK_RET_MSG(cond, ret, msg)    \
do {                                        \
    auto r = cond;                          \
    if (r != 0) {                           \
        VLOGI(LOG_TAG, msg);                \
        V_RET(ret);                        \
    }                                       \
} while (0)

#define V_CHECK_COND(cond, ret, msg)       \
do {                                        \
    if (cond) {                             \
        VLOGI(LOG_TAG, msg);                \
        V_RET(ret);                        \
    }                                       \
} while (0)

#define V_CHECK_COND_ERR(cond, ret, msg)     \
do {                                            \
    if (cond) {                                 \
        VLOGE(LOG_TAG, msg);                    \
        V_RET(ret);                            \
    }                                           \
} while (0)

#define V_CHECK_NULL(cond) if (cond == nullptr) return

#define V_CHECK_NULL_RET(cond, ret) if (cond == nullptr) V_RET(ret)

#define V_CHECK_NULL_RET_INFO(cond, ret, msg)   \
do {                                            \
    if (cond == nullptr) {                      \
        VLOGI(LOG_TAG, msg);;                   \
        V_RET(ret);                            \
    }                                           \
} while (0)

#define V_CHECK_NULL_RET_ERR(cond, ret, msg)    \
do {                                            \
    if (cond == nullptr) {                      \
        VLOGE(LOG_TAG, msg);;                   \
        V_RET(ret);                            \
    }                                           \
} while (0)

/// math
#ifndef MIN
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#endif

#ifndef MAX
#define MAX(x, y) ((x) > (y) ? (x) : (y))
#endif

#ifndef CLAMP
#define CLAMP(x, min, max) MIN(MAX(x, min), max)
#endif

/// CONVERT
#define V_TO_INT(value) static_cast<int>(value)
#define V_TO_SHORT(value) static_cast<short>(value)
#define V_TO_FLOAT(value) static_cast<float>(value)
#define V_F_TO_BOOL(value) (value > 1e-6)
#define V_F_EQUAL_ZERO(value) (fabs(value) < 1e-6)
#define V_I_TO_BOOL(value) (value == 1)
#ifndef SATURATE_CAST_SHORT
#define SATURATE_CAST_SHORT(X)                                               \
(int16_t)::std::min(                                                         \
  ::std::max(static_cast<int>(X + (X >= 0.f ? 0.5f : -0.5f)), SHRT_MIN),     \
  SHRT_MAX);
#endif

/// image
#define CHECK_OR_MAKE_GREY(frame)                                           \
do {                                                                        \
    if (!frame.has_grey() && frame.data != nullptr) {                       \
        frame.grey = VTensor(frame.width, frame.height, frame.data, INT8);   \
        if (V_F_TO_BOOL(mRtConfig->inputImageNeedCrop)) {               \
            VTensor cropped;                                                 \
            va_cv::crop(frame.grey, cropped,                                \
                vision::VRect(mRtConfig->inputCropRoiLeftTopX,             \
                      mRtConfig->inputCropRoiLeftTopY,                     \
                      mRtConfig->inputCropRoiRightBottomX,                     \
                      mRtConfig->inputCropRoiRightBottomY));                   \
            va_cv::resize(cropped, frame.grey,                              \
                va_cv::VSize(frame.width, frame.height));                   \
        }                                                                   \
    }                                                                       \
} while (0)

#define CHECK_OR_MAKE_BGR(frame)                                            \
do {                                                                        \
    if (!frame.has_rgb() && frame.data != nullptr) {                        \
        Tensor yuv(frame.width, frame.height * 3 / 2, frame.data, INT8);    \
        va_cv::cvt_color(yuv, frame.rgb,                                    \
                static_cast<int>(mRtConfig->frameConvertBgrFormat));        \
        if (V_F_TO_BOOL(mRtConfig->inputImageNeedCrop)) {                  \
            Tensor cropped;                                                 \
            va_cv::crop(frame.rgb, cropped,                                 \
                vision::VRect(mRtConfig->inputCropRoiLeftTopX,             \
                      mRtConfig->inputCropRoiLeftTopY,                     \
                      mRtConfig->inputCropRoiRightBottomX,                 \
                      mRtConfig->inputCropRoiRightBottomY));               \
            va_cv::resize(cropped, frame.rgb,                               \
                va_cv::VSize(frame.width, frame.height));                   \
        }                                                                   \
    }                                                                       \
} while (0)

#define CHECK_OR_MAKE_RGB(frame)                                            \
do {                                                                        \
    if (!frame.has_rgb() && frame.data != nullptr) {                        \
        Tensor yuv(frame.width, frame.height * 3 / 2, frame.data, INT8);    \
        va_cv::cvt_color(yuv, frame.rgb,                                    \
                static_cast<int>(mRtConfig->frameConvertBgrFormat));        \
        if (V_F_TO_BOOL(mRtConfig->inputImageNeedCrop)) {                  \
            Tensor cropped;                                                 \
            va_cv::crop(frame.rgb, cropped,                                 \
                 vision::VRect(mRtConfig->inputCropRoiLeftTopX,             \
                      mRtConfig->inputCropRoiLeftTopY,                     \
                      mRtConfig->inputCropRoiRightBottomX,                 \
                      mRtConfig->inputCropRoiRightBottomY));               \
            va_cv::resize(cropped, frame.rgb,                               \
                va_cv::VSize(frame.width, frame.height));                   \
        }                                                                   \
    }                                                                       \
} while (0)
#endif //VISION_MACRO_H
