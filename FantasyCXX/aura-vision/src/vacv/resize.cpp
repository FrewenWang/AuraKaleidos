#include "resize.h"

#ifdef USE_OPENCV
#include "opencv2/opencv.hpp"
#endif

#if defined (USE_NEON) and __ARM_NEON
#include "arm_neon.h"
#include "resize_neon.h"
#endif

#if defined (BUILD_FASTCV)
#include "fastcv.h"
#endif

#include "resize_naive.h"
#include "util/DebugUtil.h"
#include "util/TensorConverter.h"
#include "vision/util/PerfUtil.h"
#include "util/DebugUtil.h"

namespace aura::va_cv {

using namespace aura::vision;

void Resize::resize(const VTensor &src, VTensor &dst,
                    VSize dsize, double fx, double fy,
                    int interpolation) {
#if defined(BUILD_QNX) and defined (BUILD_FASTCV)
    if (src.c == 1 && src.dType == INT8) {
        resize_fastcv(src, dst, dsize, fx, fy, interpolation);
        //  DBG_PRINT_ARRAY((float*)dst.data, 10, "resize FASTCV");
    } else {
        resize_opencv(src, dst, dsize, fx, fy, interpolation);
        // DBG_PRINT_ARRAY((float*)dst.data, 10, "resize OPENCV");
    }
#elif USE_OPENCV
    resize_opencv(src, dst, dsize, fx, fy, interpolation);
#elif defined (USE_SSE)
    resize_sse(src, dst, dsize, fx, fy, interpolation);
#elif defined (USE_NEON) and __ARM_NEON
    resize_neon(src, dst, dsize, fx, fy, interpolation);
#else
    resize_naive(src, dst, dsize, fx, fy, interpolation);
#endif // USE_OPENCV
}

void Resize::resize_opencv(const VTensor &src, VTensor &dst,
                           VSize dsize, double fx, double fy,
                           int interpolation) {
#ifdef USE_OPENCV
    const auto& mat_src = vision::TensorConverter::convert_to<cv::Mat>(src);
    cv::Mat mat_dst;
    cv::resize(mat_src, mat_dst, cv::Size(dsize.w, dsize.h), fx, fy, interpolation);
    dst = vision::TensorConverter::convert_from<cv::Mat>(mat_dst, true);
#else
    resize_naive(src, dst, dsize, fx, fy, interpolation);
#endif
}

void Resize::resize_naive(const VTensor &src, VTensor &dst,
                          VSize dsize, double fx, double fy,
                          int interpolation) {

    if (interpolation != INTER_LINEAR && interpolation != INTER_CUBIC) {
        resize_opencv(src, dst, dsize, fx, fy, interpolation);
        return;
    }

    int w_in = src.w;
    int h_in = src.h;
    int c = src.c;
    int w_out = dsize.w;
    int h_out = dsize.h;
    dst.create(w_out, h_out, src.c, src.dType, src.dLayout);

    if (w_out == w_in && h_out == h_in) {
        memcpy(dst.data, src.data, sizeof(uint8_t) * w_in * h_in * c);
        return;
    }

    if (interpolation == INTER_LINEAR && (src.dType == INT8 || src.dType == FP32)) {
        if (src.dLayout == NHWC) {
            if (src.dType == vision::DType::INT8) {
                ResizeNaive::resize_naive_inter_linear_u8((char *)src.data, w_in, h_in, c,
                                                          (char *)dst.data, w_out, h_out);
            } else if (src.dType == vision::DType::FP32) {
                ResizeNaive::resize_naive_inter_linear_fp32((float *)src.data, w_in, h_in, c,
                                                            (float *)dst.data, w_out, h_out);
            }
        } else {
            int src_stride = w_in * h_in;
            int dst_stride = w_out * h_out;
            for (int i = 0; i < c; i++) {
                if (src.dType == INT8) {
                    char *src_channel_data = (char *)src.data + src_stride * i;
                    char *dst_channel_data = (char *)dst.data + dst_stride * i;
                    ResizeNaive::resize_naive_inter_linear_u8(src_channel_data, w_in, h_in, 1,
                                                              dst_channel_data, w_out, h_out);
                } else if (src.dType == FP32) {
                    float *src_channel_data = (float *)src.data + src_stride * i;
                    float *dst_channel_data = (float *)dst.data + dst_stride * i;
                    ResizeNaive::resize_naive_inter_linear_fp32(src_channel_data, w_in, h_in, 1,
                                                                dst_channel_data, w_out, h_out);
                }
            }
        }
    } else if (interpolation == INTER_CUBIC && src.dType == FP32) {
        if (src.dLayout == NHWC) {
            ResizeNaive::resize_naive_inter_cubic_fp32_hwc((float *)src.data, w_in, h_in,
                                                           (float *)dst.data, w_out, h_out);
        } else {
            ResizeNaive::resize_naive_inter_cubic_fp32_chw((float *)src.data, w_in, h_in, c,
                                                           (float *)dst.data, w_out, h_out);
        }
    } else {
        resize_opencv(src, dst, dsize, fx, fy, interpolation);
    }
}

#if defined (BUILD_FASTCV)
bool Resize::resize_fastcv(const VTensor &src, VTensor &dst, VSize dsize, double fx, double fy, int interpolation) {
    fcvInterpolationType inter;
    switch (interpolation) {
        case INTER_NEAREST:
            inter = FASTCV_INTERPOLATION_TYPE_NEAREST_NEIGHBOR;
            break;
        case INTER_LINEAR:
            inter = FASTCV_INTERPOLATION_TYPE_BILINEAR;
            break;
        case INTER_AREA:
            inter = FASTCV_INTERPOLATION_TYPE_AREA;
            break;
        default:
            inter = FASTCV_INTERPOLATION_TYPE_NEAREST_NEIGHBOR;
    }
    if (dst.empty()) {
        // AUTO_PERF(PerfUtil::global(), "resize_fastcv - create"); // 0 ms
        dst.create(dsize.w, dsize.h, src.c, src.dType, src.dLayout, DAllocType::MALLOC); // FCV_ALLOC
    }

    // AUTO_PERF(PerfUtil::global(), "resize_fastcv - fcvScaleu8"); // 1 ms
    // VLOGD("resize", "resize_fastcv src.w:%d, src.h:%d, dst.w:%d, dst.h:%d, inter:%d", src.w, src.h, dst.w, dst.h,
    //       inter);
    fcvStatus ret = fcvScaleu8(src.asUChar(), src.w, src.h, src.w * src.c,
                               dst.asUChar(), dst.w, dst.h, dst.w * dst.c, inter);
    if (ret != FASTCV_SUCCESS) {
        VLOGE("Resize", "Fail to resize_fastcv, return code = %d", ret);
    }
    return ret == FASTCV_SUCCESS;
}
#endif

#if defined (USE_NEON) and __ARM_NEON
void Resize::resize_sse(const VTensor& src, VTensor& dst,
                        VSize dsize, double fx, double fy,
                        int interpolation) {
    // todo:
}

void Resize::resize_neon(const VTensor& src, VTensor& dst,
                         VSize dsize, double fx, double fy,
                         int interpolation) {
    // 如果需要resize的不是线性插值、或者数据类型不是INT8，那么我们就不能使用NEON加速
    //
    if (interpolation != INTER_LINEAR || src.dType != INT8) {
        if (src.dLayout == NCHW) {
            resize_naive(src, dst, dsize, fx, fy, interpolation);
        } else {
            resize_opencv(src, dst, dsize, fx, fy, interpolation);
        }
        return;
    }

    int w_in  = src.w;
    int h_in  = src.h;
    int c     = src.c;
    int w_out = dsize.w;
    int h_out = dsize.h;
    dst.create(w_out, h_out, c, src.dType, src.dLayout);

    if (w_out == w_in && h_out == h_in) {
        memcpy(dst.data, src.data, sizeof(uint8_t) * w_in * h_in * c);
        return;
    }

    if (src.dLayout == NHWC) {
        ResizeNeon::resize_neon_inter_linear_three_channel((uint8_t*)src.data, w_in * 3, h_in,
                                                           (uint8_t*)dst.data, w_out * 3, h_out);
    } else {
        int src_stride = w_in * h_in;
        int dst_stride = w_out * h_out;
        for (int i = 0; i < 3; i++) {
            uint8_t* src_channel_data = (uint8_t*)src.data + src_stride * i;
            uint8_t* dst_channel_data = (uint8_t*)dst.data + dst_stride * i;
            ResizeNeon::resize_neon_inter_linear_one_channel(src_channel_data, w_in, h_in,
                                                             dst_channel_data, w_out, h_out);
        }
    }
}
#endif

void Resize::resizeNoNormalize(const VTensor &src, VTensor &dst,
                               VSize dsize, double fx, double fy,
                               int interpolation) {
#if defined(BUILD_QNX) and defined (BUILD_FASTCV)
    if (src.c == 1 && src.dType == INT8) {
        resizeWithFloatFastcv(src, dst, dsize, fx, fy, interpolation);
    } else {
        resizeWithFloatOpencv(src, dst, dsize, fx, fy, interpolation);
    }
#else
    resizeWithFloatOpencv(src, dst, dsize, fx, fy, interpolation);
#endif
}

void Resize::resizeWithFloatOpencv(const VTensor &src, VTensor &dst,
                               VSize dsize, double fx, double fy,
                               int interpolation) {
    const auto &mat_src = vision::TensorConverter::convert_to<cv::Mat>(src);
    cv::Mat mat_resized;
    cv::resize(mat_src, mat_resized, cv::Size(dsize.w, dsize.h), fx, fy, interpolation);
    // DBG_PRINT_ARRAY((char *) mat_resized.data, 100, "resizeWithFloatOpencv_resize_after");
    cv::Mat mat_resized_f;
    if (src.dType != FP32) {
        if (src.c == 1) {
            mat_resized.convertTo(mat_resized_f, CV_32FC1);
        } else if (src.c == 3) {
            mat_resized.convertTo(mat_resized_f, CV_32FC3);
        } else {
            // not supported!
            throw std::runtime_error("The tensor channel number  is not supported in resizeNoNormalize()");
        }
    } else {
        mat_resized_f = mat_resized;
    }
    dst = vision::TensorConverter::convert_from<cv::Mat>(mat_resized_f, true);
    // DBG_PRINT_ARRAY((float *) dst.data, 100, "resizeWithFloatOpencv_cvt_float_after");
}

void Resize::resizeWithFloatFastcv(const VTensor &src, VTensor &dst,
                                   VSize dsize, double fx, double fy,
                                   int interpolation) {
    VTensor resized;
    resize(src, resized, dsize, fx, fy, interpolation);
    // DBG_PRINT_ARRAY((char *) resized.data, 100, "resizeWithFloatFastcv_resize_after");
    if (resized.dType != vision::FP32) {
        dst = resized.changeDType(FP32);
    } else {
        dst = resized;
    }
    // DBG_PRINT_ARRAY((float *) dst.data, 100, "resizeWithFloatFastcv_cvt_float_after");
}

} // namespace aura::va_cv
