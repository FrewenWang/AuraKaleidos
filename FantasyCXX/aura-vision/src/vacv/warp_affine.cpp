#include "warp_affine.h"
#include <iostream>

#ifdef USE_OPENCV
#include "opencv2/opencv.hpp"
#endif

#if defined (BUILD_FASTCV)
#include "fastcv.h"
#endif

#include "util/TensorConverter.h"
#include "warp_affine_naive.h"
#include "FastcvUtil.h"

using namespace std;

namespace aura::va_cv {

using namespace aura::vision;

void WarpAffine::warp_affine(const VTensor& src, VTensor& dst,
							 const VTensor& M, VSize dsize, int flags,
							 int borderMode, const VScalar& borderValue) {
#if defined(BUILD_QNX) and defined (BUILD_FASTCV)
//    warp_affine_fastcv(src, dst, M, dsize, flags, borderMode, borderValue);
    warp_affine_opencv(src, dst, M, dsize, flags, borderMode, borderValue);
#elif USE_OPENCV
    warp_affine_opencv(src, dst, M, dsize, flags, borderMode, borderValue);
#else
#if defined (USE_NEON) and __ARM_NEON
    warp_affine_neon(src, dst, M, dsize, flags, borderMode, borderValue);
#elif defined (USE_SSE)
    warp_affine_sse(src, dst, M, dsize, flags, borderMode, borderValue);
#else
    warp_affine_naive(src, dst, M, dsize, flags, borderMode, borderValue);
#endif
#endif // USE_OPENCV
}

void WarpAffine::warp_affine(const vision::VTensor& src, vision::VTensor& dst,
							 float scale, float rot, VSize dsize,
							 const VScalar& aux_param,
							 int flags, int borderMode,
							 const VScalar& borderValue) {
#if defined(BUILD_QNX) and defined (BUILD_FASTCV)
//    warp_affine_fastcv(src, dst, scale, rot, dsize, aux_param, flags, borderMode, borderValue);
    warp_affine_opencv(src, dst, scale, rot, dsize, aux_param, flags, borderMode, borderValue);
#elif USE_OPENCV
    warp_affine_opencv(src, dst, scale, rot, dsize, aux_param, flags, borderMode, borderValue);
#else
    warp_affine_naive(src, dst, scale, rot, dsize, aux_param, flags, borderMode, borderValue);
#endif
}

void WarpAffine::warp_affine_opencv(const VTensor& src, VTensor& dst,
									const VTensor& M, VSize dsize, int flags,
									int borderMode, const VScalar& borderValue) {
#ifdef USE_OPENCV
    cv::Mat mat_src = vision::TensorConverter::convert_to<cv::Mat>(src);
    cv::Mat mat_M = vision::TensorConverter::convert_to<cv::Mat>(M);
    cv::Scalar sca_border(borderValue.v0, borderValue.v1, borderValue.v2, borderValue.v3);
    cv::Mat mat_dst;
//    cout << "[warp_affine_opencv] rot mat : " << to_string(((float *)mat_M.data)[0]) << endl;
    cv::warpAffine(mat_src, mat_dst, mat_M, cv::Size(dsize.w, dsize.h), flags, borderMode, sca_border);
//    unsigned char *d = (unsigned char *) mat_dst.data;
//    cout << "[warp_affine_opencv] : " << to_string(d[0]) << ", " << to_string(d[1]) << endl;
    dst = vision::TensorConverter::convert_from<cv::Mat>(mat_dst, true);
#else
    warp_affine_naive(src, dst, M, dsize, flags, borderMode, borderValue);
#endif
}

void WarpAffine::warp_affine_opencv(const vision::VTensor& src, vision::VTensor& dst,
									float scale, float rot, VSize dsize,
									const VScalar& aux_param,
									int flags, int borderMode,
									const VScalar& borderValue) {
#ifdef USE_OPENCV
    cv::Mat mat_src = vision::TensorConverter::convert_to<cv::Mat>(src);

    // get rotation matrix
    cv::Mat rot_mat = cv::getRotationMatrix2D(cv::Point(0, 0), rot, scale);
    // 修正
    rot_mat.at<double>(0, 2) = aux_param.v2 - rot_mat.at<double>(0, 0) * aux_param.v0 -
            rot_mat.at<double>(0, 1) * aux_param.v1;
    rot_mat.at<double>(1, 2) = aux_param.v3 - rot_mat.at<double>(1, 0) * aux_param.v0 -
            rot_mat.at<double>(1, 1) * aux_param.v1;
//    cout << "[warp_affine_opencv] rot mat : " << to_string(((float *)rot_mat.data)[0]) << endl;
    cv::Mat mat_dst;
    cv::Scalar sca_border(borderValue.v0, borderValue.v1, borderValue.v2, borderValue.v3);
    cv::warpAffine(mat_src, mat_dst, rot_mat, cv::Size(dsize.w, dsize.h), flags, borderMode, sca_border);
//    unsigned char *d = (unsigned char *) mat_dst.data;
//    cout << "[warp_affine_opencv] : " << to_string(d[0]) << ", " << to_string(d[1]) << endl;
    dst = vision::TensorConverter::convert_from<cv::Mat>(mat_dst, true);
#endif
}

VTensor WarpAffine::get_rotation_matrix_2D(const VPoint& point, float angle, float scale) {
    VTensor rotation_tensor(3, 2, 1, NCHW, FP32);

    float* m = (float*)rotation_tensor.data;

    angle *= M_PI / 180;

    double alpha = scale * cos(angle);
    double beta = scale * sin(angle);

    m[0] = alpha;
    m[1] = beta;
    m[2] = (1 - alpha) * point.x - beta * point.y;
    m[3] = -beta;
    m[4] = alpha;
    m[5] = beta * point.x + (1 - alpha) * point.y;

    return rotation_tensor;
}

void WarpAffine::warp_affine_naive(const vision::VTensor& src, vision::VTensor& dst,
								   float scale, float rot, VSize dsize,
								   const VScalar& aux_param,
								   int flags, int borderMode,
								   const VScalar& borderValue) {
    // get rotation matrix
    VTensor rot_tensor = get_rotation_matrix_2D(VPoint(0, 0), rot, scale);
    // 修正
    float* m = (float*)rot_tensor.data;
    m[2] = aux_param.v2 - m[0] * aux_param.v0 - m[1] * aux_param.v1;
    m[5] = aux_param.v3 - m[3] * aux_param.v0 - m[4] * aux_param.v1;

    warp_affine_naive(src, dst, rot_tensor, dsize, flags, borderMode, borderValue);
}

void WarpAffine::warp_affine_naive(const VTensor& src, VTensor& dst,
								   const VTensor& M, VSize dsize, int flags,
								   int borderMode, const VScalar& borderValue) {
    if ((src.dType != INT8 && src.dType != FP32)
        || borderMode != BORDER_CONSTANT
        || flags != INTER_LINEAR) {
        warp_affine_opencv(src, dst, M, dsize, flags, borderMode, borderValue);
        return;
    }

    float* m = (float*)M.data;
    double D = m[0] * m[4] - m[1] * m[3];
    D = D != 0 ? 1. / D : 0;
    double A11 = m[4] * D;
    double A22 = m[0] * D;
    m[0] = A11;
    m[1] *= -D;
    m[3] *= -D;
    m[4] = A22;
    double b1 = -m[0] * m[2] - m[1] * m[5];
    double b2 = -m[3] * m[2] - m[4] * m[5];
    m[2] = b1;
    m[5] = b2;

    int w_in  = src.w;
    int h_in  = src.h;
    int c     = src.c;
    int w_out = dsize.w;
    int h_out = dsize.h;
    dst.create(w_out, h_out, c, src.dType, src.dLayout);

    if (src.dLayout == NHWC) {
        if (src.dType == INT8) {
            WarpAffineNaive::warp_affine_naive_hwc_u8((char*)src.data, w_in, h_in, c,
                                                      (char*)dst.data, w_out, h_out, m);
        } else {
            WarpAffineNaive::warp_affine_naive_hwc_fp32((float*)src.data, w_in, h_in, c,
                                                        (float*)dst.data, w_out, h_out, m);
        }
    } else {
        int src_stride = w_in * h_in;
        int dst_stride = w_out * h_out;
        if (src.dType == INT8) {
            for (int i = 0; i < c; i++) {
                char* src_channel_data = (char*)src.data + src_stride * i;
                char* dst_channel_data = (char*)dst.data + dst_stride * i;
                WarpAffineNaive::warp_affine_naive_hwc_u8(src_channel_data, w_in, h_in, 1,
                                                          dst_channel_data, w_out, h_out, m);
            }
        } else if(src.dType == FP32) {
            for (int i = 0; i < c; i++) {
                float* src_channel_data = (float*)src.data + src_stride * i;
                float* dst_channel_data = (float*)dst.data + dst_stride * i;
                WarpAffineNaive::warp_affine_naive_hwc_fp32(src_channel_data, w_in, h_in, 1,
                                                            dst_channel_data, w_out, h_out, m);
            }
        }
    }
}

void WarpAffine::warp_affine_sse(const VTensor& src, VTensor& dst,
								 const VTensor& M, VSize dsize, int flags,
								 int borderMode, const VScalar& borderValue) {
    // todo:
}

void WarpAffine::warp_affine_neon(const VTensor& src, VTensor& dst,
								  const VTensor& M, VSize dsize, int flags,
								  int borderMode, const VScalar& borderValue) {
    // todo:

}

void WarpAffine::warp_affine_fastcv(const VTensor& src, VTensor& dst,
                                    const VTensor& M, VSize dsize, int flags,
                                    int borderMode, const VScalar& borderValue) {
#if defined (BUILD_FASTCV)
    if (dst.empty()) {
        dst.create(dsize.w, dsize.h, src.c, src.dType, src.dLayout, src.allocType);
    }
    fcvInterpolationType fcvInterType = FastcvUtil::cvtInterpolationType((VInterMode)flags);
    fcvBorderType fcvBorderType = FastcvUtil::cvtBorderType((VBorderMode)borderMode);
    fcvStatus ret = fcvTransformAffineClippedu8_v3(src.asUChar(), src.w, src.h, src.w * src.c,
                                                   (const float32_t*) M.data,
                                                   dst.asUChar(), dsize.w, dsize.h, dsize.w * src.c,
                                                   nullptr, fcvInterType, fcvBorderType, 0);
    // dstBorder = nullptr then the border isn't processed
//    const float* rm = (const float*) M.data;
//    cout << "[warp_affine_fastcv] rot mat 1 : " << to_string(rm[0]) << ", "
//         << to_string(dsize.w) << ", " << to_string(dsize.h) << endl;
    if (ret != FASTCV_SUCCESS) {
//        cout << "[warp_affine_fastcv] Fail, retry with opencv" << endl;
        warp_affine_opencv(src, dst, M, dsize, flags, borderMode, borderValue);
//    } else {
//        unsigned char *d = dst.asUChar();
//        cout << "[warp_affine_fastcv] Success : " << to_string(d[0]) << ", " << to_string(d[1]) << endl;
    }
#endif
}

void WarpAffine::warp_affine_fastcv(const vision::VTensor& src, vision::VTensor& dst,
                                    float scale, float rot, VSize dsize,
                                    const VScalar& aux_param,
                                    int flags, int borderMode,
                                    const VScalar& borderValue) {
#if defined (BUILD_FASTCV)
    if (dst.empty()) {
        dst.create(dsize.w, dsize.h, src.c, src.dType, src.dLayout, src.allocType);
    }
    // get rotation matrix
    cv::Mat rot_mat = cv::getRotationMatrix2D(cv::Point(0, 0), rot, scale);
    // 修正
    rot_mat.at<double>(0, 2) = aux_param.v2 - rot_mat.at<double>(0, 0) * aux_param.v0 -
            rot_mat.at<double>(0, 1) * aux_param.v1;
    rot_mat.at<double>(1, 2) = aux_param.v3 - rot_mat.at<double>(1, 0) * aux_param.v0 -
            rot_mat.at<double>(1, 1) * aux_param.v1;

    fcvInterpolationType fcvInterType = FastcvUtil::cvtInterpolationType((VInterMode)flags);
    fcvBorderType fcvBorderType = FastcvUtil::cvtBorderType((VBorderMode)borderMode);
    fcvStatus ret = fcvTransformAffineClippedu8_v3(src.asUChar(), src.w, src.h, src.w * src.c,
                                                   (const float32_t*) rot_mat.data,
                                                   dst.asUChar(), dsize.w, dsize.h, dsize.w * src.c,
                                                   nullptr, fcvInterType, fcvBorderType, 0);
    // dstBorder = nullptr then the border isn't processed
//    const float* rm = (const float*) rot_mat.data;
//    cout << "[warp_affine_fastcv] rot mat 2 : " << to_string(rm[0]) << ", "
//         << to_string(dsize.w) << ", " << to_string(dsize.h) <<  endl;

    if (ret != FASTCV_SUCCESS) {
//        cout << "[warp_affine_fastcv] Fail, retry with opencv" << endl;
        warp_affine_opencv(src, dst, scale, rot, dsize, aux_param, flags, borderMode, borderValue);
//    } else {
//        unsigned char *d = dst.asUChar();
//        cout << "[warp_affine_fastcv] Success" << to_string(d[0]) << ", " << to_string(d[1]) << endl;
    }
#endif
}

} // namespace va_cv
