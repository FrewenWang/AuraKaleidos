//
// Created by Frewen.Wang on 2022/8/6.
//

#include "aura/cv/ops/resize/Resize.h"
#include "opencv2/imgproc/imgproc.hpp"

namespace aura::aura_cv {

void Resize::resize(const ATensor &src, ATensor &dst, VSize dsize, double fx, double fy, int interpolation) {
#if defined(BUILD_FAST_CV) and defined(BUILD_QNX)

#elif defined (USE_NEON) and __ARM_NEON
    resize_neon(src, dst, dsize, fx, fy, interpolation);
#else

#endif
}

cv::Mat Resize::fixImageSize(cv::Mat &in, int w, int h) {
    // 获取输入图片的mat的宽高，如果原始宽高和目标宽高一致，则不需要补边
    int cols = in.cols;
    int rows = in.rows;
    if (cols == w && rows == h) {
        return in;
    }
    
    int new_w = 0;
    int new_h = 0;
    
    cv::Mat resized;
    
    float scale_width = cols * 1.f / w;
    float scale_height = rows * 1.f / h;
    
    if (scale_width > scale_height) {
        new_w = w;
        new_h = static_cast<int>(rows / scale_width);
    } else {
        new_h = h;
        new_w = static_cast<int>(cols / scale_height);
    }
    cv::resize(in, resized, cv::Size(new_w, new_h));
    
    int delta_rows = h - new_h;
    int delta_cols = w - new_w;
    int top = delta_rows / 2;
    int bottom = delta_rows - top;
    int left = delta_cols / 2;
    int right = delta_cols - left;
    cv::Mat out;
    cv::copyMakeBorder(resized, out, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar{174, 174, 174});
    return out;
}

void Resize::resizeByNaive(const ATensor &src, ATensor &dst, VSize dsize, double fx, double fy, int interpolation) {
    // if (interpolation != INTER_LINEAR && interpolation != INTER_CUBIC) {
    //     resizeByOpenCV(src, dst, dsize, fx, fy, interpolation);
    //     return;
    // }
    // int w_in = src.width;
    // int h_in = src.height;
    // int c = src.channel;
    // int w_out = dsize.w;
    // int h_out = dsize.h;
    // dst.create(w_out, h_out, c, src.dLayout, src.dType);
    throw std::invalid_argument("resizeByNaive has not implement!!!");
}

void Resize::resizeByOpenCV(const ATensor &src, ATensor &dst, VSize dsize, double fx, double fy, int interpolation) {
    const auto &mat_src = ATensor::convertTo<cv::Mat>(src);
    cv::Mat mat_dst;
    // opencv的resize的方法。传入参数：原始mat、目标mat、目标大小、fx、fy、插值器
    cv::resize(mat_src, mat_dst, cv::Size(dsize.w, dsize.h), fx, fy, interpolation);
}

void Resize::resizeByNeon(const ATensor &src, ATensor &dst, VSize dsize, double fx, double fy, int interpolation) {
    // 如果差值方式：不是双线性差值，或者数据类型不是INT5。则说明不适合使用NEON加速？？
    if (interpolation != INTER_LINEAR || src.dType != INT8) {
        if (src.dLayout == NCHW) {
            resizeByNaive(src, dst, dsize, fx, fy, interpolation);
        } else {
            resizeByOpenCV(src, dst, dsize, fx, fy, interpolation);
        }
        return;
    }
    int w_in = src.width;
    int h_in = src.height;
    int c = src.channel;
    int w_out = dsize.w;
    int h_out = dsize.h;
    dst.create(w_out, h_out, c, src.dLayout, src.dType);
    
    if (w_out == w_in && h_out == h_in) {
        memcpy(dst.data, src.data, sizeof(uint8_t) * w_in * h_in * c);
        return;
    }
    
    if (src.dLayout == NHWC) {
    
    } else {
        int src_stride = w_in * h_in;
        int dst_stride = w_out * h_out;
        for (int i = 0; i < 3; ++i) {
            uint8_t *src_channel_data = (uint8_t *) src.data + src_stride * i;
            uint8_t *dst_channel_data = (uint8_t *) dst.data + dst_stride * i;
            // ResizeNeon::resize_neon_inter_linear_one_channel(src_channel_data, w_in, h_in,
            //                                                  dst_channel_data, w_out, h_out);
        }
    }
    
}


}