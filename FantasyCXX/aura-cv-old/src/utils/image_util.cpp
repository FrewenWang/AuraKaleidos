//
// Created by frewen on 22-10-10.
//

#include "aura/cv/utils/ImageUtil.h"
#include "opencv2/imgproc/imgproc.hpp"

namespace aura::aura_cv {

cv::Mat ImageUtil::fixImageSize(cv::Mat &in, int w, int h) {
    // 获取输入数据的行、列
    int rows = in.rows;
    int cols = in.cols;

    if (cols == w && rows == h) {
        return in;
    }
    int new_w = 0;
    int new_h = 0;
    cv::Mat resized;
    // 计算新的图片的宽高的缩放比例
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
    cv::copyMakeBorder(resized, out, top, bottom, left, right,
                       cv::BORDER_CONSTANT, cv::Scalar{174, 174, 174});
    return out;
}

void ImageUtil::bgr2Yuv420NV21(unsigned char *src, unsigned char *dst, int width, int height) {
    if (src == nullptr || dst == nullptr) {
        return;
    }

    if (width % 2 != 0 || height % 2 != 0) {
        return;
    }

    static unsigned short shift = 14;
    static unsigned int coeffs[5] = {B2YI, G2YI, R2YI, B2UI, R2VI};
    static unsigned int offset = 128 << shift;

    unsigned char *y_plane = dst;
    unsigned char *vu_plane = dst + width * height;

    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; ++c) {
            int Y = (unsigned int) (src[0] * coeffs[0] + src[1] * coeffs[1] + src[2] * coeffs[2]) >> shift;
            *y_plane++ = (unsigned char) Y;

            if (r % 2 == 0 && c % 2 == 0) {
                int U = (unsigned int) ((src[0] - Y) * coeffs[3] + offset) >> shift;
                int V = (unsigned int) ((src[2] - Y) * coeffs[4] + offset) >> shift;

                vu_plane[0] = (unsigned char) V;
                vu_plane[1] = (unsigned char) U;
                vu_plane += 2;
            }
            src += 3;
        }
    }
}

cv::Mat ImageUtil::bgr2Yuv420NV21(const cv::Mat &frame) {
    int yuv_nv21_mem_len = sizeof(unsigned char) * frame.rows * 3 / 2 * frame.cols;
    auto *yuv_nv21_mem = (unsigned char *) malloc(yuv_nv21_mem_len);
    ImageUtil::bgr2Yuv420NV21(frame.data, yuv_nv21_mem, frame.cols, frame.rows);
    
    cv::Mat yuv_image;
    yuv_image.create(frame.rows * 3 / 2, frame.cols, CV_8UC1);
    memcpy(yuv_image.data, yuv_nv21_mem, yuv_nv21_mem_len);
    
    free(yuv_nv21_mem);
    return yuv_image;
}

unsigned char *ImageUtil::bgr2Yuv444Planer(cv::Mat &in, int flag) {
    //flag: 0->YUV444_p(UV) , 1->YUV444_p(VU)
    if (in.channels() != 3) { return NULL; }
    int height = in.rows, width = in.cols;
    int frameSize = height * width;
    int pu, pv;
    if (flag) {
        pv = frameSize;
        pu = pv + frameSize;
    } else {
        pu = frameSize;
        pv = pu + frameSize;
    }
    unsigned char *yuvOutput = new unsigned char[(height * width * 3)]();
    int i, j, k;
    for (i = 0, k = 0; i < height; ++i) {
        // 获取第i行的BRG数据。然后依次遍历这一行数据提取出B、G、R信息
        uchar *ptr = in.ptr<uchar>(i);
        for (j = 0; j < width; ++j) {
            int J = j * 3;
            int B = ptr[J], G = ptr[J + 1], R = ptr[J + 2];
            // 依次存储Y U V 分量
            yuvOutput[k++] = cv::saturate_cast<uchar>(0.114 * B + 0.587 * G + 0.299 * R);
            yuvOutput[pu++] = cv::saturate_cast<uchar>(0.5 * B - 0.332 * G - 0.169 * R + 128);
            yuvOutput[pv++] = cv::saturate_cast<uchar>(-0.0813 * B - 0.419 * G + 0.5 * R + 128);
        }
    }
    return yuvOutput;
}
unsigned char *ImageUtil::bgr2Yuv444SemiPlanar(cv::Mat &in, int flag) {
    //flag: 0->YUV444SemiPlanar(UV), 1->YUV444SemiPlanar(VU)
    if (in.channels() != 3) { return NULL; }
    int Uflag = 1, Vflag = 2;
    if (flag) {
        Uflag = 2;
        Vflag = 1;
    }

    int height = in.rows, width = in.cols;
    unsigned char *yuvOutput = new unsigned char[(height * width * 3)]();
    int i, j, k;

    for (i = 0, k = 0; i < height; ++i) {
        uchar *ptr = in.ptr<uchar>(i);
        for (j = 0; j < width; j++) {
            int J = j * 3;
            int B = ptr[J], G = ptr[J + 1], R = ptr[J + 2];
            float Y = 0.114 * B + 0.587 * G + 0.299 * R;
            float U = 0.5 * B - 0.332 * G - 0.169 * R + 128;
            float V = -0.0813 * B - 0.419 * G + 0.5 * R + 128;
            yuvOutput[k] = cv::saturate_cast<uchar>(Y);
            yuvOutput[k + Uflag] = cv::saturate_cast<uchar>(U);
            yuvOutput[k + Vflag] = cv::saturate_cast<uchar>(V);
            k += 3;
        }
    }
    return yuvOutput;
}

} // namespace aura::aura_cv
