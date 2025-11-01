//
// Created by Li,Wendong on 2019-05-14.
//

#include "vision/util/ImageUtil.h"
#include <fstream>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <stdarg.h>
#include <sys/stat.h>
#include <time.h>

#include "vision/core/bean/FaceInfo.h"
#include "util/math_utils.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif


namespace aura::vision {

void ImageUtil::save_img(cv::Mat &src, char *feature_name) {
    std::string save_path = "./save_img/";
    save_path.append(feature_name).append("/");
    FILE *fp = NULL;
    fp = fopen(save_path.c_str(), "w");
    if (!fp) {
        mkdir(save_path.c_str(), 0775);
    }
    fclose(fp);

    time_t t;
    srand((int) time(NULL));
    save_path.append(std::to_string(time(&t))).append("_").append(std::to_string(rand() % 1000000)).append(".jpg");
#ifdef WITH_OCV_HIGHGUI
    cv::imwrite(save_path, src);
#endif
}

float ImageUtil::calc_image_brightness(short width, short height, unsigned char *frame) {
    int elements = width * height;
    int sum = 0;
    for (int i = 0; i < elements; i++) {
        sum += frame[i];
    }
    return (float) sum / elements;
}

#ifdef BUILD_NCNN
void ImageUtil::resize_and_normalize(int src_w, int src_h, unsigned char *src_data, int pixel_type,
                                     int dst_w, int dst_h, ncnn::Mat &dst_data, float *means, float *stds) {
#ifdef NCNN
    dst_data = ncnn::Mat::from_pixels_resize(src_data, pixel_type, src_w, src_h, dst_w, dst_h);
#elif OPENCV
    //        cv::Mat mat_frame(cv::Size(src_w, src_h), CV_8UC1, frame);
    //        cv::resize(mat_frame, *_m_mat_resized, cv::Size(dst_w, dst_h), 0, 0, cv::INTER_LINEAR);
#endif
    if (means == NULL && stds == NULL) {
        cv::Mat mean;
        cv::Mat std;
        cv::Mat src_image(src_h, src_w, CV_8UC1, src_data);
        cv::meanStdDev(src_image, mean, std);
        dst_data.substract_mean_normalize((float *) mean.data, (float *) std.data);
    } else {
        dst_data.substract_mean_normalize(means, stds);
    }
}
#endif

cv::Mat ImageUtil::normalize(cv::Mat &src) {
    std::vector<double> mean;
    std::vector<double> stddev;
    mean_stddev(src, mean, stddev);
    return normalize_with_mean_stddev(src, mean, stddev);
}

cv::Mat ImageUtil::normalize_with_mean_stddev(cv::Mat &src, const std::vector<double> &mean,
                                              const std::vector<double> &stddev) {
    cv::Mat dst;
    int c = src.channels();
    if ((int) mean.size() < c || (int) stddev.size() < c) {
        return dst;
    }

    cv::Mat src_mat_f;
    switch (c) {
        case 1:
            src.convertTo(src_mat_f, CV_32FC1);
            dst = (src_mat_f - mean[0]) / (stddev[0] + 1e-6);
            break;
        case 3: {
            src.convertTo(src_mat_f, CV_32FC3);
            std::vector <cv::Mat> channels;
            cv::split(src_mat_f, channels);
            for (int i = 0; i < (int) channels.size(); ++i) {
                channels[i] = (channels[i] - mean[i]) / (stddev[i] + 1e-6);
            }
            cv::merge(channels, dst);
            break;
        }
    }
    return dst;
}

void ImageUtil::mean_stddev(cv::Mat &src, std::vector<double> &mean, std::vector<double> &stddev) {
    mean.clear();
    stddev.clear();

    cv::Mat src_mat_f;
    cv::Mat mean_mat;
    cv::Mat stddev_mat;

    int c = src.channels();

    switch (c) {
        case 1:
            src.convertTo(src_mat_f, CV_32FC1);
            cv::meanStdDev(src_mat_f, mean_mat, stddev_mat);
            break;
        case 3:
            src.convertTo(src_mat_f, CV_32FC3);
            cv::meanStdDev(src_mat_f, mean_mat, stddev_mat);
            break;
        default:
            return;
    }

    for (int i = 0; i < c; ++i) {
        mean.emplace_back(((double *) mean_mat.data)[i]);
        stddev.emplace_back(((double *) stddev_mat.data)[i]);
    }
}

void ImageUtil::bgr2nv21(unsigned char *src, unsigned char *dst, int width, int height) {
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

cv::Mat ImageUtil::bgr2nv21(const cv::Mat &frame) {
    int yuv_nv21_mem_len = sizeof(unsigned char) * frame.rows * 3 / 2 * frame.cols;
    auto *yuv_nv21_mem = (unsigned char *) malloc(yuv_nv21_mem_len);
    ImageUtil::bgr2nv21(frame.data, yuv_nv21_mem, frame.cols, frame.rows);

    cv::Mat yuv_image;
    yuv_image.create(frame.rows * 3 / 2, frame.cols, CV_8UC1);
    memcpy(yuv_image.data, yuv_nv21_mem, yuv_nv21_mem_len);

    free(yuv_nv21_mem);
    return yuv_image;
}

void ImageUtil::bgr2uyvy(const cv::Mat &rgb, cv::Mat &uyvy) {
    static unsigned short shift = 14;
    static unsigned int coeffs[5] = {B2YI, G2YI, R2YI, B2UI, R2VI};
    static unsigned int offset = 128 << shift;
    for (int ih = 0; ih < rgb.rows; ih++) {
        const auto rgbRowPtr = rgb.ptr<uint8_t>(ih);
        auto yuvRowPtr = uyvy.ptr<uint8_t>(ih);
        for (int iw = 0; iw < rgb.cols; iw = iw + 2) {
            const int rgbColIdxBytes = iw * rgb.elemSize();
            const int yuvColIdxBytes = iw * uyvy.elemSize();
            const uint8_t B1 = rgbRowPtr[rgbColIdxBytes + 0];
            const uint8_t G1 = rgbRowPtr[rgbColIdxBytes + 1];
            const uint8_t R1 = rgbRowPtr[rgbColIdxBytes + 2];
            const uint8_t B2 = rgbRowPtr[rgbColIdxBytes + 3];
            const uint8_t G2 = rgbRowPtr[rgbColIdxBytes + 4];
            const uint8_t R2 = rgbRowPtr[rgbColIdxBytes + 5];
            const unsigned int Y1 = (unsigned int) (B1 * coeffs[0] + G1 * coeffs[1] + R1 * coeffs[2]) >> shift;
            const unsigned int U = (unsigned int) ((B1 - Y1) * coeffs[3] + offset) >> shift;
            const unsigned int V = (unsigned int) ((R1 - Y1) * coeffs[4] + offset) >> shift;
            const unsigned int Y2 = (unsigned int) (B2 * coeffs[0] + G2 * coeffs[1] + R2 * coeffs[2]) >> shift;
            yuvRowPtr[yuvColIdxBytes + 0] = cv::saturate_cast<uint8_t>(U);
            yuvRowPtr[yuvColIdxBytes + 1] = cv::saturate_cast<uint8_t>(Y1);
            yuvRowPtr[yuvColIdxBytes + 2] = cv::saturate_cast<uint8_t>(V);
            yuvRowPtr[yuvColIdxBytes + 3] = cv::saturate_cast<uint8_t>(Y2);
        }
    }
}

cv::Mat ImageUtil::fix_image_size(cv::Mat &in, int w, int h) {
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
    cv::copyMakeBorder(resized, out, top, bottom, left, right,
                       cv::BORDER_CONSTANT, cv::Scalar{174, 174, 174});
    return out;
}

void ImageUtil::cvt_color(cv::Mat &src, cv::Mat &dst, int code) {
    if (code == cv::COLOR_YUV2BGR_YV12) {
#if __ARM_NEON
        int w = src.cols;
        int h = src.rows * 2 / 3;
        dst.create(h, w, CV_8UC3);
        cvt_color_yuv2bgr_yv12(src.data, dst.data, w, h);
#else
        cv::cvtColor(src, dst, code);
#endif // __ARM_NEON
    } else if (code == cv::COLOR_YUV2BGR_NV21) {
#if __ARM_NEON
        int w = src.cols;
        int h = src.rows * 2 / 3;
        dst.create(h, w, CV_8UC3);
        cvt_color_yuv2bgr_nv21(src.data, dst.data, w, h);
#else
        cv::cvtColor(src, dst, code);
#endif // __ARM_NEON
    } else {
        cv::cvtColor(src, dst, code);
    }
}

void ImageUtil::cvt_color_yuv2rgb(unsigned char *src, int *dst, int width, int height) {
    int length = width * height;
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int y = (0xff & ((int) src[i * width + j]));
            int u = (0xff & ((int) src[length + (i >> 1) * width + (j & ~1) + 0]));
            int v = (0xff & ((int) src[length + (i >> 1) * width + (j & ~1) + 1]));
            y = y < 16 ? 16 : y;
            double dr = 1.164 * (y - 16) + 1.596 * (v - 128);
            double dg = 1.164 * (y - 16) - 0.813 * (v - 128) - 0.391 * (u - 128);
            double db = 1.164 * (y - 16) + 2.018 * (u - 128);
            int r = static_cast<int>(round(dr));
            int g = static_cast<int>(round(dg));
            int b = static_cast<int>(round(db));
            r = r < 0 ? 0 : (r > 255 ? 255 : r);
            g = g < 0 ? 0 : (g > 255 ? 255 : g);
            b = b < 0 ? 0 : (b > 255 ? 255 : b);
            dst[i * width + j] = 0xff000000 + (b << 16) + (g << 8) + r;
        }
    }
}

#if __ARM_NEON
void ImageUtil::cvt_color_yuv2bgr_yv12(unsigned char *src, unsigned char *dst, int w, int h, int bIdx) {
    if (src == nullptr || dst == nullptr) {
        return;
    }

    if (w % 2 != 0 || h % 2 != 0) {
        return;
    }

    const int stride = w;

    int y_stride = w * h;
    int v_stride = w * h / 4;

    unsigned char *y1 = src;
    unsigned char *v1 = y1 + y_stride;
    unsigned char *u1 = v1 + v_stride;

    int8x8_t cvr = vdup_n_s8 (ITUR_BT_602_CVR);
    int8x8_t cvg = vdup_n_s8 (-ITUR_BT_602_CVG);
    int8x8_t cug = vdup_n_s8 (-ITUR_BT_602_CUG);
    int8x8_t cub = vdup_n_s8 (ITUR_BT_602_CUB);
    int16x8_t cy = vdupq_n_s16 (ITUR_BT_602_CY);
    uint8x8_t uvoffset = vdup_n_u8 (128);
    uint8x8_t yoffset = vdup_n_u8 (16);
    int16x8_t round_offset = vdupq_n_s16(1 << (ITUR_BT_602_SHIFT - 1));

    for (int j = 0; j < h; j += 2) {
        unsigned char *row1 = dst + j * w * 3;
        unsigned char *row2 = row1 + w * 3;

        const unsigned char *y2 = y1 + stride;

        for (int i = 0; i < w / 16; i += 1, row1 += 48, row2 += 48) {

            const unsigned char* ly1 = y1 + i * 16;
            const unsigned char* ly2 = y2 + i * 16;
            uint8x8_t vu = vld1_u8(u1 + i * 8);
            uint8x8_t vv = vld1_u8(v1 + i * 8);
            uint8x8x2_t vy1 = vld2_u8(ly1);
            uint8x8x2_t vy2 = vld2_u8(ly2);

            uint16x8_t vu16 = vsubl_u8(vu, uvoffset); // widen subtract
            uint16x8_t vv16 = vsubl_u8(vv, uvoffset);
            int8x8_t svu = vqmovn_s16(vreinterpretq_s16_u16(vu16)); // convert to signed integer
            int8x8_t svv = vqmovn_s16(vreinterpretq_s16_u16(vv16));

            uint16x8x2_t vy116, vy216;
            vy116.val[0] = vsubl_u8(vy1.val[0], yoffset);
            vy116.val[1] = vsubl_u8(vy1.val[1], yoffset);
            vy216.val[0] = vsubl_u8(vy2.val[0], yoffset);
            vy216.val[1] = vsubl_u8(vy2.val[1], yoffset);
            int16x8x2_t svy1, svy2;
            svy1.val[0] = vreinterpretq_s16_u16(vy116.val[0]);
            svy1.val[1] = vreinterpretq_s16_u16(vy116.val[1]);
            svy2.val[0] = vreinterpretq_s16_u16(vy216.val[0]);
            svy2.val[1] = vreinterpretq_s16_u16(vy216.val[1]);

            int16x8_t gu = vmull_s8(svu, cug);
            int16x8_t gv = vmull_s8(svv, cvg);
            int16x8_t ruv = vmull_s8(svv, cvr);
            int16x8_t buv = vmull_s8(svu, cub);
            int16x8_t guv = vaddq_s16(gu, gv);
            ruv = vqaddq_s16(ruv, round_offset);
            buv = vqaddq_s16(buv, round_offset);
            guv = vqaddq_s16(guv, round_offset);

            int16x8x2_t vy1_2, vy2_2;
            vy1_2.val[0] = vmulq_s16(svy1.val[0], cy);
            vy1_2.val[1] = vmulq_s16(svy1.val[1], cy);
            vy2_2.val[0] = vmulq_s16(svy2.val[0], cy);
            vy2_2.val[1] = vmulq_s16(svy2.val[1], cy);

            int16x8x2_t vb1_2, vg1_2, vr1_2, vb2_2, vg2_2, vr2_2;
            vb1_2.val[0] = vaddq_s16(vy1_2.val[0], buv);
            vg1_2.val[0] = vaddq_s16(vy1_2.val[0], guv);
            vr1_2.val[0] = vaddq_s16(vy1_2.val[0], ruv);

            vb1_2.val[1] = vaddq_s16(vy1_2.val[1], buv);
            vg1_2.val[1] = vaddq_s16(vy1_2.val[1], guv);
            vr1_2.val[1] = vaddq_s16(vy1_2.val[1], ruv);

            vb2_2.val[0] = vaddq_s16(vy2_2.val[0], buv);
            vg2_2.val[0] = vaddq_s16(vy2_2.val[0], guv);
            vr2_2.val[0] = vaddq_s16(vy2_2.val[0], ruv);

            vb2_2.val[1] = vaddq_s16(vy2_2.val[1], buv);
            vg2_2.val[1] = vaddq_s16(vy2_2.val[1], guv);
            vr2_2.val[1] = vaddq_s16(vy2_2.val[1], ruv);


            uint16x8x2_t svb1_2, svg1_2, svr1_2, svb2_2, svg2_2, svr2_2;
            svb1_2.val[0] = vreinterpretq_u16_s16(vqabsq_s16(vb1_2.val[0]));
            svb1_2.val[1] = vreinterpretq_u16_s16(vqabsq_s16(vb1_2.val[1]));
            svg1_2.val[0] = vreinterpretq_u16_s16(vqabsq_s16(vg1_2.val[0]));
            svg1_2.val[1] = vreinterpretq_u16_s16(vqabsq_s16(vg1_2.val[1]));
            svr1_2.val[0] = vreinterpretq_u16_s16(vqabsq_s16(vr1_2.val[0]));
            svr1_2.val[1] = vreinterpretq_u16_s16(vqabsq_s16(vr1_2.val[1]));
            svb2_2.val[0] = vreinterpretq_u16_s16(vqabsq_s16(vb2_2.val[0]));
            svb2_2.val[1] = vreinterpretq_u16_s16(vqabsq_s16(vb2_2.val[1]));
            svg2_2.val[0] = vreinterpretq_u16_s16(vqabsq_s16(vg2_2.val[0]));
            svg2_2.val[1] = vreinterpretq_u16_s16(vqabsq_s16(vg2_2.val[1]));
            svr2_2.val[0] = vreinterpretq_u16_s16(vqabsq_s16(vr2_2.val[0]));
            svr2_2.val[1] = vreinterpretq_u16_s16(vqabsq_s16(vr2_2.val[1]));

            uint8x8x2_t v8b1_2, v8g1_2, v8r1_2, v8b2_2, v8g2_2, v8r2_2;
            v8b1_2.val[0] = vqshrn_n_u16(svb1_2.val[0], ITUR_BT_602_SHIFT);
            v8b1_2.val[1] = vqshrn_n_u16(svb1_2.val[1], ITUR_BT_602_SHIFT);
            v8g1_2.val[0] = vqshrn_n_u16(svg1_2.val[0], ITUR_BT_602_SHIFT);
            v8g1_2.val[1] = vqshrn_n_u16(svg1_2.val[1], ITUR_BT_602_SHIFT);
            v8r1_2.val[0] = vqshrn_n_u16(svr1_2.val[0], ITUR_BT_602_SHIFT);
            v8r1_2.val[1] = vqshrn_n_u16(svr1_2.val[1], ITUR_BT_602_SHIFT);

            v8b2_2.val[0] = vqshrn_n_u16(svb2_2.val[0], ITUR_BT_602_SHIFT);
            v8b2_2.val[1] = vqshrn_n_u16(svb2_2.val[1], ITUR_BT_602_SHIFT);
            v8g2_2.val[0] = vqshrn_n_u16(svg2_2.val[0], ITUR_BT_602_SHIFT);
            v8g2_2.val[1] = vqshrn_n_u16(svg2_2.val[1], ITUR_BT_602_SHIFT);
            v8r2_2.val[0] = vqshrn_n_u16(svr2_2.val[0], ITUR_BT_602_SHIFT);
            v8r2_2.val[1] = vqshrn_n_u16(svr2_2.val[1], ITUR_BT_602_SHIFT);

            uint8x8_t vzero = vdup_n_u8(0);

            uint8x16_t v8b1_11 = vcombine_u8(vzero, v8b1_2.val[0]);
            uint8x16_t v8b1_12 = vcombine_u8(vzero, v8b1_2.val[1]);
            uint8x16_t v8g1_11 = vcombine_u8(vzero, v8g1_2.val[0]);
            uint8x16_t v8g1_12 = vcombine_u8(vzero, v8g1_2.val[1]);
            uint8x16_t v8r1_11 = vcombine_u8(vzero, v8r1_2.val[0]);
            uint8x16_t v8r1_12 = vcombine_u8(vzero, v8r1_2.val[1]);

            uint8x16_t v8b2_11 = vcombine_u8(vzero, v8b2_2.val[0]);
            uint8x16_t v8b2_12 = vcombine_u8(vzero, v8b2_2.val[1]);
            uint8x16_t v8g2_11 = vcombine_u8(vzero, v8g2_2.val[0]);
            uint8x16_t v8g2_12 = vcombine_u8(vzero, v8g2_2.val[1]);
            uint8x16_t v8r2_11 = vcombine_u8(vzero, v8r2_2.val[0]);
            uint8x16_t v8r2_12 = vcombine_u8(vzero, v8r2_2.val[1]);

#if __aarch64__
            uint8x16_t v8b1_1 = vzip2q_u8(v8b1_11, v8b1_12);
            uint8x16_t v8g1_1 = vzip2q_u8(v8g1_11, v8g1_12);
            uint8x16_t v8r1_1 = vzip2q_u8(v8r1_11, v8r1_12);

            uint8x16_t v8b2_1 = vzip2q_u8(v8b2_11, v8b2_12);
            uint8x16_t v8g2_1 = vzip2q_u8(v8g2_11, v8g2_12);
            uint8x16_t v8r2_1 = vzip2q_u8(v8r2_11, v8r2_12);
#else
            uint8x16_t v8b1_1 = vzipq_u8(v8b1_11, v8b1_12).val[1];
            uint8x16_t v8g1_1 = vzipq_u8(v8g1_11, v8g1_12).val[1];
            uint8x16_t v8r1_1 = vzipq_u8(v8r1_11, v8r1_12).val[1];

            uint8x16_t v8b2_1 = vzipq_u8(v8b2_11, v8b2_12).val[1];
            uint8x16_t v8g2_1 = vzipq_u8(v8g2_11, v8g2_12).val[1];
            uint8x16_t v8r2_1 = vzipq_u8(v8r2_11, v8r2_12).val[1];
#endif // __aarch64__

            uint8x16x3_t bgr1, bgr2;
            bgr1.val[0] = v8b1_1;
            bgr1.val[1] = v8g1_1;
            bgr1.val[2] = v8r1_1;

            bgr2.val[0] = v8b2_1;
            bgr2.val[1] = v8g2_1;
            bgr2.val[2] = v8r2_1;

            vst3q_u8(row1, bgr1);
            vst3q_u8(row2, bgr2);
        }

        y1 += stride * 2;
        u1 += w / 2;
        v1 += w / 2;
    }
}

void ImageUtil::cvt_color_yuv2bgr_nv21(unsigned char *src, unsigned char *dst, int w, int h, int bIdx) {
    if (src == nullptr || dst == nullptr) {
        return;
    }

    if (w % 2 != 0 || h % 2 != 0) {
        return;
    }

    const int stride = w;

    int y_stride = w * h;

    unsigned char *y1 = src;
    unsigned char *vu1 = y1 + y_stride;

    int8x8_t cvr = vdup_n_s8 (ITUR_BT_602_CVR);
    int8x8_t cvg = vdup_n_s8 (-ITUR_BT_602_CVG);
    int8x8_t cug = vdup_n_s8 (-ITUR_BT_602_CUG);
    int8x8_t cub = vdup_n_s8 (ITUR_BT_602_CUB);
    int16x8_t cy = vdupq_n_s16 (ITUR_BT_602_CY);
    uint8x8_t uvoffset = vdup_n_u8 (128);
    uint8x8_t yoffset = vdup_n_u8 (16);
    int16x8_t round_offset = vdupq_n_s16(1 << (ITUR_BT_602_SHIFT - 1));

    for (int j = 0; j < h; j += 2) {
        unsigned char *row1 = dst + j * w * 3;
        unsigned char *row2 = row1 + w * 3;

        const unsigned char *y2 = y1 + stride;

        for (int i = 0; i < w / 16; i += 1, row1 += 48, row2 += 48) {

            const unsigned char* ly1 = y1 + i * 16;
            const unsigned char* ly2 = y2 + i * 16;
            uint8x8x2_t vvu = vld2_u8(vu1 + i * 16);
            uint8x8_t vu = vvu.val[1];
            uint8x8_t vv = vvu.val[0];
            uint8x8x2_t vy1 = vld2_u8(ly1);
            uint8x8x2_t vy2 = vld2_u8(ly2);

            uint16x8_t vu16 = vsubl_u8(vu, uvoffset); // widen subtract
            uint16x8_t vv16 = vsubl_u8(vv, uvoffset);
            int8x8_t svu = vqmovn_s16(vreinterpretq_s16_u16(vu16)); // convert to signed integer
            int8x8_t svv = vqmovn_s16(vreinterpretq_s16_u16(vv16));

            uint16x8x2_t vy116, vy216;
            vy116.val[0] = vsubl_u8(vy1.val[0], yoffset);
            vy116.val[1] = vsubl_u8(vy1.val[1], yoffset);
            vy216.val[0] = vsubl_u8(vy2.val[0], yoffset);
            vy216.val[1] = vsubl_u8(vy2.val[1], yoffset);
            int16x8x2_t svy1, svy2;
            svy1.val[0] = vreinterpretq_s16_u16(vy116.val[0]);
            svy1.val[1] = vreinterpretq_s16_u16(vy116.val[1]);
            svy2.val[0] = vreinterpretq_s16_u16(vy216.val[0]);
            svy2.val[1] = vreinterpretq_s16_u16(vy216.val[1]);

            int16x8_t gu = vmull_s8(svu, cug);
            int16x8_t gv = vmull_s8(svv, cvg);
            int16x8_t ruv = vmull_s8(svv, cvr);
            int16x8_t buv = vmull_s8(svu, cub);
            int16x8_t guv = vaddq_s16(gu, gv);
            ruv = vaddq_s16(ruv, round_offset);
            buv = vaddq_s16(buv, round_offset);
            guv = vaddq_s16(guv, round_offset);

            int16x8x2_t vy1_2, vy2_2;
            vy1_2.val[0] = vmulq_s16(svy1.val[0], cy);
            vy1_2.val[1] = vmulq_s16(svy1.val[1], cy);
            vy2_2.val[0] = vmulq_s16(svy2.val[0], cy);
            vy2_2.val[1] = vmulq_s16(svy2.val[1], cy);

            int16x8x2_t vb1_2, vg1_2, vr1_2, vb2_2, vg2_2, vr2_2;
            vb1_2.val[0] = vaddq_s16(vy1_2.val[0], buv);
            vg1_2.val[0] = vaddq_s16(vy1_2.val[0], guv);
            vr1_2.val[0] = vaddq_s16(vy1_2.val[0], ruv);

            vb1_2.val[1] = vaddq_s16(vy1_2.val[1], buv);
            vg1_2.val[1] = vaddq_s16(vy1_2.val[1], guv);
            vr1_2.val[1] = vaddq_s16(vy1_2.val[1], ruv);

            vb2_2.val[0] = vaddq_s16(vy2_2.val[0], buv);
            vg2_2.val[0] = vaddq_s16(vy2_2.val[0], guv);
            vr2_2.val[0] = vaddq_s16(vy2_2.val[0], ruv);

            vb2_2.val[1] = vaddq_s16(vy2_2.val[1], buv);
            vg2_2.val[1] = vaddq_s16(vy2_2.val[1], guv);
            vr2_2.val[1] = vaddq_s16(vy2_2.val[1], ruv);


            uint16x8x2_t svb1_2, svg1_2, svr1_2, svb2_2, svg2_2, svr2_2;
            svb1_2.val[0] = vreinterpretq_u16_s16(vqabsq_s16(vb1_2.val[0]));
            svb1_2.val[1] = vreinterpretq_u16_s16(vqabsq_s16(vb1_2.val[1]));
            svg1_2.val[0] = vreinterpretq_u16_s16(vqabsq_s16(vg1_2.val[0]));
            svg1_2.val[1] = vreinterpretq_u16_s16(vqabsq_s16(vg1_2.val[1]));
            svr1_2.val[0] = vreinterpretq_u16_s16(vqabsq_s16(vr1_2.val[0]));
            svr1_2.val[1] = vreinterpretq_u16_s16(vqabsq_s16(vr1_2.val[1]));
            svb2_2.val[0] = vreinterpretq_u16_s16(vqabsq_s16(vb2_2.val[0]));
            svb2_2.val[1] = vreinterpretq_u16_s16(vqabsq_s16(vb2_2.val[1]));
            svg2_2.val[0] = vreinterpretq_u16_s16(vqabsq_s16(vg2_2.val[0]));
            svg2_2.val[1] = vreinterpretq_u16_s16(vqabsq_s16(vg2_2.val[1]));
            svr2_2.val[0] = vreinterpretq_u16_s16(vqabsq_s16(vr2_2.val[0]));
            svr2_2.val[1] = vreinterpretq_u16_s16(vqabsq_s16(vr2_2.val[1]));

            uint8x8x2_t v8b1_2, v8g1_2, v8r1_2, v8b2_2, v8g2_2, v8r2_2;
            v8b1_2.val[0] = vqshrn_n_u16(svb1_2.val[0], ITUR_BT_602_SHIFT);
            v8b1_2.val[1] = vqshrn_n_u16(svb1_2.val[1], ITUR_BT_602_SHIFT);
            v8g1_2.val[0] = vqshrn_n_u16(svg1_2.val[0], ITUR_BT_602_SHIFT);
            v8g1_2.val[1] = vqshrn_n_u16(svg1_2.val[1], ITUR_BT_602_SHIFT);
            v8r1_2.val[0] = vqshrn_n_u16(svr1_2.val[0], ITUR_BT_602_SHIFT);
            v8r1_2.val[1] = vqshrn_n_u16(svr1_2.val[1], ITUR_BT_602_SHIFT);

            v8b2_2.val[0] = vqshrn_n_u16(svb2_2.val[0], ITUR_BT_602_SHIFT);
            v8b2_2.val[1] = vqshrn_n_u16(svb2_2.val[1], ITUR_BT_602_SHIFT);
            v8g2_2.val[0] = vqshrn_n_u16(svg2_2.val[0], ITUR_BT_602_SHIFT);
            v8g2_2.val[1] = vqshrn_n_u16(svg2_2.val[1], ITUR_BT_602_SHIFT);
            v8r2_2.val[0] = vqshrn_n_u16(svr2_2.val[0], ITUR_BT_602_SHIFT);
            v8r2_2.val[1] = vqshrn_n_u16(svr2_2.val[1], ITUR_BT_602_SHIFT);

            uint8x8_t vzero = vdup_n_u8(0);

            uint8x16_t v8b1_11 = vcombine_u8(vzero, v8b1_2.val[0]);
            uint8x16_t v8b1_12 = vcombine_u8(vzero, v8b1_2.val[1]);
            uint8x16_t v8g1_11 = vcombine_u8(vzero, v8g1_2.val[0]);
            uint8x16_t v8g1_12 = vcombine_u8(vzero, v8g1_2.val[1]);
            uint8x16_t v8r1_11 = vcombine_u8(vzero, v8r1_2.val[0]);
            uint8x16_t v8r1_12 = vcombine_u8(vzero, v8r1_2.val[1]);

            uint8x16_t v8b2_11 = vcombine_u8(vzero, v8b2_2.val[0]);
            uint8x16_t v8b2_12 = vcombine_u8(vzero, v8b2_2.val[1]);
            uint8x16_t v8g2_11 = vcombine_u8(vzero, v8g2_2.val[0]);
            uint8x16_t v8g2_12 = vcombine_u8(vzero, v8g2_2.val[1]);
            uint8x16_t v8r2_11 = vcombine_u8(vzero, v8r2_2.val[0]);
            uint8x16_t v8r2_12 = vcombine_u8(vzero, v8r2_2.val[1]);

#if __aarch64__
            uint8x16_t v8b1_1 = vzip2q_u8(v8b1_11, v8b1_12);
            uint8x16_t v8g1_1 = vzip2q_u8(v8g1_11, v8g1_12);
            uint8x16_t v8r1_1 = vzip2q_u8(v8r1_11, v8r1_12);

            uint8x16_t v8b2_1 = vzip2q_u8(v8b2_11, v8b2_12);
            uint8x16_t v8g2_1 = vzip2q_u8(v8g2_11, v8g2_12);
            uint8x16_t v8r2_1 = vzip2q_u8(v8r2_11, v8r2_12);
#else
            uint8x16_t v8b1_1 = vzipq_u8(v8b1_11, v8b1_12).val[1];
            uint8x16_t v8g1_1 = vzipq_u8(v8g1_11, v8g1_12).val[1];
            uint8x16_t v8r1_1 = vzipq_u8(v8r1_11, v8r1_12).val[1];

            uint8x16_t v8b2_1 = vzipq_u8(v8b2_11, v8b2_12).val[1];
            uint8x16_t v8g2_1 = vzipq_u8(v8g2_11, v8g2_12).val[1];
            uint8x16_t v8r2_1 = vzipq_u8(v8r2_11, v8r2_12).val[1];
#endif // __aarch64__

            uint8x16x3_t bgr1, bgr2;
            bgr1.val[0] = v8b1_1;
            bgr1.val[1] = v8g1_1;
            bgr1.val[2] = v8r1_1;

            bgr2.val[0] = v8b2_1;
            bgr2.val[1] = v8g2_1;
            bgr2.val[2] = v8r2_1;

            vst3q_u8(row1, bgr1);
            vst3q_u8(row2, bgr2);
        }

        y1 += stride * 2;
        vu1 += w;
    }
}
#endif // __ARM_NEON

void ImageUtil::by_landmark_get_warp_affine(const VPoint *src_landmarks, const std::vector <VPoint> &base_landmarks,
                                            cv::Mat &rotation_matrix) {
    float lefteyex = 0;
    float lefteyey = 0;
    float righteyex = 0;
    float righteyey = 0;
    for (int i = FLM_61_R_EYE_LEFT_CORNER; i < FLM_71_NOSE_BRIDGE1; i++) {
        lefteyex += src_landmarks[i].x;
        lefteyey += src_landmarks[i].y;
    }
    for (int i = FLM_51_L_EYE_LEFT_CORNER; i < FLM_61_R_EYE_LEFT_CORNER; i++) {
        righteyex += src_landmarks[i].x;
        righteyey += src_landmarks[i].y;
    }
    lefteyex /= 10.0f;
    lefteyey /= 10.0f;
    righteyex /= 10.0f;
    righteyey /= 10.0f;
    float dx = righteyex - lefteyex;
    float dy = righteyey - lefteyey;
    // 得到一个目标弧度值
    float srcDegrees = static_cast<float>((std::atan2(dy, dx) * ANGEL_180 / M_PI) - ANGEL_180);

//        /**
//        *  计算 Base 数据的眼部中心点
//        */
//        float base_left_eye_x = 0;
//        float base_left_eye_y = 0;
//        float base_right_eye_x = 0;
//        float base_right_eye_y = 0;
//        for (int i = FLM_61_R_EYE_LEFT_CORNER; i < FLM_71_NOSE_BRIDGE1; i++) {
//
//            base_left_eye_x += base_landmarks[i].x;
//            base_left_eye_y += base_landmarks[i].y;
//
//        }
//        for (int i = FLM_51_L_EYE_LEFT_CORNER; i < FLM_61_R_EYE_LEFT_CORNER; i++) {
//            base_right_eye_x += base_landmarks[i].x;
//            base_right_eye_y += base_landmarks[i].y;
//        }
//        base_left_eye_x /= 10;
//        base_left_eye_y /= 10;
//        base_right_eye_x /= 10;
//        base_right_eye_y /= 10;
//        float base_dx = base_right_eye_x - base_left_eye_x;
//        float base_dy = base_right_eye_y - base_left_eye_y;
//        // 根据弧度值计算角度
//        int baseDegrees = int(std::atan2(base_dy, base_dx) * Config::ANGEL_180 / M_PI) - Config::ANGEL_180;

    /**
     *   计算 Base 的 x,y 均值
     */
    float base_mean_x = 0.0f;
    float base_mean_y = 0.0f;
    for (int j = 0; j < LM_2D_106_COUNT; ++j) {
        base_mean_x += base_landmarks[j].x;
        base_mean_y += base_landmarks[j].y;
    }
    base_mean_x = base_mean_x / LM_2D_106_COUNT;
    base_mean_y = base_mean_y / LM_2D_106_COUNT;

    /**
     *  计算输入 landmark 的 x 均值
     */
    float input_mean_x = 0.0f;
    float input_mean_y = 0.0f;
    for (int k = 0; k < 106; ++k) {
        input_mean_x += src_landmarks[k].x;
        input_mean_y += src_landmarks[k].y;
    }
    input_mean_x = input_mean_x / LM_2D_106_COUNT;
    input_mean_y = input_mean_y / LM_2D_106_COUNT;

    /**
     * 计算 base 根号平方和
     */
    float sum_base_x = 0.0f;
    float sum_base_y = 0.0f;
    for (int l = 0; l < LM_2D_106_COUNT; ++l) {
        sum_base_x += std::pow(base_landmarks[l].x - base_mean_x, 2);
        sum_base_y += std::pow(base_landmarks[l].y - base_mean_y, 2);
    }
    float base_sign = std::sqrt(sum_base_x + sum_base_y);
    /**
     * 计算输入的根号平方和
     */
    float sum_input_x = 0.0f;
    float sum_input_y = 0.0f;
    for (int l = 0; l < 106; ++l) {
        sum_input_x += std::pow(src_landmarks[l].x - input_mean_x, 2);
        sum_input_y += std::pow(src_landmarks[l].y - input_mean_y, 2);
    }
    float input_sign = std::sqrt(sum_input_x + sum_input_y);
    float scale =
            (base_sign / LM_2D_106_COUNT) / (input_sign / LM_2D_106_COUNT);

    rotation_matrix = cv::getRotationMatrix2D(cv::Point(0, 0), srcDegrees, scale);
    rotation_matrix.at<double>(0, 2) =
            base_mean_x - rotation_matrix.at<double>(0, 0) * input_mean_x -
            rotation_matrix.at<double>(0, 1) * input_mean_y;
    rotation_matrix.at<double>(1, 2) =
            base_mean_y - rotation_matrix.at<double>(1, 0) * input_mean_x -
            rotation_matrix.at<double>(1, 1) * input_mean_y;
}

void ImageUtil::get_warp_params(const VPoint *landmark, const VPoint *ref_landmark, float &scale, float &rot,
                                va_cv::VScalar &aux_coeff) {
    // 根据眼部关键点求脸部转正角度
    float left_eye_cx = 0.f;
    float left_eye_cy = 0.f;
    for (int i = FLM_61_R_EYE_LEFT_CORNER; i < FLM_71_NOSE_BRIDGE1; i++) {
        left_eye_cx += landmark[i].x;
        left_eye_cy += landmark[i].y;
    }
    left_eye_cx /= 10.f;
    left_eye_cy /= 10.f;

    float right_eye_cx = 0.f;
    float right_eye_cy = 0.f;
    for (int i = FLM_51_L_EYE_LEFT_CORNER; i < FLM_61_R_EYE_LEFT_CORNER; i++) {
        right_eye_cx += landmark[i].x;
        right_eye_cy += landmark[i].y;
    }
    right_eye_cx /= 10.f;
    right_eye_cy /= 10.f;

    auto dx = right_eye_cx - left_eye_cx;
    auto dy = right_eye_cy - left_eye_cy;
    rot = static_cast<float>((std::atan2(dy, dx) * 180.f / M_PI) - 180.f);

    // 计算检测人脸的关键点平均值
    float lmk_mean_x = 0.f;
    float lmk_mean_y = 0.f;
    for (int i = 0; i < LM_2D_106_COUNT; ++i) {
        lmk_mean_x += landmark[i].x;
        lmk_mean_y += landmark[i].y;
    }
    lmk_mean_x = lmk_mean_x / LM_2D_106_COUNT;
    lmk_mean_y = lmk_mean_y / LM_2D_106_COUNT;

    // 计算检测人脸的关键点到中心点的距离之和
    float dist_x = 0.0f;
    float dist_y = 0.0f;
    for (int l = 0; l < 106; ++l) {
        dist_x += std::pow(landmark[l].x - lmk_mean_x, 2);
        dist_y += std::pow(landmark[l].y - lmk_mean_y, 2);
    }
    float dist = std::sqrt(dist_x + dist_y);

    // 计算参考关键点的平均值
    float ref_lmk_mean_x = 0.f;
    float ref_lmk_mean_y = 0.f;
    for (int j = 0; j < LM_2D_106_COUNT; ++j) {
        ref_lmk_mean_x += ref_landmark[j].x;
        ref_lmk_mean_y += ref_landmark[j].y;
    }
    ref_lmk_mean_x = ref_lmk_mean_x / LM_2D_106_COUNT;
    ref_lmk_mean_y = ref_lmk_mean_y / LM_2D_106_COUNT;

    // 计算参考关键点到中心点的距离之和
    float ref_dist_x = 0.0f;
    float ref_dist_y = 0.0f;
    for (int i = 0; i < LM_2D_106_COUNT; ++i) {
        ref_dist_x += std::pow(ref_landmark[i].x - ref_lmk_mean_x, 2);
        ref_dist_y += std::pow(ref_landmark[i].y - ref_lmk_mean_y, 2);
    }
    float ref_dist = std::sqrt(ref_dist_x + ref_dist_y);

    // 计算缩放系数
    scale = ref_dist / dist;

    // 修正参数
    aux_coeff.v0 = lmk_mean_x;
    aux_coeff.v1 = lmk_mean_y;
    aux_coeff.v2 = ref_lmk_mean_x;
    aux_coeff.v3 = ref_lmk_mean_y;
}

void ImageUtil::get_warp_params(const VPoint *landmark, const VPoint *ref_landmark, VTensor &rot_matrix) {
    // 当前帧检测的 Landmark
    std::vector<float> lmk_x(LM_2D_106_COUNT);
    std::vector<float> lmk_y(LM_2D_106_COUNT);
    for (int i = 0; i < LM_2D_106_COUNT; ++i) {
        // 当前帧检测的 Landmark
        lmk_x[i] = landmark[i].x;
        lmk_y[i] = landmark[i].y;
    }

    // 基准的 Landmark
    static std::vector<float> ref_lmk_x(LM_2D_106_COUNT);
    static std::vector<float> ref_lmk_y(LM_2D_106_COUNT);
    for (int i = 0; i < LM_2D_106_COUNT; ++i) {
        // 基准的 Landmark
        ref_lmk_x[i] = ref_landmark[i].x;
        ref_lmk_y[i] = ref_landmark[i].y;
    }

    float m01 =
            std::inner_product(std::begin(lmk_x), std::end(lmk_x), std::begin(lmk_x), 0.f) +
            std::inner_product(std::begin(lmk_y), std::end(lmk_y), std::begin(lmk_y), 0.f);
    float m02 = 0.0f;
    float m03 = std::accumulate(lmk_x.begin(), lmk_x.begin() + LM_2D_106_COUNT, 0.f);
    float m04 = std::accumulate(lmk_y.begin(), lmk_y.begin() + LM_2D_106_COUNT, 0.f);
    float m05 =
            (std::inner_product(std::begin(ref_lmk_x), std::end(ref_lmk_x),
                                std::begin(lmk_x), 0.f) +
             std::inner_product(std::begin(lmk_y), std::end(lmk_y),
                                std::begin(ref_lmk_y),
                                0.f)) *
            -1;

    float m11 = 0;
    float m12 = m01;
    float m13 = m04 * -1;
    float m14 = m03;
    float m15 =
            std::inner_product(std::begin(ref_lmk_x), std::end(ref_lmk_x),
                               std::begin(lmk_y), 0.f) -
            std::inner_product(std::begin(lmk_x), std::end(lmk_x), std::begin(ref_lmk_y),
                               0.f);

    float m21 = m03;
    float m22 = m04 * -1;
    float m23 = LM_2D_106_COUNT;
    float m24 = 0;
    float m25 =
            std::accumulate(ref_lmk_x.begin(), ref_lmk_x.begin() + LM_2D_106_COUNT,
                            0.f) * -1;

    float m31 = m04;
    float m32 = m03;
    float m33 = 0;
    float m34 = LM_2D_106_COUNT;
    float m35 =
            std::accumulate(ref_lmk_y.begin(), ref_lmk_y.begin() + LM_2D_106_COUNT,
                            0.f) * -1;

    float m16_arr[16]{m01, m02, m03, m04, m11, m12, m13, m14, m21, m22, m23, m24, m31, m32, m33, m34};
    float m4_arr[4]{m05 * -1, m15 * -1, m25 * -1, m35 * -1};

    // 矩阵求逆
    float m_dst[16];
    if (!MathUtils::matrix_4x4_invert(m16_arr, m_dst)) {
        return;
    }

    // 矩阵相乘
    float m_results[4];
    for (int i = 0; i < 4; ++i) {
        m_results[i] = m_dst[i * 4 + 0] * m4_arr[0] +
                       m_dst[i * 4 + 1] * m4_arr[1] +
                       m_dst[i * 4 + 2] * m4_arr[2] +
                       m_dst[i * 4 + 3] * m4_arr[3];
    }

    // 仿射变换矩阵
    rot_matrix.create(3, 2, FP32);
    auto *m_data = (float *) rot_matrix.data;
    m_data[0] = m_results[0];
    m_data[1] = m_results[1] * -1;
    m_data[2] = m_results[2];
    m_data[3] = m_results[1];
    m_data[4] = m_results[0];
    m_data[5] = m_results[3];
}


int ImageUtil::get_landmarks_x_min(VPoint *landmarks, std::vector<int> list) {
    int value = -1;
    for (auto iter = list.begin(); iter != list.end(); ++iter) {
        if (value == -1) {
            value = landmarks[*iter].x;
        } else if (value > landmarks[*iter].x) {
            value = landmarks[*iter].x;
        }
    }
    return value;
}

int ImageUtil::get_landmarks_x_max(VPoint *landmarks, std::vector<int> list) {
    int value = -1;
    for (auto iter = list.begin(); iter != list.end(); ++iter) {
        if (value == -1) {
            value = landmarks[*iter].x;
        } else if (value < landmarks[*iter].x) {
            value = landmarks[*iter].x;
        }
    }
    return value;
}

int ImageUtil::get_landmarks_y_min(VPoint *landmarks, std::vector<int> list) {
    int value = -1;
    for (auto iter = list.begin(); iter != list.end(); ++iter) {
        if (value == -1) {
            value = landmarks[*iter].y;
        } else if (value > landmarks[*iter].y) {
            value = landmarks[*iter].y;
        }
    }
    return value;
}

int ImageUtil::get_landmarks_y_max(VPoint *landmarks, std::vector<int> list) {
    int value = -1;
    for (auto iter = list.begin(); iter != list.end(); ++iter) {
        if (value == -1) {
            value = landmarks[*iter].y;
        } else if (value < landmarks[*iter].y) {
            value = landmarks[*iter].y;
        }
    }
    return value;
}

} // namespace aura::vision