#ifndef VISION_CVT_COLOR_H
#define VISION_CVT_COLOR_H

#include "opencv2/opencv.hpp"

namespace xperf {

const unsigned int R2YI = 4899;
const unsigned int G2YI = 9617;
const unsigned int B2YI = 1868;
const unsigned int B2UI = 9241;
const unsigned int R2VI = 11682;

class ImageProcessUtil {
public:
    static cv::Mat bgr_to_yuv(const cv::Mat& frame) {
        int yuv_nv21_mem_len = sizeof(unsigned char) * frame.rows*3/2 * frame.cols;
        unsigned char* yuv_nv21_mem = (unsigned char*)malloc(yuv_nv21_mem_len);
        bgr2nv21(frame.data, yuv_nv21_mem, frame.cols, frame.rows);

        cv::Mat yuv_image;
        yuv_image.create(frame.rows * 3 / 2, frame.cols, CV_8UC1);
        memcpy(yuv_image.data, yuv_nv21_mem, yuv_nv21_mem_len);

        free(yuv_nv21_mem);
        return yuv_image;
    }

    static void bgr2nv21(unsigned char *src, unsigned char *dst, int width, int height) {
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

    static cv::Mat fix_image_size(cv::Mat &in, int w, int h) {
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
};

} // namespace xperf

#endif //VISION_CVT_COLOR_H
