#pragma once
#include "opencv2/opencv.hpp"

class ImageUtil {
public:
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

private:
    static const unsigned int R2YI = 4899;
    static const unsigned int G2YI = 9617;
    static const unsigned int B2YI = 1868;
    static const unsigned int B2UI = 9241;
    static const unsigned int R2VI = 11682;    
};