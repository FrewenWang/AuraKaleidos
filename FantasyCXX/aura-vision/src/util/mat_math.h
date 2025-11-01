
#ifndef VISION_NATIVE_MAT_MATH_H
#define VISION_NATIVE_MAT_MATH_H

#include <stdint.h>
#include <opencv2/core/core.hpp>
#include "vision/core/common/VStructs.h"

namespace aura::vision {

    typedef struct {
        uint32_t dim_x;        /*!< \brief Width of buffer in X dimension in elements. */
        uint32_t dim_y;        /*!< \brief Height of buffer in Y dimension in elements. */
        int32_t stride_y;     /*!< \brief Stride in Y dimension in bytes. */
    } params2d_t;

    class MatMath {

    public:

        static void scale_image_bilinear(const uint8_t src[], params2d_t &src_addr, uint8_t dst[],
                                         params2d_t &dst_addr, float x_scale, float y_scale,
                                         float src_offset_x, float src_offset_y, int16_t dst_offset_x,
                                         int16_t dst_offset_y);

        static void mean_stddev(uint8_t base_ptr[],
                                params2d_t &addrs, float *fmean, float *fstddev,
                                uint32_t *pixels_processed, float *fcurrent_sum);

        static void convert_to_bidiag(const int rows, const int cols, float *u,
                                      float *v, float *diag, float *superdiag);

        static int bidiag_to_diag(const int rows, const int cols, float *u,
                                  float *v, float *diag, float *superdiag);

        static void sort_singular_values(const int rows, const int cols, float *u,
                                         float *v, float *singular_values);

        static void svd(const int rows, const int cols, float *a, float *u,
                        float *v, float *u1, float *diag, float *superdiag);

        static int cholesky(int enable_test, const int order, float *a, float *l);

        static void cholesky_solver(const int order, float *l, float *y, float *b,
                                    float *x);

        static void mat_mul(float *x1, const int r1, const int c1, float *x2,
                            const int c2, float *y);

        static void mat_trans(const float *x, const int rows, const int cols,
                              float *y);

        /**
         * 查找每列最大和最小值
         * @param data 图片数据
         * @param rows 行数
         * @param cols 列数
         * @param max  最大值
         * @param min  最小值
         */
        static void find_col_max_min(cv::Mat &mat, int rows, int cols, float *max, float *min);

        /**
         * 图片排列是 H-W-C 且格式是BGR 对图片进行减均值除方差
         * @param input_img 输入图片
         * @param rows 行数
         * @param cols 列数
         * @param mean 均值
         * @param std  方差
         */
        static void color_normalization(float *input_img, int rows, int cols, int channels,
                                        std::vector<float> &mean, std::vector<float> &std, NORMAL_ALG algorithm = MUL);

        /**
         * Convert HWC image to CHW
         * Image type must be CV32FC1
         * @param pblob
         * @param img
         */
        static void convert_hwc_to_chw(cv::Mat &img, float *pblob);
    };

}
#endif //VISION_NATIVE_MAT_MATH_H
