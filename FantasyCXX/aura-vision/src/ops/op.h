#pragma once

#include <opencv2/opencv.hpp>

namespace aura::vision {
namespace op {

struct VSize {
    int w;
    int h;
    VSize() : w(0), h(0) {}
    VSize(int _w, int _h) : w(_w), h(_h) {}
};

struct VScalar {
    double v0;
    double v1;
    double v2;
    double v3;

    VScalar() : v0(0), v1(0), v2(0), v3(0) {}
};

// 插值模式
enum VInterMode {
    INTER_NEAREST = 0,
    INTER_LINEAR = 1,
    INTER_CUBIC = 2,
    INTER_AREA = 3,
    INTER_LANCZOS4 = 4,
    INTER_MAX = 7,
    WARP_INVERSE_MAP = 16
};

// 边界模式
enum VBorderMode {
    BORDER_REPLICATE = 1,
    BORDER_CONSTANT = 0,
    BORDER_REFLECT = 2,
    BORDER_WRAP = 3,
    BORDER_REFLECT_101 = 4,
    BORDER_REFLECT101 = 4,
    BORDER_TRANSPARENT = 5,
    BORDER_DEFAULT = 4,
    BORDER_ISOLATED = 16
};

// 模板匹配模式
enum VMatchMode {
    TM_SQDIFF = 0,
    TM_SQDIFF_NORMED = 1,
    TM_CCORR = 2,
    TM_CCORR_NORMED = 3,
    TM_CCOEFF = 4,
    TM_CCOEFF_NORMED = 5
};

// 颜色转换模式
//enum InputImageFormat {
//    COLOR_GRAY2RGB = 8,
//    COLOR_GRAY2BGR = COLOR_GRAY2RGB,
//    COLOR_YUV2RGB_NV12 = 90,
//    COLOR_YUV2BGR_NV12 = 91,
//    COLOR_YUV2RGB_NV21 = 92,
//    COLOR_YUV2BGR_NV21 = 93,
//    COLOR_YUV2RGBA_NV12 = 94,
//    COLOR_YUV2BGRA_NV12 = 95,
//    COLOR_YUV2RGBA_NV21 = 96,
//    COLOR_YUV2BGRA_NV21 = 97,
//    COLOR_YUV2BGR_YV12 = 99
//};

// ---------------------------------------------------------------------------------------------------------------------
// 常规操作 Operator
// ---------------------------------------------------------------------------------------------------------------------

void memcpy(void *a, const void *b, size_t c);

// ---------------------------------------------------------------------------------------------------------------------
// 图像处理 Operator
// ---------------------------------------------------------------------------------------------------------------------

/**
 * @brief 仿射变换
 * @param src 输入
 * @param dst 输出
 * @param M 仿射变换矩阵
 * @param dsize 目标尺寸
 * @param flags 插值模式
 * @param borderMode 边界模式
 * @param borderValue 边界值
 */
void warpAffine(cv::Mat &src, cv::Mat &dst, cv::Mat &M, cv::Size &dsize,
                const int &flags = cv::INTER_LINEAR, const int &borderMode = cv::BORDER_CONSTANT,
                const cv::Scalar &borderValue = cv::Scalar());

} // namespace op
} // namespace aura::vision