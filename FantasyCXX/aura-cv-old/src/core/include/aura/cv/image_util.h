//
// Created by frewen on 22-10-10.
//
#pragma once

#include "opencv2/core/core.hpp"

namespace aura::aura_cv {

const static unsigned int R2YI = 4899;
const static unsigned int G2YI = 9617;
const static unsigned int B2YI = 1868;
const static unsigned int B2UI = 9241;
const static unsigned int R2VI = 11682;

const static unsigned char ITUR_BT_602_CY = 37;
const static unsigned char ITUR_BT_602_CUB = 65;
const static unsigned char ITUR_BT_602_CUG = 13;
const static unsigned char ITUR_BT_602_CVG = 26;
const static unsigned char ITUR_BT_602_CVR = 51;
const static unsigned char ITUR_BT_602_SHIFT = 5;

class image_util {
public:
    /**
     * BGR转换为NV21格式
     * @param src 输入BGR图像数据
     * @param dst 输出NV21图像数据
     * @param width 图像宽度
     * @param height 图像高度
     */
    static void bgr2Yuv420NV21(unsigned char *src, unsigned char *dst, int width, int height);

    /**
     * @brief BGR转换为NV21格式
     * @param frame 输入BGR图像数据
     * @return 转化而成的nv21数据
     */
    static cv::Mat bgr2Yuv420NV21(const cv::Mat &frame);

    /**
     * 针对图片进行补边或者缩放
     * @param in 输出的图片数据
     * @param w  目标的宽度
     * @param h  目标的宽度
     * @return
     */
    static cv::Mat fixImageSize(cv::Mat &in, int w, int h);

    /**
     * BRG数据转化成为 YUV 444 Planar
     * @param in  输出的图片数据
     * @param flag
     * flag = 0  0->YUV444Planer(UV)
     * I444（属于 YUV 444 Planar） I444 属于 YUV 444 Planar 的一种。
     * YUV分量分别存放，先是 width * height 长度的Y，后面跟 width * height 长度的U， 最后是w * h长度的V，总长度为 width * height * 3。
     * flag = 1 1->YUV444Planer(VU)
     * YV24 属于 YUV 444 Planar 的一种。
     * YUV 分量分别存放，先是w * h长度的Y，后面跟w * h长度的V， 最后是w * h长度的U，总长度为w * height * 3。
     * 与 I444 不同的是，YV24 是先排列 V。
     * @return
     */
    static unsigned char *bgr2Yuv444Planer(cv::Mat &in, int flag = 0);

    /**
     * BRG数据转化成为 YUV 444 SemiPlanar
     * @param in
     * @param flag
     * @return
     */
    static unsigned char *bgr2Yuv444SemiPlanar(cv::Mat &in, int flag = 0);

    static unsigned char *bgr2Yuv422Pr(cv::Mat &in, int flag = 0);

    static unsigned char *bgr2Yuv422PYv16(cv::Mat &in, int flag = 0);

    static unsigned char *bgr2Yuv420I420(cv::Mat &in, int flag = 0);

    static unsigned char *bgr2Yuv420Nv12(cv::Mat &in, int flag = 0);

    static unsigned char *bgr2Yuv422Semip(cv::Mat &in, int flag = 0);
};

} // namespace aura::aura_cv
