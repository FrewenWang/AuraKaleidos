//
// Created by Frewen.Wang on 2022/9/24.
//
#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "aura/cv/ops/resize/Resize.h"
#include "aura/cv/utils/ImageUtil.h"
#include "aura/cv/ops/cvt_color/Convert.h"
#include "aura/aura_utils/utils/PrintUtil.h"

using namespace aura::cv;
using namespace aura::utils;

int main(int argc, char **argv) {
    // C++ 进行日志打印
    std::cout << "begin execute AuraCV Examples" << std::endl;

    std::string img_path = "./1704698709.jpg";

    cv::Mat image = cv::imread(img_path);
    // cv::imshow("image", image);
    // cv::waitKey(0);
    cv::Mat fixedImage = Resize::fixImageSize(image, 1600, 1300);
    cv::Mat yuv420NV21 = ImageUtil::bgr2Yuv420NV21(fixedImage);

    // 打印前100个像素
    // 174 174 174 174 174 174 174 174 174 174 174 174 174 174 174 174 174 174 174 174 174 174 174 174 174 174 174 174
    // 174 174 174 174 174 174 174 174 174 174 174 174 174 174 174 174 174 174 174 174 174 174
    PrintUtil::printArray(yuv420NV21.data, 50, "nv21");

    // 进行(NV21转灰度图)的格式转换
    cv::Mat grayFromNV21;
    Convert::cvtColor(image, grayFromNV21, ColorCvtFormat::COLOR_BGR2GRAY);
    // 打印前20个像素
    // opencv2.4.13的转换像素值
    // 20 23 26 26 25 26 28 31 24 79 131 156 148 96 42 25 17 117 160 134 145 166 108 22 32 28 26 28 29 24 18 15 14 14
    // 14 14 14 14 13 13 19 19 18 18 17 17 17 16 11 17
    // opencv4.6.0的转换像素值
    // 20 23 26 26 25 26 28 31 24 79 131 156 148 96 42 25 17 117 160 134 145 166 108 22 32 28 26 28 29 24 18 15 14 14
    // 14 14 14 14 13 13 19 19 18 18 17 17 17 16 11 17
    PrintUtil::printArray(grayFromNV21.data, 50, "grayFromNV21");


    std::cout << "end execute AuraCV Examples" << std::endl;
    return 0;
}