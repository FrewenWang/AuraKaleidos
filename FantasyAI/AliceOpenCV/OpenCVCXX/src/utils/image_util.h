//
// Created by Frewen.Wong on 2021/10/14.
//

#ifndef OPENCV_DEMO_IMAGE_UTIL_H
#define OPENCV_DEMO_IMAGE_UTIL_H

#include <opencv2/core/mat.hpp>

namespace vision {
/**
 * 图像处理相关的工具类
 * 所有的图像处理按照宽高进行处理
 */
class ImageUtil {
public:
    /**
     * Opencv 平均池化
     * @param in
     * @return
     */
    static cv::Mat average_pooling(cv::Mat in);

    /**
     * 使用高斯滤波器（3×3 大小，标准差 s=1.3）来对图片进行降噪处理
     * 高斯滤波器是一种可以使图像平滑的滤波器，用于去除噪声。可用于去除噪声的滤波器还有中值滤波器，平滑滤波器、LoG 滤波器。
     * 高斯滤波器将中心像素周围的像素按照高斯分布加权平均进行平滑化。这样的（二维）权值通常被称为卷积核或者滤波器。
     * 但是，由于图像的长宽可能不是滤波器大小的整数倍，因此我们需要在图像的边缘补0。这种方法称作 Zero Padding。并且权值（卷积核）要进行归一化操作(∑g = 1)。
     * 权值 g(x,y,s) = 1/ (s*sqrt(2 * pi)) * exp( - (x^2 + y^2) / (2*s^2))
     * 标准差 s = 1.3 的 8 近邻 高斯滤波器如下：
     *              1 2 1
     *  K =  1/16 [ 2 4 2 ]
     *              1 2 1
     * @param in
     * @return
     */
    static cv::Mat gaussian_filter(cv::Mat in);

};


}

#endif //OPENCV_DEMO_IMAGE_UTIL_H
