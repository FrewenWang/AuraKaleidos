//
// Created by on 2021/7/31.
// 代码参考：https://geek-docs.com/opencv/opencv-examples/huiduhua.html
//

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

/**
 * 什么叫Opencv灰度化？灰度是一种图像亮度的表示方法，将彩色图像转化成为灰度图像的过程称为图像的灰度化处理。
  * 在RGB模型中，如果R=G=B时，则彩色表示一种灰度颜色，其中R=G=B的值叫灰度值，因此，灰度图像每个像素只需一个字节存放灰度值（又称强度值、亮度值），灰度范围为0-255。
  * 1 灰度化的方法
  *     1.1 分量法
  *     1.2 最大值法
  *     1.3 平均值法
  *     1.4 加权平均法
 * @param argc 参数个数
 * @param argv 参数是参数
 * @return
 */
int main(int argc, const char *argv[]) {
    // 分量法
    // 将彩色图像中的三分量的亮度作为三个灰度图像的灰度值，可根据应用需要选取一种灰度图像。
    // f1(i,j)=R(i,j)f2(i,j)=G(i,j)f3(i,j)=B(i,j)
    // 其中fk(i,j)(k=1,2,3)为转换后的灰度图像在（i,j）处的灰度值。


    // 最大值法
    // 将彩色图像中的三分量亮度的最大值作为灰度图的灰度值。
    // f(i,j)=max(R(i,j),G(i,j),B(i,j))

    // 平均值法
    // 将彩色图像中的三分量亮度求平均得到一个灰度值。

    // 加权平均法：
    // 根据重要性及其它指标，将三个分量以不同的权值进行加权平均。
    // 由于人眼对绿色的敏感最高，对蓝色敏感最低，因此，按下式对RGB三分量进行加权平均能得到较合理的灰度图像。
    // f(i,j)=0.30R(i,j)+0.59G(i,j)+0.11B(i,j)
    // 通过下式计算：
    // Y = 0.2126 R + 0.7152 G + 0.0722 B
    // 需要注意的是：OpenCV中的像素排列的是 BGR
    // 下面的方法是加权平均法
    // OpenCV的imread方法
    cv::Mat img = cv::imread("/Users/frewen/03.ProgramStudy/20.AI/01.WorkSpace/NyxAILearning/NyxOpenCV/OpenCVCXX/image1.jpg",
                             cv::IMREAD_COLOR);

    int width = img.rows;
    int height = img.cols;

    // 使用opencv的zeros生成一个全0的Mat对象
    cv::Mat out = cv::Mat::zeros(height, width, CV_8UC1);

    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < height; ++j) {
            out.at<uchar>(i, j) = (int) (img.at<cv::Vec3b>(i, j)[0] * 0.0722
                                         + img.at<cv::Vec3b>(i, j)[1] * 0.7152
                                         + img.at<cv::Vec3b>(i, j)[2] * 0.2126);
        }
    }
    //cv::imwrite("out.jpg", out);
    cv::imshow("result", out);
    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}