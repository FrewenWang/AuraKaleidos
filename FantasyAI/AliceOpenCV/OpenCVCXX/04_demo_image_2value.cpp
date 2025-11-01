//
// Created by on 2021/7/31.
// 代码参考：https://geek-docs.com/opencv/opencv-examples/huiduhua.html
//

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

/**
 *
 *
 * @param argc 参数个数
 * @param argv 参数是参数
 * @return
 */
int main(int argc, const char *argv[]) {
    // 读取一张采色图像
    cv::Mat img = cv::imread("", cv::IMREAD_COLOR);


    int width = img.rows;
    int height = img.cols;

    int th = 128;

    cv::Mat out = cv::Mat::zeros(width, height, CV_8UC1)

    return 0;
}