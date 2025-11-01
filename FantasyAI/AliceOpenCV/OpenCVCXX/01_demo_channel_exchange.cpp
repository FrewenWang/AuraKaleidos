//
// Created by on 2021/7/31.
// 代码参考：https://geek-docs.com/opencv/opencv-examples/channel-exchange.html
//

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

/**
 * @param argc 参数个数
 * @param argv 参数是参数
 * @return
 */
int main(int argc, const char *argv[]) {

    // 使用OpenCV读取一张图片
    // 这个地方需要使用绝对路径。否则会报下面的错误：
    // libc++abi: terminating with uncaught exception of type cv::Exception:
    // OpenCV(4.5.3) /Users/frewen/03.ProgramStudy/15.CLang/04.Resources/01.OpenCV/opencv-4.5.3/modules/highgui/src/window.cpp:1006:
    // error: (-215:Assertion failed) size.width>0 && size.height>0 in function 'imshow'
    cv::Mat image = cv::imread("/Users/frewen/03.ProgramStudy/20.AI/01.WorkSpace/NyxAILearning/NyxOpenCV/OpenCVCXX/image1.jpg",
                               cv::IMREAD_COLOR);
    int width = image.rows;
    int height = image.cols;
    cv::Mat out = image.clone();

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            unsigned char tmp = out.at<cv::Vec3b>(i, j)[0];
            out.at<cv::Vec3b>(i, j)[0] = image.at<cv::Vec3b>(i, j)[2];
            out.at<cv::Vec3b>(i, j)[2] = tmp;
        }
    }

    // cv::imwrite("out.jpg", out);
    cv::imshow("image_out", out);
    cv::waitKey(0);
    cv::destroyAllWindows();
}