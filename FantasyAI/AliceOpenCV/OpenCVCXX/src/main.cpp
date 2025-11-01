#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include "utils/image_util.h"

using namespace std;
using namespace cv;
using namespace vision;

/**
 *
 * 代码参考：https://geek-docs.com/opencv/opencv-examples/opencv-examples.html
 * @return
 */
int main() {
    cv::Mat img = cv::imread(
            "/Users/frewen/03.ProgramStudy/20.AI/01.WorkSpace/NyxAILearning/NyxOpenCV/OpenCVCXX/image1.jpg",
            cv::IMREAD_COLOR);
    if (img.empty()) {
        printf("Image not loaded");
        return -1;
    }
    //imshow("image", img);
    //waitKey(0);
    //return 0;

    /**
     * 将图片按照固定大小网格分割，网格内的像素值取网格内所有像素的平均值。我们将这种把图片使用均等大小网格分割，
     * 并求网格内代表值的操作称为池化（Pooling）。池化操作是卷积神经网络（Convolutional Neural Network）中重要的图像处理方式。
     * 平均池化按照下式定义：
     */
    cv::Mat averagePoolingOut = ImageUtil::average_pooling(img);
    /**
     *
     */
    cv::Mat gaussianFilterOut = ImageUtil::gaussian_filter(img);

    imshow("image", averagePoolingOut);
    waitKey(0);
    return 0;
}
