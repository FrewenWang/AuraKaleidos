//
// Created by Frewen.Wong on 2021/10/14.
//

#include "image_util.h"

cv::Mat vision::ImageUtil::average_pooling(cv::Mat in) {
    // 计算输入图片资源的行列数
    int width = in.cols;  // 图像的列数为图像的宽度
    int height = in.rows;  // 图像的行数为图像的高度
    // 初始化一个具有指定大小和类型(三通道)的零数组。
    // 一般的图像文件格式使用的是 Unsigned 8bits，cv::Mat矩阵对应的参数类型就是
    // CV_8UC1，CV_8UC2，CV_8UC3。（最后的1、2、3表示通道数，譬如RGB3通道就用CV_8UC3）
    cv::Mat out = cv::Mat::zeros(height, width, CV_8UC3);
    int r = 8;
    double v = 0;

    for (int j = 0; j < height; j += r) {
        for (int i = 0; i < width; i += r) {
            for (int c = 0; c < 3; c++) {
                v = 0;
                for (int _j = 0; _j < r; _j++) {
                    for (int _i = 0; _i < r; _i++) {
                        v += (double) in.at<cv::Vec3b>(j + _j, i + _i)[c];
                    }
                }
                v /= (r * r);
                for (int _j = 0; _j < r; _j++) {
                    for (int _i = 0; _i < r; _i++) {
                        out.at<cv::Vec3b>(j + _j, i + _i)[c] = (uchar) v;
                    }
                }
            }
        }
    }


    return out;
}

cv::Mat vision::ImageUtil::gaussian_filter(cv::Mat in) {
    //  获取输入图像尺寸的宽度和高度
    int width = in.cols;
    int height = in.rows;
    // 定义一个 0值 输出的Mat对象
    cv::Mat out = cv::Mat::zeros(height, width, CV_8UC3);

    // 使用高斯滤波器（3×3 大小，标准差 s=1.3）来对图片进行降噪处理
    double s = 1.3;
    int k_size = 3;
    int p = floor(k_size / 2);
    int x = 0, y = 0;
    double k_sum = 0;

    // 定义一个 3*3 的卷积核.并且初始化卷积核里面的数据
    float k[k_size][k_size];
    for (int i = 0; i < k_size; i++) {
        for (int j = 0; j < k_size; j++) {
            x = i - p;
            y = j - p;
            k[i][j] = 1 / (s * sqrt(2 * M_PI)) * exp(-(x * x + y * y) / (2 * s * s));
            k_sum += k[i][j];
        }
    }
    //
    for (int j = 0; j < k_size; j++) {
        for (int i = 0; i < k_size; i++) {
            k[j][i] /= k_sum;
        }
    }

    // filtering
    double v = 0;

    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            for (int c = 0; c < 3; c++) {
                v = 0;
                for (int _j = -p; _j < p + 1; _j++) {
                    for (int _i = -p; _i < p + 1; _i++) {
                        if (((j + _j) >= 0) && ((i + _i) >= 0)) {
                            v += (double) in.at<cv::Vec3b>(j + _j, i + _i)[c] * k[_j + p][_i + p];
                        }
                    }
                }
                out.at<cv::Vec3b>(j, i)[c] = v;
            }
        }
    }

    return out;
}
