//
// Created by LiWendong on 23-3-15.
//

#pragma once

#include <opencv2/opencv.hpp>

namespace aura::vision {
namespace op {

class WarpAffine {
public:
    static void warpAffine(cv::Mat &src, cv::Mat &dst, cv::Mat &M, cv::Size &dsize,
                           const int &flags = cv::INTER_LINEAR, const int &borderMode = cv::BORDER_CONSTANT,
                           const cv::Scalar &borderValue = cv::Scalar());
};

} // namespace op
} // namespace aura::vision