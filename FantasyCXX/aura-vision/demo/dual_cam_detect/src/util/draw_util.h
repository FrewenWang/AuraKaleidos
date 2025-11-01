#pragma once

#include <string>

#include "opencv2/opencv.hpp"
#include "vision/VisionAbility.h"

using namespace vision;
class DrawUtil {
public:
    static cv::Mat& drawline(cv::Mat &image, const GestureInfo *gesture, const cv::Scalar &rect_scalar, int offset);
    static void draw(cv::Mat &image, VisionResult* result, int id);
};