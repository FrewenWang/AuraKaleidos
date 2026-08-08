#include "utils/image_util.h"

#include <opencv2/core.hpp>

#include <cassert>
#include <stdexcept>

int main() {
    const cv::Mat constant_image(10, 13, CV_8UC3, cv::Scalar(24, 48, 96));

    const cv::Mat pooled = vision::ImageUtil::average_pooling(constant_image);
    assert(pooled.size() == constant_image.size());
    assert(pooled.at<cv::Vec3b>(9, 12) == cv::Vec3b(24, 48, 96));

    const cv::Mat filtered = vision::ImageUtil::gaussian_filter(constant_image);
    assert(filtered.size() == constant_image.size());
    assert(filtered.at<cv::Vec3b>(5, 5)[0] > 20);

    bool rejected = false;
    try {
        vision::ImageUtil::average_pooling(cv::Mat());
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    assert(rejected);
    return 0;
}
