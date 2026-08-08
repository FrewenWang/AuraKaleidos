#include "Helpers.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <stdexcept>

namespace alice::inference {

std::vector<float> load_image_nchw(
    const std::string& filename,
    std::int64_t channels,
    std::int64_t height,
    std::int64_t width) {
    if (channels != 1 && channels != 3) {
        throw std::invalid_argument("Only one-channel and three-channel inputs are supported");
    }
    if (height <= 0 || width <= 0) {
        throw std::invalid_argument("The model must have static positive image dimensions");
    }

    cv::Mat image = cv::imread(filename, cv::IMREAD_COLOR);
    if (image.empty()) {
        throw std::runtime_error("Unable to load image: " + filename);
    }

    cv::resize(image, image, cv::Size(static_cast<int>(width), static_cast<int>(height)));
    if (channels == 1) {
        cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
    } else {
        cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    }
    image.convertTo(image, CV_32F, 1.0 / 255.0);

    const std::size_t plane_size = static_cast<std::size_t>(height * width);
    std::vector<float> output(static_cast<std::size_t>(channels) * plane_size);
    if (channels == 1) {
        std::copy_n(image.ptr<float>(), plane_size, output.begin());
        return output;
    }

    std::vector<cv::Mat> planes;
    cv::split(image, planes);
    for (std::int64_t channel = 0; channel < channels; ++channel) {
        std::copy_n(
            planes[static_cast<std::size_t>(channel)].ptr<float>(),
            plane_size,
            output.begin() + channel * static_cast<std::int64_t>(plane_size));
    }
    return output;
}

}  // namespace alice::inference
