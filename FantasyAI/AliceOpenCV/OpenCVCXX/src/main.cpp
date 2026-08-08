#include <opencv2/imgcodecs.hpp>

#include <iostream>
#include <string>

#include "utils/image_util.h"

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) {
        std::cerr << "Usage: OpenCVDemo <input-image> [output-image]\n";
        return 2;
    }

    const cv::Mat image = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Unable to load image: " << argv[1] << '\n';
        return 1;
    }

    const cv::Mat output = vision::ImageUtil::average_pooling(image);
    const std::string output_path = argc == 3 ? argv[2] : "average_pooling.png";
    if (!cv::imwrite(output_path, output)) {
        std::cerr << "Unable to write image: " << output_path << '\n';
        return 1;
    }
    std::cout << "Wrote " << output_path << '\n';
    return 0;
}
