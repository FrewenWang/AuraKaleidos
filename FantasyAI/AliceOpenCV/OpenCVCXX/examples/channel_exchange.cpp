#include <opencv2/imgcodecs.hpp>

#include <iostream>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: opencv_channel_exchange <input> <output>\n";
        return 2;
    }
    cv::Mat image = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Unable to load image: " << argv[1] << '\n';
        return 1;
    }
    for (auto iterator = image.begin<cv::Vec3b>(); iterator != image.end<cv::Vec3b>(); ++iterator) {
        std::swap((*iterator)[0], (*iterator)[2]);
    }
    return cv::imwrite(argv[2], image) ? 0 : 1;
}
