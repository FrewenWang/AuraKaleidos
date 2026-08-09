#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: opencv_binary_threshold <input> <output>\n";
        return 2;
    }
    const cv::Mat gray = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    if (gray.empty()) {
        std::cerr << "Unable to load image: " << argv[1] << '\n';
        return 1;
    }
    cv::Mat binary;
    cv::threshold(gray, binary, 128, 255, cv::THRESH_BINARY);
    return cv::imwrite(argv[2], binary) ? 0 : 1;
}
