#include <iostream>
#include <string>
#include "speed_test/detection_speed.h"
#include "accuracy_test/detection_accuracy.h"

using namespace xperf;

int main(int argc, char** argv) {
    std::string img_path;
    if (argc < 3) {
        std::cout << "need at least 2 params" << std::endl;
    }
    std::string test_type(argv[1]);
    if (test_type == "speed") {
        img_path = argv[2];
        DetectionSpeed::test_setup(img_path);
        DetectionSpeed::test_execute();
        DetectionSpeed::test_cleanup();
    } else if (test_type == "accuracy") {
        std::string eval_feature(argv[2]);
        std::string eval_img_path(argv[3]);
        std::string eval_result_path(argv[4]);
        DetectionAccuracy::test_setup();
        DetectionAccuracy::test_execute(eval_feature, eval_img_path, eval_result_path);
        DetectionAccuracy::test_cleanup();
    }


}