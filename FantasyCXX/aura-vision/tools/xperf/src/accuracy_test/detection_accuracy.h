#ifndef VISION_ACCURACY_PROFILER_H
#define VISION_ACCURACY_PROFILER_H

#include <string>

namespace xperf {

class DetectionAccuracy {
public:
    static void test_setup();
    static void test_execute(std::string eval_feature, std::string eval_img_path, std::string eval_result_path);
    static void test_cleanup();
};

} // namespace xperf

#endif //VISION_ACCURACY_PROFILER_H
