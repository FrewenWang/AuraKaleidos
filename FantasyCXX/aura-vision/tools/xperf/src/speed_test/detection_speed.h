#ifndef VISION_DETECTION_SPEED_H
#define VISION_DETECTION_SPEED_H

#include <string>

namespace xperf {

#define DETECT_FUNC(id) {DetectFunctor<id>(), GET_TAG(id)}

class DetectionSpeed {
public:
    static void test_setup(std::string img_path);
    static void test_cleanup();
//    static void profile_setup();
//    static void profile_cleanup();
    static void test_execute();
};

} // namespace xperf

#endif //VISION_DETECTION_SPEED_H
