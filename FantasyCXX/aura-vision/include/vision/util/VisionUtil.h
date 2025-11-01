#ifndef VISION_VISION_UTIL_H
#define VISION_VISION_UTIL_H

#include "vision/core/common/VMacro.h"
#include "vision/core/common/VStructs.h"

namespace aura::vision {

class VA_PUBLIC VisionUtil {
public:
    static void yuv_to_rgb(const void* src, void* dst, int w, int h);
    static void save_image(const char* path, const void* frame, int w, int h);

    static void euler_to_rotation_matrix(float yaw, float pitch, float roll, float *vector_angle);
    static void face_angle_optimal(const VAngle& angle_a, const VAngle& angle_b, std::vector<float>& output);
};

} // namespace vision

#endif //VISION_VISION_UTIL_H
