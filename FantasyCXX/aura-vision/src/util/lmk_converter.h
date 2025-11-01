#ifndef VISION_LANDMARK_CONVERTER_H
#define VISION_LANDMARK_CONVERTER_H

#include "vision/core/bean/FaceInfo.h"

namespace aura::vision {

class LmkConverter {
public:
    static void get_68_point(FaceInfo* face);
};

} // namespace aura::vision

#endif //VISION_LANDMARK_CONVERTER_H
