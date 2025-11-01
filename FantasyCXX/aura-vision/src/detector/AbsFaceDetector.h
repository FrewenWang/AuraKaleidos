
#pragma once

#include "AbsDetector.h"

namespace aura::vision {

class AbsFaceDetector : public AbsDetector<FaceInfo> {
public:
    int doDetect(VisionRequest *request, VisionResult *result) override;
};

} // namespace vision
