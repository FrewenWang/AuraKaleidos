
#pragma once

#include "AbsDetector.h"

namespace aura::vision {

class AbsGestureDetector : public AbsDetector<GestureInfo> {
public:
    int doDetect(VisionRequest *request, VisionResult *result) override;
};

} // namespace vision
