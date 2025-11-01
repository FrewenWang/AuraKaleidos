
#pragma once

#include "AbsDetector.h"

namespace aura::vision {

class AbsBodyDetector : public AbsDetector<BodyInfo> {
public:
    int doDetect(VisionRequest *request, VisionResult *result) override;
};

} // namespace vision
