
#include "vision/core/request/FaceRequest.h"
#include "vision/config/runtime_config/RtConfig.h"

namespace aura::vision {

FaceRequest::FaceRequest(RtConfig* cfg)
    : AbsVisionRequest(),
      faceCount(static_cast<int>(cfg->faceMaxCount)), landmark2d106(nullptr) {
}

FaceRequest::~FaceRequest() { landmark2d106 = nullptr;
}

short FaceRequest::tag() const {
    return TAG;
}

void FaceRequest::clear() {
    AbsVisionRequest::clear();
    landmark2d106 = nullptr;
}

void FaceRequest::clearAll() {
    clear();
}

} // namespace aura::vision