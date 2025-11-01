
#include "vision/core/request/GestureRequest.h"
#include "vision/config/runtime_config/RtConfig.h"

namespace aura::vision {

GestureRequest::GestureRequest(RtConfig* cfg)
    : AbsVisionRequest(),
      gestureCount(V_TO_INT(cfg->gestureMaxCount)){
}

GestureRequest::~GestureRequest() = default;

short GestureRequest::tag() const {
    return TAG;
}

void GestureRequest::clear() {
    AbsVisionRequest::clear();
}

void GestureRequest::clearAll() {
    clear();
}

} // namespace aura::vision