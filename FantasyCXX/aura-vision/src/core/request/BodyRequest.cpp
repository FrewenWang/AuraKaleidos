
#include "vision/core/request/BodyRequest.h"
#include "vision/config/runtime_config/RtConfig.h"

namespace aura::vision {

BodyRequest::BodyRequest(RtConfig* cfg)
    : AbsVisionRequest(),
      bodyCount(V_TO_INT(cfg->bodyMaxCount)){
}

BodyRequest::~BodyRequest() = default;

short BodyRequest::tag() const {
    return TAG;
}

void BodyRequest::clear() {
    AbsVisionRequest::clear();
}

void BodyRequest::clearAll() {
    clear();
}

} // namespace aura::vision