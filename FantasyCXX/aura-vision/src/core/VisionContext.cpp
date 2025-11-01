#include "vision/core/VisionContext.h"

#include <detector/AbsBodyDetector.h>

#include "config/runtime_config/RtConfigSource1.h"
#include "config/runtime_config/RtConfigSource2.h"
#include "vision/core/common/VConstants.h"

namespace aura::vision {

vision::RtConfig *VisionContext::getRtConfig(const int &sourceId) {
    switch (sourceId) {
        case SOURCE_1:
            return new RtConfigSource1();
        case SOURCE_2:
            return new RtConfigSource2();
        default:
            return new RtConfig();
    }
}

} // namespace aura::vision
