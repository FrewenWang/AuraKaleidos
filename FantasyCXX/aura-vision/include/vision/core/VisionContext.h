//
// Created by sunwenming01 on 22-8-23.
//

#pragma once

#include "vision/config/runtime_config/RtConfig.h"

namespace aura::vision {

class VisionContext {

public:
    static RtConfig *getRtConfig(const int &sourceId);
};

}
