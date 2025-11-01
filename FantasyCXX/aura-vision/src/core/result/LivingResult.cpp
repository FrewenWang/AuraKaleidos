
#include "vision/core/result/LivingResult.h"

namespace aura::vision {

LivingResult::LivingResult(RtConfig *cfg) : AbsVisionResult() {
    livingCount = static_cast<int>(cfg->livingMaxCount);
    livingInfos = new LivingInfo *[livingCount];
    useInternalMem = V_F_TO_BOOL(cfg->useInternalMem);
    if (useInternalMem) {
        for (int i = 0; i < livingCount; i++) {
            livingInfos[i] = new LivingInfo();
        }
    } else {
        for (int i = 0; i < livingCount; i++) {
            livingInfos[i] = nullptr;
        }
    }
}

LivingResult::~LivingResult() {
    if (useInternalMem) {
        for (int i = 0; i < livingCount; i++) {
            delete livingInfos[i];
            livingInfos[i] = nullptr;
        }
    }
    delete[] livingInfos;
    livingInfos = nullptr;
}

void LivingResult::resize(int count) {
    if (livingCount == count) {
        return;
    }
    if (useInternalMem) {
        for (int i = 0; i < livingCount; i++) {
            delete livingInfos[i];
            livingInfos[i] = nullptr;
        }
    }
    delete[] livingInfos;
    livingInfos = nullptr;

    livingCount = count;
    livingInfos = new LivingInfo *[livingCount];
    if (useInternalMem) {
        for (int i = 0; i < livingCount; i++) {
            livingInfos[i] = new LivingInfo();
        }
    } else {
        for (int i = 0; i < livingCount; i++) {
            livingInfos[i] = nullptr;
        }
    }
}

short LivingResult::tag() const {
    return TAG;
}

void LivingResult::clear() {
    AbsVisionResult::clear();
    if (livingInfos != nullptr) {
        LivingInfo *info = nullptr;
        for (int i = 0; i < livingCount; i++) {
            info = livingInfos[i];
            if (info != nullptr) {
                info->clear();
            }
        }
    }
}

void LivingResult::clearAll() {
    AbsVisionResult::clear();
    if (livingInfos != nullptr) {
        LivingInfo *info = nullptr;
        for (int i = 0; i < livingCount; i++) {
            info = livingInfos[i];
            if (info != nullptr) {
                info->clearAll();
            }
        }
    }
}

void LivingResult::toString(std::stringstream &ss) const {
    ss << "\n[LivingResult] ========================================\n";
    for (int i = 0; i < livingCount; i++) {
        LivingInfo *info = livingInfos[i];
        if (info == nullptr || !info->hasLiving()) {
            continue;
        }
        info->toString(ss);
    }
}

} // namespace aura::vision
