
#include "vision/core/result/GestureResult.h"

namespace aura::vision {

GestureResult::GestureResult(RtConfig* cfg) : AbsVisionResult() {
    gestureMaxCount = static_cast<int>(cfg->gestureMaxCount);
    gestureInfos = new GestureInfo *[gestureMaxCount];
    useInternalMem = V_F_TO_BOOL(cfg->useInternalMem);
    if (useInternalMem) {
        for (int i = 0; i < gestureMaxCount; i++) {
            gestureInfos[i] = new GestureInfo();
        }
    } else {
        for (int i = 0; i < gestureMaxCount; i++) {
            gestureInfos[i] = nullptr;
        }
    }
}

GestureResult::~GestureResult() {
    if (useInternalMem) {
        for (int i = 0; i < gestureMaxCount; i++) {
            delete gestureInfos[i];
            gestureInfos[i] = nullptr;
        }
    }
    delete[] gestureInfos;
    gestureInfos = nullptr;
}

void GestureResult::resize(int count) {
    if (gestureMaxCount == count) {
        return;
    }
    if (useInternalMem) {
        for (int i = 0; i < gestureMaxCount; i++) {
            delete gestureInfos[i];
            gestureInfos[i] = nullptr;
        }
    }
    delete[] gestureInfos;
    gestureInfos = nullptr;

    gestureMaxCount = count;
    gestureInfos = new GestureInfo* [gestureMaxCount];
    if (useInternalMem) {
        for (int i = 0; i < gestureMaxCount; i++) {
            gestureInfos[i] = new GestureInfo();
        }
    } else {
        for (int i = 0; i < gestureMaxCount; i++) {
            gestureInfos[i] = nullptr;
        }
    }
}

bool GestureResult::noGesture() const {
    for (int i = 0; i < gestureMaxCount; i++) {
        if (gestureInfos[i]->hasGesture()) {
            return false;
        }
    }
    return true;
}

short GestureResult::gestureCount() {
    short count = 0;
    for (int i = 0; i < gestureMaxCount; i++) {
        if (gestureInfos[i]->hasGesture()) {
            count++;
        }
    }
    return count;
}

short GestureResult::tag() const {
    return TAG;
}

void GestureResult::clear() {
    AbsVisionResult::clear();
    if (gestureInfos != nullptr) {
        GestureInfo *info = nullptr;
        for (int i = 0; i < gestureMaxCount; i++) {
            info = gestureInfos[i];
            if (info != nullptr) {
                info->clear();
            }
        }
    }
}

void GestureResult::clearAll() {
    AbsVisionResult::clear();
    if (gestureInfos != nullptr) {
        GestureInfo *info = nullptr;
        for (int i = 0; i < gestureMaxCount; i++) {
            info = gestureInfos[i];
            if (info != nullptr) {
                info->clear_all();
            }
        }
    }
}

void GestureResult::toString(std::stringstream &ss) const {
    ss << "\n[GestureResult] ========================================\n";
    for (int i = 0; i < gestureMaxCount; i++) {
        GestureInfo *info = gestureInfos[i];
        if (info == nullptr || !info->hasGesture()) {
            continue;
        }
        info->toString(ss);
    }
}

} // namespace aura::vision
