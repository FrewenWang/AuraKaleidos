#include "vision/core/result/BodyResult.h"

namespace aura::vision {

BodyResult::BodyResult(RtConfig *cfg) : AbsVisionResult() {
    bodyMaxCount = static_cast<int>(cfg->bodyMaxCount);
    pBodyInfos = new BodyInfo *[bodyMaxCount];
    _use_internal_mem = V_F_TO_BOOL(cfg->useInternalMem);
    if (_use_internal_mem) {
        for (int i = 0; i < bodyMaxCount; i++) {
            pBodyInfos[i] = new BodyInfo();
        }
    } else {
        for (int i = 0; i < bodyMaxCount; i++) {
            pBodyInfos[i] = nullptr;
        }
    }
}

BodyResult::~BodyResult() {
    if (_use_internal_mem) {
        for (int i = 0; i < bodyMaxCount; i++) {
            delete pBodyInfos[i];
            pBodyInfos[i] = nullptr;
        }
    }
    delete[] pBodyInfos;
    pBodyInfos = nullptr;
}

void BodyResult::resize(int count) {
    if (bodyMaxCount == count) {
        return;
    }
    if (_use_internal_mem) {
        for (int i = 0; i < bodyMaxCount; i++) {
            delete pBodyInfos[i];
            pBodyInfos[i] = nullptr;
        }
    }
    delete[] pBodyInfos;
    pBodyInfos = nullptr;

    bodyMaxCount = count;
    pBodyInfos = new BodyInfo* [bodyMaxCount];
    if (_use_internal_mem) {
        for (int i = 0; i < bodyMaxCount; i++) {
            pBodyInfos[i] = new BodyInfo();
        }
    } else {
        for (int i = 0; i < bodyMaxCount; i++) {
            pBodyInfos[i] = nullptr;
        }
    }
}

bool BodyResult::noBody() const {
    for (int i = 0; i < bodyMaxCount; i++) {
        if (pBodyInfos[i]->hasBody()) {
            return false;
        }
    }
    return true;
}

bool BodyResult::hasBody() const {
    return !noBody();
}

short BodyResult::tag() const {
    return TAG;
}

void BodyResult::clear() {
    AbsVisionResult::clear();

    if (pBodyInfos != nullptr) {
        BodyInfo *info = nullptr;
        for (int i = 0; i < bodyMaxCount; i++) {
            info = pBodyInfos[i];
            if (info != nullptr) {
                info->clear();
            }
        }
    }
}

void BodyResult::clearAll() {
    AbsVisionResult::clear();
    if (pBodyInfos != nullptr) {
        BodyInfo *info = nullptr;
        for (int i = 0; i < bodyMaxCount; i++) {
            info = pBodyInfos[i];
            if (info != nullptr) {
                info->clearAll();
            }
        }
    }
}

void BodyResult::copy(AbsVisionResult *src) {
    auto* fr = (BodyResult *) src;
    for (int i = 0; i < bodyMaxCount; i++) {
        pBodyInfos[i]->copy(*(fr->pBodyInfos[i]));
    }
}

short BodyResult::bodyCount() const {
    short count = 0;
    for (int i = 0; i < bodyMaxCount; i++) {
        if (pBodyInfos[i]->id > 0) {
            count++;
        }
    }
    return count;
}

void BodyResult::toString(std::stringstream &ss) const {
    ss << "\n[BodyResult] ========================================\n";
    for (int i = 0; i < bodyMaxCount; i++) {
        BodyInfo *info = pBodyInfos[i];
        if (info == nullptr || !info->hasBody()) {
            continue;
        }
        info->toString(ss);
    }
}

} // namespace aura::vision
