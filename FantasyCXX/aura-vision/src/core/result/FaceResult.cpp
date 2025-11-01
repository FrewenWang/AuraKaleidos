
#include "vision/core/result/FaceResult.h"

namespace aura::vision {

FaceResult::FaceResult(RtConfig* cfg) : AbsVisionResult() {
    faceMaxCount = static_cast<int>(cfg->faceMaxCount);
    faceInfos = new FaceInfo *[faceMaxCount];
    useInternalMem = V_F_TO_BOOL(cfg->useInternalMem);
    if (useInternalMem) {
        for (int i = 0; i < faceMaxCount; i++) {
            faceInfos[i] = new FaceInfo();
        }
    } else {
        for (int i = 0; i < faceMaxCount; i++) {
            faceInfos[i] = nullptr;
        }
    }
}

FaceResult::~FaceResult() {
    if (useInternalMem) {
        for (int i = 0; i < faceMaxCount; i++) {
            delete faceInfos[i];
            faceInfos[i] = nullptr;
        }
    }
    delete[] faceInfos;
    faceInfos = nullptr;
}

void FaceResult::resize(int count) {
    if (faceMaxCount == count) {
        return;
    }
    if (useInternalMem) {
        for (int i = 0; i < faceMaxCount; i++) {
            delete faceInfos[i];
            faceInfos[i] = nullptr;
        }
    }
    delete[] faceInfos;
    faceInfos = nullptr;

    faceMaxCount = count;
    faceInfos = new FaceInfo* [faceMaxCount];
    if (useInternalMem) {
        for (int i = 0; i < faceMaxCount; i++) {
            faceInfos[i] = new FaceInfo();
        }
    } else {
        for (int i = 0; i < faceMaxCount; i++) {
            faceInfos[i] = nullptr;
        }
    }
}

// 注销过时方法
//bool FaceResult::faceOccluded() const {
//    int state = static_cast<int>(faceInfos[0]->stateFaceCoverSingle);
//    return state == F_QUALITY_COVER_NORMAL;
//}
//
//bool FaceResult::faceLive() const {
//    int state = static_cast<int>(faceInfos[0]->stateNoInteractLiving);
//    return state == F_NO_INTERACT_LIVING_LIVING;
//}

bool FaceResult::noFace() const {
    for (int i = 0; i < faceMaxCount; i++) {
        if (faceInfos[i]->hasFace()) {
            return false;
        }
    }
    return true;
}

bool FaceResult::hasFace() const {
    return !noFace();
}

short FaceResult::faceCount() {
    short count = 0;
    for (int i = 0; i < faceMaxCount; i++) {
        if (faceInfos[i]->hasFace()) {
            count++;
        }
    }
    return count;
}

short FaceResult::tag() const {
    return TAG;
}

void FaceResult::clear() {
    AbsVisionResult::clear();

    // 单人脸模式下（当前帧检测到的人脸数==1），清空状态和属性数据，保留人脸id和人脸框信息，用于下一帧跟踪
    // 多人脸模式下（当前帧检测到的人脸数>1），跟踪策略不适应（会引入bug），因此需要全部清空，包括face id和人脸框
    // @wangyan：更优雅的设计或兼容多人脸的跟踪策略
    // add by wangzhijiang 目前重新设计设计多人脸的跟踪策略。故不再根据是否检测到多个人，来选择不同的清空方式
    // if (faceInfos != nullptr) {
    //     FaceInfo *info = nullptr;
    //     int face_count_detected = 0;
    //     for (int i = 0; i < faceMaxCount; ++i) {
    //         info = faceInfos[i];
    //         if (info != nullptr && info->id > 0) {
    //             face_count_detected += 1;
    //         }
    //     }
    //
    //     for (int i = 0; i < faceMaxCount; i++) {
    //         info = faceInfos[i];
    //         if (info != nullptr) {
    //             if (face_count_detected == 1) {
    //                 info->clear();
    //             } else {
    //                 info->clearAll();
    //             }
    //         }
    //     }
    // }
    FaceInfo *info = nullptr;
    for (int i = 0; i < faceMaxCount; i++) {
        info = faceInfos[i];
        if (info != nullptr) {
            info->clear();
        }
    }

}

void FaceResult::clearAll() {
    AbsVisionResult::clear();
    if (faceInfos != nullptr) {
        FaceInfo *info = nullptr;
        for (int i = 0; i < faceMaxCount; i++) {
            info = faceInfos[i];
            if (info != nullptr) {
                info->clearAll();
            }
        }
    }
}

void FaceResult::copy(AbsVisionResult *src) {
    auto* fr = (FaceResult *) src;
    for (int i = 0; i < faceMaxCount; i++) {
        faceInfos[i]->copy(*(fr->faceInfos[i]));
    }
}

void FaceResult::toString(std::stringstream &ss) const {
    ss << "\n[FaceResult] ========================================\n";
    for (int i = 0; i < faceMaxCount; i++) {
        FaceInfo *info = faceInfos[i];
        if (info == nullptr || !info->hasFace()) {
            continue;
        }
        info->toString(ss);
    }
}

} // namespace aura::vision
