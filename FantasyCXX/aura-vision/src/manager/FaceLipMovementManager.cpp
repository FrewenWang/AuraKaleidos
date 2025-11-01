#include "FaceLipMovementManager.h"

#include "util/SystemClock.h"
#include "vision/manager/VisionManagerRegistry.h"
#include "vision/util/MouthUtil.h"
#include <cmath>

namespace aura::vision {

static const char *TAG = "FaceLipMovementManager";

FaceLipMovementStrategy::FaceLipMovementStrategy(RtConfig *cfg) {
    this->rtConfig = cfg;
    init();
}

FaceLipMovementStrategy::~FaceLipMovementStrategy() { clear(); }

void FaceLipMovementStrategy::clear() {
    lastLipDistance = DEFAULT_LAST_LIP_DISTANCE;
}

void FaceLipMovementStrategy::checkResetState(int windowTime) {
    curTime = SystemClock::nowMillis();
    durTime = curTime - lastChangeTime;
    if (durTime >= windowTime) {
        lastChangeTime = curTime;
        lastLipDistance = DEFAULT_LAST_LIP_DISTANCE;
        lastSingleLip = false;
        change = 0.0f;
        singleLip = false;
        curTime = 0;
        durTime = 0;
    }
}

void FaceLipMovementStrategy::execute(FaceInfo *face) {
    int windowTime = (int) rtConfig->lipMovementWindowTime;
    windowLen = windowTime / 100;

    if (face->stateMouthCoverSingle == FaceQualityStatus::F_QUALITY_COVER_MOUTH_HIGH) {
        checkResetState(windowTime);
        VLOGW(TAG, "lip movement execute failed with mouthCover=%d", face->stateMouthCoverSingle);
        return;
    }

    // 上下唇距离取100（上嘴唇下）和103（下嘴唇上）之间的距离
    float lipsDistanceMouth = MouthUtil::getLipDistanceWithMouthLandmark(face);
    float lipsDistanceLandmark = MouthUtil::getLipDistance(face);
    float distance = std::max(lipsDistanceMouth, lipsDistanceLandmark);
    // 缩放到指定尺寸的唇部张开幅度
    float lipRefRect = MouthUtil::getLipDistanceRefRect(distance, std::abs(face->rectLT.y - face->rectRB.y));

    LipValues values(lipRefRect, face->optimizedHeadDeflection.yaw, face->optimizedHeadDeflection.pitch);
    lipDeque.push_back(values);
    while (lipDeque.size() > windowLen) {
        lipDeque.pop_front();
    }

    lipDequeLen = (short) lipDeque.size();
    for (int i = 0; i < lipDequeLen; i++) {
        if (i == 0) {
            maxLip = lipDeque[0];
            minLip = lipDeque[0];
        }
        if (lipDeque[i].lipDistance > maxLip.lipDistance) {
            maxLip = lipDeque[i];
        }
        if (lipDeque[i].lipDistance < minLip.lipDistance) {
            minLip = lipDeque[i];
        }
    }
    VLOGD(TAG, "max %s  min %s size[%d]", maxLip.toString().c_str(), minLip.toString().c_str(), lipDeque.size());

    lipThreshold = 10;
    postChange = std::max(std::abs(maxLip.yaw - minLip.yaw), std::abs(maxLip.pitch - minLip.pitch));
    if (postChange > 5 or (maxLip.yaw > 37 and postChange < 5)) {
        lipThreshold = 12;
    }
    if (postChange > 10 or (maxLip.yaw > 37 and postChange > 5) or maxLip.yaw > 40) {
        lipThreshold = 15;
    }
    if (postChange > 15) {
        lipThreshold = 18;
    }
    if (maxLip.yaw > 37 and postChange > 8) {
        lipThreshold = 20;
    }
    if (maxLip.yaw > 40 and postChange > 5) {
        lipThreshold = 20;
    }
    if (maxLip.yaw > 50 or postChange > 20) {
        lipThreshold = 25;
    }
    if (postChange > 30) {
        lipThreshold = 30;
    }

    // 刚开始检测的第一帧，或者丢失人脸后重新开始检测的第一帧不进行唇动判断，默认单帧结果为NONE
    singleLip = false;
    change = 0.0f;
    if (lipDeque.size() > 1) {
        change = static_cast<float>(abs(maxLip.lipDistance - minLip.lipDistance));
        singleLip = change >= lipThreshold;
        face->stateLipMovementSingle = singleLip ? FaceLipMovementStatus::F_LIP_MOVING
                                                   : FaceLipMovementStatus::F_LIP_MOVEMENT_NONE;
    } else {
        face->stateLipMovementSingle = FaceLipMovementStatus::F_LIP_MOVEMENT_NONE;
    }

    curTime = SystemClock::nowMillis();
    durTime = curTime - lastChangeTime;
    if ((durTime < windowTime && singleLip) || durTime >= windowTime) {
        lastSingleLip = singleLip;
        lastChangeTime = curTime;
    }

    face->stateLipMovement = lastSingleLip ? FaceLipMovementStatus::F_LIP_MOVING
                                           : FaceLipMovementStatus::F_LIP_MOVEMENT_NONE;

    VLOGI(TAG, "lip_movement[%ld], d=[%f] lipRefRect=[%f], change=[%f], postChange[%f], threshold=[%f], lip=[%d], "
               "single=[%d], windowTime=[%d]", face->id, distance, lipRefRect, change, postChange, lipThreshold,
          face->stateLipMovement, face->stateLipMovementSingle, windowTime);
}

void FaceLipMovementStrategy::onNoFace(VisionRequest *request, FaceInfo *face) {
    // 丢失人脸在时间窗口内保持丢失前的唇动状态，超过时间窗口后将唇动置为NONE，与有人脸时的策略保持一致
    int windowTime = (int) rtConfig->lipMovementWindowTime;
    checkResetState(windowTime);

    face->stateLipMovementSingle = FaceLipMovementStatus::F_LIP_MOVEMENT_NONE;
    face->stateLipMovement = lastSingleLip ? FaceLipMovementStatus::F_LIP_MOVING
                                             : FaceLipMovementStatus::F_LIP_MOVEMENT_NONE;
    VLOGI(TAG, "[when no face] lip_movement[%ld], lipsDistance=[%f] lipRefRect=[%f], change=[%f], threshold=[%f], "
               "lip=[%d], single=[%d], windowTime=[%d]",
          face->id, 0.f, 0.f, 0.f, lipThreshold, face->stateLipMovement, face->stateLipMovementSingle, windowTime);
}

void FaceLipMovementStrategy::init() {
    if (rtConfig->sourceId == Source::SOURCE_1) {
        lipThreshold = MouthUtil::THRESHOLD_LIP_MOVEMENT_DMS;
    } else if (rtConfig->sourceId == Source::SOURCE_2) {
        lipThreshold = MouthUtil::THRESHOLD_LIP_MOVEMENT_OMS;
    } else {
        lipThreshold = MouthUtil::THRESHOLD_LIP_MOVEMENT_DEFAULT;
    }
}

FaceLipMovementManager::FaceLipMovementManager() {

}

bool FaceLipMovementManager::preDetect(VisionRequest *request, VisionResult *result) {
    bool checkLandmarkDetect = VA_GET_DETECTED(ABILITY_FACE_LANDMARK);
    V_CHECK_COND(!checkLandmarkDetect, Error::PREPARE_ERR, "LipMovement would be scheduled after LandmarkManager");
    VA_CHECK_DETECTED(ABILITY_FACE_LIP_MOVEMENT);
}

void FaceLipMovementManager::doDetect(VisionRequest *request, VisionResult *result) {
    VA_SET_DETECTED(ABILITY_FACE_LIP_MOVEMENT);
    // 执行多人脸策略
    execute_face_strategy<FaceLipMovementStrategy>(result, faceLipMovementMap, mRtConfig);
}

void FaceLipMovementManager::onNoFace(VisionRequest *request, VisionResult *result, FaceInfo *face) {
    if (face != nullptr && (!faceLipMovementMap.empty())) {
        auto iter = faceLipMovementMap.find(face->id);
        if (iter != faceLipMovementMap.end()) {
            iter->second->onNoFace(request, face);
        }
    }
}

void FaceLipMovementManager::clear() {
    for (auto &info: faceLipMovementMap) {
        if (info.second) {
            info.second->clear();
            FaceLipMovementStrategy::recycle(info.second);
        }
    }
    faceLipMovementMap.clear();
}

void FaceLipMovementManager::deinit() {
    AbsVisionManager::deinit();
}

FaceLipMovementManager::~FaceLipMovementManager() { clear(); }

// NOLINTNEXTLINE
REGISTER_VISION_MANAGER("FaceLipMovementManager", ABILITY_FACE_LIP_MOVEMENT, []() {
    return std::dynamic_pointer_cast<AbsVisionManager>(std::make_shared<FaceLipMovementManager>());
});

} // namespace vision