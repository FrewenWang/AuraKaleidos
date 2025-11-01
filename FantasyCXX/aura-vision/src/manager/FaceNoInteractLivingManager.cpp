
#include <algorithm>
#include <set>
#include "FaceNoInteractLivingManager.h"
#include "detector/FaceNoInteractLivingDetector.h"
#include "vision/manager/VisionManagerRegistry.h"

namespace aura::vision {

static const char *TAG = "FaceNoInteractLivingManager";

/**
 * @brief 无感活体滑窗策略初始化。默认情况下，无感活体检测为固定滑窗策略。5帧里面有3帧为活体，则满足活体要求。
 * @param cfg
 */
FaceNoInteractLivingStrategy::FaceNoInteractLivingStrategy(RtConfig *cfg)
    : noInteractLivingWindow(cfg->sourceId, cfg->faceNoInteractLiveWindowLen, cfg->faceNoInteractDefaultDutyFactor,
                           AbsVisionManager::DEFAULT_END_DUTY_FACTOR, F_NO_INTERACT_LIVING_LIVING) {
    this->rtConfig = cfg;
    setup_sliding_window();
}

FaceNoInteractLivingStrategy::~FaceNoInteractLivingStrategy() {}

void FaceNoInteractLivingStrategy::setup_sliding_window() {
    noInteractLivingWindow.set_window_adjust_strategy(WindowAdjustStrategy::Fixed);
}

void FaceNoInteractLivingStrategy::execute(FaceInfo *face) {
    // 获取当前人脸的无感活体状态的滑窗结果.
    // 之前的逻辑：只有当无感活体滑窗结果为通过，且当前帧的无感活体单帧结果为通过，才通过。
    // 新修改逻辑：去掉判断单帧无感活体结果判断，能力层不要侵入上层业务需求。
    // 在FaceId的业务需求中，可以在业务层在自己的策略中增加单帧结果判断（因为FaceId必须当前单帧为通过，才算完全通过）
    if (noInteractLivingWindow.update(face->stateNoInteractLivingSingle, nullptr)) {
        face->stateNoInteractLiving = F_NO_INTERACT_LIVING_LIVING;
    } else {
        face->stateNoInteractLiving = F_NO_INTERACT_LIVING_ATTACK;
    }
    VLOGI(TAG, "face_liveness[%ld], mode=[%s], stateSingle=%s, state=%f, score=%f", face->id,
          static_cast<int>(rtConfig->cameraLightType) == CAMERA_LIGHT_TYPE_IR ? "IR" : "RGB",
          face->stateNoInteractLivingSingle == F_NO_INTERACT_LIVING_LIVING ? "Live" : "Attack",
          face->stateNoInteractLiving, face->scoreNoInteractLiving);
}

void FaceNoInteractLivingStrategy::onNoFace(VisionRequest *request, FaceInfo *face) {
    // 无感活体单帧状态为F_NO_INTERACT_LIVING_UNKNOWN，此时仅更新滑窗，无感活体状态为攻击
    if (noInteractLivingWindow.update(face->stateNoInteractLivingSingle, nullptr)) {
        face->stateNoInteractLiving = F_NO_INTERACT_LIVING_LIVING;
    } else {
        face->stateNoInteractLiving = F_NO_INTERACT_LIVING_ATTACK;
    }
    VLOGI(TAG, "[when no face] face_liveness[%ld], mode=[%s], stateSingle=%s, state=%f, score=%f", face->id,
          static_cast<int>(rtConfig->cameraLightType) == CAMERA_LIGHT_TYPE_IR ? "IR" : "RGB",
          face->stateNoInteractLivingSingle == F_NO_INTERACT_LIVING_LIVING ? "Live" : "Attack",
          face->stateNoInteractLiving, face->scoreNoInteractLiving);
}

void FaceNoInteractLivingStrategy::clear() {
    noInteractLivingWindow.clear();
}

FaceNoInteractLivingManager::FaceNoInteractLivingManager() {
    detector = std::make_shared<FaceLivenessDetector>();
}

FaceNoInteractLivingManager::~FaceNoInteractLivingManager() {
    clear();
}

void FaceNoInteractLivingManager::init(RtConfig *cfg) {
    mRtConfig = cfg;
    detector->init(cfg);
}

void FaceNoInteractLivingManager::deinit() {
    AbsVisionManager::deinit();
    if (detector != nullptr) {
        detector->deinit();
        detector = nullptr;
    }
}

void FaceNoInteractLivingManager::clear() {
    for (auto &info : noInteractLivingMap) {
        if (info.second) {
            info.second->clear();
            FaceNoInteractLivingStrategy::recycle(info.second);
        }
    }
    noInteractLivingMap.clear();
}

bool FaceNoInteractLivingManager::preDetect(VisionRequest *request, VisionResult *result) {
    VA_CHECK_DETECTED(ABILITY_FACE_NO_INTERACTIVE_LIVING);
}

void FaceNoInteractLivingManager::doDetect(VisionRequest *request, VisionResult *result) {
    VA_SET_DETECTED(ABILITY_FACE_NO_INTERACTIVE_LIVING);
    V_CHECK_NULL(result);

    // 无感活体检测
    detector->detect(request, result);

    // 执行多人脸策略
    execute_face_strategy<FaceNoInteractLivingStrategy>(result, noInteractLivingMap, mRtConfig);
}

void FaceNoInteractLivingManager::onNoFace(VisionRequest *request, VisionResult *result, FaceInfo *face) {
    if (face != nullptr && (!noInteractLivingMap.empty())) {
        auto iter = noInteractLivingMap.find(face->id);
        if (iter != noInteractLivingMap.end()) {
            iter->second->onNoFace(request, face);
        }
    }
}

REGISTER_VISION_MANAGER("FaceNoInteractLivingManager", ABILITY_FACE_NO_INTERACTIVE_LIVING, []() {
    return std::dynamic_pointer_cast<AbsVisionManager>(std::make_shared<FaceNoInteractLivingManager>());
});

} // namespace vision
