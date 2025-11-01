

#include "FaceDangerousDriveManager.h"

#include <algorithm>

#include "detector/FaceDangerousDriveDetector.h"
#include "vision/manager/VisionManagerRegistry.h"

namespace aura::vision {

static const char *TAG = "FaceDangerousDrivingManager";

FaceDangerousDriveStrategy::FaceDangerousDriveStrategy(RtConfig *cfg)
    : _record_state(false),
      smokeWindow(cfg->sourceId, AbsVisionManager::DEFAULT_WINDOW_LENGTH,
                  AbsVisionManager::DEFAULT_TRIGGER_DUTY_FACTOR, AbsVisionManager::DEFAULT_END_DUTY_FACTOR,
                  F_DANGEROUS_SMOKE),
      drinkWindow(cfg->sourceId, AbsVisionManager::DEFAULT_WINDOW_LENGTH,
                  AbsVisionManager::DEFAULT_TRIGGER_DUTY_FACTOR, AbsVisionManager::DEFAULT_END_DUTY_FACTOR,
                  F_DANGEROUS_DRINK),
      silenceWindow(cfg->sourceId, AbsVisionManager::DEFAULT_WINDOW_LENGTH,
                    AbsVisionManager::DEFAULT_TRIGGER_DUTY_FACTOR, AbsVisionManager::DEFAULT_END_DUTY_FACTOR,
                    F_DANGEROUS_SILENCE),
      openMouthWindow(cfg->sourceId, AbsVisionManager::DEFAULT_WINDOW_LENGTH,
                      AbsVisionManager::DEFAULT_TRIGGER_DUTY_FACTOR, AbsVisionManager::DEFAULT_END_DUTY_FACTOR,
                      F_DANGEROUS_OPEN_MOUTH),
      maskCoverWindow(cfg->sourceId, AbsVisionManager::DEFAULT_WINDOW_LENGTH,
                      AbsVisionManager::DEFAULT_TRIGGER_DUTY_FACTOR, AbsVisionManager::DEFAULT_END_DUTY_FACTOR,
                      F_DANGEROUS_MASK_COVER),
      coverMouthWindow(cfg->sourceId, AbsVisionManager::DEFAULT_WINDOW_LENGTH,
                       AbsVisionManager::DEFAULT_TRIGGER_DUTY_FACTOR, AbsVisionManager::DEFAULT_END_DUTY_FACTOR,
                       F_DANGEROUS_COVER_MOUTH),
      smokeBurningWindow(cfg->sourceId, AbsVisionManager::DEFAULT_WINDOW_LENGTH,
                         AbsVisionManager::DEFAULT_TRIGGER_DUTY_FACTOR) {
    this->rtConfig = cfg;
    setupSlidingWindow();
}

FaceDangerousDriveStrategy::~FaceDangerousDriveStrategy() {
    FaceDangerousDriveStrategy::clear();
}

void FaceDangerousDriveStrategy::setupSlidingWindow() {
    auto stageParas = std::make_shared<StageParameters>();
    for (int i = AbsVisionManager::WINDOW_LOWER_FPS; i <= AbsVisionManager::WINDOW_UPPER_FPS; ++i) {
        stageParas->stage_routine[i] = {static_cast<int>(round(i * AbsVisionManager::DEFAULT_W_LENGTH_RATIO_1_0)),
                                        AbsVisionManager::DEFAULT_W_DUTY_FACTOR};
    }

    smokeWindow.set_trigger_expire_time((long)rtConfig->smokeTriggerExpireTime);
    drinkWindow.set_trigger_expire_time((long)rtConfig->drinkTriggerExpireTime);
    silenceWindow.set_trigger_expire_time((long)rtConfig->silenceTriggerExpireTime);
    openMouthWindow.set_trigger_expire_time((long)rtConfig->openMouthTriggerExpireTime);
    maskCoverWindow.set_trigger_expire_time((long)rtConfig->maskTriggerExpireTime);
    coverMouthWindow.set_trigger_expire_time((long)rtConfig->coverMouthTriggerExpireTime);

    smokeWindow.set_fps_stage_parameters(stageParas);
    smokeBurningWindow.set_fps_stage_parameters(stageParas);
    drinkWindow.set_fps_stage_parameters(stageParas);
    silenceWindow.set_fps_stage_parameters(stageParas);
    openMouthWindow.set_fps_stage_parameters(stageParas);
    maskCoverWindow.set_fps_stage_parameters(stageParas);
    coverMouthWindow.set_fps_stage_parameters(stageParas);
}

void FaceDangerousDriveStrategy::execute(FaceInfo *face) {
    // 赋值默认结果
    face->stateDangerDrive = F_DANGEROUS_NONE;

    // 如下逻辑暂时注释。如果用于检测喝水的时候工程化规避策略。记录历史人脸关键点数据，当没有人脸的时候检测喝水
    // if (!_record_state) {
    //     _record_state = true;
    //     memcpy(lastLandmark2D106, face->landmark2D106, LM_2D_106_COUNT * sizeof(VPoint));
    // } else {
    //     for (int i = 0; i < LM_2D_106_COUNT; ++i) {
    //         lastLandmark2D106[i].x = lastLandmark2D106[i].x * 0.9f + face->landmark2D106[i].x * 0.1f;
    //         lastLandmark2D106[i].y = lastLandmark2D106[i].y * 0.9f + face->landmark2D106[i].y * 0.1f;
    //     }
    // }

    // 滑窗过滤-抽烟、喝水。如果pitch角度小于pitch角度(低头不检测抽烟喝水)
    if (kFaceDangerAngleLimitSwitch
        && (face->headDeflection.pitch < rtConfig->dangerDrivePitchLower
            || face->headDeflection.yaw > rtConfig->dangerDriveYawUpper
            || face->headDeflection.yaw < rtConfig->dangerDriveYawLower)) {
        smokeWindow.update(F_DANGEROUS_NONE, &face->smokeVState);
        face->stateSmokeBurning = V_TO_SHORT(smokeBurningWindow.update(F_SMOKE_BURNING_UNKNOWN));
        drinkWindow.update(F_DANGEROUS_NONE, &face->drinkVState);
    } else {
        smokeWindow.update(face->stateDangerDriveSingle, &face->smokeVState);
        face->stateSmokeBurning = V_TO_SHORT(smokeBurningWindow.update(face->stateSmokeBurningSingle));
        drinkWindow.update(face->stateDangerDriveSingle, &face->drinkVState);
    }

    // 滑窗过滤-比嘘有pitch角度限制，避免低头时误检比嘘
    bool ret_silence = false;
    if (kFaceDangerAngleLimitSwitch && face->headDeflection.pitch < rtConfig->dangerDrivePitchLower) {
        ret_silence = silenceWindow.update(F_DANGEROUS_NONE, &face->silenceVState);
        VLOGD(TAG, "invalid frame ignored for silence out of threshold");
    } else {
        ret_silence = silenceWindow.update(face->stateDangerDriveSingle, &face->silenceVState);
    }

    // 滑窗过滤-张嘴
    bool ret_open_mouth = openMouthWindow.update(face->stateDangerDriveSingle, &face->openMouthVState);
    bool retMask = maskCoverWindow.update(face->stateDangerDriveSingle, &face->maskMouthVState);
    bool retCoverMouth = coverMouthWindow.update(face->stateDangerDriveSingle, &face->coverMouthVState);

    // 当抽烟状态是start开始进入触发状态的时候，清除其他的互斥逻辑滑窗
    if (face->smokeVState.state == VSlidingState::START) {
        drinkWindow.clear_buffer();
        coverMouthWindow.clear_buffer();
        maskCoverWindow.clear_buffer();
        face->stateDangerDrive = F_DANGEROUS_SMOKE;
    }

    if (face->drinkVState.state == VSlidingState::START) {
        smokeWindow.clear_buffer();
        smokeBurningWindow.clear_buffer();
        coverMouthWindow.clear_buffer();
        maskCoverWindow.clear_buffer();
        face->stateDangerDrive = F_DANGEROUS_DRINK;
    }

    if (face->coverMouthVState.state == VSlidingState::START) {
        drinkWindow.clear_buffer();
        smokeWindow.clear_buffer();
        smokeBurningWindow.clear_buffer();
        maskCoverWindow.clear_buffer();
        face->stateDangerDrive = F_DANGEROUS_COVER_MOUTH;
    }

    // 如果戴口罩满足了滑窗，则需要清空喝水和抽烟的滑窗
    if (face->maskMouthVState.state == VSlidingState::START) {
        drinkWindow.clear_buffer();
        smokeWindow.clear_buffer();
        smokeBurningWindow.clear_buffer();
        coverMouthWindow.clear_buffer();
        face->stateDangerDrive = F_DANGEROUS_MASK_COVER;
    }

    // 如果比嘘满足滑窗
    if (ret_silence) {
        face->stateDangerDrive = F_DANGEROUS_SILENCE;
        silenceWindow.clear_buffer();
    }

    // 如果张嘴满足滑窗。则设置
    if (ret_open_mouth) {
        face->stateDangerDrive = F_DANGEROUS_OPEN_MOUTH;
        openMouthWindow.clear_buffer();
    }
    VLOGI(TAG, "danger_drive[%ld] stateSingle=[%d] state=[%d]", face->id, face->stateDangerDriveSingle,
          face->stateDangerDrive);
}

void FaceDangerousDriveStrategy::onNoFace(VisionRequest *request, FaceInfo *face) {
    // 赋值默认结果
    face->stateDangerDrive = F_DANGEROUS_NONE;

    // 滑窗过滤-抽烟、喝水。如果pitch角度小于pitch角度(低头不检测抽烟喝水)
    smokeWindow.update(F_DANGEROUS_NONE, &face->smokeVState);
    face->stateSmokeBurning = V_TO_SHORT(smokeBurningWindow.update(F_SMOKE_BURNING_UNKNOWN));
    drinkWindow.update(F_DANGEROUS_NONE, &face->drinkVState);

    // 滑窗过滤-比嘘有pitch角度限制，避免低头时误检比嘘
    bool ret_silence = silenceWindow.update(F_DANGEROUS_NONE, &face->silenceVState);
    VLOGD(TAG, "[when no face] invalid frame ignored for silence out of threshold");

    // 滑窗过滤-张嘴
    bool ret_open_mouth = openMouthWindow.update(F_DANGEROUS_NONE, &face->openMouthVState);
    bool retMask = maskCoverWindow.update(F_DANGEROUS_NONE, &face->maskMouthVState);
    bool retCoverMouth = coverMouthWindow.update(F_DANGEROUS_NONE, &face->coverMouthVState);

    // TODO 按目前逻辑不会是start状态，因此该部分代码不会执行。暂时保留待 @王志江 后续考量
    if (face->smokeVState.state == VSlidingState::START) {
        drinkWindow.clear_buffer();
        coverMouthWindow.clear_buffer();
        maskCoverWindow.clear_buffer();
        face->stateDangerDrive = F_DANGEROUS_SMOKE;
    }

    if (face->drinkVState.state == VSlidingState::START) {
        smokeWindow.clear_buffer();
        smokeBurningWindow.clear_buffer();
        coverMouthWindow.clear_buffer();
        maskCoverWindow.clear_buffer();
        face->stateDangerDrive = F_DANGEROUS_DRINK;
    }

    if (face->coverMouthVState.state == VSlidingState::START) {
        drinkWindow.clear_buffer();
        smokeWindow.clear_buffer();
        smokeBurningWindow.clear_buffer();
        maskCoverWindow.clear_buffer();
        face->stateDangerDrive = F_DANGEROUS_COVER_MOUTH;
    }

    // 如果戴口罩满足了滑窗，则需要清空喝水和抽烟的滑窗
    if (face->maskMouthVState.state == VSlidingState::START) {
        drinkWindow.clear_buffer();
        smokeWindow.clear_buffer();
        smokeBurningWindow.clear_buffer();
        coverMouthWindow.clear_buffer();
        face->stateDangerDrive = F_DANGEROUS_MASK_COVER;
    }

    // 如果比嘘满足滑窗
    if (ret_silence) {
        face->stateDangerDrive = F_DANGEROUS_SILENCE;
        silenceWindow.clear_buffer();
    }

    // 如果张嘴满足滑窗。则设置
    if (ret_open_mouth) {
        face->stateDangerDrive = F_DANGEROUS_OPEN_MOUTH;
        openMouthWindow.clear_buffer();
    }

    VLOGI(TAG, "[when no face] danger_drive[%ld] stateSingle=[%d] state=[%d]", face->id, face->stateDangerDriveSingle,
          face->stateDangerDrive);
}

/**
 * runtime_config发生变化时候，更新危险驾驶相关触发时间
 * @param key
 * @param value
 */
void FaceDangerousDriveStrategy::onConfigUpdated(int key, float value) {
    switch (key) {
        case STRATEGY_TRIGGER_EXPIRE_TIME_SMOKE:
            smokeWindow.set_trigger_expire_time((long)rtConfig->smokeTriggerExpireTime);
            break;
        case STRATEGY_TRIGGER_EXPIRE_TIME_DRINK:
            drinkWindow.set_trigger_expire_time((long)rtConfig->drinkTriggerExpireTime);
            break;
        case STRATEGY_TRIGGER_EXPIRE_TIME_SILENCE:
            silenceWindow.set_trigger_expire_time((long)rtConfig->silenceTriggerExpireTime);
            break;
        case STRATEGY_TRIGGER_EXPIRE_TIME_OPEN_MOUTH:
            openMouthWindow.set_trigger_expire_time((long)rtConfig->openMouthTriggerExpireTime);
            break;
        case STRATEGY_TRIGGER_EXPIRE_TIME_MASK:
            maskCoverWindow.set_trigger_expire_time((long)rtConfig->maskTriggerExpireTime);
            break;
        case STRATEGY_TRIGGER_EXPIRE_TIME_COVER_MOUTH:
            coverMouthWindow.set_trigger_expire_time((long)rtConfig->coverMouthTriggerExpireTime);
            break;
        default:
            break;
    }
}

void FaceDangerousDriveStrategy::clear() {
    smokeWindow.clear();
    drinkWindow.clear();
    silenceWindow.clear();
    openMouthWindow.clear();
    maskCoverWindow.clear();
    coverMouthWindow.clear();
    smokeBurningWindow.clear();
    _record_state = false;
}

FaceDangerousDriveManager::FaceDangerousDriveManager() : detector(nullptr) {
    detector = std::make_shared<FaceDangerousDriveDetector>();
}

FaceDangerousDriveManager::~FaceDangerousDriveManager() {
    clear();
}

void FaceDangerousDriveManager::init(RtConfig *cfg) {
    mRtConfig = cfg;
    detector->init(cfg);
}

void FaceDangerousDriveManager::deinit() {
    AbsVisionManager::deinit();
    if (detector != nullptr) {
        detector->deinit();
        detector = nullptr;
    }
}

void FaceDangerousDriveManager::clear() {
    for (auto &info : dangerDriveStrategyMap) {
        if (info.second) {
            info.second->clear();
            FaceDangerousDriveStrategy::recycle(info.second);
        }
    }
    dangerDriveStrategyMap.clear();
}

bool FaceDangerousDriveManager::preDetect(VisionRequest *request, VisionResult *result) {
    VA_CHECK_DETECTED(ABILITY_FACE_DANGEROUS_DRIVING);
}

void FaceDangerousDriveManager::doDetect(VisionRequest *request, VisionResult *result) {
    VA_SET_DETECTED(ABILITY_FACE_DANGEROUS_DRIVING);
    // 危险驾驶检测
    detector->detect(request, result);
    // 执行多人脸策略
    execute_face_strategy<FaceDangerousDriveStrategy>(result, dangerDriveStrategyMap, mRtConfig);
}

void FaceDangerousDriveManager::onNoFace(VisionRequest *request, VisionResult *result, FaceInfo *face) {
    if (face != nullptr && (!dangerDriveStrategyMap.empty())) {
        auto iter = dangerDriveStrategyMap.find(face->id);
        if (iter != dangerDriveStrategyMap.end()) {
            iter->second->onNoFace(request, face);
        }
    }
}

void FaceDangerousDriveManager::onConfigUpdated(int key, float value) {
    for (auto &info : dangerDriveStrategyMap) {
        info.second->onConfigUpdated(key, value);
    }
}

REGISTER_VISION_MANAGER("FaceDangerousDrivingManager", ABILITY_FACE_DANGEROUS_DRIVING, []() {
    return std::dynamic_pointer_cast<AbsVisionManager>(std::make_shared<FaceDangerousDriveManager>());
});

} // namespace aura::vision
