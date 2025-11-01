

#include "FaceCallManager.h"

#include <algorithm>
#include <set>

#include "detector/FaceCallDetector.h"
#include "vision/util/log.h"
#include "vision/manager/VisionManagerRegistry.h"

namespace aura::vision {

static const char *TAG = "FaceCallManager";

FaceCallStrategy::FaceCallStrategy(RtConfig *cfg)
    : faceCallWindow(cfg->sourceId, AbsVisionManager::DEFAULT_WINDOW_LENGTH,
                     AbsVisionManager::DEFAULT_TRIGGER_DUTY_FACTOR, AbsVisionManager::DEFAULT_END_DUTY_FACTOR,
                     F_CALL_CALLING),
      gestureNearbyWindow(cfg->sourceId, AbsVisionManager::DEFAULT_WINDOW_LENGTH,
                          AbsVisionManager::DEFAULT_TRIGGER_DUTY_FACTOR, AbsVisionManager::DEFAULT_END_DUTY_FACTOR,
                          F_CALL_GESTURE_NEARBY) {
    this->rtConfig = cfg;
    setupSlidingWindow();
}

FaceCallStrategy::~FaceCallStrategy() {
    clear();
}

void FaceCallStrategy::clear() {
    faceCallWindow.clear();
    gestureNearbyWindow.clear();
}

void FaceCallStrategy::setupSlidingWindow() {
    auto callParas = std::make_shared<StageParameters>();
    auto gestureParas = std::make_shared<StageParameters>();
    for (int i = AbsVisionManager::WINDOW_LOWER_FPS; i <= AbsVisionManager::WINDOW_UPPER_FPS; ++i) {
        callParas->stage_routine[i] = {static_cast<int>(round(i * AbsVisionManager::DEFAULT_W_LENGTH_RATIO_2_0)),
                                       AbsVisionManager::DEFAULT_W_DUTY_FACTOR};
        gestureParas->stage_routine[i] = {static_cast<int>(round(i * AbsVisionManager::DEFAULT_W_LENGTH_RATIO_1_0)),
                                          AbsVisionManager::DEFAULT_W_DUTY_FACTOR};
    }

    faceCallWindow.set_trigger_need_time((long)rtConfig->callTriggerNeedTime);
    faceCallWindow.set_trigger_expire_time((long)rtConfig->callTriggerExpireTime);
    gestureNearbyWindow.set_trigger_need_time((long)rtConfig->callTriggerNeedTime);
    gestureNearbyWindow.set_trigger_expire_time((long)rtConfig->callTriggerExpireTime);

    faceCallWindow.set_fps_stage_parameters(callParas);
    gestureNearbyWindow.set_fps_stage_parameters(gestureParas);
}

void FaceCallStrategy::execute(FaceInfo *face) {
    face->stateCallSingle = F_CALL_NONE;
    // 工程策略：
    // 如果左右耳都是非打电话。则单帧打电话状态为非打电话
    // 如果左右耳有一个耳朵打电话。则单帧结果为打电话状态
    // 否则，为NearbyEar状态(耳朵边有其他遮挡物)
    if ((face->stateCallLeftSingle == F_CALL_NONE) && (face->stateCallRightSingle == F_CALL_NONE)) {
        face->stateCallSingle = F_CALL_NONE;
    } else {
        if ((face->stateCallLeftSingle == F_CALL_CALLING) || (face->stateCallRightSingle == F_CALL_CALLING)) {
            face->stateCallSingle = F_CALL_CALLING;
        } else {
            face->stateCallSingle = F_CALL_GESTURE_NEARBY;
        }
    }
    // 更新滑窗数据
    bool callResult = faceCallWindow.update(face->stateCallSingle, &face->phoneCallVState);
    bool gestureNearbyResult = gestureNearbyWindow.update(face->stateCallSingle, &face->gestureNearbyEar);

    if (callResult) {
        face->stateCall = F_CALL_CALLING;
    } else if (gestureNearbyResult) {
        face->stateCall = F_CALL_GESTURE_NEARBY;
    } else {
        face->stateCall = F_CALL_NONE;
    }
    VLOGI(TAG, "face_call[%ld] stateSingle=[state:%d,L:%d,R:%d],score=[L:%f,R:%f],state=[%f],vState=[%d,%d,%d]",
          face->id, face->stateCallSingle, face->stateCallLeftSingle, face->stateCallRightSingle,
          face->scoreCallLeftSingle, face->scoreCallRightSingle, face->stateCall, face->phoneCallVState.state,
          face->phoneCallVState.continue_time, face->phoneCallVState.trigger_count);
}

void FaceCallStrategy::onNoFace(VisionRequest *request, FaceInfo *face) {
    // 新的一帧开始时未清除打电话单帧结果，在手动置为 NONE
    face->stateCallLeftSingle = F_CALL_NONE;
    face->stateCallRightSingle = F_CALL_NONE;
    face->stateCallSingle = F_CALL_NONE;

    // 更新滑窗数据
    bool callResult = faceCallWindow.update(F_CALL_NONE, &face->phoneCallVState);
    bool gestureNearbyResult = gestureNearbyWindow.update(F_CALL_NONE, &face->gestureNearbyEar);

    if (callResult) {
        face->stateCall = F_CALL_CALLING;
    } else if (gestureNearbyResult) {
        face->stateCall = F_CALL_GESTURE_NEARBY;
    } else {
        face->stateCall = F_CALL_NONE;
    }
    VLOGI(TAG,
          "[when no face] face_call[%ld] stateSingle=[state:%d,left:%d,right:%d] ,state=[%f],vState=[%d,%d,%d],"
          "score=[%f]",
          face->id, face->stateCallSingle, face->stateCallLeftSingle, face->stateCallRightSingle, face->stateCall,
          face->phoneCallVState.state, face->phoneCallVState.continue_time, face->phoneCallVState.trigger_count,
          face->scoreCallLeftSingle);
}

void FaceCallStrategy::onConfigUpdated(int key, float value) {
    switch (key) {
        case STRATEGY_TRIGGER_EXPIRE_TIME_CALL:
            faceCallWindow.set_trigger_expire_time(V_TO_INT(rtConfig->callTriggerExpireTime));
            gestureNearbyWindow.set_trigger_expire_time(V_TO_INT(rtConfig->callTriggerExpireTime));
            break;
        case STRATEGY_TRIGGER_NEED_TIME_CALL:
            faceCallWindow.set_trigger_need_time(V_TO_INT(rtConfig->callTriggerNeedTime));
            gestureNearbyWindow.set_trigger_need_time(V_TO_INT(rtConfig->callTriggerNeedTime));
            break;
        default:
            break;
    }
}

FaceCallManager::FaceCallManager() {
    detector = std::make_shared<FaceCallDetector>();
}

FaceCallManager::~FaceCallManager() = default;

void FaceCallManager::init(RtConfig *cfg) {
    mRtConfig = cfg;
    detector->init(cfg);
}

void FaceCallManager::deinit() {
    AbsVisionManager::deinit();
    if (detector != nullptr) {
        detector->deinit();
        detector = nullptr;
    }
}

bool FaceCallManager::preDetect(VisionRequest *request, VisionResult *result) {
    VA_CHECK_DETECTED(ABILITY_FACE_CALL);
}

void FaceCallManager::doDetect(VisionRequest *request, VisionResult *result) {
    VA_SET_DETECTED(ABILITY_FACE_CALL);
    // 检测打电话
    detector->detect(request, result);
    // 执行多人脸滑窗策略
    execute_face_strategy<FaceCallStrategy>(result, faceCallStrategyMap, mRtConfig);
}

void FaceCallManager::clear() {
    for (auto &info : faceCallStrategyMap) {
        if (info.second) {
            info.second->clear();
            FaceCallStrategy::recycle(info.second);
        }
    }
    faceCallStrategyMap.clear();
}

void FaceCallManager::onNoFace(VisionRequest *request, VisionResult *result, FaceInfo *face) {
    if (face != nullptr && (!faceCallStrategyMap.empty())) {
        auto iter = faceCallStrategyMap.find(face->id);
        if (iter != faceCallStrategyMap.end()) {
            iter->second->onNoFace(request, face);
        }
    }
}

void FaceCallManager::onConfigUpdated(int key, float value) {
    for (auto &info : faceCallStrategyMap) {
        info.second->onConfigUpdated(key, value);
    }
}

REGISTER_VISION_MANAGER("FaceCallManager", ABILITY_FACE_CALL, []() {
    return std::dynamic_pointer_cast<AbsVisionManager>(std::make_shared<FaceCallManager>());
});

} // namespace aura::vision
