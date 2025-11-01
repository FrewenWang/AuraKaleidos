
#include "FaceAttributeManager.h"
#include <set>
#include "vision/util/log.h"
#include "vision/manager/VisionManagerRegistry.h"

namespace aura::vision {

const static char *TAG = "FaceAttributeManager";

FaceAttributeStrategy::FaceAttributeStrategy(RtConfig *cfg)
    : _age_window(cfg->sourceId, AbsVisionManager::DEFAULT_WINDOW_LENGTH,
                  AbsVisionManager::DEFAULT_TRIGGER_DUTY_FACTOR),
      _gender_window(cfg->sourceId, AbsVisionManager::DEFAULT_WINDOW_LENGTH,
                     AbsVisionManager::DEFAULT_TRIGGER_DUTY_FACTOR),
      _race_window(cfg->sourceId, AbsVisionManager::DEFAULT_WINDOW_LENGTH,
                   AbsVisionManager::DEFAULT_TRIGGER_DUTY_FACTOR),
      _glass_window(cfg->sourceId, AbsVisionManager::DEFAULT_WINDOW_LENGTH,
                    AbsVisionManager::DEFAULT_TRIGGER_DUTY_FACTOR) {
    this->rtConfig = cfg;
    setupSlidingWindow();
}

FaceAttributeStrategy::~FaceAttributeStrategy() {
    clear();
}

void FaceAttributeStrategy::setupSlidingWindow() {
    auto stageParas = std::make_shared<StageParameters>();
    for (int i = AbsVisionManager::WINDOW_LOWER_FPS; i <= AbsVisionManager::WINDOW_UPPER_FPS; ++i) {
        stageParas->stage_routine[i] = {static_cast<int>(round(i * AbsVisionManager::DEFAULT_W_LENGTH_RATIO_1_0)),
                                        AbsVisionManager::DEFAULT_W_DUTY_FACTOR};
    }
    _age_window.set_fps_stage_parameters(stageParas);
    _gender_window.set_fps_stage_parameters(stageParas);
    _race_window.set_fps_stage_parameters(stageParas);
    _glass_window.set_fps_stage_parameters(stageParas);
}

void FaceAttributeStrategy::clear() {
    _age_window.clear();
    _gender_window.clear();
    _race_window.clear();
    _glass_window.clear();
}

void FaceAttributeStrategy::execute(FaceInfo *face) {
    if (face == nullptr) {
        return;
    }
    face->stateAge = static_cast<float>(_age_window.update(face->stateAgeSingle));
    face->stateGender = static_cast<float>(_gender_window.update(face->stateGenderSingle));
    face->stateRace = static_cast<float>(_race_window.update(face->stateRaceSingle));
    face->stateGlass = static_cast<float>(_glass_window.update(face->stateGlassSingle));
    VLOGI(TAG, "face_attribute[%ld] stateSingle=[gender:%f,age:%f,glass:%f] state=[gender:%f,age:%f,glass:%f]",
          face->id, face->stateGenderSingle, face->stateAgeSingle, face->stateGlassSingle, face->stateGender,
          face->stateAge, face->stateGlass);
}

void FaceAttributeStrategy::onNoFace(VisionRequest *request, FaceInfo *face) {
    face->stateAge = static_cast<float>(_age_window.update(F_ATTR_UNKNOWN));
    face->stateGender = static_cast<float>(_gender_window.update(F_ATTR_UNKNOWN));
    face->stateRace = static_cast<float>(_race_window.update(F_ATTR_UNKNOWN));
    face->stateGlass = static_cast<float>(_glass_window.update(F_ATTR_UNKNOWN));
    VLOGI(TAG,
          "[when no face] face_attribute[%ld] stateSingle=[gender:%f,age:%f,glass:%f] "
          "state=[gender:%f,age:%f,glass:%f]",
          face->id, face->stateGenderSingle, face->stateAgeSingle, face->stateGlassSingle, face->stateGender,
          face->stateAge, face->stateGlass);
}

FaceAttributeManager::FaceAttributeManager() {
    detector = std::make_shared<FaceAttributeDetector>();
}

FaceAttributeManager::~FaceAttributeManager() {}

void FaceAttributeManager::init(RtConfig *cfg) {
    mRtConfig = cfg;
    detector->init(cfg);
}

void FaceAttributeManager::deinit() {
    AbsVisionManager::deinit();
    if (detector != nullptr) {
        detector->deinit();
        detector = nullptr;
    }
}

bool FaceAttributeManager::preDetect(VisionRequest *request, VisionResult *result) {
    VA_CHECK_DETECTED(ABILITY_FACE_ATTRIBUTE);
}

void FaceAttributeManager::doDetect(VisionRequest *request, VisionResult *result) {
    VA_SET_DETECTED(ABILITY_FACE_ATTRIBUTE);
    // 根据上一帧检测的人脸遮挡结果，以及设置的固定检测帧率。判断当前帧是否需要Detect
    if (checkFpsFixedDetect(FIXED_DETECT_DURATION, forceDetectCurFrame)) {
        detector->detect(request, result);
    } else {
        VLOGD(TAG, "no need detect curFrame");
        return;
    }
    // 执行多人脸策略
    execute_face_strategy<FaceAttributeStrategy>(result, attributeStrategyMap, mRtConfig);

    // 性别年龄识别，没有必要太高帧率。直接默认使用固定帧率检测。不进行每一帧强制检测
    forceDetectCurFrame = false;
}

void FaceAttributeManager::clear() {
    AbsVisionManager::clear();
    for (auto &info : attributeStrategyMap) {
        if (info.second) {
            info.second->clear();
            FaceAttributeStrategy::recycle(info.second);
        }
    }
    attributeStrategyMap.clear();
    forceDetectCurFrame = true;
}

void FaceAttributeManager::onNoFace(VisionRequest *request, VisionResult *result, FaceInfo *face) {
    if (face != nullptr && (!attributeStrategyMap.empty())) {
        auto iter = attributeStrategyMap.find(face->id);
        if (iter != attributeStrategyMap.end()) {
            iter->second->onNoFace(request, face);
        }
    }
}

// NOLINTNEXTLINE
REGISTER_VISION_MANAGER("FaceAttributeManager", ABILITY_FACE_ATTRIBUTE, []() {
    return std::dynamic_pointer_cast<AbsVisionManager>(std::make_shared<FaceAttributeManager>());
});

} // namespace aura::vision