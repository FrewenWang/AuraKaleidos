

#include "FaceEmotionManager.h"
#include <set>

#include "vision/util/log.h"
#include "util/sliding_window.h"
#include "vision/manager/VisionManagerRegistry.h"

namespace aura::vision {

static const char *TAG = "FaceEmotionManager";

FaceEmotionStrategy::FaceEmotionStrategy(RtConfig *cfg)
    : faceEmotionWindow(cfg->sourceId, AbsVisionManager::DEFAULT_WINDOW_LENGTH,
                        AbsVisionManager::DEFAULT_TRIGGER_DUTY_FACTOR) {
    this->rtConfig = cfg;
    setupSlidingWindow();
}

FaceEmotionStrategy::~FaceEmotionStrategy() {}

void FaceEmotionStrategy::setupSlidingWindow() {
    auto stageParas = std::make_shared<StageParameters>();
    for (int i = AbsVisionManager::WINDOW_LOWER_FPS; i <= AbsVisionManager::WINDOW_UPPER_FPS; ++i) {
        stageParas->stage_routine[i] = {static_cast<int>(round(i * AbsVisionManager::DEFAULT_W_LENGTH_RATIO_1_0)),
                                        AbsVisionManager::DEFAULT_W_DUTY_FACTOR};
    }
    faceEmotionWindow.set_fps_stage_parameters(stageParas);
}

void FaceEmotionStrategy::execute(FaceInfo *face) {
    // 人脸遮挡时在Detector中已经将单帧和多帧结果置为UNKNOW，此处直接向滑窗中填充单帧结果
    int emotionSingle = static_cast<int>(face->stateEmotionSingle); // 表情检测单帧数据
    face->stateEmotion = faceEmotionWindow.update(emotionSingle);   // 更新滑窗
    VLOGI(TAG, "face_emotion[%ld] stateSingle=[%f], score=[%f],state=[%f] faceCoverSingle[%d]", face->id,
          face->stateEmotionSingle, face->scoreEmotionSingle, face->stateEmotion, face->stateFaceCoverSingle);
}

void FaceEmotionStrategy::onNoFace(VisionRequest *request, FaceInfo *face) {
    face->stateEmotion = faceEmotionWindow.update(FaceAttributeStatus::F_ATTR_UNKNOWN);
    VLOGI(TAG, "[when no face] face_emotion[%ld] stateSingle=[%f], state=[%f] score=[%f]", face->id,
          face->stateEmotionSingle, face->stateEmotion, face->scoreEmotionSingle);
}

void FaceEmotionStrategy::clear() {
    faceEmotionWindow.clear();
}

FaceEmotionManager::FaceEmotionManager() {
    detector = std::make_shared<FaceEmotionDetector>();
}

FaceEmotionManager::~FaceEmotionManager() {
    clear();
};

void FaceEmotionManager::init(RtConfig *cfg) {
    mRtConfig = cfg;
    detector->init(cfg);
}

void FaceEmotionManager::deinit() {
    AbsVisionManager::deinit();
    if (detector != nullptr) {
        detector->deinit();
        detector = nullptr;
    }
}

bool FaceEmotionManager::preDetect(VisionRequest *request, VisionResult *result) {
    VA_CHECK_DETECTED(ABILITY_FACE_EMOTION);
}

void FaceEmotionManager::doDetect(VisionRequest *request, VisionResult *result) {
    VA_SET_DETECTED(ABILITY_FACE_EMOTION);

    // 根据上一帧检测的人脸情绪结果，以及设置的固定检测帧率。判断当前帧是否需要Detect
    if (checkFpsFixedDetect(FIXED_DETECT_DURATION, forceDetectCurFrame)) {
        detector->detect(request, result);
    } else {
        VLOGD(TAG, "no need detect curFrame");
        return;
    }
    // 执行多人脸策略
    execute_face_strategy<FaceEmotionStrategy>(result, emotionStrategyMap, mRtConfig);

    // 人脸属性识别，没有必要太高帧率。直接默认使用固定帧率检测。不进行每一帧强制检测
    forceDetectCurFrame = false;
}

void FaceEmotionManager::clear() {
    AbsVisionManager::clear();
    for (auto &info: emotionStrategyMap) {
        if (info.second) {
            info.second->clear();
            FaceEmotionStrategy::recycle(info.second);
        }
    }
    forceDetectCurFrame = true;
    emotionStrategyMap.clear();
}

void FaceEmotionManager::onNoFace(VisionRequest *request, VisionResult *result, FaceInfo *face) {
    if (face != nullptr && (!emotionStrategyMap.empty())) {
        auto iter = emotionStrategyMap.find(face->id);
        if (iter != emotionStrategyMap.end()) {
            iter->second->onNoFace(request, face);
        }
    }
}

// NOLINTNEXTLINE
REGISTER_VISION_MANAGER("FaceEmotionManager", ABILITY_FACE_EMOTION, []() {
    return std::dynamic_pointer_cast<AbsVisionManager>(std::make_shared<FaceEmotionManager>());
});

} // namespace aura::vision