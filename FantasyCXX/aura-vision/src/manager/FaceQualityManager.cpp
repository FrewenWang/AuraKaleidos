
#include <algorithm>
#include <set>
#include "FaceQualityManager.h"
#include "vision/manager/VisionManagerRegistry.h"

namespace aura::vision {

static const char *TAG = "FaceQualityManager";

/**
 * 构造函数
 * 注意：构造函数初始化变量的初始化顺序和声明顺序保持一致
 * @param cfg
 */
FaceQualityStrategy::FaceQualityStrategy(RtConfig *cfg)
    : faceCoverWindow(cfg->sourceId, AbsVisionManager::DEFAULT_WINDOW_LENGTH,
                      AbsVisionManager::DEFAULT_TRIGGER_DUTY_FACTOR, AbsVisionManager::DEFAULT_END_DUTY_FACTOR,
                      F_QUALITY_COVER_HIGH),
      leftEyeCoverWindow(cfg->sourceId, AbsVisionManager::DEFAULT_WINDOW_LENGTH,
                         AbsVisionManager::DEFAULT_TRIGGER_DUTY_FACTOR, AbsVisionManager::DEFAULT_END_DUTY_FACTOR,
                         F_QUALITY_COVER_LEFT_EYE),
      rightEyeCoverWindow(cfg->sourceId, AbsVisionManager::DEFAULT_WINDOW_LENGTH,
                          AbsVisionManager::DEFAULT_TRIGGER_DUTY_FACTOR, AbsVisionManager::DEFAULT_END_DUTY_FACTOR,
                          F_QUALITY_COVER_RIGHT_EYE),
      mouthCoverWindow(cfg->sourceId, AbsVisionManager::DEFAULT_WINDOW_LENGTH,
                       AbsVisionManager::DEFAULT_TRIGGER_DUTY_FACTOR, AbsVisionManager::DEFAULT_END_DUTY_FACTOR,
                       F_QUALITY_COVER_MOUTH_HIGH),
      blurWindow(cfg->sourceId, AbsVisionManager::DEFAULT_WINDOW_LENGTH, AbsVisionManager::DEFAULT_TRIGGER_DUTY_FACTOR,
                 AbsVisionManager::DEFAULT_END_DUTY_FACTOR, F_QUALITY_BLUR_HIGH),
      noiseWindow(cfg->sourceId, AbsVisionManager::DEFAULT_WINDOW_LENGTH,
                  AbsVisionManager::DEFAULT_TRIGGER_DUTY_FACTOR, AbsVisionManager::DEFAULT_END_DUTY_FACTOR,
                  F_QUALITY_NOISE_HIGH) {
    this->rtConfig = cfg;
    setupSlidingWindow();
}

FaceQualityStrategy::~FaceQualityStrategy() {
    clear();
}

void FaceQualityStrategy::setupSlidingWindow() {
    auto stageParas = std::make_shared<StageParameters>();
    for (int i = AbsVisionManager::WINDOW_LOWER_FPS; i <= AbsVisionManager::WINDOW_UPPER_FPS; ++i) {
        stageParas->stage_routine[i] = {static_cast<int>(round(i * AbsVisionManager::DEFAULT_W_LENGTH_RATIO_1_0)),
                                        AbsVisionManager::DEFAULT_W_DUTY_FACTOR};
    }

    blurWindow.set_window_adjust_strategy(WindowAdjustStrategy::Fixed);
    noiseWindow.set_window_adjust_strategy(WindowAdjustStrategy::Fixed);
    leftEyeCoverWindow.set_fps_stage_parameters(stageParas);
    rightEyeCoverWindow.set_fps_stage_parameters(stageParas);
    mouthCoverWindow.set_fps_stage_parameters(stageParas);
    faceCoverWindow.set_fps_stage_parameters(stageParas);
}

void FaceQualityStrategy::execute(FaceInfo *face) {
    // 更新滑窗
    face->stateBlur = blurWindow.update(face->stateBlurSingle, nullptr) ? FaceQualityStatus::F_QUALITY_BLUR_HIGH :
                                                                          FaceQualityStatus::F_QUALITY_BLUR_NORMAL;
    face->stateNoise = noiseWindow.update(face->stateNoiseSingle, nullptr) ? FaceQualityStatus::F_QUALITY_NOISE_HIGH :
                                                                             FaceQualityStatus::F_QUALITY_NOISE_NORMAL;
    face->stateLeftEyeCover = leftEyeCoverWindow.update(face->leftEyeCoverSingle, nullptr) ?
                                      FaceQualityStatus::F_QUALITY_COVER_LEFT_EYE :
                                      FaceQualityStatus::F_QUALITY_COVER_LEFT_EYE_NORMAL;
    face->stateRightEyeCover = rightEyeCoverWindow.update(face->rightEyeCoverSingle, nullptr) ?
                                       FaceQualityStatus::F_QUALITY_COVER_RIGHT_EYE :
                                       FaceQualityStatus::F_QUALITY_COVER_RIGHT_EYE_NORMAL;
    face->stateMouthCover = mouthCoverWindow.update(face->stateMouthCoverSingle, nullptr) ?
                                    FaceQualityStatus::F_QUALITY_COVER_MOUTH_HIGH :
                                    FaceQualityStatus::F_QUALITY_COVER_MOUTH_NORMAL;
    face->stateFaceCover = faceCoverWindow.update(face->stateFaceCoverSingle, nullptr) ?
                                   FaceQualityStatus::F_QUALITY_COVER_HIGH :
                                   FaceQualityStatus::F_QUALITY_COVER_NORMAL;
    VLOGI(TAG,
          "face_quality[%s] "
          "stateSingle=[%s,leftEye:%d,rightEye:%d,mouth:%d],state=[%s,left_eye:%d,right_eye:%d,mouth:%d]",
          std::to_string(face->id).c_str(),
          face->stateFaceCoverSingle == F_QUALITY_COVER_HIGH ? "cover" : "good", face->leftEyeCoverSingle,
          face->rightEyeCoverSingle, face->stateMouthCoverSingle,
          face->stateFaceCover == F_QUALITY_COVER_HIGH ? "cover" : "good", face->stateLeftEyeCover,
          face->stateRightEyeCover, face->stateMouthCover);
}

/**
 * 无人脸的时候，往无感活体的滑窗内填充F_QUALITY_UNKNOWN，逐渐降低滑窗结果
 * @param request
 * @param face
 */
void FaceQualityStrategy::onNoFace(VisionRequest *request, FaceInfo *face) {
    face->stateBlur = blurWindow.update(F_QUALITY_UNKNOWN, nullptr) ? FaceQualityStatus::F_QUALITY_BLUR_HIGH :
                                                                      FaceQualityStatus::F_QUALITY_BLUR_NORMAL;
    face->stateNoise = noiseWindow.update(F_QUALITY_UNKNOWN, nullptr) ? FaceQualityStatus::F_QUALITY_NOISE_HIGH :
                                                                        FaceQualityStatus::F_QUALITY_NOISE_NORMAL;
    face->stateLeftEyeCover = leftEyeCoverWindow.update(F_QUALITY_UNKNOWN, nullptr) ?
                                      FaceQualityStatus::F_QUALITY_COVER_LEFT_EYE :
                                      FaceQualityStatus::F_QUALITY_COVER_LEFT_EYE_NORMAL;
    face->stateRightEyeCover = rightEyeCoverWindow.update(F_QUALITY_UNKNOWN, nullptr) ?
                                       FaceQualityStatus::F_QUALITY_COVER_RIGHT_EYE :
                                       FaceQualityStatus::F_QUALITY_COVER_RIGHT_EYE_NORMAL;
    face->stateMouthCover = mouthCoverWindow.update(F_QUALITY_UNKNOWN, nullptr) ?
                                    FaceQualityStatus::F_QUALITY_COVER_MOUTH_HIGH :
                                    FaceQualityStatus::F_QUALITY_COVER_MOUTH_NORMAL;
    face->stateFaceCover = faceCoverWindow.update(F_QUALITY_UNKNOWN, nullptr) ?
                                   FaceQualityStatus::F_QUALITY_COVER_HIGH :
                                   FaceQualityStatus::F_QUALITY_COVER_NORMAL;
    VLOGI(TAG,
          "[when no face] face_quality[%s] stateSingle=[%s,leftEye:%d,rightEye:%d,mouth:%d],"
          "state=[%s,leftEye:%d,rightEye:%d,mouth:%d]",
          std::to_string(face->id).c_str(),
          face->stateFaceCoverSingle == F_QUALITY_COVER_HIGH ? "cover" : "good", face->leftEyeCoverSingle,
          face->rightEyeCoverSingle, face->stateMouthCoverSingle,
          face->stateFaceCover == F_QUALITY_COVER_HIGH ? "cover" : "good", face->stateLeftEyeCover,
          face->stateRightEyeCover, face->stateMouthCover);
}

void FaceQualityStrategy::clear() {
    faceCoverWindow.clear();
    leftEyeCoverWindow.clear();
    rightEyeCoverWindow.clear();
    mouthCoverWindow.clear();
    blurWindow.clear();
    noiseWindow.clear();
}

FaceQualityManager::FaceQualityManager() {
    detector = std::make_shared<FaceQualityDetector>();
}

FaceQualityManager::~FaceQualityManager() {
    clear();
}

void FaceQualityManager::init(RtConfig *cfg) {
    mRtConfig = cfg;
    detector->init(cfg);
}

void FaceQualityManager::deinit() {
    AbsVisionManager::deinit();
    if (detector != nullptr) {
        detector->deinit();
        detector = nullptr;
    }
}

bool FaceQualityManager::preDetect(VisionRequest *request, VisionResult *result) {
    VA_CHECK_DETECTED(ABILITY_FACE_QUALITY);
}

void FaceQualityManager::doDetect(VisionRequest *request, VisionResult *result) {
    VA_SET_DETECTED(ABILITY_FACE_QUALITY);

    // 根据上一帧检测的人脸遮挡结果，以及设置的固定检测帧率。判断当前帧是否需要Detect
    if (checkFpsFixedDetect(FIXED_DETECT_DURATION, forceDetectCurFrame)) {
        detector->detect(request, result);
    } else {
        VLOGD(TAG, "no need detect curFrame");
        return;
    }
    // 执行多人脸策略
    execute_face_strategy<FaceQualityStrategy>(result, faceQualityMap, mRtConfig);

    // TODO 下面的代码太过繁琐。需要优化。放在此处不是很合理
    // 进行记录上一帧检测到人脸遮挡的结果。如果上一帧检测到遮挡，则下一帧强制进行检测
    forceDetectCurFrame = false;
    int detectFaceCount = result->faceCount();
    auto faceInfos = result->getFaceResult()->faceInfos;
    for (int i = 0; i < detectFaceCount; ++i) {
        auto face = faceInfos[i];
        if ((face->stateFaceCoverSingle == F_QUALITY_COVER_HIGH)) {
            forceDetectCurFrame = true;
            break;
        }
    }
}

void FaceQualityManager::onNoFace(VisionRequest *request, VisionResult *result, FaceInfo *face) {
    if (face != nullptr && (!faceQualityMap.empty())) {
        auto iter = faceQualityMap.find(face->id);
        if (iter != faceQualityMap.end()) {
            iter->second->onNoFace(request, face);
        }
    }
}

void FaceQualityManager::clear() {
    AbsVisionManager::clear();
    for (auto &info: faceQualityMap) {
        if (info.second) {
            info.second->clear();
            FaceQualityStrategy::recycle(info.second);
        }
    }
    faceQualityMap.clear();
}

REGISTER_VISION_MANAGER("FaceQualityManager", ABILITY_FACE_QUALITY, []() {
    return std::dynamic_pointer_cast<AbsVisionManager>(std::make_shared<FaceQualityManager>());
});

} // namespace vision