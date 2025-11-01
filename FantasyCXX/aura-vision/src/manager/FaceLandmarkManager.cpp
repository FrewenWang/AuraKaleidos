

#include "FaceLandmarkManager.h"
#include "detector/FaceLandmarkDetector.h"
#include "vision/manager/VisionManagerRegistry.h"

namespace aura::vision {

FaceLandmarkManager::FaceLandmarkManager() :
      _no_face_window(SOURCE_UNKNOWN, DEFAULT_WINDOW_LENGTH, DEFAULT_TRIGGER_DUTY_FACTOR, DEFAULT_END_DUTY_FACTOR,
                      F_NONE) {
    detector = std::make_shared<FaceLandmarkDetector>();
    setupSlidingWindow();
}

void FaceLandmarkManager::init(RtConfig *cfg) {
    mRtConfig = cfg;
    detector->init(cfg);
    _no_face_window.setSourceId(mRtConfig->sourceId);
}

void FaceLandmarkManager::deinit() {
    AbsVisionManager::deinit();
    if (detector != nullptr) {
        detector->deinit();
        detector = nullptr;
    }
}

bool FaceLandmarkManager::preDetect(VisionRequest *request, VisionResult *result) {
    VA_CHECK_DETECTED(ABILITY_FACE_LANDMARK);
}

void FaceLandmarkManager::doDetect(VisionRequest *request, VisionResult *result) {
    VA_SET_DETECTED(ABILITY_FACE_LANDMARK);

    detector->detect(request, result);

    if (result->hasFace()) {
        _no_face_window.clear();
    } else {
        FaceInfo *face = result->getFaceResult()->faceInfos[0];
        if (face) {
            _no_face_window.update(F_NONE, &face->_no_face_state);
        }
    }
}

void FaceLandmarkManager::setupSlidingWindow() {
    auto stageParas = std::make_shared<StageParameters>();
    for (int i = WINDOW_LOWER_FPS; i <= WINDOW_UPPER_FPS; ++i) {
        stageParas->stage_routine[i] = {static_cast<int>(round(i * DEFAULT_W_LENGTH_RATIO_1_5)), DEFAULT_W_DUTY_FACTOR};
    }
    _no_face_window.set_fps_stage_parameters(stageParas);
}

void FaceLandmarkManager::clear() {
    _no_face_window.clear();
}

REGISTER_VISION_MANAGER("FaceLandmarkManager", ABILITY_FACE_LANDMARK, []() {
    return std::dynamic_pointer_cast<AbsVisionManager>(std::make_shared<FaceLandmarkManager>());
});

} // namespace vision