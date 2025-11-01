//
// Created by v_zhangtieshuai on 2020/4/15.
//

#include "CameraCoverManager.h"
#include "opencv2/opencv.hpp"
#include "vision/manager/VisionManagerRegistry.h"

namespace aura::vision {

static const char *TAG = "CameraCoverManager";

static const float VISION_EPSINON = 0.000001;
#define VISION_FACE_IMG_COVER_0_CMP(x) ((x) >= std::abs(EPSINON))

void CameraCoverManager::clear() {
    cameraCoverWindow.clear();
}

CameraCoverManager::CameraCoverManager() :
        cameraCoverWindow(SOURCE_UNKNOWN, DEFAULT_WINDOW_LENGTH, DEFAULT_TRIGGER_DUTY_FACTOR,
                          DEFAULT_END_DUTY_FACTOR, true) {
    setupSlidingWindow();
}

void CameraCoverManager::init(RtConfig *cfg) {
    mRtConfig = cfg;
    // 目前CameraCoverManager里面根据是Source1还是Source1来判断是使用对应Detector进行检测
    // 后续还可能根据不同Source下的RGB还是IR摄像头设置不同的Detector
    // 或者后续根据VisSourceConfig里面的配置来决定使用不同的Detector
    if (mRtConfig->sourceId == SOURCE_1) {
        detector = std::make_shared<Source1CameraCoverDetector>();
    } else if (mRtConfig->sourceId == SOURCE_2) {
        detector = std::make_shared<Source2CameraCoverDetector>();
    } else {
        VLOGW(TAG, "camera_cover detect warning for invalid source: %d", mRtConfig->sourceId);
        return;
    }
    detector->init(cfg);
    cameraCoverWindow.setSourceId(mRtConfig->sourceId);
}

void CameraCoverManager::deinit() {
    AbsVisionManager::deinit();
    if (detector != nullptr) {
        detector->deinit();
        detector = nullptr;
    }
}

void CameraCoverManager::setupSlidingWindow() {
    auto stageParas = std::make_shared<StageParameters>();
    for (int i = WINDOW_LOWER_FPS; i <= WINDOW_UPPER_FPS; ++i) {
        stageParas->stage_routine[i] = {static_cast<int>(round(i * DEFAULT_W_LENGTH_RATIO_1_0)), DEFAULT_W_DUTY_FACTOR};
    }
    cameraCoverWindow.set_fps_stage_parameters(stageParas);
}

bool CameraCoverManager::preDetect(VisionRequest *request, VisionResult *result) {
    VA_CHECK_DETECTED(ABILITY_CAMERA_COVER);
}

void CameraCoverManager::doDetect(VisionRequest *request, VisionResult *result) {
    VA_SET_DETECTED(ABILITY_CAMERA_COVER);

    // 判断如果之前(上一帧)有人脸，且之前滑窗是遮挡结果滑窗，则需要清空滑窗(此策略逻辑暂时注释掉保留，待后续性能优化可能放开)
    /* if (face_result->hasFace()) {
        if (face_result->faceInfos[0]->stateSource2CameraCover == ImageCoverStatus::F_IMAGE_COVER_BAD) {
            _image_cover_window.clear();
        }
        // 然后当前帧直接设置为无遮挡的单帧结果。同样单帧结果入滑窗(此举为减少每帧都进行遮挡滑窗判断)
        face_result->faceInfos[0]->source2CameraCoverSingle = ImageCoverStatus::F_IMAGE_COVER_GOOD;
        int triggered = _image_cover_window.update(false, nullptr);
        face_result->faceInfos[0]->stateSource2CameraCover = triggered ? ImageCoverStatus::F_IMAGE_COVER_BAD
                                                               : ImageCoverStatus::F_IMAGE_COVER_GOOD;
        return;
    } else {*/

    if (mRtConfig->sourceId == Source::SOURCE_2 &&
        checkFpsFixedDetect(FIXED_DETECT_DURATION, forceDetectCurFrame) == false) {
        VLOGD(TAG, "no need detect curFrame");
        return ;
    }

    detector->doDetect(request, result);

    // 根据当前摄像头的闪光灯类型获取不同的阈值，如果不需要打印，可注释掉
    if (mRtConfig->cameraLightType == CAMERA_LIGHT_TYPE_IR) {
        cameraCoverThreshold = mRtConfig->cameraCoverThresholdIr;
    } else {
        cameraCoverThreshold = mRtConfig->cameraCoverThresholdRgb;
    }
    auto *face = result->getFaceResult()->faceInfos[0];
    // 判断滑窗结果
    int triggered = cameraCoverWindow.update(face->stateCameraCoverSingle, nullptr);
    face->stateCameraCover = triggered ? ImageCoverStatus::F_IMAGE_COVER_BAD : ImageCoverStatus::F_IMAGE_COVER_GOOD;
    VLOGI(TAG, "camera_cover source=[%d] stateSingle=[%d], state=[%d], score=[%f], lightType[%f] threshold=[%f]",
          mRtConfig->sourceId, face->stateCameraCoverSingle, face->stateCameraCover, face->scoreCameraCoverSingle,
          mRtConfig->cameraLightType, cameraCoverThreshold);
}

REGISTER_VISION_MANAGER("CameraCoverManager", ABILITY_CAMERA_COVER, []() {
    return std::dynamic_pointer_cast<AbsVisionManager>(std::make_shared<CameraCoverManager>());
});

} // namespace aura::vision