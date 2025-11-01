#include "FaceEyeCenterManager.h"

#include <algorithm>
#include <cmath>
#include "vision/util/log.h"
#include "util/SystemClock.h"
#include "vision/manager/VisionManagerRegistry.h"

namespace aura::vision {

static const char *TAG = "FaceEyeCenterManager";

FaceEyeCenterStrategy::FaceEyeCenterStrategy(RtConfig *cfg)
    : _left_eyelid_distance_window(cfg->sourceId, EYE_CENTER_WINDOW_LENGTH,
                                   AbsVisionManager::DEFAULT_TRIGGER_DUTY_FACTOR,
                                   AbsVisionManager::DEFAULT_END_DUTY_FACTOR, true),
      _right_eyelid_distance_window(cfg->sourceId, EYE_CENTER_WINDOW_LENGTH,
                                    AbsVisionManager::DEFAULT_TRIGGER_DUTY_FACTOR,
                                    AbsVisionManager::DEFAULT_END_DUTY_FACTOR, true),
      leftEyeCloseWindow(cfg->sourceId, EYE_CENTER_WINDOW_LENGTH, AbsVisionManager::DEFAULT_TRIGGER_DUTY_FACTOR,
                         AbsVisionManager::DEFAULT_END_DUTY_FACTOR, true),
      rightEyeCloseWindow(cfg->sourceId, EYE_CENTER_WINDOW_LENGTH, AbsVisionManager::DEFAULT_TRIGGER_DUTY_FACTOR,
                          AbsVisionManager::DEFAULT_END_DUTY_FACTOR, true),
      totalEyeCloseWindow(cfg->sourceId, EYE_CENTER_WINDOW_LENGTH, AbsVisionManager::DEFAULT_TRIGGER_DUTY_FACTOR,
                          AbsVisionManager::DEFAULT_END_DUTY_FACTOR, true) {
    this->rtConfig = cfg;
    setupSlidingWindow();
}

FaceEyeCenterStrategy::~FaceEyeCenterStrategy() {
    clear();
}

void FaceEyeCenterStrategy::clear() {
    _left_eyelid_distance_window.clear();
    _right_eyelid_distance_window.clear();

    // clear的时候除了清除滑窗。还要清除滑窗均值等数据
    _left_eyelid_close_distance_threshold = 5.f;
    _right_eyelid_close_distance_threshold = 5.f;
}

void FaceEyeCenterStrategy::setupSlidingWindow() {
    _left_eyelid_distance_window.set_trigger_expire_time(rtConfig->closeEyeTriggerExpireTime);
    _right_eyelid_distance_window.set_trigger_expire_time(rtConfig->closeEyeTriggerExpireTime);
    leftEyeCloseWindow.set_trigger_expire_time(rtConfig->closeEyeTriggerExpireTime);
    rightEyeCloseWindow.set_trigger_expire_time(rtConfig->closeEyeTriggerExpireTime);
    totalEyeCloseWindow.set_trigger_expire_time(rtConfig->closeEyeTriggerExpireTime);
}

void FaceEyeCenterStrategy::execute(FaceInfo *face) {
    if (face->eyeEyelidDistanceLeft == 0 && face->eyeEyelidDistanceRight == 0) {
        VLOGD(TAG, "no need execute eyelid open ratio");
        return;
    }
    if (executeV2) {
        // 使用V2版本的滑窗结果
        executeCloseEyeWindowV2(face);
    } else {
        executeCloseEyeWindowV1(face);
    }
}

void FaceEyeCenterStrategy::executeCloseEyeWindowV1(FaceInfo *face) {
    // left eye close state single frame
    float _left_eyelid_distance_mean = 0.f;
    float _left_eye_open_ratio = 0.f;
    if (face->eyeEyelidDistanceLeft > _left_eyelid_close_distance_threshold) {
        // 如果睁眼距离大于阈值。则认为是睁眼.更新睁闭眼滑窗结果
        _left_eyelid_distance_window.update(face->eyeEyelidDistanceLeft, nullptr);
        // 计算睁眼滑窗的平均值
        _left_eyelid_distance_mean = _left_eyelid_distance_window.get_window_mean_value();
        // 计算睁眼
        _left_eye_open_ratio = face->eyeEyelidDistanceLeft / _left_eyelid_distance_mean;
        // 左眼睁闭眼距离的阈值 = 滑窗的平均值 * 80%
        // _left_eyelid_close_distance_threshold = _left_eyelid_distance_mean * 0.8;
        // 如果阈值设置为平均距离的阈值的80%。 会导致这个平均值会一直被拉低。所以我们设置阈值就是为平均值
        // 风险：如果前期开合比较大，后续可能一直都无法更新滑窗，并且开合度一直比较低。
        _left_eyelid_close_distance_threshold = _left_eyelid_distance_mean;
        // 如果睁眼的比例大于比例大约0.8。则认为就是睁眼。否则则认为是闭眼
        face->eyeEyelidStatusLeft = _left_eye_open_ratio > 0.8 ? FaceEyeStatus::F_EYE_OPEN : FaceEyeStatus::F_EYE_CLOSE;
    } else {
        face->eyeEyelidStatusLeft = FaceEyeStatus::F_EYE_CLOSE;
        // 计算睁眼滑窗的平均值
        _left_eyelid_distance_mean = _left_eyelid_distance_window.get_window_mean_value();
        if (fabs(_left_eyelid_distance_mean) < 1e-6) {
            // 当距离滑窗无数据时，默认开和度为0。
            // 当检出眼部开合距离一直小于阈值，致使滑窗一直无法添加数据，此时应默认开合度为0
            _left_eye_open_ratio = 0.f;
        } else {
            _left_eye_open_ratio = face->eyeEyelidDistanceLeft / _left_eyelid_distance_mean;
        }
    }
    face->eyeEyelidOpeningDistanceMeanLeft = _left_eyelid_distance_mean;
    face->eyeEyelidOpenRatioLeft = _left_eye_open_ratio;

    // right eye close state single frame
    float _right_eyelid_distance_mean = 0.f;
    float _right_eye_open_ratio = 0.f;
    if (face->eyeEyelidDistanceRight > _right_eyelid_close_distance_threshold) {
        // 如果睁眼距离大于阈值
        _right_eyelid_distance_window.update(face->eyeEyelidDistanceRight, nullptr);
        _right_eyelid_distance_mean = _right_eyelid_distance_window.get_window_mean_value();
        _right_eye_open_ratio = face->eyeEyelidDistanceRight / _right_eyelid_distance_mean;
        // 睁闭眼距离的阈值 = 滑窗的平均值 * 80%
        // 如果阈值设置为平均距离的阈值的80%。 会导致这个平均值会一直被拉低。所以我们设置阈值就是为平均值
        // 风险：如果前期开合比较大，后续可能一直都无法更新滑窗，并且开合度一直比较低。
        // _right_eyelid_close_distance_threshold = _right_eyelid_distance_mean * 0.8;
        _right_eyelid_close_distance_threshold = _right_eyelid_distance_mean;
        face->eyeEyelidStatusRight =
                _right_eye_open_ratio > 0.8 ? FaceEyeStatus::F_EYE_OPEN : FaceEyeStatus::F_EYE_CLOSE;
    } else {
        face->eyeEyelidStatusRight = FaceEyeStatus::F_EYE_CLOSE;
        _right_eyelid_distance_mean = _right_eyelid_distance_window.get_window_mean_value();
        if (fabs(_right_eyelid_distance_mean) < 1e-6) {
            // 当距离滑窗无数据时，默认开和度为0。
            // 当检出眼部开合距离一直小于阈值，致使滑窗一直无法添加数据，此时应默认开合度为0
            _right_eye_open_ratio = 0.f;
        } else {
            _right_eye_open_ratio = face->eyeEyelidDistanceRight / _right_eyelid_distance_mean;
        }
    }
    face->eyeEyelidOpeningDistanceMeanRight = _right_eyelid_distance_mean;
    face->eyeEyelidOpenRatioRight = _right_eye_open_ratio;

    VLOGI(TAG,
          "eye_center[%ld] left_eye_status=[%d], eyelid_distance=[%f],eyelid_distance_mean=[%f], "
          "threshold=[%f], open_ratio=[%f] \n "
          "right_eye_status=[%d], eyelid_distance=[%f],eyelid_distance_mean=[%f], "
          "threshold=[%f], open_ratio=[%f]",
          face->id, face->leftEyeDetectSingle, face->eyeEyelidDistanceLeft, _left_eyelid_distance_mean,
          _left_eyelid_close_distance_threshold, _left_eye_open_ratio, face->rightEyeDetectSingle,
          face->eyeEyelidDistanceRight, _right_eyelid_distance_mean, _right_eyelid_close_distance_threshold,
          _right_eye_open_ratio);

    // both eyes close state single frame
    if (face->eyeEyelidStatusRight == FaceEyeStatus::F_EYE_CLOSE
        && face->eyeEyelidStatusLeft == FaceEyeStatus::F_EYE_CLOSE) {
        face->eyeState = FaceEyeStatus::F_EYE_CLOSE;
    } else {
        face->eyeState = FaceEyeStatus::F_EYE_OPEN;
    }
}
void FaceEyeCenterStrategy::executeCloseEyeWindowV2(FaceInfo *face) {
    // left eye close state single frame
    float _left_eyelid_distance_mean = 0.f;
    float _left_eye_open_ratio = 0.f;
    if (face->eyeEyelidDistanceLeft > _left_eyelid_close_distance_threshold) {
        // 如果睁眼距离大于阈值。则认为是睁眼.更新睁闭眼滑窗结果
        _left_eyelid_distance_window.update(face->eyeEyelidDistanceLeft, nullptr);
        // 计算睁眼滑窗的平均值
        _left_eyelid_distance_mean = _left_eyelid_distance_window.get_window_mean_value();
        // 计算睁眼
        _left_eye_open_ratio = face->eyeEyelidDistanceLeft / _left_eyelid_distance_mean;
        // 左眼睁闭眼距离的阈值 = 滑窗的平均值 * 80%
        // _left_eyelid_close_distance_threshold = _left_eyelid_distance_mean * 0.8;
        // 如果阈值设置为平均距离的阈值的80%。 会导致这个平均值会一直被拉低。所以我们设置阈值就是为平均值
        // 风险：如果前期开合比较大，后续可能一直都无法更新滑窗，并且开合度一直比较低。
        _left_eyelid_close_distance_threshold = _left_eyelid_distance_mean;
        // 如果睁眼的比例大于比例大约0.8。则认为就是睁眼。否则则认为是闭眼
        face->eyeEyelidStatusLeft = _left_eye_open_ratio > 0.8 ? FaceEyeStatus::F_EYE_OPEN : FaceEyeStatus::F_EYE_CLOSE;
    } else {
        face->eyeEyelidStatusLeft = FaceEyeStatus::F_EYE_CLOSE;
        // 计算睁眼滑窗的平均值
        _left_eyelid_distance_mean = _left_eyelid_distance_window.get_window_mean_value();
        if (fabs(_left_eyelid_distance_mean) < 1e-6) {
            // 当距离滑窗无数据时，默认开和度为0。
            // 当检出眼部开合距离一直小于阈值，致使滑窗一直无法添加数据，此时应默认开合度为0
            _left_eye_open_ratio = 0.f;
        } else {
            _left_eye_open_ratio = face->eyeEyelidDistanceLeft / _left_eyelid_distance_mean;
        }
    }
    face->eyeEyelidOpeningDistanceMeanLeft = _left_eyelid_distance_mean;
    face->eyeEyelidOpenRatioLeft = _left_eye_open_ratio;

    // right eye close state single frame
    float _right_eyelid_distance_mean = 0.f;
    float _right_eye_open_ratio = 0.f;
    if (face->eyeEyelidDistanceRight > _right_eyelid_close_distance_threshold) {
        // 如果睁眼距离大于阈值
        _right_eyelid_distance_window.update(face->eyeEyelidDistanceRight, nullptr);
        _right_eyelid_distance_mean = _right_eyelid_distance_window.get_window_mean_value();
        _right_eye_open_ratio = face->eyeEyelidDistanceRight / _right_eyelid_distance_mean;
        // 睁闭眼距离的阈值 = 滑窗的平均值 * 80%
        // 如果阈值设置为平均距离的阈值的80%。 会导致这个平均值会一直被拉低。所以我们设置阈值就是为平均值
        // 风险：如果前期开合比较大，后续可能一直都无法更新滑窗，并且开合度一直比较低。
        // _right_eyelid_close_distance_threshold = _right_eyelid_distance_mean * 0.8;
        _right_eyelid_close_distance_threshold = _right_eyelid_distance_mean;
    } else {
        _right_eyelid_distance_mean = _right_eyelid_distance_window.get_window_mean_value();
        if (fabs(_right_eyelid_distance_mean) < 1e-6) {
            // 当距离滑窗无数据时，默认开和度为0。
            // 当检出眼部开合距离一直小于阈值，致使滑窗一直无法添加数据，此时应默认开合度为0
            _right_eye_open_ratio = 0.f;
        } else {
            _right_eye_open_ratio = face->eyeEyelidDistanceRight / _right_eyelid_distance_mean;
        }
    }
    face->eyeEyelidOpeningDistanceMeanRight = _right_eyelid_distance_mean;
    face->eyeEyelidOpenRatioRight = _right_eye_open_ratio;
    localLeftEyeCloseSingle = false;
    localRightEyeCloseSingle = false;
    localEyeCloseSingle = false;
    // 如果左眼检测不到眼睛。但是右眼检测到闭眼
    if (face->leftEyeDetectSingle == FaceEyeDetectStatus::EYE_UNAVAILABLE
        && face->rightEyeDetectSingle == FaceEyeDetectStatus::PUPIL_UNAVAILABLE) {
        localRightEyeCloseSingle = true;
    }
    if (face->rightEyeDetectSingle == FaceEyeDetectStatus::EYE_UNAVAILABLE
        && face->leftEyeDetectSingle == FaceEyeDetectStatus::PUPIL_UNAVAILABLE) {
        localLeftEyeCloseSingle = true;
    }
    // 根据算法策略计算出来的单帧睁闭眼的结果
    if ((face->leftEyeDetectSingle == FaceEyeDetectStatus::PUPIL_UNAVAILABLE
         && face->rightEyeDetectSingle == FaceEyeDetectStatus::PUPIL_UNAVAILABLE)
        || localLeftEyeCloseSingle || localRightEyeCloseSingle) {
        localEyeCloseSingle = true;
    }

    // 检测到眼睛，但是未检测到瞳孔，就算闭眼。如果未检测到眼睛也不算鼻咽。避免闭眼过扰
    auto leftEyeClose = face->rightEyeDetectSingle == FaceEyeDetectStatus::PUPIL_UNAVAILABLE;
    if (leftEyeCloseWindow.update(leftEyeClose, nullptr)) {
        face->eyeEyelidStatusLeft = FaceEyeStatus::F_EYE_CLOSE;
    } else {
        face->eyeEyelidStatusLeft = FaceEyeStatus::F_EYE_OPEN;
    }
    // 检测到眼睛，但是未检测到瞳孔，就算闭眼。如果未检测到眼睛也不算鼻咽。避免闭眼过扰
    auto rightEyeClose = face->rightEyeDetectSingle != FaceEyeDetectStatus::PUPIL_AVAILABLE;
    if (rightEyeCloseWindow.update(rightEyeClose, nullptr)) {
        face->eyeEyelidStatusRight = FaceEyeStatus::F_EYE_CLOSE;
    } else {
        face->eyeEyelidStatusRight = FaceEyeStatus::F_EYE_OPEN;
    }

    if (totalEyeCloseWindow.update(localEyeCloseSingle, nullptr)) {
        face->eyeState = FaceEyeStatus::F_EYE_CLOSE;
    } else {
        face->eyeState = FaceEyeStatus::F_EYE_OPEN;
    }

    VLOGI(TAG,
          "eye_center[%ld] left_eye_status=[%d], eyelid_distance=[%f],eyelid_distance_mean=[%f], "
          "threshold=[%f], open_ratio=[%f] \n "
          "right_eye_status=[%d], eyelid_distance=[%f],eyelid_distance_mean=[%f], "
          "threshold=[%f], open_ratio=[%f] eyeState=[%f]",
          face->id, face->leftEyeDetectSingle, face->eyeEyelidDistanceLeft, _left_eyelid_distance_mean,
          _left_eyelid_close_distance_threshold, _left_eye_open_ratio, face->rightEyeDetectSingle,
          face->eyeEyelidDistanceRight, _right_eyelid_distance_mean, _right_eyelid_close_distance_threshold,
          _right_eye_open_ratio, face->eyeState);
}

void FaceEyeCenterStrategy::onNoFace(VisionRequest *request, FaceInfo *face) {
    if (executeV2) {
         onNoFaceV2(request,face);
    } else {
        onNoFaceV1(request,face);
    }
}
void FaceEyeCenterStrategy::onNoFaceV1(VisionRequest *request, FaceInfo *face) {
    // left eye close state single frame
    float _left_eyelid_distance_mean = 0.f;
    float _left_eye_open_ratio = 0.f;
    face->eyeEyelidStatusLeft = FaceEyeStatus::F_EYE_CLOSE;
    // 计算睁眼滑窗的平均值
    _left_eyelid_distance_mean = _left_eyelid_distance_window.get_window_mean_value();
    if (fabs(_left_eyelid_distance_mean) < 1e-6) {
        // 当距离滑窗无数据时，默认开和度为0。
        // 当检出眼部开合距离一直小于阈值，致使滑窗一直无法添加数据，此时应默认开合度为0
        _left_eye_open_ratio = 0.f;
    } else {
        _left_eye_open_ratio = face->eyeEyelidDistanceLeft / _left_eyelid_distance_mean;
    }
    face->eyeEyelidOpeningDistanceMeanLeft = _left_eyelid_distance_mean;
    face->eyeEyelidOpenRatioLeft = _left_eye_open_ratio;
    VLOGI(TAG, "[when no face] eye_center[%ld] left_eye_status=[%d], eyelid_distance=[%f],eyelid_distance_mean=[%f], "
               "threshold=[%f], open_ratio=[%f]", face->id, face->leftEyeDetectSingle,
          face->eyeEyelidDistanceLeft, _left_eyelid_distance_mean, _left_eyelid_close_distance_threshold,
          _left_eye_open_ratio);

    // right eye close state single frame
    float _right_eyelid_distance_mean = 0.f;
    float _right_eye_open_ratio = 0.f;
    face->eyeEyelidStatusRight = FaceEyeStatus::F_EYE_CLOSE;
    _right_eyelid_distance_mean = _right_eyelid_distance_window.get_window_mean_value();
    if (fabs(_right_eyelid_distance_mean) < 1e-6) {
        // 当距离滑窗无数据时，默认开和度为0。
        // 当检出眼部开合距离一直小于阈值，致使滑窗一直无法添加数据，此时应默认开合度为0
        _right_eye_open_ratio = 0.f;
    } else {
        _right_eye_open_ratio = face->eyeEyelidDistanceRight / _right_eyelid_distance_mean;
    }
    face->eyeEyelidOpeningDistanceMeanRight = _right_eyelid_distance_mean;
    face->eyeEyelidOpenRatioRight = _right_eye_open_ratio;
    VLOGI(TAG, "[when no face] eye_center[%ld] right_eye_status=[%d], eyelid_distance=[%f],eyelid_distance_mean=[%f], "
               "threshold=[%f], open_ratio=[%f]", face->id, face->rightEyeDetectSingle,
          face->eyeEyelidDistanceRight, _right_eyelid_distance_mean, _right_eyelid_close_distance_threshold,
          _right_eye_open_ratio);

    // both eyes close state single frame
    if (face->eyeEyelidStatusRight == FaceEyeStatus::F_EYE_CLOSE &&
        face->eyeEyelidStatusLeft == FaceEyeStatus::F_EYE_CLOSE) {
        face->eyeState = FaceEyeStatus::F_EYE_CLOSE;
    } else {
        face->eyeState = FaceEyeStatus::F_EYE_OPEN;
    }
}
void FaceEyeCenterStrategy::onNoFaceV2(VisionRequest *request, FaceInfo *face) {
    // left eye close state single frame
    float _left_eyelid_distance_mean = 0.f;
    float _left_eye_open_ratio = 0.f;
    face->eyeEyelidStatusLeft = FaceEyeStatus::F_EYE_CLOSE;
    // 计算睁眼滑窗的平均值
    _left_eyelid_distance_mean = _left_eyelid_distance_window.get_window_mean_value();
    if (fabs(_left_eyelid_distance_mean) < 1e-6) {
        // 当距离滑窗无数据时，默认开和度为0。
        // 当检出眼部开合距离一直小于阈值，致使滑窗一直无法添加数据，此时应默认开合度为0
        _left_eye_open_ratio = 0.f;
    } else {
        _left_eye_open_ratio = face->eyeEyelidDistanceLeft / _left_eyelid_distance_mean;
    }
    face->eyeEyelidOpeningDistanceMeanLeft = _left_eyelid_distance_mean;
    face->eyeEyelidOpenRatioLeft = _left_eye_open_ratio;
    VLOGI(TAG,
          "[when no face] eye_center[%ld] left_eye_status=[%d], eyelid_distance=[%f],eyelid_distance_mean=[%f], "
          "threshold=[%f], open_ratio=[%f]",
          face->id, face->leftEyeDetectSingle, face->eyeEyelidDistanceLeft, _left_eyelid_distance_mean,
          _left_eyelid_close_distance_threshold, _left_eye_open_ratio);

    // right eye close state single frame
    float _right_eyelid_distance_mean = 0.f;
    float _right_eye_open_ratio = 0.f;
    face->eyeEyelidStatusRight = FaceEyeStatus::F_EYE_CLOSE;
    _right_eyelid_distance_mean = _right_eyelid_distance_window.get_window_mean_value();
    if (fabs(_right_eyelid_distance_mean) < 1e-6) {
        // 当距离滑窗无数据时，默认开和度为0。
        // 当检出眼部开合距离一直小于阈值，致使滑窗一直无法添加数据，此时应默认开合度为0
        _right_eye_open_ratio = 0.f;
    } else {
        _right_eye_open_ratio = face->eyeEyelidDistanceRight / _right_eyelid_distance_mean;
    }
    face->eyeEyelidOpeningDistanceMeanRight = _right_eyelid_distance_mean;
    face->eyeEyelidOpenRatioRight = _right_eye_open_ratio;
    VLOGI(TAG,
          "[when no face] eye_center[%ld] right_eye_status=[%d], eyelid_distance=[%f],eyelid_distance_mean=[%f], "
          "threshold=[%f], open_ratio=[%f]",
          face->id, face->rightEyeDetectSingle, face->eyeEyelidDistanceRight, _right_eyelid_distance_mean,
          _right_eyelid_close_distance_threshold, _right_eye_open_ratio);

    if (rightEyeCloseWindow.update(false, nullptr)) {
        face->eyeEyelidStatusRight = FaceEyeStatus::F_EYE_CLOSE;
    } else {
        face->eyeEyelidStatusRight = FaceEyeStatus::F_EYE_OPEN;
    }

    if (leftEyeCloseWindow.update(false, nullptr)) {
        face->eyeEyelidStatusLeft = FaceEyeStatus::F_EYE_CLOSE;
    } else {
        face->eyeEyelidStatusLeft = FaceEyeStatus::F_EYE_OPEN;
    }

    if (totalEyeCloseWindow.update(false, nullptr)) {
        face->eyeState = FaceEyeStatus::F_EYE_CLOSE;
    } else {
        face->eyeState = FaceEyeStatus::F_EYE_OPEN;
    }
}

FaceEyeCenterManager::FaceEyeCenterManager() {
    _detector = std::make_shared<FaceEyeCenterDetector>();
}

FaceEyeCenterManager::~FaceEyeCenterManager() {
    clear();
}

void FaceEyeCenterManager::init(RtConfig *cfg) {
    mRtConfig = cfg;
    _detector->init(cfg);
}

void FaceEyeCenterManager::deinit() {
    AbsVisionManager::deinit();
    if (_detector != nullptr) {
        _detector->deinit();
        _detector = nullptr;
    }
}

bool FaceEyeCenterManager::preDetect(VisionRequest *request, VisionResult *result) {
    VA_CHECK_DETECTED(ABILITY_FACE_EYE_CENTER);
}

void FaceEyeCenterManager::doDetect(VisionRequest *request, VisionResult *result) {
    VA_SET_DETECTED(ABILITY_FACE_EYE_CENTER);

    _detector->detect(request, result);
    // 执行多人练策略
    execute_face_strategy<FaceEyeCenterStrategy>(result, _eye_center_map, mRtConfig);
}

void FaceEyeCenterManager::clear() {
    for (auto &info: _eye_center_map) {
        if (info.second) {
            info.second->clear();
            FaceEyeCenterStrategy::recycle(info.second);
        }
    }
    _eye_center_map.clear();
}

void FaceEyeCenterManager::onNoFace(VisionRequest *request, VisionResult *result, FaceInfo *face) {
    if (face != nullptr && (!_eye_center_map.empty())) {
        auto iter = _eye_center_map.find(face->id);
        if (iter != _eye_center_map.end()) {
            iter->second->onNoFace(request, face);
        }
    }
}

REGISTER_VISION_MANAGER("FaceEyeCenterManager", ABILITY_FACE_EYE_CENTER, []() {
    return std::dynamic_pointer_cast<AbsVisionManager>(std::make_shared<FaceEyeCenterManager>());
});

} // namespace aura::vision
