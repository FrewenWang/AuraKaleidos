

#include "FaceEyeWakingManager.h"
#include "FaceLandmarkManager.h"
#include "vision/manager/VisionManagerRegistry.h"

namespace aura::vision {
FaceEyeWakingManager::FaceEyeWakingManager()
    : _wake_window(SOURCE_UNKNOWN, AbsVisionManager::DEFAULT_WINDOW_LENGTH,
                   AbsVisionManager::DEFAULT_TRIGGER_DUTY_FACTOR, AbsVisionManager::DEFAULT_END_DUTY_FACTOR,
                   F_EYE_WAKING) {
    setupSlidingWindow();
}

FaceEyeWakingManager::~FaceEyeWakingManager() {
    clear();
}

void FaceEyeWakingManager::init(RtConfig *cfg) {
    mRtConfig = cfg;
    AbsVisionManager::init(cfg);
    _wake_window.setSourceId(mRtConfig->sourceId);
}

void FaceEyeWakingManager::deinit() {
    AbsVisionManager::deinit();
}

void FaceEyeWakingManager::setupSlidingWindow() {
    auto stageParas = std::make_shared<StageParameters>();
    for (int i = WINDOW_LOWER_FPS; i <= WINDOW_UPPER_FPS; ++i) {
        stageParas->stage_routine[i] = {static_cast<int>(round(i * DEFAULT_W_LENGTH_RATIO_3_0)), DEFAULT_W_DUTY_FACTOR};
    }
    _wake_window.set_fps_stage_parameters(stageParas);
}

bool FaceEyeWakingManager::preDetect(VisionRequest *request, VisionResult *result) {
    VA_CHECK_DETECTED(ABILITY_FACE_EYE_WAKING);
}

void FaceEyeWakingManager::doDetect(VisionRequest *request, VisionResult *result) {
    VA_SET_DETECTED(ABILITY_FACE_EYE_WAKING);

    //    PERF_TICK(result->get_perf_util(), "face_eye_center");
    _fi = result->getFaceResult()->faceInfos[0];

    // 如果是跟踪延续而非当前帧检测出来的人脸则不执行下面策略
    if (_fi->faceType != FaceDetectType::F_TYPE_DETECT) {
        return;
    }

    // TODO 眼神追踪相关逻辑
    // FaceEyeHelper::instance()->detect(request, result);
    //    IovNcnnFaceEyeCenterDetector::instance()->single_face_detect(request->_width, request->_height, request->_frame,
    //                                                                 _fi, result->get_perf_util());
    //    PERF_TOCK(result->get_perf_util(), "face_eye_center");

    //    PERF_TICK(result->get_perf_util(), "face_eye_wake");

    // 获取眼球质点角度，检测质点失败则直接结束
    if (!detect_eye_angle(_fi)) {
        _wake_window.update(F_EYE_WAKING_NONE, nullptr);
        //        PERF_TOCK(result->get_perf_util(), "face_eye_wake");
        return;
    }

    // 获取人脸旋转角度
    detect_face_angle(_fi);
    _is_eye_wake = F_EYE_WAKING_NONE;

    if (mRtConfig->releaseMode > 0) {
        if (_eye_direction == _k_waking_eye_forward && _face_direction == _k_waking_face_forward) {
            _is_eye_wake = F_EYE_WAKING;
        } else if (_eye_direction == _k_waking_eye_left && _face_direction == _k_waking_face_right) {
            _is_eye_wake = F_EYE_WAKING;
        } else if (_eye_direction == _k_waking_eye_right && _face_direction == _k_waking_face_left) {
            _is_eye_wake = F_EYE_WAKING;
        }
    } else {
        if (_face_direction == _k_waking_face_forward) {
            _is_eye_wake = F_EYE_WAKING;
        }
    }

    // 增加滑窗滑窗判断
    bool ret = _wake_window.update(_is_eye_wake, nullptr);
    if (ret) {
        _fi->eyeWaking = F_EYE_WAKING;
    } else {
        _fi->eyeWaking = F_EYE_WAKING_NONE;
    }

    //    PERF_TOCK(result->get_perf_util(), "face_eye_wake");
}

/**
 * 获取眼球角度
 * @param fi
 */
bool FaceEyeWakingManager::detect_eye_angle(FaceInfo *fi) {
    bool has_left_eye = true;
    bool has_right_eye = true;
    if (fi->eyeCentroidLeft.x <= 0 && fi->eyeCentroidLeft.y <= 0) {
        has_left_eye = false;
    }

    if (fi->eyeCentroidRight.x <= 0 && fi->eyeCentroidRight.y <= 0) {
        has_right_eye = false;
    }

    if (!has_left_eye && !has_right_eye) {
        return false;
    }

    VPoint *landmark106_2d = fi->landmark2D106;
    _left_eye_dis = (landmark106_2d[FLM_54_L_EYE_RIGHT_CORNER].x - landmark106_2d[FLM_51_L_EYE_LEFT_CORNER].x) / 2;
    _right_eye_dis = (landmark106_2d[FLM_64_R_EYE_RIGHT_CORNER].x - landmark106_2d[FLM_61_R_EYE_LEFT_CORNER].x) / 2;

    if (has_left_eye && has_right_eye) {
        _eye_dis = (fi->eyeCentroidRight.x - landmark106_2d[FLM_70_R_EYE_CENTER].x + fi->eyeCentroidLeft.x
                    - landmark106_2d[FLM_60_L_EYE_CENTER].x)
                   / 2;
    } else if (has_left_eye) {
        _eye_dis = fi->eyeCentroidLeft.x - landmark106_2d[FLM_60_L_EYE_CENTER].x;
    } else {
        _eye_dis = fi->eyeCentroidRight.x - landmark106_2d[FLM_70_R_EYE_CENTER].x;
    }

    _eye_radius = (_left_eye_dis + _right_eye_dis) / 2;

    _eye_direction = _k_waking_eye_forward;
    if (_eye_dis < _eye_radius * mRtConfig->wakingEyeAngleThreshold
        && _eye_dis > -1 * (_eye_radius * mRtConfig->wakingEyeAngleThreshold)) {
        _eye_direction = _k_waking_eye_forward;
    } else if (_eye_dis > _eye_radius * mRtConfig->wakingEyeAngleThreshold) {
        _eye_direction = _k_waking_eye_left;
    } else {
        _eye_direction = _k_waking_eye_right;
    }

    return true;
}

/**
 * 获取头部方向
 * @param fi
 */
void FaceEyeWakingManager::detect_face_angle(FaceInfo *fi) {
    _yaw = fi->headDeflection.yaw;
    _face_direction = _k_waking_face_forward;
    if (_yaw < mRtConfig->wakingFaceAngle && _yaw > -1 * mRtConfig->wakingFaceAngle) {
        _face_direction = _k_waking_face_forward;
    } else if (_yaw <= -1 * mRtConfig->wakingFaceAngle) {
        _face_direction = _k_waking_face_left;
    } else {
        _face_direction = _k_waking_face_right;
    }
}

void FaceEyeWakingManager::clear() {
    _wake_window.clear();
}

void FaceEyeWakingManager::onNoFace(VisionRequest *request, VisionResult *result, FaceInfo *face) {
    if (face != nullptr) {
        face->eyeWaking = _wake_window.update(F_EYE_WAKING_NONE, nullptr);
    }
}

REGISTER_VISION_MANAGER("FaceEyeWakingManager", ABILITY_FACE_EYE_WAKING, []() {
    return std::dynamic_pointer_cast<AbsVisionManager>(std::make_shared<FaceEyeWakingManager>());
});

} // namespace aura::vision
