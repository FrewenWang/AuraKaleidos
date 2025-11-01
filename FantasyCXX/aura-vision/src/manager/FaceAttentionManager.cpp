
#include "FaceAttentionManager.h"

#include <cmath>

#include "FaceLandmarkManager.h"
#include "vision/util/log.h"
#include "vision/manager/VisionManagerRegistry.h"

namespace aura::vision {

static const char *TAG = "FaceAttentionManager";

FaceAttentionStrategy::FaceAttentionStrategy(RtConfig *cfg)
    : _left_window(cfg->sourceId, AbsVisionManager::DEFAULT_WINDOW_LENGTH,
                   AbsVisionManager::DEFAULT_TRIGGER_DUTY_FACTOR, AbsVisionManager::DEFAULT_END_DUTY_FACTOR,
                   F_ATTENTION_LOOK_LEFT),
      _right_window(cfg->sourceId, AbsVisionManager::DEFAULT_WINDOW_LENGTH,
                    AbsVisionManager::DEFAULT_TRIGGER_DUTY_FACTOR, AbsVisionManager::DEFAULT_END_DUTY_FACTOR,
                    F_ATTENTION_LOOK_RIGHT),
      _up_window(cfg->sourceId, AbsVisionManager::DEFAULT_WINDOW_LENGTH, AbsVisionManager::DEFAULT_TRIGGER_DUTY_FACTOR,
                 AbsVisionManager::DEFAULT_END_DUTY_FACTOR, F_ATTENTION_LOOK_UP),
      _down_window(cfg->sourceId, AbsVisionManager::DEFAULT_WINDOW_LENGTH,
                   AbsVisionManager::DEFAULT_TRIGGER_DUTY_FACTOR, AbsVisionManager::DEFAULT_END_DUTY_FACTOR,
                   F_ATTENTION_LOOK_DOWN) {
    this->rtConfig = cfg;
    setupSlidingWindow();
    // 初始化Camera镜像反转的标志变量
    cameraMirror = V_F_TO_BOOL(cfg->cameraImageMirror);
}

FaceAttentionStrategy::~FaceAttentionStrategy() {
    clear();
}

void FaceAttentionStrategy::clear() {
    _left_window.clear();
    _right_window.clear();
    _up_window.clear();
    _down_window.clear();
}

void FaceAttentionStrategy::setupSlidingWindow() {
    auto stageParas = std::make_shared<StageParameters>();
    for (int i = AbsVisionManager::WINDOW_LOWER_FPS; i <= AbsVisionManager::WINDOW_UPPER_FPS; ++i) {
        stageParas->stage_routine[i] = {static_cast<int>(round(i * AbsVisionManager::DEFAULT_W_LENGTH_RATIO_1_5)),
                                        AbsVisionManager::DEFAULT_W_DUTY_FACTOR};
    }

    _left_window.set_trigger_need_time((long)rtConfig->attentionTriggerNeedTime);
    _left_window.set_trigger_expire_time((long)rtConfig->attentionTriggerExpireTime);
    _right_window.set_trigger_need_time((long)rtConfig->attentionTriggerNeedTime);
    _right_window.set_trigger_expire_time((long)rtConfig->attentionTriggerExpireTime);
    _up_window.set_trigger_need_time((long)rtConfig->attentionTriggerNeedTime);
    _up_window.set_trigger_expire_time((long)rtConfig->attentionTriggerExpireTime);
    _down_window.set_trigger_need_time((long)rtConfig->attentionTriggerNeedTime);
    _down_window.set_trigger_expire_time((long)rtConfig->attentionTriggerExpireTime);

    _left_window.set_fps_stage_parameters(stageParas);
    _right_window.set_fps_stage_parameters(stageParas);
    _up_window.set_fps_stage_parameters(stageParas);
    _down_window.set_fps_stage_parameters(stageParas);
}

VAngle FaceAttentionStrategy::calibrate_head_pose(FaceInfo *face_info) {
    auto _landmark_2d_106 = face_info->landmark2D106;
    // 获取左右眼睛瞳孔的距离.
    float pupil_w = _landmark_2d_106[FLM_69_R_EYE_PUPIL].x - _landmark_2d_106[FLM_59_L_EYE_PUPIL].x;
    float pupil_h = _landmark_2d_106[FLM_69_R_EYE_PUPIL].y - _landmark_2d_106[FLM_59_L_EYE_PUPIL].y;
    // 计算左右眼睛瞳孔之间的空间距离
    float pupil_length = fabs(sqrt(pupil_w * pupil_w + pupil_h * pupil_h));

    // 摄像头感光元器件的像素水平长度: 摄像头焦距/(ccd_width/frame_width)
    float ccd_focus_length_pixel_x = rtConfig->cameraFocalLength / (rtConfig->cameraCcdWidth / rtConfig->frameWidth);
    // 垂直 Z 轴的物理距离
    // 前置条件，瞳孔件距离是60(现实场景经验数据)
    // 计算人脸和摄像头交集平面的距离
    float face_dist =
            (60.f / cos(face_info->headDeflection.yaw / ANGEL_180 * M_PI) * ccd_focus_length_pixel_x) / pupil_length;
    // 根据成像来计算人脸的宽度和高度
    float face_w_ccd = (rtConfig->frameWidth / 2.0 - _landmark_2d_106[FLM_73_NOSE_BRIDGE3].x) * rtConfig->cameraCcdWidth
                       / rtConfig->frameWidth;
    float face_h_ccd = (rtConfig->frameWidth / 2.0 - _landmark_2d_106[FLM_73_NOSE_BRIDGE3].y) * rtConfig->cameraCcdWidth
                       / rtConfig->frameWidth;
    // 人与主光轴的距离
    float face_z_x = face_dist * face_w_ccd / rtConfig->cameraFocalLength;
    float face_z_y = face_dist * face_h_ccd / rtConfig->cameraFocalLength;
    // 正前方的偏转角
    float target_pt_yaw = atan2(rtConfig->cameraPositionX + face_z_x, face_dist);
    float target_pt_pitch = atan2(rtConfig->cameraPositionY + face_z_y, face_dist);

    VAngle angle;

    // 根据Camera硬件是否镜像来针对不同的计算人脸角度的yaw角度
    if (cameraMirror) {
        angle.yaw = RADIAN_2_ANGLE_FACTOR * target_pt_yaw + face_info->headDeflection.yaw;
    } else {
        // 如果是镜像反转的摄像头，则需要跟pitch一样进行取反
        angle.yaw = RADIAN_2_ANGLE_FACTOR * target_pt_yaw - face_info->headDeflection.yaw;
    }
    angle.pitch = RADIAN_2_ANGLE_FACTOR * target_pt_pitch + face_info->headDeflection.pitch;
    angle.roll = 0.0F;

    return angle;
}

int FaceAttentionStrategy::get_head_state(VAngle &angle) {
    if (angle.pitch < rtConfig->attentionHeadUpAngle && angle.pitch > rtConfig->attentionHeadDownAngle
        && angle.yaw < rtConfig->attentionHeadLeftAngle && angle.yaw > rtConfig->attentionHeadRightAngle) {
        return F_ATTENTION_LOOK_FORWARD;
    }

    if (angle.yaw > rtConfig->attentionHeadLeftAngle) { // 左转角度是正
        return F_ATTENTION_LOOK_LEFT;
    } else if (angle.yaw < rtConfig->attentionHeadRightAngle) { // 右转角度是负数
        return F_ATTENTION_LOOK_RIGHT;
    } else if (angle.pitch > rtConfig->attentionHeadUpAngle) { // 向上角度是正数
        return F_ATTENTION_LOOK_UP;
    } else if (angle.pitch < rtConfig->attentionHeadDownAngle) { // 向下角度是负数
        return F_ATTENTION_LOOK_DOWN;
    }
    return F_ATTENTION_NONE;
}

void FaceAttentionStrategy::execute(FaceInfo *face) {
    // 修正头部姿态角 && 判断当前帧的头部方向。根据摄像头位置以及硬件参数进行人脸角度矫正
    auto faceAngle = calibrate_head_pose(face);
    int head_status = get_head_state(faceAngle);

    bool is_left_look = static_cast<bool>(_left_window.update(head_status, &face->_left_attention_state));
    bool is_right_look = static_cast<bool>(_right_window.update(head_status, &face->_right_attention_state));
    bool is_up_look = static_cast<bool>(_up_window.update(head_status, &face->_up_attention_state));
    bool is_down_look = static_cast<bool>(_down_window.update(head_status, &face->_down_attention_state));

    if (is_left_look) {
        face->stateAttention = F_ATTENTION_LOOK_LEFT;
    } else if (is_right_look) {
        face->stateAttention = F_ATTENTION_LOOK_RIGHT;
    } else if (is_up_look) {
        face->stateAttention = F_ATTENTION_LOOK_UP;
    } else if (is_down_look) {
        face->stateAttention = F_ATTENTION_LOOK_DOWN;
    } else {
        face->stateAttention = F_ATTENTION_LOOK_FORWARD;
    }
    VLOGI(TAG, "face_attention[%ld] headStatusSingle=[%d],state=[%f],faceAngle=[%f,%f,%f]", face->id, head_status,
          face->stateAttention, faceAngle.yaw, faceAngle.pitch, faceAngle.roll);
}

void FaceAttentionStrategy::onNoFace(VisionRequest *request, FaceInfo *face) {
    bool is_left_look = static_cast<bool>(_left_window.update(F_ATTENTION_NONE, &face->_left_attention_state));
    bool is_right_look = static_cast<bool>(_right_window.update(F_ATTENTION_NONE, &face->_right_attention_state));
    bool is_up_look = static_cast<bool>(_up_window.update(F_ATTENTION_NONE, &face->_up_attention_state));
    bool is_down_look = static_cast<bool>(_down_window.update(F_ATTENTION_NONE, &face->_down_attention_state));

    if (is_left_look) {
        face->stateAttention = F_ATTENTION_LOOK_LEFT;
    } else if (is_right_look) {
        face->stateAttention = F_ATTENTION_LOOK_RIGHT;
    } else if (is_up_look) {
        face->stateAttention = F_ATTENTION_LOOK_UP;
    } else if (is_down_look) {
        face->stateAttention = F_ATTENTION_LOOK_DOWN;
    } else {
        face->stateAttention = F_ATTENTION_LOOK_FORWARD;
    }
    VLOGI(TAG, "[when no face] face_attention[%ld] headStatusSingle=[%d],state=[%f],faceAngle=[%f,%f,%f]", face->id,
          F_ATTENTION_NONE, face->stateAttention, 0, 0, 0);
}

void FaceAttentionStrategy::onConfigUpdated(int key, float value) {
    switch (key) {
        case STRATEGY_TRIGGER_NEED_TIME_ATTENTION:
            _left_window.set_trigger_need_time((long)rtConfig->attentionTriggerNeedTime);
            _right_window.set_trigger_need_time((long)rtConfig->attentionTriggerNeedTime);
            _up_window.set_trigger_need_time((long)rtConfig->attentionTriggerNeedTime);
            _down_window.set_trigger_need_time((long)rtConfig->attentionTriggerNeedTime);
            break;
        case STRATEGY_TRIGGER_EXPIRE_TIME_ATTENTION:
            _left_window.set_trigger_expire_time((long)rtConfig->attentionTriggerNeedTime);
            _right_window.set_trigger_expire_time((long)rtConfig->attentionTriggerNeedTime);
            _up_window.set_trigger_expire_time((long)rtConfig->attentionTriggerNeedTime);
            _down_window.set_trigger_expire_time((long)rtConfig->attentionTriggerNeedTime);
            break;
        default:
            break;
    }
}

FaceAttentionManager::FaceAttentionManager() {}

FaceAttentionManager::~FaceAttentionManager() {
    clear();
}

void FaceAttentionManager::clear() {
    for (auto &info : _attention_strategy_map) {
        if (info.second) {
            info.second->clear();
            FaceAttentionStrategy::recycle(info.second);
        }
    }
    _attention_strategy_map.clear();
}

void FaceAttentionManager::deinit() {
    AbsVisionManager::deinit();
}

void FaceAttentionManager::onNoFace(VisionRequest *request, VisionResult *result, FaceInfo *face) {
    if (face != nullptr && (!_attention_strategy_map.empty())) {
        auto iter = _attention_strategy_map.find(face->id);
        if (iter != _attention_strategy_map.end()) {
            iter->second->onNoFace(request, face);
        }
    }
}

bool FaceAttentionManager::preDetect(VisionRequest *request, VisionResult *result) {
    return !result->isAbilityExec(ABILITY_FACE_ATTENTION);
}

void FaceAttentionManager::doDetect(VisionRequest *request, VisionResult *result) {
    result->setAbilityExec(ABILITY_FACE_ATTENTION);
    // 执行多人练策略
    execute_face_strategy<FaceAttentionStrategy>(result, _attention_strategy_map, mRtConfig);
}

void FaceAttentionManager::onConfigUpdated(int key, float value) {
    for (auto &info : _attention_strategy_map) {
        info.second->onConfigUpdated(key, value);
    }
}

// NOLINTNEXTLINE
REGISTER_VISION_MANAGER("FaceAttentionManager", ABILITY_FACE_ATTENTION, []() {
    return std::dynamic_pointer_cast<AbsVisionManager>(std::make_shared<FaceAttentionManager>());
});

} // namespace vision
