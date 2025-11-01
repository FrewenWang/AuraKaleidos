

#include "FaceSmileManager.h"

#include <math.h>
#include <numeric>
#include "util/math_utils.h"
#include "vision/manager/VisionManagerRegistry.h"

namespace aura::vision {
FaceSmileStrategy::FaceSmileStrategy(RtConfig* cfg) {
    this->rtConfig = cfg;
}

FaceSmileStrategy::~FaceSmileStrategy() {
    clear();
}

void FaceSmileStrategy::clear() {
    _mouth_arr.clear();
}

void FaceSmileStrategy::execute(FaceInfo* face) {
    // 计算嘴角上扬的位置
    vision::SimpleSize mouth = calculate_mouth_size(face);

    float mouth_value = 0;
    if (mouth.width != 0) {
        mouth_value = mouth.height / mouth.width;
    }

    // 判断是否微笑
    bool smile = face_smile_detect(face, mouth_value);
    face->stateEmotion = smile ? F_ATTR_EMOTION_HAPPY : F_ATTR_EMOTION_OTHER;
}

void FaceSmileStrategy::onNoFace(VisionRequest* request, FaceInfo* face) {
    // 嘴部高宽比
    float mouth_value = 0;
    // 判断是否微笑
    bool smile = face_smile_detect(face, mouth_value);
    face->stateEmotion = smile ? F_ATTR_EMOTION_HAPPY : F_ATTR_EMOTION_OTHER;
}

bool FaceSmileStrategy::face_smile_detect(const FaceInfo *fi, float mouth_value) {
    int state_fatigue = static_cast<int>(fi->stateFatigue);
    if (!rtConfig->get_switch(ABILITY_FACE_FATIGUE)) {
        state_fatigue = 0;
    }

    float smile_threshold = _k_smile_def_threshold;
    if (mouth_value >= _k_smile_def_threshold && state_fatigue == FaceFatigueStatus::F_FATIGUE_NONE) {
        _mouth_arr.push_back(mouth_value);
        smile_threshold = std::accumulate(_mouth_arr.begin(), _mouth_arr.end(), 0.0) / _mouth_arr.size();
    }

    if (state_fatigue == FaceFatigueStatus::F_FATIGUE_YAWN_EYECLOSE) {
        _mouth_arr.clear();
        return false;
    }

    if (mouth_value <= smile_threshold * _k_smile_threshold_rate) {
        return true;
    }

    return false;
}

SimpleSize FaceSmileStrategy::calculate_mouth_size(FaceInfo *info) {
    float height_left = fabs(info->landmark2D106[FLM_88_MOUTH_TOP_LIP_TOP].y
                            - info->landmark2D106[FLM_99_MOUTH_TOP_LIP_BOTTOM_TOP].y);

    float height_right = fabs(info->landmark2D106[FLM_102_MOUTH_LOWER_LIP_TOP_BOTTOM].y
                             - info->landmark2D106[FLM_91_MOUTH_LOWER_LIP_BOTTOM].y);

    float height = (height_left + height_right) / 2;

    float width = fabs(info->landmark2D106[FLM_86_MOUTH_LEFT_CORNER].x
                      - info->landmark2D106[FLM_90_MOUTH_RIGHT_CORNER].x);

    SimpleSize size = {width, height};
    return size;
}

bool FaceSmileManager::preDetect(VisionRequest *request, VisionResult *result) {
    VA_CHECK_DETECTED(ABILITY_FACE_SMILE);
}

void FaceSmileManager::doDetect(VisionRequest *request, VisionResult *result) {
    VA_SET_DETECTED(ABILITY_FACE_SMILE);

    // 执行多人脸策略
    execute_face_strategy<FaceSmileStrategy>(result, _face_smile_map, mRtConfig);
}

void FaceSmileManager::onNoFace(VisionRequest *request, VisionResult *result, FaceInfo *face) {
    if (face != nullptr && (!_face_smile_map.empty())) {
        auto iter = _face_smile_map.find(face->id);
        if (iter != _face_smile_map.end()) {
            iter->second->onNoFace(request, face);
        }
    }
}

void FaceSmileManager::clear() {
    for(auto& info : _face_smile_map) {
        if (info.second) {
            info.second->clear();
            FaceSmileStrategy::recycle(info.second);
        }
    }
    _face_smile_map.clear();
}

void FaceSmileManager::deinit() {
    AbsVisionManager::deinit();
}

FaceSmileManager::~FaceSmileManager() {
    clear();
}

// NOLINTNEXTLINE
REGISTER_VISION_MANAGER("FaceSmileManager", ABILITY_FACE_SMILE,[]() {
    return std::dynamic_pointer_cast<AbsVisionManager>(std::make_shared<FaceSmileManager>());
});

} // namespace vision