//
// Created by wangzhijiang on 2022/05/05.
//
#ifdef BUILD_3D_LANDMARK

#include "Face3dLandmarkManager.h"
#include "vision/manager/VisionManagerRegistry.h"

namespace aura::vision {

const static char *TAG = "Face3dLandmarkManager";
/**
 * 人脸3D Landmark由于算法侧SDK暂时没有完成开发，故先增加编译开关关闭人脸3D Landmark的检测
 */
Face3dLandmarkManager::Face3dLandmarkManager() {
    _detector = std::make_shared<Face3dLandmarkDetector>();
}

void Face3dLandmarkManager::init(RtConfig *cfg) {
    mRtConfig = cfg;
    _detector->init(cfg);
}

void Face3dLandmarkManager::deinit() {
    AbsVisionManager::deinit();
    if (_detector != nullptr) {
        _detector->deinit();
        _detector = nullptr;
    }
}

bool Face3dLandmarkManager::preDetect(VisionRequest *request, VisionResult *result) {
    bool checkLandmarkDetect = VA_GET_DETECTED(ABILITY_FACE_LANDMARK);
    V_CHECK_COND(!checkLandmarkDetect, Error::PREPARE_ERR, "3DLandmark should be scheduled after LandmarkManager");
    // 判断眼球中心点模型是否开启。如果开启。则需要先执行眼球中心点模型检测
    if (mRtConfig->get_switch(ABILITY_FACE_EYE_CENTER)) {
        bool checkEyeCenterDetect = VA_GET_DETECTED(ABILITY_FACE_EYE_CENTER);
        V_CHECK_COND(!checkEyeCenterDetect, Error::PREPARE_ERR, "3DLandmark should scheduled after EyeCenterManager");
    }
    // 判断人脸3D 关键点的逻辑是否重复执行
    VA_CHECK_DETECTED(ABILITY_FACE_3D_LANDMARK);
}

void Face3dLandmarkManager::doDetect(VisionRequest *request, VisionResult *result) {
    VA_SET_DETECTED(ABILITY_FACE_LANDMARK);
    _detector->detect(request, result);
}

void Face3dLandmarkManager::clear() {
    // 集度新需求：取消订阅能力的时候，不需要重置视线标定。重置的主动发起通过AbilityCmd接口设置。
    // if (_detector == nullptr) {
    //     VLOGE(TAG, "Face3dLandmarkDetector not init,no need clear!!!");
    //     return;
    // }
    // bool rst = _detector->resetEyeGazeCalib();
    // if (!rst) {
    //     VLOGE(TAG, "Face3dLandmarkDetector resetEyeGazeCalib Error!!! rst:%d", rst);
    // }
}

bool Face3dLandmarkManager::onAbilityCmd(int cmd) {
    VLOGI(TAG, "Face3dLandmarkDetector onAbilityCmd cmd:%d", cmd);
    switch (cmd) {
        case CMD_RESET_GAZE_CALIB: {
            // 如果收到上层通知重置视线标定的CMD，则进行重置视线标定
            if (_detector == nullptr) {
                VLOGE(TAG, "not init,no need reset!!!");
                return false;
            }
            bool rst = _detector->resetEyeGazeCalib();
            if (!rst) {
                VLOGE(TAG, "onAbilityCmd resetEyeGazeCalib Error!!! rst:%d", rst);
            }
            return rst;
        }
        default:
            break;
    }
    return false;
}

REGISTER_VISION_MANAGER("FaceLandmark3dManager", ABILITY_FACE_3D_LANDMARK, []() {
    return std::dynamic_pointer_cast<AbsVisionManager>(std::make_shared<Face3dLandmarkManager>());
});

} // namespace aura::vision

#endif