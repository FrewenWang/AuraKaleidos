#include "PlayPhoneManager.h"

#include "util/math_utils.h"
#include "vision/manager/VisionManagerRegistry.h"
#include <math.h>
#include <numeric>

namespace aura::vision {

static const char *TAG = "PlayPhoneManager";

PlayPhoneStrategy::PlayPhoneStrategy(RtConfig *cfg)
    : playPhoneWindow(cfg->sourceId, AbsVisionManager::DEFAULT_WINDOW_LENGTH,
                      AbsVisionManager::DEFAULT_TRIGGER_DUTY_FACTOR, AbsVisionManager::DEFAULT_END_DUTY_FACTOR,
                      G_PLAY_PHONE_STATUS_PLAYING) {
    this->rtConfig = cfg;
    setupSlidingWindow();
}

PlayPhoneStrategy::~PlayPhoneStrategy() = default;

void PlayPhoneStrategy::setupSlidingWindow() {
    auto stageParas = std::make_shared<StageParameters>();
    for (int i = AbsVisionManager::WINDOW_LOWER_FPS; i <= AbsVisionManager::WINDOW_UPPER_FPS; ++i) {
        stageParas->stage_routine[i] = {static_cast<int>(round(i * AbsVisionManager::DEFAULT_W_LENGTH_RATIO_1_0)),
                                        AbsVisionManager::DEFAULT_W_DUTY_FACTOR};
    }
    playPhoneWindow.set_fps_stage_parameters(stageParas);
}

void PlayPhoneStrategy::execute(GestureInfo *gesture) {}

void PlayPhoneStrategy::execute(FaceInfo *pFaceInfo, GestureInfo *pGestureInfo) {
    // 算法模型检测的规避策略:当算法模型同时检测出来打电话和玩手机，则屏蔽玩手机的结果检出
    if (V_TO_SHORT(pFaceInfo->stateCallSingle) == F_CALL_CALLING) {
        VLOGD(TAG, "play_phone ignore play phone state because of calling");
        pGestureInfo->statePlayPhoneSingle = G_PLAY_PHONE_STATUS_NONE;
    }

    if (pFaceInfo->stateDangerDriveSingle == F_DANGEROUS_DRINK && pFaceInfo->dangerDriveConfidence > 0.9) {
        VLOGD(TAG, "play_phone ignored because of drink and conf is %f",pFaceInfo->dangerDriveConfidence);
        pGestureInfo->statePlayPhoneSingle = G_PLAY_PHONE_STATUS_NONE;
    }

    bool isPlayingPhone = playPhoneWindow.update(pGestureInfo->statePlayPhoneSingle, &pGestureInfo->playPhoneVState);
    if (isPlayingPhone) {
        pGestureInfo->statePlayPhone = G_PLAY_PHONE_STATUS_PLAYING;
    } else {
        pGestureInfo->statePlayPhone = G_PLAY_PHONE_STATUS_NONE;
    }
    VLOGD(TAG, "play_phone[%d] singleState=[%d], state=[%d],vState=[%d,%d,%d]", pGestureInfo->id,
          pGestureInfo->statePlayPhoneSingle, pGestureInfo->statePlayPhone, pGestureInfo->playPhoneVState.state,
          pGestureInfo->playPhoneVState.continue_time, pGestureInfo->playPhoneVState.trigger_count);
}

void PlayPhoneStrategy::clear() {
    playPhoneWindow.clear();
}

PlayPhoneManager::PlayPhoneManager() {}

bool PlayPhoneManager::preDetect(VisionRequest *request, VisionResult *result) {
    bool checkGestureRect = VA_GET_DETECTED(ABILITY_GESTURE_RECT);
    V_CHECK_COND(!checkGestureRect, Error::PREPARE_ERR, "PlayPhone detect would be scheduled after GestureRect");
    VA_CHECK_DETECTED(ABILITY_PLAY_PHONE_DETECT);
}

void PlayPhoneManager::doDetect(VisionRequest *request, VisionResult *result) {
    VA_SET_DETECTED(ABILITY_PLAY_PHONE_DETECT);

    // 获取需要检测的手势框的检测个数
    for (int i = 0; i < mRtConfig->gestureNeedDetectCount; ++i) {
        // 执行滑窗结果检测
        playPhoneStrategy->execute(result->getFaceResult()->faceInfos[i], result->getGestureResult()->gestureInfos[i]);
    }
}

void PlayPhoneManager::init(RtConfig *cfg) {
    mRtConfig = cfg;
    playPhoneStrategy = new PlayPhoneStrategy(cfg);
}

void PlayPhoneManager::deinit() {
    AbsVisionManager::deinit();
    if (playPhoneStrategy != nullptr) {
        delete playPhoneStrategy;
        playPhoneStrategy = nullptr;
    }
}

void PlayPhoneManager::clear() {
    if (playPhoneStrategy != nullptr) {
        playPhoneStrategy->clear();
        // 务必注意，自己new的对象不能调用recycle。尤其是生命周期长的对象
        // PlayPhoneStrategy::recycle(playPhoneStrategy);
    }
}

PlayPhoneManager::~PlayPhoneManager() {
    clear();
    delete playPhoneStrategy;
}

REGISTER_VISION_MANAGER("PlayPhoneManager", ABILITY_PLAY_PHONE_DETECT, []() {
    return std::dynamic_pointer_cast<AbsVisionManager>(std::make_shared<PlayPhoneManager>());
});

} // namespace vision