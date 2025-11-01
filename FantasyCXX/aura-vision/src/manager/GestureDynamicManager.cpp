
#include "GestureDynamicManager.h"
#include "vision/util/log.h"
#include "util/SystemClock.h"
#include "vision/manager/VisionManagerRegistry.h"

namespace aura::vision {

static constexpr char *TAG = (char *) "GestureDynamicManager";

void GestureDynamicManager::clear() {
    gestureStaticWindow.clear();
    waveStaticWindow.clear();
    gestureDynamicWindow.clear();
    startPinchTime = 0;
    startGraspTime = 0;
    startLeftWaveTime = 0;
    startRightWaveTime = 0;
    lastTriggerDynamicTime = 0;
}

GestureDynamicManager::GestureDynamicManager() : nowTime(0),
                                                 curStaticGestureType(GESTURE_NO_DETECT),
                                                 curStaticGestureWaveType(GESTURE_NO_DETECT),
                                                 lastTriggerDynamicTime(0),
                                                 startPinchTime(0),
                                                 startGraspTime(0),
                                                 startLeftWaveTime(0),
                                                 startRightWaveTime(0),
                                                 gestureStaticWindow(SOURCE_UNKNOWN, DEFAULT_WINDOW_LEN,
                                                                     DEFAULT_DUTY_FACTOR),
                                                 waveStaticWindow(SOURCE_UNKNOWN, DEFAULT_WINDOW_LEN,
                                                                  DEFAULT_DUTY_FACTOR),
                                                 gestureDynamicWindow(SOURCE_UNKNOWN, DEFAULT_DYNAMIC_WINDOW_LEN,
                                                                      DEFAULT_DYNAMIC_DUTY_FACTOR) {
    gestMap[GESTURE_FIST_RAW] = GESTURE_FIST;
    gestMap[GESTURE_1_RAW] = GESTURE_1;
    gestMap[GESTURE_2_RAW] = GESTURE_2;
    gestMap[GESTURE_3_RAW] = GESTURE_3;
    gestMap[GESTURE_4_RAW] = GESTURE_4;
    gestMap[GESTURE_5_RAW] = GESTURE_5;
    gestMap[GESTURE_OK_RAW] = GESTURE_OK;
    gestMap[GESTURE_THUMB_UP_RAW] = GESTURE_THUMB_UP;
    gestMap[GESTURE_THUMB_DOWN_RAW] = GESTURE_THUMB_DOWN;
    gestMap[GESTURE_THUMB_RIGHT_RAW] = GESTURE_THUMB_RIGHT;
    gestMap[GESTURE_THUMB_LEFT_RAW] = GESTURE_THUMB_LEFT;
    gestMap[GESTURE_LEFT5_RAW] = GESTURE_LEFT5;
    gestMap[GESTURE_RIGHT5_RAW] = GESTURE_RIGHT5;
    setup_sliding_window();
}

void GestureDynamicManager::init(RtConfig *cfg) {
    mRtConfig = cfg;
    gestureStaticWindow.setSourceId(mRtConfig->sourceId);
    waveStaticWindow.setSourceId(mRtConfig->sourceId);
    gestureDynamicWindow.setSourceId(mRtConfig->sourceId);
}

void GestureDynamicManager::deinit() {
    AbsVisionManager::deinit();
}

void GestureDynamicManager::setup_sliding_window() {
    gestureStaticWindow.set_window_adjust_strategy(WindowAdjustStrategy::Fixed);
    waveStaticWindow.set_window_adjust_strategy(WindowAdjustStrategy::Fixed);
    gestureDynamicWindow.set_window_adjust_strategy(WindowAdjustStrategy::Fixed);
}

bool GestureDynamicManager::preDetect(VisionRequest *request, VisionResult *result) {
    VA_CHECK_DETECTED(ABILITY_GESTURE_DYNAMIC);
}

void GestureDynamicManager::doDetect(VisionRequest *request, VisionResult *result) {
    if (request == nullptr || result == nullptr) {
        return;
    }
    VA_SET_DETECTED(ABILITY_GESTURE_DYNAMIC);
    if (SystemClock::nowMillis() - lastTriggerDynamicTime < WATI_TIME_LIMIT) {
        return;
    }

    GestureInfo *gesture_info = result->getGestureResult()->gestureInfos[0];
    if (gesture_info == nullptr) {
        return;
    }
    // 执行动态手势滑窗，获取手势类型
    curStaticGestureType = GESTURE_NONE;
    int gest_idx = gestureStaticWindow.update(gesture_info->staticTypeSingle);
    if (gestMap.find(gest_idx) != gestMap.end()) {
        curStaticGestureType = gestMap[gest_idx];
        gestureStaticWindow.clear();
    }

    // 检测到手势4或手势5时，根据特定landmark关键点判断是左5还是右5
    int revise_static_hand_type = static_cast<int>(gesture_info->staticTypeSingle);
    if (revise_static_hand_type == GESTURE_5_RAW || revise_static_hand_type == GESTURE_4_RAW) {
        if (gesture_info->landmark21[GLM_4_THUMB4].x < gesture_info->landmark21[GLM_20_LITTLE_THUMB4].x) {
            revise_static_hand_type = GESTURE_RIGHT5_RAW;
        } else {
            revise_static_hand_type = GESTURE_LEFT5_RAW;
        }
    }

    VLOGD(TAG, "the specific type of static gestureType5：%d", revise_static_hand_type);

    curStaticGestureWaveType = GESTURE_NONE;
    int wave_gest_idx = waveStaticWindow.update((float) revise_static_hand_type);
    if (gestMap.find(wave_gest_idx) != gestMap.end()) {
        curStaticGestureWaveType = gestMap[wave_gest_idx];
        waveStaticWindow.clear();
    }
    // 动态手势类型设置默认值
    gesture_info->dynamicType = GESTURE_DYNAMIC_NONE;
    // 获取当前时间戳
    nowTime = SystemClock::nowMillis();
    // 是否检测到了动态手势
    bool is_detected = false;
    // 左挥手检测
    is_detected = by_dynamic_type_detect(GESTURE_DYNAMIC_LEFT_WAVE,
                                         GESTURE_RIGHT5, GESTURE_LEFT5,
                                         curStaticGestureWaveType,
                                         startLeftWaveTime, gesture_info->dynamicType);

    // 右挥手检测
    is_detected = by_dynamic_type_detect(GESTURE_DYNAMIC_RIGHT_WAVE,
                                         GESTURE_LEFT5, GESTURE_RIGHT5,
                                         curStaticGestureWaveType,
                                         startRightWaveTime, gesture_info->dynamicType);

    // 抓检测
    is_detected = by_dynamic_type_detect(GESTURE_DYNAMIC_GRASP,
                                         GESTURE_5, GESTURE_FIST,
                                         curStaticGestureType,
                                         startGraspTime, gesture_info->dynamicType);

    // 捏检测
    is_detected = by_dynamic_type_detect(GESTURE_DYNAMIC_PINCH,
                                         GESTURE_5, GESTURE_OK,
                                         curStaticGestureType,
                                         startPinchTime, gesture_info->dynamicType);
    if (!is_detected) {
        VLOGI(TAG, "not detect dynamic gesture!!");
    }
    gesture_info->dynamicType = gestureDynamicWindow.update(gesture_info->dynamicType);
    if (gesture_info->dynamicType > GESTURE_DYNAMIC_NONE) {
        clear();
        lastTriggerDynamicTime = SystemClock::nowMillis();
    }
    VLOGD(TAG, "gesture staticTypeSingle=[%d], dynamicType=[%d]", gesture_info->staticTypeSingle,
          gesture_info->dynamicType);
}

bool GestureDynamicManager::by_dynamic_type_detect(int detectDynamicType, int triggerStartType, int triggerEndType,
                                                   int curStaticGestureType, std::int64_t &triggerStartTime, int &outputType) {
    if (curStaticGestureType == triggerStartType) {
        triggerStartTime = nowTime;
    }
    if (triggerStartTime != 0) {
        if (nowTime - triggerStartTime <= DETECT_TIME_LIMIT) {
            if (curStaticGestureType == triggerEndType) {
                outputType = detectDynamicType;
                return true;
            }
        } else {
            triggerStartTime = 0;
        }
    }
    return false;
}
// NOLINTNEXTLINE
REGISTER_VISION_MANAGER("GestureDynamicManager", ABILITY_GESTURE_DYNAMIC,[]() {
    return std::dynamic_pointer_cast<AbsVisionManager>(std::make_shared<GestureDynamicManager>());
});

} // namespace vision