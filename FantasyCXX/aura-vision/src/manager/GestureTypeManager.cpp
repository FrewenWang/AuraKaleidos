

#include "GestureTypeManager.h"

#include <algorithm>

#include "vision/manager/VisionManagerRegistry.h"

static constexpr char *TAG = (char *) "GestureTypeManager";

namespace aura::vision {

GestureTypeManager::GestureTypeManager()
        : gestureWindow(SOURCE_UNKNOWN, DEFAULT_WINDOW_LENGTH, DEFAULT_TRIGGER_DUTY_FACTOR) {
    _gest_map[GESTURE_FIST_RAW] = GESTURE_FIST;
    _gest_map[GESTURE_1_RAW] = GESTURE_1;
    _gest_map[GESTURE_2_RAW] = GESTURE_2;
    _gest_map[GESTURE_3_RAW] = GESTURE_3;
    _gest_map[GESTURE_4_RAW] = GESTURE_4;
    _gest_map[GESTURE_5_RAW] = GESTURE_5;
    _gest_map[GESTURE_OK_RAW] = GESTURE_OK;
    _gest_map[GESTURE_THUMB_UP_RAW] = GESTURE_THUMB_UP;
    _gest_map[GESTURE_THUMB_DOWN_RAW] = GESTURE_THUMB_DOWN;
    _gest_map[GESTURE_THUMB_RIGHT_RAW] = GESTURE_THUMB_RIGHT;
    _gest_map[GESTURE_THUMB_LEFT_RAW] = GESTURE_THUMB_LEFT;
    _gest_map[GESTURE_HEART_RAW] = GESTURE_HEART;
    setupSlidingWindow();
}

GestureTypeManager::~GestureTypeManager() {
    clear();
}

void GestureTypeManager::init(RtConfig *cfg) {
    AbsVisionManager::init(cfg);
    gestureWindow.setSourceId(cfg->sourceId);
}

void GestureTypeManager::deinit() {
    AbsVisionManager::deinit();
}

void GestureTypeManager::setupSlidingWindow() {
    auto stageParas = std::make_shared<StageParameters>();
    for (int i = WINDOW_LOWER_FPS; i <= WINDOW_UPPER_FPS; ++i) {
        stageParas->stage_routine[i] = {static_cast<int>(round(i * DEFAULT_W_LENGTH_RATIO_1_5)), DEFAULT_W_DUTY_FACTOR};
    }
    gestureWindow.set_fps_stage_parameters(stageParas);
}

bool GestureTypeManager::preDetect(VisionRequest *request, VisionResult *result) {
    return !result->isAbilityExec(ABILITY_GESTURE_TYPE);
}

void GestureTypeManager::doDetect(VisionRequest *request, VisionResult *result) {
    result->setAbilityExec(ABILITY_GESTURE_TYPE);
    GestureInfo *info = result->getGestureResult()->gestureInfos[0];
    info->staticType = GESTURE_NONE;
    int gest_idx = gestureWindow.update((float) info->staticTypeSingle);
    if (_gest_map.find(gest_idx) != _gest_map.end()) {
        info->staticType = _gest_map[gest_idx];
        // 手势满足滑窗之后实时输出。不需要清除滑窗
        // if (info->staticType != GESTURE_LIKE) {
        //     if (_cfg->releaseMode == PRODUCT) {
        //         gestureWindow.clear();
        //     }
        // }
    } else {
        info->staticType = GESTURE_NONE;
    }

    /**
     * TODO ZHUSHISONG  暂时保留使用landmark进行额外校验的逻辑，后续根据算法模型效果选择删除
     */
    auto points = info->landmark21;
    if (points != nullptr) {
        auto ymin = (int) points[0].y;
        for (int i = 1; i < 21; ++i) {
            if (ymin > points[i].y) {
                ymin = (int) points[i].y;
            }
        }
        auto px1 = (int) (points[0].x);
        auto px2 = (int) (std::max(std::max(points[6].x, points[10].x), points[14].x));
        auto p4x = (int) (points[4].x);
        auto p4y = (int) (points[4].y);

        if (info->staticType == GESTURE_THUMB_UP) {
            if (px1 <= px2) {
                if (p4y == ymin && p4x > px1 && p4x < px2) {
                    info->staticType = GESTURE_THUMB_UP;
                    // 手势满足滑窗之后实时输出。不需要清除滑窗
                    // if (_cfg->releaseMode == PRODUCT) {
                    //     gestureWindow.clear();
                    // }
                } else {
                    info->staticType = GESTURE_NONE;
                }
            } else {
                if (p4y == ymin && p4x > px2 && p4x < px1) {
                    info->staticType = GESTURE_THUMB_UP;
                    // 手势满足滑窗之后实时输出。不需要清除滑窗
                    // if (_cfg->releaseMode == PRODUCT) {
                    //     gestureWindow.clear();
                    // }
                } else {
                    info->staticType = GESTURE_NONE;
                }
            }
        }
    }
    VLOGD(TAG, "gesture staticTypeSingle=[%d], staticType=[%d]", info->staticTypeSingle, info->staticType);
}

void GestureTypeManager::clear() { gestureWindow.clear();
}

void GestureTypeManager::onNoGesture(VisionRequest *request, VisionResult *result, GestureInfo *gesture) {
    gestureWindow.update(GESTURE_NO_DETECT);
}

REGISTER_VISION_MANAGER("GestureTypeManager", ABILITY_GESTURE_TYPE, []() {
    return std::dynamic_pointer_cast<AbsVisionManager>(std::make_shared<GestureTypeManager>());
});

} // namespace vision