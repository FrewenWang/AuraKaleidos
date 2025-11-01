
#include "FaceInteractLivingManager.h"

#include "util/SystemClock.h"
#include "vision/manager/VisionManagerRegistry.h"
#include "vision/util/MouthUtil.h"

namespace aura::vision {

static const char *TAG = "FaceInteractLivingStrategy";

FaceInteractLivingStrategy::FaceInteractLivingStrategy(RtConfig* cfg) {
    this->rtConfig = cfg;
    lastDetectType = 0;

    timeout = INTERACT_LIVING_DETECT_OUTPUT;

    lastDistance = 0.0;
    lastStatus = 0;

    finished = false;
    isFirst = true;
    firstLipDistance = 0.0f;
    openMouthCount = 0;
    startTime = 0;
    isLarger = false;

    status = DETECT_TYPE_ERROR;

    nowStatusPitch = 0;
    nowStatusYaw = 0;
    lastStatusPitch = 0;
    lastStatusYaw = 0;
    leftTime = 0;
    rightTime = 0;
    leftFineValue = HEAD_BEHAVIOR_SHAKE_LEFT_DEFLECT;
    rightFineValue = HEAD_BEHAVIOR_SHAKE_RIGHT_DEFLECT;
    downFineValue = HEAD_BEHAVIOR_NOD_DEFLECT_MIN;

    eyeCloseCount = 0;

    distance = 0.0f;
    fineValue = 0.0f;
    thresholdLow = 0.0f;
    thresholdHigh = 0.0f;
}

void FaceInteractLivingStrategy::clear() {
    lastDistance = 0.0;
    lastStatus = 0;
    finished = false;
    isFirst = true;
    firstLipDistance = 0.0f;
    openMouthCount = 0;
    startTime = 0;
    isLarger = false;

    status = DETECT_TYPE_ERROR;

    nowStatusPitch = 0;
    nowStatusYaw = 0;
    lastStatusPitch = 0;
    lastStatusYaw = 0;
    leftTime = 0;
    rightTime = 0;

    eyeCloseCount = 0;
}

void FaceInteractLivingStrategy::is_first_detect(int detect_type) {
    if (lastDetectType != detect_type) {
        isFirst = true;
        lastDetectType = detect_type;
    }
}

int FaceInteractLivingStrategy::detect(int detectType, FaceInfo *face) {
    if (isFirst) {
        startTime = SystemClock::nowMillis();
        finished = false;
    }

    switch (detectType) {
        case INTERACT_LIVING_ACTION_HEAD_LEFT:
        case INTERACT_LIVING_ACTION_HEAD_RIGHT: {
            status = detectLeftOrRightRotate(detectType, face->headDeflection);
            break;
        }

        case INTERACT_LIVING_ACTION_SHAKE_HEAD: {
            status = detectShakeHead(face->headDeflection);
            break;
        }

        case INTERACT_LIVING_ACTION_CLOSE_EYES: {
            // 睁闭眼使用的人脸关键点的睁闭眼阈值。以及头部偏转角
            status = detectCloseEye(face->eyeCloseConfidence, face->headDeflection);
            break;
        }

        case INTERACT_LIVING_ACTION_OPEN_MOUTH: {
            // 张嘴的使用的是。计算的是人脸关键点的上下嘴唇的距离
            // 上下唇距离取100（上嘴唇下）和103（下嘴唇上）之间的距离
            float lipsDistance = MouthUtil::getLipDistance(face);
            // 缩放到指定尺寸的唇部张开幅度
            float lipRefRect = MouthUtil::getLipDistanceRefRect(lipsDistance,
                                                                 std::abs(face->rectLT.y - face->rectRB.y));
            status = detectOpenMouth(lipRefRect, face->headDeflection);
            break;
        }
        default: {
            status = DETECT_TYPE_ERROR;
            break;
        }
    }
    return status;
}

bool FaceInteractLivingStrategy::checkOriginAngle(VAngle &head_pos) {
    if (head_pos.pitch < rtConfig->livingStartPitchUpper
        && head_pos.pitch > rtConfig->livingStartPitchLower
        && head_pos.yaw < rtConfig->livingStartYawUpper
        && head_pos.yaw > rtConfig->livingStartYawLower
        && head_pos.roll < rtConfig->livingStartRollUpper
        && head_pos.roll > rtConfig->livingStartRollLower) {
        return true;
    } else {
        return false;
    }
}

int FaceInteractLivingStrategy::detectLeftOrRightRotate(int detectType, VAngle &headPos) {
    if (isFirst) {
        // 开始检测有感活体的时候。判断当前人脸是否处于居中位置。否则提示人脸初始角度不对。居中则提示人脸最优角度完成
        if (!checkOriginAngle(headPos)) {
            return FACE_CHECK_ORIGIN_ANGLE_FAIL;
        }
        lastDistance = 0.0;
        lastStatus = 0.0;
        isFirst = false;
        return FACE_CHECK_ORIGIN_ANGLE_SUCCESS;
    }
    // 如果开始执行有感活体检测。
    if (!finished) {
        // 根据不同的检测类型，设置相关的参数值
        if (detectType == INTERACT_LIVING_ACTION_HEAD_LEFT) { // 检测头部左转
            distance = headPos.yaw;
            fineValue = INTERACT_LIVING_HEAD_LEFT_FINE_VALUE;
            // 如果是向左转头。连续两帧角度变化大于5度，小于15度。
            thresholdLow = INTERACT_LIVING_HEAD_LEFT_THRESHOLD_LOW;
            thresholdHigh = INTERACT_LIVING_HEAD_LEFT_THRESHOLD_HIGH;
            isLarger = true;
        } else if (detectType == INTERACT_LIVING_ACTION_HEAD_RIGHT) { // 检测头部右转
            // 获取向右转头的yaw的距离
            distance = headPos.yaw;
            fineValue = INTERACT_LIVING_HEAD_RIGHT_FINE_VALUE;
            // 如果是向右转头。连续两帧角度变化大于5度，小于15度。
            thresholdLow = INTERACT_LIVING_HEAD_RIGHT_THRESHOLD_LOW;
            thresholdHigh = INTERACT_LIVING_HEAD_RIGHT_THRESHOLD_HIGH;
            isLarger = false;
        } else {
            return DETECT_TYPE_ERROR;
        }
        // 根据设置的参数数据，检测状态
        lastStatus = headRotateLivenessStatus();
    }
    // 返回这一帧的检测结果
    return lastStatus;
}

int FaceInteractLivingStrategy::detectShakeHead(VAngle &headPos) {
    if (isFirst) {
        // 开始检测有感活体的时候。判断当前人脸是否处于居中位置。否则提示人脸初始角度不对。居中则提示人脸最优角度完成
        if (!checkOriginAngle(headPos)) {
            return FACE_CHECK_ORIGIN_ANGLE_FAIL;
        }
        // 记录初始头部姿态的pitch角度和yaw角度
        lastStatusPitch = headPos.pitch;
        lastStatusYaw = headPos.yaw;
        isFirst = false;
        return FACE_CHECK_ORIGIN_ANGLE_SUCCESS;
    } else {
        // 记录此次的头部姿态的pitch角度和yaw角度
        nowStatusPitch = headPos.pitch;
        nowStatusYaw = headPos.yaw;

        changPitchValue = std::fabs(nowStatusPitch - lastStatusPitch);
        changYawValue = std::fabs(nowStatusYaw - lastStatusYaw);
        lastStatusPitch = nowStatusPitch;
        lastStatusYaw = nowStatusYaw;

        // 如果摇头的时候上下pitch角度大约点头的角度阈值。并且pitch角度大于yaw角度。说明是点头动作。提示动作错误
        if (changPitchValue > downFineValue && changPitchValue > changYawValue) {
            return ACTION_INCORRECT;
        } else {

            if (headPos.yaw < leftFineValue) {
                leftTime = SystemClock::nowMillis();
            } else if (headPos.yaw > rightFineValue) {
                rightTime = SystemClock::nowMillis();
            }
            // 如果左右摇头之前的时间差为5秒内。则提示有感活体成功
            long time_space = fabs(leftTime - rightTime);
            //
            if (time_space > 0 && time_space < HEAD_BEHAVIOR_SHAKE_MAX_TIME) {
                leftTime = 0;
                rightTime = 0;
                return DETECT_SUCCESS;
            } else {
                return ACTION_CHECK_START;
            }
        }
    }

    return ACTION_CHECK_START;
}

int FaceInteractLivingStrategy::detectCloseEye(float eyeThreshold, VAngle &headPos) {
    if (isFirst) {
        // 开始检测有感活体的时候。判断当前人脸是否处于居中位置。否则提示人脸初始角度不对。居中则提示人脸最优角度完成
        if (!checkOriginAngle(headPos)) {
            return FACE_CHECK_ORIGIN_ANGLE_FAIL;
        }
        lastDistance = 0.0f;
        lastStatus = 0.0f;
        isFirst = false;
        eyeCloseCount = 0;
        return FACE_CHECK_ORIGIN_ANGLE_SUCCESS;
    }

    if (eyeCloseCount < INTERACT_LIVING_EYE_OR_MOUTH_DETECT_COUNT) {
        //  如果闭眼阈值小于 eyeThreshold。则认为是闭眼
        bool eyeStatus = eyeThreshold < INTERACT_LIVING_EYE_THRESHOLD ? true : false;
        if (eyeStatus) {
            ++eyeCloseCount;
        } else {
            eyeCloseCount = 0;
        }
        return ACTION_CHECK_START;
    } else {
        // 闭眼10次则认为有感活体闭眼成功。
        eyeCloseCount = 0;
        return DETECT_SUCCESS;
    }
}

int FaceInteractLivingStrategy::detectOpenMouth(float lipDistance, VAngle &headPos) {
    if (isFirst) {
        if (!checkOriginAngle(headPos)) {
            return FACE_CHECK_ORIGIN_ANGLE_FAIL;
        }
        firstLipDistance = lipDistance;
        isFirst = false;
        openMouthCount = 0;
        return FACE_CHECK_ORIGIN_ANGLE_SUCCESS;
    }
    auto change = (float) std::abs(lipDistance - firstLipDistance);
    VLOGD(TAG, "firstLipDistance[%f] lip[%f] change[%f] count[%d]",
          firstLipDistance, lipDistance, change, openMouthCount);
    if (change > MouthUtil::THRESHOLD_LIP_MOVEMENT_DEFAULT) {
        openMouthCount++;
    }
    // 有3帧变化幅度超过给定阈值，则认为触发张张嘴动作
    if (openMouthCount < 3) {
        return ACTION_CHECK_START;
    } else {
        return DETECT_SUCCESS;
    }
}

int FaceInteractLivingStrategy::headRotateLivenessStatus() {
    short stat = ACTION_CHECK_START;
    // 有感活体单帧单帧运动的距离
    float changedValue = distance - lastDistance;
    lastDistance = distance;

    // HeadDeflection的值是左正右负. 所以大于最小转角数。小于最大转角度数
    if (changedValue > thresholdLow && changedValue < thresholdHigh) {
        stat = LEFT_OR_DOWN; // 正在左转、低头
    }

    if (changedValue < -thresholdLow && changedValue > -thresholdHigh) {
        stat = RIGHT_OR_UP; // 正在右转、抬头
    }

    VLOGD(TAG, "distance[%f] lastDistance[%f] change[%f] isLarger[%d] cameraImageMirror[%f]",
          distance, lastDistance, changedValue, isLarger, rtConfig->cameraImageMirror);

    // 如果单帧转角速度太快。则提示异常转角速度太快
    if (changedValue > thresholdHigh) {
        return LEFT_OR_DOWN_FAST; // 异常:左转、低头速度太快
    }

    if (changedValue < -thresholdHigh) {
        return RIGHT_OR_UP_FAST; // 异常:右转、抬头速度太快
    }

    if ((lastStatus == LEFT_OR_DOWN && stat == RIGHT_OR_UP) ||
        (lastStatus == RIGHT_OR_UP && stat == RIGHT_OR_UP)) {
        return ACTION_INCORRECT; // 前后两帧方向不一致时，动作不正确
    }

    // 算法模型原始输出是：左正右负、下正上负
    // 需要转化为标准的坐标系原则：即遵循用户体验的：左正右负、上正下负
    // 带镜像摄像头：如果是向左转头，且转角大于 fineValue，则检测成功。
    if (isLarger && distance > fineValue) {
        finished = true;
        return DETECT_SUCCESS;
    }
    // 带镜像摄像头：如果是向右转头，且转角小于 fineValue，则检测成功。
    if (!isLarger && distance < fineValue) {
        finished = true;
        return DETECT_SUCCESS; // 成功
    }

    return stat;
}

void FaceInteractLivingStrategy::execute(FaceInfo *face) {
    face->stateInteractLiving = (float) detect(V_TO_INT(rtConfig->interactLivingAction), face);
    VLOGI(TAG, "FaceId[%s], action[%d] state[%f]", std::to_string(face->id).c_str(),
          V_TO_INT(rtConfig->interactLivingAction), face->stateInteractLiving);
}

void FaceInteractLivingStrategy::onNoFace(VisionRequest *request, FaceInfo *face) {
    // 有感活体检测要求全程有人脸配合，当人脸丢失则直接返回错误
    face->stateInteractLiving = DETECT_TYPE_ERROR;
    VLOGI(TAG, "[when no face] FaceId: %s, stateInteractLiving: %f",
          std::to_string(face->id).c_str(), face->stateInteractLiving);
}

FaceInteractLivingStrategy::~FaceInteractLivingStrategy() {
    if (rtConfig) {
        delete rtConfig;
        rtConfig = nullptr;
    }
}

FaceInteractLivingManager::~FaceInteractLivingManager() {
    clear();
}

void FaceInteractLivingManager::clear() {
    for(auto& info : interactLivingMap) {
        if (info.second) {
            info.second->clear();
            FaceInteractLivingStrategy::recycle(info.second);
        }
    }
    interactLivingMap.clear();
}

void FaceInteractLivingManager::init(RtConfig* cfg) {
    mRtConfig = cfg;
}

void FaceInteractLivingManager::deinit() {
    clear();
    mRtConfig = nullptr;
}

bool FaceInteractLivingManager::preDetect(VisionRequest *request, VisionResult *result) {
    VA_CHECK_DETECTED(ABILITY_FACE_INTERACTIVE_LIVING);
}

void FaceInteractLivingManager::doDetect(VisionRequest *request, VisionResult *result) {
    VA_SET_DETECTED(ABILITY_FACE_INTERACTIVE_LIVING);

    // 活体检测
//    PERF_TICK(result->get_perf_util(), "face_interact_living");
    // 执行多人脸策略
    execute_face_strategy<FaceInteractLivingStrategy>(result, interactLivingMap, mRtConfig);
//    PERF_TOCK(result->get_perf_util(), "face_interact_living");
}

void FaceInteractLivingManager::onNoFace(VisionRequest *request, VisionResult *result, FaceInfo *face) {
    if (face != nullptr && (!interactLivingMap.empty())) {
        auto iter = interactLivingMap.find(face->id);
        if (iter != interactLivingMap.end()) {
            iter->second->onNoFace(request, face);
        }
    }
}

// NOLINTNEXTLINE
REGISTER_VISION_MANAGER("FaceInteractLivingManager", ABILITY_FACE_INTERACTIVE_LIVING,[]() {
    return std::dynamic_pointer_cast<AbsVisionManager>(std::make_shared<FaceInteractLivingManager>());
});

} // namespace aura::vision
