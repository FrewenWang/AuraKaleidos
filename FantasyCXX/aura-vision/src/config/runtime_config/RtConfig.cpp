#include "vision/config/runtime_config/RtConfig.h"
#include "vision/manager/VisionManagerRegistry.h"

namespace aura::vision {

static const char *TAG = "RtConfig";

/**
 * 运行时候配置参数静态全局配置变量
 */
short RtConfig::scheduleHtp = HTP0;
/**
 * 运行时配置参数。在基类的Config的构造函数里面进行实例化每个参数
 * 若各个去掉有自己自定义的配置，则在子类构造函数里面重写配置参数
 */
RtConfig::RtConfig() {
    sourceId = -1;
    // ======================== 输入图像参数. ========================================
    frameWidth = 1280;
    frameHeight = 720;
    frameFormat = static_cast<float>(FrameFormat::YUV_422_UYVY);
    frameConvertGrayFormat = COLOR_YUV2GRAY_NV21;
    frameConvertRgbFormat = COLOR_YUV2RGB_NV21;
    frameConvertBgrFormat = COLOR_YUV2BGR_NV21;

    // ======================== 人脸检测个数参数设置 ====================================
    faceMaxCount = 5;                           // 最多支持人脸数5个
    gestureMaxCount = 5;                        // 最多支持手势数5个
    bodyMaxCount = 2;                           // 最多支持肢体数2个
    livingMaxCount = 1;                         // 最多支持活体数(猫狗婴儿)1个
    faceNeedDetectCount = 5;                    // 主线默认业务检测5个人脸
    gestureNeedDetectCount = 5;                 // 主线默认业务检测1个手势
    bodyNeedDetectCount = 2;                    // 主线默认业务检测2个肢体(头肩)
    livingNeedDetectCount = 1;                  // 主线默认业务检测2个活体

    // camera参数
    cameraFocalLength = 4.907f;             // 焦距,
    cameraCcdWidth = 5.108845f;             // 感光元件宽度
    cameraCcdHeight = 4.0650910;            // 感光元件高度
    cameraLightType = CAMERA_LIGHT_TYPE_IR; // 采光类型
    cameraLightSwapMode = CAMERA_SWAP_MODE_AUTO;   // 采光类型是自动模式还是手动模式切换
    cameraPositionX = 0.f;                  // 摄像头位置x
    cameraPositionY = -200.f;               // 摄像头位置y
    cameraImageMirror = true;               // 摄像头是否存在镜像反转,默认摄像头一般是镜像的
    // ============================  车辆信息相关的阈值和参数  ================================================
    eyeGazeCalibSwitcher = false;           // 3D 关键点一点标定的默认开关为false
    speedThreshold = 0.f;                   // 3D 关键点一点标定的车速阈值
    steeringWheelAngleThreshold = 0.f;      // 3D 关键点一点标定的方向盘角度阈值

    // ======================= 设置各种目标检测的检测阈值  ======================================
    faceRectThreshold = 0.2f;                           // 人脸框检测阈值
    faceRectMinPixelThreshold = 110.0f; // 最小人脸框阈值，小于阈值则表示没有检测到人脸
    landmarkDetectScene = 0.f;
    landmarkThresholdNormal = 0.26f;
    landmarkThresholdFaceid = 0.85f;
    faceFeatureCompareThreshold = 0.51f;
    faceCoverFeatureCompareThreshold = 0.330f;
    noInteractiveLivingRgbThreshold = 0.8f;         //新RGB模型使用置信度0.8
    noInteractiveLivingIrThreshold = 0.8f;
    eyeCenterThreshold = 0.4f;                      // 主线瞳孔中心点模型检测到瞳孔的阈值
    eyeDetectedThreshold = 0.5f;                    // 主线眼球中心点模型检测到人眼的阈值
    callThreshold = 0.5f;
    gestureRectThreshold = 0.3f;
    gestureLandmarkThreshold = 0.6f;
    _s_gesture_lm_type_threshold = 0.85f;
    faceUpFrontYawLower = -15.f;
    faceUpFrontYawUpper = 15.f;
    faceUpFrontPitchLower = -15.f;
    faceUpFrontPitchUpper = 30.f;
    smallEyeThreshold = 0.3f; // 小眼睛参数

    // 是否在无人脸时使用上一帧变换矩阵检测当前帧图像
    usePrevFrameToDetectDms = false;

    // 危险驾驶角度限制. -7.f代表是低头
    dangerDrivePitchLower = -7.f; // 危险驾驶行为检测pitch角度限制
    dangerDriveYawLower = -30.f;    // 危险驾驶行为检测人脸yaw角度下限
    dangerDriveYawUpper = 30.f;     // 危险驾驶行为检测人脸yaw角度上限

    // 视线偏移角度阈值。主线阈值: 左右45度、上下30度 (需根据不同车型进行角度换算)
    attentionHeadRightAngle = -45.f;
    attentionHeadLeftAngle = 45.f;
    attentionHeadDownAngle = -30.f;
    attentionHeadUpAngle = 30.f;

    // ============================
    // 疲劳检测相关的阈值限定（左正右负、上正下负）=============================================
    eyeCloseAngleLimitSwitch = true; // 疲劳检测中闭眼的角度阈值检查的开关，默认开启
    openEyeBlink = false;            // eye blink检查的开关，默认关
    eyeClosePitchUpper = 30.f;        // 闭眼检测的人脸上下pitch角度限制
    eyeCloseYawLower = -30.f;           // 闭眼检测人脸左右yaw角度的下限
    eyeCloseYawUpper = 30.f;            // 闭眼检测人脸左右yaw角度的上限

    // 点摇头参数配置 --angle
    shakeExtremumDistanceAngle = 5.0f;
    nodExtremumDistanceAngle = 5.5f;
    shakeExtremumNumberAngle = 4;
    nodExtremumNumberAngle = 3;
    // 点摇头参数配置 --landmark
    shakeExtremumDistanceLandmark = 15.0f;
    nodExtremumDistanceLandmark = 12.0f;
    shakeExtremumNumberLandmark = 3;
    nodExtremumNumberLandmark = 4;

    // 眼神唤醒参数
    wakingFaceAngle = 5.f;
    wakingEyeAngleThreshold = 0.12f;
    wakingMode = 0.f;

    // ========================  滑窗策略的滑窗长参数设置  ====================
    faceQualityCoverWindowLen = 10;
    faceQualityCoverFrames = 10;
    faceQualityDefaultDutyFactor = 0.7f;

    // 无感活体滑窗策略参数
    faceNoInteractLiveWindowLen = 5;
    faceNoInteractMinLiveFrames = 3;
    faceNoInteractDefaultDutyFactor = 0.6f;

    // ===================== 有感活体动作相关配置参数（初始角度阈值限定) =====================
    interactLivingAction = INTERACT_LIVING_ACTION_NONE;
    livingStartYawUpper = 10.f;
    livingStartYawLower = -10.f;
    livingStartPitchUpper = 10.f;
    livingStartPitchLower = -10.f;
    livingStartRollUpper = 10.f;
    livingStartRollLower = -10.f;

    // ======================== 滑窗策略的触发时间以及过期时间，时间单位:ms =====================
    livingDetectTriggerNeedTime = 0.f;
    playPhoneTriggerNeedTime = 0.f;             // 玩手机检测触发时间
    playPhoneTriggerExpireTime = 0.f;           // 玩手机检测超时时间
    callTriggerNeedTime = 0.f;                  // 危险打电话触发时间（默认2秒）
    callTriggerExpireTime = 0.f;                // 危险打电话超时时间（默认6秒）
    attentionTriggerNeedTime = 0.f;
    attentionTriggerExpireTime = 0.f;
    smokeTriggerExpireTime = 0.f;
    drinkTriggerExpireTime = 0.f;
    silenceTriggerExpireTime = 0.f;
    openMouthTriggerExpireTime = 0.f;
    closeEyeTriggerExpireTime = 0.f;
    yawnTriggerExpireTime = 0.f;
    maskTriggerExpireTime = 0.f;
    coverMouthTriggerExpireTime = 0.f;

    // 主驾驶检测区域ROI参数
    useDriverRoiPositionFilter = 0.f;
    driverRoiPositionX = 0.f;
    driverRoiPositionY = 0.f;
    useDriverRoiRectFilter = 0.f;
    driverRoiLeftTopX = 0.f;
    driverRoiLeftTopY = 0.f;
    driverRoiRightBottomX = frameWidth;
    driverRoiRightBottomY = frameHeight;

    // 输入图像裁剪参数
    inputImageNeedCrop = 0.f;
    inputCropRoiLeftTopX = 0.f;
    inputCropRoiLeftTopY = 0.f;
    inputCropRoiRightBottomX = frameWidth;
    inputCropRoiRightBottomY = frameHeight;

    // =============== 感知能力计算调度策略 =======================
    faceDetectMethod = FaceDetectMethod::DETECT;        // (目前只支持模型检测)人脸框框检测方法：模型检测、人脸跟随、模板匹配
    gestureUseRoi = false;                              // 手势检测采用160*160中间区域
    callUseMutexMode = true;                            // 打电话检测左右耳互斥模式
    threadAffinityPolicy = TP_NORMAL;                   // 是否采用绑定大核模式
    releaseMode = PRODUCT;                              // 输出模式，BENCHMARK_TEST:指标测试，DEMO:调试演示，PRODUCT：产线
    scheduleMethod = SchedulerMethod::NAIVE;            // 1: Naive, 2: DAG
    scheduleDagThreadCount = 2;                         // 调度器线程数
    predictorThreadCount = 4;                           // 推理器线程数
    fixedFrameDetectSwitcher = false;                   // 感知能力层设置启用间隔帧检测的策略开关。默认关闭

    productNumber = ProductNumber::JIDU_MOBILE;         // 项目编号
    logLevel = 6.f;                                     // VERBOSE
    useInternalMem = 1.f;

    // 唇动检测滑窗时间，单位ms
    lipMovementWindowTime = 1000;
};

RtConfig::~RtConfig() = default;

void RtConfig::init() {
    init_params();
    init_switches();
}

void RtConfig::deinit() {
    _configs.clear();
    _switches.clear();
    _managers = nullptr;
}

void RtConfig::init_params() {
    _configs[FRAME_WIDTH] = &frameWidth;
    _configs[FRAME_HEIGHT] = &frameHeight;
    _configs[FRAME_FORMAT] = &frameFormat;
    _configs[FRAME_CONVERT_GRAY_FORMAT] = &frameConvertGrayFormat;
    _configs[FRAME_CONVERT_BGR_FORMAT] = &frameConvertBgrFormat;
    _configs[FRAME_CONVERT_RGB_FORMAT] = &frameConvertRgbFormat;

    _configs[FACE_MAX_COUNT] = &faceMaxCount;
    _configs[GESTURE_MAX_COUNT] = &gestureMaxCount;
    _configs[BODY_MAX_COUNT] = &bodyMaxCount;
    _configs[FACE_NEED_CHECK_COUNT] = &faceNeedDetectCount;
    _configs[BIOLOGY_CATEGORY_COUNT] = &livingMaxCount;

    // ============================  Camera硬件内参标定相关  ================================================
    _configs[CAMERA_FOCAL_LENGTH] = &cameraFocalLength;
    _configs[CAMERA_CCD_WIDTH] = &cameraCcdWidth;
    _configs[CAMERA_CCD_HEIGHT] = &cameraCcdHeight;
    _configs[CAMERA_LIGHT_TYPE] = &cameraLightType;
    _configs[CAMERA_LIGHT_TYPE_SWAP_MODE] = &cameraLightSwapMode;
    _configs[CAMERA_POSITION_X] = &cameraPositionX;
    _configs[CAMERA_POSITION_Y] = &cameraPositionY;
    _configs[CAMERA_FOCAL_LENGTH_PIXEL_X] = &cameraFocalLengthPixelX;
    _configs[CAMERA_FOCAL_LENGTH_PIXEL_Y] = &cameraFocalLengthPixelY;
    _configs[CAMERA_OPTICAL_CENTER_X] = &cameraOpticalCenterX;
    _configs[CAMERA_OPTICAL_CENTER_Y] = &cameraOpticalCenterY;
    _configs[CAMERA_DISTORTION_K1] = &cameraDistortionK1;
    _configs[CAMERA_DISTORTION_K2] = &cameraDistortionK2;
    _configs[CAMERA_DISTORTION_K3] = &cameraDistortionK3;
    _configs[CAMERA_DISTORTION_P1] = &cameraDistortionP1;
    _configs[CAMERA_DISTORTION_P2] = &cameraDistortionP2;
    _configs[CAMERA_DISTORTION_P3] = &cameraDistortionP3;

    // ============================  车辆信息相关阈值和参数  ================================================
    _configs[EYE_GAZE_CALIB_SWITCHER] = &eyeGazeCalibSwitcher;
    _configs[SPEED_THRESHOLD] = &speedThreshold;
    _configs[STEERING_WHEEL_ANGLE_THRESHOLD] = &steeringWheelAngleThreshold;

    // ======================= 设置各种目标检测的检测阈值  ==================================================
    _configs[FACE_RECT_THRESHOLD] = &faceRectThreshold;
    _configs[FACE_RECT_MIN_PIXEL_THRESHOLD] = &faceRectMinPixelThreshold;
    _configs[LANDMARK_DETECT_SCENE] = &landmarkDetectScene;
    _configs[FACE_LMK_THRESHOLD_NORMAL] = &landmarkThresholdNormal;
    _configs[FACE_LMK_THRESHOLD_FACEID] = &landmarkThresholdFaceid;
    _configs[FACE_FEATURE_COMPARE_THRESHOLD] = &faceFeatureCompareThreshold;
    _configs[FACE_COVER_FEATURE_COMPARE_THRESHOLD] = &faceCoverFeatureCompareThreshold;
    _configs[NO_INTERACT_LIVING_RGB_THRESHOLD] = &noInteractiveLivingRgbThreshold;
    _configs[NO_INTERACT_LIVING_IR_THRESHOLD] = &noInteractiveLivingIrThreshold;
    _configs[FACE_EYE_CENTER_THRESHOLD] = &eyeCenterThreshold;
    /**
     *眼球中心点模型，是否检测到眼睛阈值，命名使用小写大写风格
     */
    _configs[EYE_DETECTED_THRESHOLD] = &eyeDetectedThreshold;
    _configs[CALL_THRESHOLD] = &callThreshold;
    _configs[GESTURE_RECT_THRESHOLD] = &gestureRectThreshold;
    _configs[GESTURE_LMK_THRESHOLD] = &gestureLandmarkThreshold;
    _configs[GESTURE_LMK_TYPE_THRESHOLD] = &_s_gesture_lm_type_threshold;
    _configs[FACE_UP_FRONT_YAW_MIN] = &faceUpFrontYawLower;
    _configs[FACE_UP_FRONT_YAW_MAX] = &faceUpFrontYawUpper;
    _configs[FACE_UP_FRONT_PITCH_MIN] = &faceUpFrontPitchLower;
    _configs[FACE_UP_FRONT_PITCH_MAX] = &faceUpFrontPitchUpper;
    _configs[SMALL_EYE_THRESHOLD] = &smallEyeThreshold;
    _configs[CAMERA_COVER_THRESHOLD_IR] = &cameraCoverThresholdIr;
    _configs[CAMERA_COVER_THRESHOLD_RGB] = &cameraCoverThresholdRgb;

    _configs[USE_PREV_FRAME_TO_DETECT_DMS] = &usePrevFrameToDetectDms;
    _configs[DANGER_DRIVE_PITCH_LOWER] = &dangerDrivePitchLower;
    _configs[DANGER_DRIVE_YAW_LOWER] = &dangerDriveYawLower;
    _configs[DANGER_DRIVE_YAW_UPPER] = &dangerDriveYawUpper;

    // 设置 DMS 疲劳检测闭眼相关的参数配置
    _configs[USE_FACE_EYE_CLOSE_ANGLE_LIMIT_SWITCH] = &eyeCloseAngleLimitSwitch;
    _configs[OPEN_EYE_BLINK] = &openEyeBlink;
    _configs[EYE_CLOSE_PITCH_UPPER] = &eyeClosePitchUpper;
    _configs[EYE_CLOSE_YAW_UPPER] = &eyeCloseYawLower;
    _configs[EYE_CLOSE_YAW_LOWER] = &eyeCloseYawUpper;

    _configs[ATTENTION_HEAD_RIGHT_ANGLE] = &attentionHeadRightAngle;
    _configs[ATTENTION_HEAD_LEFT_ANGLE] = &attentionHeadLeftAngle;
    _configs[ATTENTION_HEAD_DOWN_ANGLE] = &attentionHeadDownAngle;
    _configs[ATTENTION_HEAD_UP_ANGLE] = &attentionHeadUpAngle;

    _configs[SHAKE_PEAK_ANGLE] = &shakeExtremumDistanceAngle;
    _configs[SHAKE_PEAK_ANGLE_NUM] = &shakeExtremumNumberAngle;
    _configs[NOD_PEAK_ANGLE] = &nodExtremumDistanceAngle;
    _configs[NOD_PEAK_ANGLE_NUM] = &nodExtremumNumberAngle;
    _configs[SHAKE_PEAK_LMK] = &shakeExtremumDistanceLandmark;
    _configs[NOD_PEAK_LMK] = &nodExtremumDistanceLandmark;
    _configs[SHAKE_PEAK_LMK_NUM] = &shakeExtremumNumberLandmark;
    _configs[NOD_PEAK_LMK_NUM] = &nodExtremumNumberLandmark;

    _configs[WAKING_FACE_ANGLE] = &wakingFaceAngle;
    _configs[WAKING_EYE_ANGLE_THRESHOLD] = &wakingEyeAngleThreshold;
    _configs[WAKING_MODE] = &wakingMode;

    _configs[NO_INTERACT_LIVE_WINDOW_LEN] = &faceNoInteractLiveWindowLen;
    _configs[NO_INTERACT_MIN_LIVE_FRAMES] = &faceNoInteractMinLiveFrames;
    _configs[NO_INTERACT_DEFAULT_DUTY_FACTOR] = &faceNoInteractDefaultDutyFactor;

    _configs[FACE_QUALITY_COVER_WINDOW_LEN] = &faceQualityCoverWindowLen;
    _configs[FACE_QUALITY_COVER_FRAMES] = &faceQualityCoverFrames;
    _configs[FACE_QUALITY_DEFAULT_DUTY_FACTOR] = &faceQualityDefaultDutyFactor;

    _configs[INTERACT_LIVING_ACTION] = &interactLivingAction;
    _configs[LIVING_START_YAW_LOWER] = &livingStartYawLower;
    _configs[LIVING_START_YAW_UPPER] = &livingStartYawUpper;
    _configs[LIVING_START_PITCH_LOWER] = &livingStartPitchLower;
    _configs[LIVING_START_PITCH_UPPER] = &livingStartPitchUpper;
    _configs[LIVING_START_ROLL_LOWER] = &livingStartRollLower;
    _configs[LIVING_START_ROLL_UPPER] = &livingStartRollUpper;

    _configs[STRATEGY_TRIGGER_NEED_TIME_CALL] = &callTriggerNeedTime;
    _configs[STRATEGY_TRIGGER_NEED_TIME_ATTENTION] = &attentionTriggerNeedTime;
    _configs[STRATEGY_TRIGGER_NEED_TIME_PLAY_PHONE] = &playPhoneTriggerNeedTime;
    _configs[STRATEGY_TRIGGER_EXPIRE_TIME_PLAY_PHONE] = &playPhoneTriggerExpireTime;
    _configs[STRATEGY_TRIGGER_EXPIRE_TIME_SMOKE] = &smokeTriggerExpireTime;
    _configs[STRATEGY_TRIGGER_EXPIRE_TIME_DRINK] = &drinkTriggerExpireTime;
    _configs[STRATEGY_TRIGGER_EXPIRE_TIME_SILENCE] = &silenceTriggerExpireTime;
    _configs[STRATEGY_TRIGGER_EXPIRE_TIME_OPEN_MOUTH] = &openMouthTriggerExpireTime;
    _configs[STRATEGY_TRIGGER_EXPIRE_TIME_CALL] = &callTriggerExpireTime;
    _configs[STRATEGY_TRIGGER_EXPIRE_TIME_CLOSE_EYE] = &closeEyeTriggerExpireTime;
    _configs[STRATEGY_TRIGGER_EXPIRE_TIME_YAWN] = &yawnTriggerExpireTime;
    _configs[STRATEGY_TRIGGER_EXPIRE_TIME_ATTENTION] = &attentionTriggerExpireTime;
    _configs[STRATEGY_TRIGGER_EXPIRE_TIME_MASK] = &maskTriggerExpireTime;
    _configs[STRATEGY_TRIGGER_EXPIRE_TIME_COVER_MOUTH] = &coverMouthTriggerExpireTime;


    _configs[USE_DRIVER_ROI_POSITION_FILTER] = &useDriverRoiPositionFilter;
    _configs[DRIVER_ROI_POSITION_X] = &driverRoiPositionX;
    _configs[DRIVER_ROI_POSITION_Y] = &driverRoiPositionY;

    _configs[USE_DRIVER_ROI_FILTER] = &useDriverRoiRectFilter;
    _configs[DRIVER_ROI_LT_X] = &driverRoiLeftTopX;
    _configs[DRIVER_ROI_LT_Y] = &driverRoiLeftTopY;
    _configs[DRIVER_ROI_RB_X] = &driverRoiRightBottomX;
    _configs[DRIVER_ROI_RB_Y] = &driverRoiRightBottomY;
    _configs[IMAGE_ROI_CROP] = &inputImageNeedCrop;
    _configs[IMAGE_ROI_LT_X] = &inputCropRoiLeftTopX;
    _configs[IMAGE_ROI_LT_Y] = &inputCropRoiLeftTopY;
    _configs[IMAGE_ROI_RB_X] = &inputCropRoiRightBottomX;
    _configs[IMAGE_ROI_RB_Y] = &inputCropRoiRightBottomY;

    // =============== 感知能力计算调度策略 =======================
    _configs[GESTURE_USE_ROI] = &gestureUseRoi;
    _configs[CALL_USE_MUTEX_MODE] = &callUseMutexMode;
    _configs[THREAD_AFFINITY_POLICY] = &threadAffinityPolicy;
    _configs[RELEASE_MODE] = &releaseMode;
    _configs[MAX_NUM_THREADS] = &predictorThreadCount;
    _configs[PRODUCT_NUMBER] = &productNumber;
    _configs[NATIVE_LOG_LEVEL] = &logLevel;
    _configs[USE_INTERNAL_MEM] = &useInternalMem;
    // 设置当前摄像头是否存在镜像反转的运行时配置，支持上层配置
    _configs[CAMERA_CAMERA_IMAGE_MIRROR] = &cameraImageMirror;

    _configs[SCHEDULE_METHOD] = &scheduleMethod;
    _configs[SCHEDULE_DAG_THREAD_COUNT] = &scheduleDagThreadCount;

    _configs[LIP_MOVEMENT_WINDOW_TIME] = &lipMovementWindowTime;
}

void RtConfig::init_switches() {
    _switches[ABILITY_ALL] = true;
    _switches[ABILITY_FACE] = false;
    _switches[ABILITY_FACE_DETECTION] = false;
    _switches[ABILITY_FACE_RECT] = false;
    _switches[ABILITY_FACE_LANDMARK] = false;
    _switches[ABILITY_FACE_DANGEROUS_DRIVING] = false;
    _switches[ABILITY_FACE_FATIGUE] = false;
    _switches[ABILITY_FACE_ATTENTION] = false;
    _switches[ABILITY_FACE_HEAD_BEHAVIOR] = false;
    _switches[ABILITY_FACE_EYE_TRACKING] = false;
    _switches[ABILITY_FACE_EYE_WAKING] = false;
    _switches[ABILITY_FACE_SMILE] = false;
    _switches[ABILITY_FACE_EMOTION] = false;
    _switches[ABILITY_FACE_QUALITY] = false;
    _switches[ABILITY_CAMERA_COVER] = false;
    _switches[ABILITY_FACE_INTERACTIVE_LIVING] = false;
    _switches[ABILITY_FACE_FEATURE] = false;
    _switches[ABILITY_FACE_CALL] = false;
    _switches[ABILITY_FACE_ATTRIBUTE] = false;
    _switches[ABILITY_FACE_AR_HUD] = false;
    _switches[ABILITY_FACE_NO_INTERACTIVE_LIVING] = false;
    _switches[ABILITY_GESTURE] = false;
    _switches[ABILITY_GESTURE_RECT] = false;
    _switches[ABILITY_GESTURE_LANDMARK] = false;
    _switches[ABILITY_GESTURE_TYPE] = false;
    _switches[ABILITY_GESTURE_DYNAMIC] = false;
    _switches[ABILITY_BODY] = false;
    _switches[ABILITY_PERSON_VEHICLE] = false;
    _switches[ABILITY_PERSON_VEHICLE_RECT] = false;
    _switches[ABILITY_FRAME_BRIGHTNESS] = false;
    _switches[ABILITY_IMAGE_BRIGHTNESS] = false;
}

void RtConfig::set_config(int key, float value) {
    if (key < 0 || key >= KEY_LENGTH) {
        VLOGE(TAG, "setConfig: key(%d) error!", key);
        return;
    }

    if (_configs.find(key) == _configs.end()) {
        return;
    }

    *_configs[key] = value;

    for (auto &m: _managers->getManagers()) {
        m->onConfigUpdated(key, value);
    }

    // set log level
    if (key == NATIVE_LOG_LEVEL) {
        VLOGI(TAG, "set log level to %d", V_TO_INT(value));
        Logger::setLogLevel(static_cast<LogLevel>(V_TO_INT(value)));
    }
}

float RtConfig::get_config(int key) {
    if (key < 0 || key >= KEY_LENGTH) {
        return -1.f;
    }
    auto it = _configs.find(key);
    if (it == _configs.end()) {
        return -1.f;
    } else {
        return *it->second;
    }
}

void RtConfig::set_switch(short ability, bool enable) {
    _switches[ability] = enable;
    auto manager = _managers->getManager(ability);
    if (manager && !enable) {
        manager->clear();
    }
    // VLOGD(TAG, "Set switch (%d -> %s)", ability, enable && manager ? "true" : "false");
}

void RtConfig::set_switches(const std::unordered_map<short, bool> &switches) {
    for (const auto &sw: switches) {
        set_switch(sw.first, sw.second);
    }
}

bool RtConfig::get_switch(short ability) {
    auto it = _switches.find(ability);
    return it == _switches.end() ? false : it->second;
}

const std::unordered_map<short, bool> &RtConfig::get_switches() {
    return _switches;
}

void RtConfig::inject_manager_registry(const std::shared_ptr<VisionManagerRegistry> &managers) {
    _managers = managers;
}

std::shared_ptr<VisionManagerRegistry> RtConfig::getManagerRegistry() const {
    return _managers;
}

} // namespace aura::vision