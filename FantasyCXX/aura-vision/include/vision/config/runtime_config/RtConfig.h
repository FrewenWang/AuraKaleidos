#pragma  once

#include <memory>
#include <unordered_map>
#include <vector>

namespace aura::vision {

enum ParamKey : short {
    // =================== 图像数据帧相关 ===================
    FRAME_WIDTH = 0,
    FRAME_HEIGHT,
    FRAME_FORMAT,
    FRAME_CONVERT_GRAY_FORMAT,
    FRAME_CONVERT_BGR_FORMAT,
    FRAME_CONVERT_RGB_FORMAT,

    // ============= 目标检测数量控制 =====================
    FACE_MAX_COUNT,
    GESTURE_MAX_COUNT,
    BODY_MAX_COUNT,
    LIVING_MAX_COUNT,
    HEAD_SHOULDER_MAX_COUNT,
    FACE_NEED_CHECK_COUNT,
    GESTURE_NEED_CHECK_COUNT,
    BODY_NEED_CHECK_COUNT,
    LIVING_NEED_CHECK_COUNT,
    // ==============  Camera硬件内参标定相关  ====================
    CAMERA_FOCAL_LENGTH,
    CAMERA_CCD_WIDTH,
    CAMERA_CCD_HEIGHT,
    CAMERA_LIGHT_TYPE,
    CAMERA_LIGHT_TYPE_SWAP_MODE,
    CAMERA_POSITION_X,
    CAMERA_POSITION_Y,
    CAMERA_FOCAL_LENGTH_PIXEL_X,
    CAMERA_FOCAL_LENGTH_PIXEL_Y,
    CAMERA_OPTICAL_CENTER_X,
    CAMERA_OPTICAL_CENTER_Y,
    CAMERA_DISTORTION_K1,
    CAMERA_DISTORTION_K2,
    CAMERA_DISTORTION_K3,
    CAMERA_DISTORTION_P1,
    CAMERA_DISTORTION_P2,
    CAMERA_DISTORTION_P3,
    CAMERA_CAMERA_IMAGE_MIRROR,    //当前摄像头是否是镜像的

    // ==============  车辆信息相关阈值与变量  ====================
    EYE_GAZE_CALIB_SWITCHER,
    SPEED_THRESHOLD,
    STEERING_WHEEL_ANGLE_THRESHOLD,

    // =============  设置各种目标检测的检测阈值  ======================
    FACE_RECT_THRESHOLD,
    FACE_RECT_MIN_PIXEL_THRESHOLD,  // 最小人脸框阈值，小于阈值则表示没有检测到人脸
    LANDMARK_DETECT_SCENE,                              // 人脸关键点检测场景
    FACE_LMK_THRESHOLD_NORMAL,
    FACE_LMK_THRESHOLD_FACEID,
    FACE_FEATURE_COMPARE_THRESHOLD,
    FACE_COVER_FEATURE_COMPARE_THRESHOLD,
    NO_INTERACT_LIVING_RGB_THRESHOLD,
    NO_INTERACT_LIVING_IR_THRESHOLD,
    FACE_EYE_CENTER_THRESHOLD,
    CALL_THRESHOLD,
    GESTURE_RECT_THRESHOLD,
    GESTURE_LMK_THRESHOLD,
    GESTURE_LMK_TYPE_THRESHOLD,
    FACE_UP_FRONT_YAW_MIN,
    FACE_UP_FRONT_YAW_MAX,
    FACE_UP_FRONT_PITCH_MIN,
    FACE_UP_FRONT_PITCH_MAX,
    SMALL_EYE_THRESHOLD,
    EYE_DETECTED_THRESHOLD,                 //是否检测到眼睛，置信度配置
    CAMERA_COVER_THRESHOLD_IR,
    CAMERA_COVER_THRESHOLD_RGB,

    USE_PREV_FRAME_TO_DETECT_DMS,
    DANGER_DRIVE_PITCH_LOWER,
    DANGER_DRIVE_PITCH_UPPER,
    DANGER_DRIVE_YAW_LOWER,
    DANGER_DRIVE_YAW_UPPER,

    ATTENTION_HEAD_RIGHT_ANGLE, // 注意力右转头的角度
    ATTENTION_HEAD_LEFT_ANGLE,  // 注意力左转头的角度
    ATTENTION_HEAD_DOWN_ANGLE,  // 注意力低头的角度
    ATTENTION_HEAD_UP_ANGLE,    // 注意力抬头的角度

    // 疲劳检测相关的配置开关
    USE_FACE_EYE_CLOSE_ANGLE_LIMIT_SWITCH,
    OPEN_EYE_BLINK,
    EYE_CLOSE_PITCH_UPPER,                  // 闭眼检测的人脸上下pitch角度限制
    EYE_CLOSE_YAW_UPPER,                    // 闭眼检测人脸左右yaw角度的下限
    EYE_CLOSE_YAW_LOWER,                    // 闭眼检测人脸左右yaw角度的上限

    // ============================ 多模行为点摇头的参数设置 ============================
    SHAKE_PEAK_ANGLE,
    NOD_PEAK_ANGLE,
    SHAKE_PEAK_ANGLE_NUM,
    NOD_PEAK_ANGLE_NUM,
    SHAKE_PEAK_LMK,
    NOD_PEAK_LMK,
    SHAKE_PEAK_LMK_NUM,
    NOD_PEAK_LMK_NUM,

    // ======================= 眼神唤醒检测的参数设置  ========================
    WAKING_FACE_ANGLE,
    WAKING_EYE_ANGLE_THRESHOLD,
    WAKING_MODE,

    // ========================  滑窗策略的滑窗长参数设置  ====================
    // 无感活体滑窗长度、最小帧数、占空比
    NO_INTERACT_LIVE_WINDOW_LEN,
    NO_INTERACT_MIN_LIVE_FRAMES,
    NO_INTERACT_DEFAULT_DUTY_FACTOR,
    // 人脸质量滑窗长度、最小帧数、占空比
    FACE_QUALITY_COVER_WINDOW_LEN,
    FACE_QUALITY_COVER_FRAMES,
    FACE_QUALITY_DEFAULT_DUTY_FACTOR,

    // ========================  有感活体动作相关配置参数（初始角度阈值限定)====================
    INTERACT_LIVING_ACTION,
    LIVING_START_YAW_LOWER,     // 有感活体开始条件的 yaw 范围下限
    LIVING_START_YAW_UPPER,     // 有感活体开始条件的 yaw 范围上限
    LIVING_START_PITCH_LOWER,   // 有感活体开始条件的 pitch 范围下限
    LIVING_START_PITCH_UPPER,   // 有感活体开始条件的 pitch 范围上限
    LIVING_START_ROLL_LOWER,    // 有感活体开始条件的 roll 范围下限
    LIVING_START_ROLL_UPPER,    // 有感活体开始条件的 roll 范围上限

    // ============================ 滑窗策略的触发时间以及过期时间，时间单位:ms =======================================
    STRATEGY_TRIGGER_NEED_TIME_CALL,
    STRATEGY_TRIGGER_NEED_TIME_ATTENTION,
    STRATEGY_TRIGGER_EXPIRE_TIME_SMOKE,
    STRATEGY_TRIGGER_EXPIRE_TIME_DRINK,
    STRATEGY_TRIGGER_EXPIRE_TIME_SILENCE,
    STRATEGY_TRIGGER_EXPIRE_TIME_OPEN_MOUTH,
    STRATEGY_TRIGGER_EXPIRE_TIME_CALL,
    STRATEGY_TRIGGER_EXPIRE_TIME_CLOSE_EYE,
    STRATEGY_TRIGGER_EXPIRE_TIME_YAWN,
    STRATEGY_TRIGGER_EXPIRE_TIME_ATTENTION,
    STRATEGY_TRIGGER_EXPIRE_TIME_MASK,
    STRATEGY_TRIGGER_EXPIRE_TIME_COVER_MOUTH,

    //  ========================= 设置图像检测的ROI相关信息 ===============================
    USE_DRIVER_ROI_POSITION_FILTER,             // 设置主驾ROI点的开关
    DRIVER_ROI_POSITION_X,          // 设置主驾ROI点的X轴
    DRIVER_ROI_POSITION_Y,          // 设置主驾ROI点的Y轴
    USE_DRIVER_ROI_FILTER,     // 主驾驶限定人脸检测区域
    DRIVER_ROI_LT_X,       // 主驾驶 roi left-top 的x坐标
    DRIVER_ROI_LT_Y,       // 主驾驶 roi left-top 的y坐标
    DRIVER_ROI_RB_X,       // 主驾驶 roi right-bottom 的x坐标
    DRIVER_ROI_RB_Y,       // 主驾驶 roi right-bottom 的y坐标
    IMAGE_ROI_CROP,        // 输入图像特定区域裁剪
    IMAGE_ROI_LT_X,        // 裁剪框left-top 的x坐标
    IMAGE_ROI_LT_Y,        // 裁剪框left-top 的y坐标
    IMAGE_ROI_RB_X,        // 裁剪框right-bottom 的x坐标
    IMAGE_ROI_RB_Y,        // 裁剪框right-bottom 的y坐标

    // =============== 感知能力计算调度策略 =======================
    GESTURE_USE_ROI,                        // 手势检测使用ROI策略
    CALL_USE_MUTEX_MODE,                    // 打电话使用左右耳互斥模式
    THREAD_AFFINITY_POLICY,                 // 设置线程亲和性策略
    RELEASE_MODE,          // 发版模式
    MAX_NUM_THREADS,       // 允许的并行计算线程数
    PRODUCT_NUMBER,        // 产品车型
    NATIVE_LOG_LEVEL,      // native 层日志等级
    USE_INTERNAL_MEM,      // 是否由 sdk 内部管理内存
    SCHEDULE_METHOD,                        // 原子能力执行调度策略
    SCHEDULE_DAG_THREAD_COUNT,              // DAG 调度器线程数量

    //宠物与婴儿检测能力的配置
    STRATEGY_TRIGGER_NEED_TIME_PLAY_PHONE,
    STRATEGY_TRIGGER_EXPIRE_TIME_PLAY_PHONE,
    BIOLOGY_CATEGORY_COUNT,
    BIOLOGY_CATEGORY_THRESHOLD,
    STRATEGY_TRIGGER_NEED_TIME_CATEGORY,

    // 唇动检测滑窗时间，单位ms
    LIP_MOVEMENT_WINDOW_TIME,

    KEY_LENGTH    // 标记ParamKey的数量，所有有效字段需要定义在该字段之前

};

class VisionManagerRegistry;

class RtConfig {
public:
    int sourceId;
    // 输入图像参数
    float frameWidth;
    float frameHeight;
    float frameFormat;
    float frameConvertGrayFormat;
    float frameConvertRgbFormat;
    float frameConvertBgrFormat;

    // 检测参数
    float faceMaxCount;                 // 最多支持人脸数(算法能力)
    float gestureMaxCount;              // 最多支持手势数(算法能力)
    float bodyMaxCount;                 // 最多支持肢体数(算法能力)
    float livingMaxCount;               // 最多支持活体数(算法能力)
    float faceNeedDetectCount;          // 人脸需要检测的数量。（上层业务需求配置）
    float gestureNeedDetectCount;       // 手势需要检测的数量。（上层业务需求配置）
    float bodyNeedDetectCount;          // 肢体需要检测的数量。（上层业务需求配置）
    float livingNeedDetectCount;        // 活体需要检测的数量。（上层业务需求配置）

    // ============================  Camera硬件内参标定相关  ================================================
    // 注意：如果有DMS的注意力检测功能 务必需要配置根据如下这些参数：
    // 参照摄像头规格说明
    float cameraFocalLength;    // 焦距,单位mm
    // 需要摄像头厂商提供。如果没有提供。根据摄像头的FOV进行估计：
    // 水平视场角FOV_W = 2*arctan(0.5*ccd_width/cameraFocalLength)
    // 垂直视场角FOV_H = 2*arctan(0.5*ccd_height/cameraFocalLength)
    float cameraCcdWidth;  // 感光元件宽度,单位mm
    float cameraCcdHeight; // 感光元件高度,单位mm
    // 实车标定：人脸和摄像头的平面交集的点 与 摄像头的相对距离
    // 以前方摄像头所在位置构建平面，视线望向正前方形成交集的点。由摄像头作为左上角远点。看交集的点的坐标
    float cameraPositionX;     // 摄像头位置x
    float cameraPositionY;     // 摄像头位置y
    float cameraFocalLengthPixelX;
    float cameraFocalLengthPixelY;
    float cameraOpticalCenterX;
    float cameraOpticalCenterY;
    float cameraDistortionK1;
    float cameraDistortionK2;
    float cameraDistortionK3;
    float cameraDistortionP1;
    float cameraDistortionP2;
    float cameraDistortionP3;
    float cameraImageMirror;        // 摄像头是否是镜像的
    float cameraLightType;          // 采光类型: RGB or IR
    float cameraLightSwapMode;

    // ============================  车辆信息相关的阈值和参数  ================================================
    float eyeGazeCalibSwitcher;
    float speedThreshold;
    float steeringWheelAngleThreshold;

    // ======================= 设置各种目标检测的检测阈值  =====================================================
    float faceRectThreshold;
    float faceRectMinPixelThreshold;    // 最小人脸框阈值，小于阈值则表示没有检测到人脸
    float landmarkDetectScene;                          // 人脸检测场景，两类：normal=0 和 faceid=1，faceid 场景下关键点阈值不同
    float landmarkThresholdNormal;                      // 人脸关键点检测场景：Normal
    float landmarkThresholdFaceid;                      // 人脸关键点检测场景：FaceId
    float faceFeatureCompareThreshold;                  // 人脸特征值比对阈值，主线默认0.51
    float faceCoverFeatureCompareThreshold;             // 人脸遮挡情况下的比对阈值，主线模型0.33 用于戴口罩进行人脸识别
    float noInteractiveLivingRgbThreshold;                    // 无感活体RGB模型阈值
    float noInteractiveLivingIrThreshold;                     // 无感活体IR模型阈值
    float eyeCenterThreshold;                                 // 主线瞳孔中心点模型检测到瞳孔的阈值
    float eyeDetectedThreshold;                               // 主线瞳孔中心点模型检测到人眼的阈值
    float callThreshold;
    float gestureRectThreshold;
    float gestureLandmarkThreshold;
    float _s_gesture_lm_type_threshold;
    float smallEyeThreshold;                                // 小眼睛参数
    float faceUpFrontYawLower;
    float faceUpFrontYawUpper;
    float faceUpFrontPitchLower;
    float faceUpFrontPitchUpper;
    float cameraCoverThresholdIr;
    float cameraCoverThresholdRgb;

    // ======================= 设置各种目标检测的检测阈值  ======================================
    // 危险驾驶检测约束
    float usePrevFrameToDetectDms; // 是否在无人脸时使用上一帧变换矩阵检测当前帧图像是否喝水
    // ============================  危险驾驶角度限制(左正右负、上正下负)  =======================
    float dangerDrivePitchLower;    // 危险驾驶行为检测pitch角度下限
    float dangerDrivePitchUpper;    // 危险驾驶行为检测pitch角度下限
    float dangerDriveYawLower;      // 危险驾驶行为检测人脸yaw角度下限
    float dangerDriveYawUpper;      // 危险驾驶行为检测人脸yaw角度上限

    // ============================ 疲劳检测相关的阈值限定（左正右负、上正下负）====================
    float eyeCloseAngleLimitSwitch;     // 疲劳检测中闭眼的角度阈值检查的开关，默认开启
    float openEyeBlink;                     // 感知能力eyeBlink设置
    float eyeClosePitchUpper;           // 闭眼检测的人脸上下pitch角度限制
    float eyeCloseYawLower;             // 闭眼检测人脸左右yaw角度的下限
    float eyeCloseYawUpper;             // 闭眼检测人脸左右yaw角度的上限

    // ============================ 视线偏移角度阈值 ============================
    // 注意力角度阈值计算是根据yaw和pitch角度以及人脸相对摄像头角度计算所得  数值范围：左正右负、上正下负
    float attentionHeadRightAngle;
    float attentionHeadLeftAngle;
    float attentionHeadDownAngle;
    float attentionHeadUpAngle;

    // ============================ 多模行为点摇头的参数设置 ============================
    // 点摇头参数配置 --angle
    float shakeExtremumDistanceAngle;
    float nodExtremumDistanceAngle;
    float shakeExtremumNumberAngle;
    float nodExtremumNumberAngle;
    // 点摇头参数配置 --landmark
    float shakeExtremumDistanceLandmark;
    float nodExtremumDistanceLandmark;
    float shakeExtremumNumberLandmark;
    float nodExtremumNumberLandmark;

    // ======================= 眼神唤醒检测的参数设置  ========================
    float wakingFaceAngle;
    float wakingEyeAngleThreshold;
    float wakingMode;

    // ========================  滑窗策略的滑窗长参数设置  ====================
    // 无感活体滑窗策略参数
    float faceNoInteractLiveWindowLen;
    float faceNoInteractMinLiveFrames;
    float faceNoInteractDefaultDutyFactor;
    // 人脸遮挡滑窗策略参数
    float faceQualityCoverWindowLen;
    float faceQualityCoverFrames;
    float faceQualityDefaultDutyFactor;

    // ========================  有感活体动作相关配置参数（初始角度阈值限定)====================
    float interactLivingAction;            // 有感活体动作
    float livingStartYawUpper;
    float livingStartYawLower;            // 有感活体开始条件的 yaw 范围下限
    float livingStartPitchUpper;          // 有感活体开始条件的 yaw 范围上限
    float livingStartPitchLower;
    float livingStartRollUpper;
    float livingStartRollLower;

    // ======================== 滑窗策略的触发时间以及过期时间，时间单位:ms =====================
    float livingDetectTriggerNeedTime;
    float playPhoneTriggerNeedTime;
    float playPhoneTriggerExpireTime;
    float callTriggerNeedTime;
    float callTriggerExpireTime;
    float attentionTriggerNeedTime;
    float attentionTriggerExpireTime;
    float smokeTriggerExpireTime;
    float drinkTriggerExpireTime;
    float silenceTriggerExpireTime;
    float openMouthTriggerExpireTime;
    float closeEyeTriggerExpireTime;
    float yawnTriggerExpireTime;
    float maskTriggerExpireTime;
    float coverMouthTriggerExpireTime;

    // ======================== 感知能力图像驾驶区域设置 ==================================
    // 主驾驶检测区域ROI参数
    float useDriverRoiPositionFilter;
    float driverRoiPositionX;
    float driverRoiPositionY;
    float useDriverRoiRectFilter;
    float driverRoiLeftTopX;
    float driverRoiLeftTopY;
    float driverRoiRightBottomX;
    float driverRoiRightBottomY;
    // 输入图像裁剪参数
    float inputImageNeedCrop;
    float inputCropRoiLeftTopX;
    float inputCropRoiLeftTopY;
    float inputCropRoiRightBottomX;
    float inputCropRoiRightBottomY;

    // =============== 感知能力计算调度策略 =======================
    float faceDetectMethod;
    float gestureUseRoi;
    float callUseMutexMode;                          // 打电话左右耳互斥策略
    float threadAffinityPolicy;                      // 设置线程亲和性策略
    float predictorThreadCount;
    float scheduleMethod;                            // 感知能力检测schedule的方式：Naive、Dag
    float scheduleDagThreadCount;                    // 感知能力检测调度的线程数
    static short scheduleHtp;                        // 感知能力设置指定HTP(指定HTP0或者HTP1)
    float fixedFrameDetectSwitcher;                  // 感知能力层设置启用间隔帧检测的策略开关


    float logLevel;                 // 日志级别
    float productNumber;            // 项目编号
    float releaseMode;
    float useInternalMem;           // 是否使用外部 buffer 作为 FaceInfo、GestureInfo 的数据存储

    // 唇动检测滑窗时间，单位ms
    float lipMovementWindowTime;

    RtConfig();
    ~RtConfig();

    void init();

    void deinit();

    void set_config(int key, float value);
    float get_config(int key);

    void set_switch(short ability, bool enable);
    void set_switches(const std::unordered_map<short, bool>& switches);
    bool get_switch(short ability);
    const std::unordered_map<short, bool>& get_switches();

    void inject_manager_registry(const std::shared_ptr<VisionManagerRegistry>& managers);
    std::shared_ptr<VisionManagerRegistry> getManagerRegistry() const;

private:
    void init_params();
    void init_switches();

    std::unordered_map<short, bool> _switches;
    std::unordered_map<short, float*> _configs;
    std::shared_ptr<VisionManagerRegistry> _managers;
};

} // namespace vision
