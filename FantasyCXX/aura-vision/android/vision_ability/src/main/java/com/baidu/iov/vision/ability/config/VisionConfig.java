package com.baidu.iov.vision.ability.config;

import com.baidu.iov.vision.ability.util.ALog;

/**
 * 视觉能力配置类
 * <p>
 * 开发测试过程中，可用 VAST 工具修改，实际项目 Release 版本中，需将实际配置写死在该类中
 * <p>
 * create by v_liuyong01 on 2019/2/20.
 */
public class VisionConfig {

    // ---------------------------------------------------------------------------------------------
    //  通用配置
    public static void setDebug(boolean debug) {
        ALog.DEBUG = debug;
    }

    // ---------------------------------------------------------------------------------------------
    // 缓存配置
    public static final byte REQUEST_CACHE_SIZE = 1;
    public static final byte RESULT_CACHE_SIZE = REQUEST_CACHE_SIZE;
    public static final byte TASK_CACHE_SIZE = 1;

    // ---------------------------------------------------------------------------------------------
    // 线程池配置

    // 各线程池线程数量 默认值
    public static final byte EXECUTOR_TASK_QUEUE_SIZE = 1;
    public static final byte EXECUTOR_TASK_THREAD_SIZE = 1;
//    public static final byte EXECUTOR_DISK_THREAD_SIZE = 1;
//    public static final byte EXECUTOR_NETWORK_THREAD_SIZE = 1;

    // 各线程池线程数量
    public static byte executorTaskQueueSize = EXECUTOR_TASK_QUEUE_SIZE;
    public static byte executorTaskThreadSize = EXECUTOR_TASK_THREAD_SIZE;
//    public static int executorDiskThreadSize = EXECUTOR_DISK_THREAD_SIZE;
//    public static int executorNetworkThreadSize = EXECUTOR_NETWORK_THREAD_SIZE;

    // ---------------------------------------------------------------------------------------------
    // 能力模型配置

    // 与 Native 层的请求编号保持一致：
    public static final short ABILITY_UNKNOWN = -1;
    public static final short ABILITY_ALL = 1;
    public static final short ABILITY_FACE = 1000;
    public static final short ABILITY_FACE_DETECTION = 1001;
    public static final short ABILITY_FACE_RECT = ABILITY_FACE + 10;
    public static final short ABILITY_FACE_LANDMARK = ABILITY_FACE + 20;
    public static final short ABILITY_FACE_LANDMARK_68P = ABILITY_FACE + 21;
    public static final short ABILITY_FACE_2DTO3D = ABILITY_FACE + 30;
    public static final short ABILITY_FACE_DANGEROUS_DRIVING = ABILITY_FACE + 40;
    public static final short ABILITY_FACE_FATIGUE = ABILITY_FACE + 50;
    public static final short ABILITY_FACE_ATTENTION = ABILITY_FACE + 60;
    public static final short ABILITY_FACE_HEAD_BEHAVIOR = ABILITY_FACE + 70;
    public static final short ABILITY_FACE_EYE_TRACKING = ABILITY_FACE + 80;
    public static final short ABILITY_FACE_EYE_WAKING = ABILITY_FACE + 81;
    public static final short ABILITY_FACE_EYE_GAZE = ABILITY_FACE + 82;
    public static final short ABILITY_FACE_EYE_CENTER = ABILITY_FACE + 83;
    public static final short ABILITY_FACE_MOUTH_LANDMARK = ABILITY_FACE + 85;
    public static final short ABILITY_FACE_INTERACTIVE_LIVING = ABILITY_FACE + 90;
    public static final short ABILITY_FACE_FEATURE = ABILITY_FACE + 100;
    public static final short ABILITY_FACE_CALL = ABILITY_FACE + 110;
    public static final short ABILITY_FACE_AR_HUD = ABILITY_FACE + 120;
    public static final short ABILITY_FACE_NO_INTERACTIVE_LIVING = ABILITY_FACE + 130;
    public static final short ABILITY_FACE_ATTRIBUTE = ABILITY_FACE + 140;
    public static final short ABILITY_FACE_SMILE = ABILITY_FACE + 150;
    public static final short ABILITY_FACE_EMOTION = ABILITY_FACE + 160;
    public static final short ABILITY_FACE_QUALITY = ABILITY_FACE + 170;
    public static final short ABILITY_SOURCE2_CAMERA_COVER = ABILITY_FACE + 180;

    public static final short ABILITY_GESTURE = 2000;
    public static final short ABILITY_GESTURE_RECT = ABILITY_GESTURE + 10;
    public static final short ABILITY_GESTURE_LANDMARK = ABILITY_GESTURE + 20;
    public static final short ABILITY_GESTURE_TYPE = ABILITY_GESTURE + 30;
    // 动态手势
    public static final short ABILITY_GESTURE_DYNAMIC = ABILITY_GESTURE + 40;


    public static final short ABILITY_FRAME = 3000;
    public static final short ABILITY_FRAME_BRIGHTNESS = ABILITY_FRAME + 10;

    public static final short ABILITY_VIS_FACE = 4000;
    public static final short ABILITY_VIS_FACE_RECT = ABILITY_VIS_FACE + 10;
    public static final short ABILITY_VIS_FACE_LANDMARK = ABILITY_VIS_FACE + 20;
    public static final short ABILITY_VIS_FACE_NO_INTERACTIVE_LIVING = ABILITY_VIS_FACE + 30;
    public static final short ABILITY_VIS_FACE_RECOGNIZE = ABILITY_VIS_FACE + 40;
    public static final short ABILITY_VIS_FACE_EMOTION = ABILITY_VIS_FACE + 50;
    public static final short ABILITY_VIS_FACE_DANGEROUS = ABILITY_VIS_FACE + 60;

    public static final short ABILITY_VIS_GESTURE = 5000;
    public static final short ABILITY_VIS_GESTURE_RECT = ABILITY_VIS_GESTURE + 10;
    public static final short ABILITY_VIS_GESTURE_CATEGORY = ABILITY_VIS_GESTURE + 20;

    public static final short ABILITY_BODY = 6000;

    public static final short ABILITY_PERSON_VEHICLE = 7000;
    public static final short ABILITY_PERSON_VEHICLE_RECT = ABILITY_PERSON_VEHICLE + 10;

    public static final short ABILITY_FACE_RECONSTRUCT = 8000;

    // assets 中的 Iov 能力配置文件路径
    public static String IOV_VISION_CONFIG_FILEPATH = "vision_native_config.json";

    public static final short EXEC_MODE_SERIAL = 0; // 每帧串行执行
    public static final short EXEC_MODE_PIPELINE = 1; // 支持流水线执行
    public static short execMode = EXEC_MODE_SERIAL;

    // ---------------------------------------------------------------------------------------------
    // 请求相关配置
    public static short desireFrameWidth = 600;
    public static short desireFrameHeight = 600;

    // ---------------------------------------------------------------------------------------------
    // 结果相关配置
    public static final int CALLBACK_THREAD_MAIN = 1;
    public static final int CALLBACK_THREAD_CURRENT = 2;

    /**
     * 针对奇瑞车机保存底图的角度（左上右下）
     */
    public static final float[] faceUpFrontThreshold = new float[]{30, 8, 18, 15};
    /**
     * 结果对象返回所在线程，可指定在主线程、原计算操作子线程中返回，默认检测线程
     */
    public static int callbackThreadMode = CALLBACK_THREAD_CURRENT;

    // ---------------------------------------------------------------------------------------------
    // 运行时可配置项

    // 如果有需要在 Java 层保留每个配置项的变量，则可以使用 ConfigHolder
    //    private static class ConfigHolder {
    //        public float value;
    //        ConfigHolder(float v) {
    //            value = v;
    //        }
    //    }

    public static final short FRAME_WIDTH = 1280;
    public static final short FRAME_HEIGHT = 720;
//    public static ConfigHolder frameWidth = new ConfigHolder(FRAME_WIDTH);
//    public static ConfigHolder frameHeight = new ConfigHolder(FRAME_HEIGHT);

    public static final short FACE_MAX_COUNT = 5;
    public static final short GESTURE_MAX_COUNT = 1;
    public static final short BODY_MAX_COUNT = 1;
    public static final short FACE_NEED_CHECK_COUNT = 1;

    public static final short CAMERA_LIGHT_TYPE_RGB = 0; // 可见光类型
    public static final short CAMERA_LIGHT_TYPE_IR = 1;  // 红外光类型
    public static final float CAMERA_FOCAL_LENGTH = 2.1F;
    public static final float CAMERA_CCD_WIDTH = 1.78F;
    public static final float CAMERA_CCD_HEIGHT = 1.378F;
    public static final float CAMERA_POSITION_X = 0.F;
    public static final float CAMERA_POSITION_Y = -2.F;

    public static final float FACE_RECT_THRESHOLD = 0.6F;
    public static final short LMK_DETECT_SCENARIO_NORMAL = 0;
    public static final short LMK_DETECT_SCENARIO_FACEID = 1;
    public static final float FACE_LANDMARK_THRESHOLD_NORMAL = 0.8F;
    public static final float FACE_LANDMARK_THRESHOLD_FACEID = 0.95F;
    public static final float FACE_FEATURE_COMPARE_THRESHOLD = 0.51F;
    public static final float FACE_COVER_FEATURE_COMPARE_THRESHOLD = 0.33F;
    public static final float NO_INTERACTIVE_LIVING_RGB_THRESHOLD = 49F;
    public static final float NO_INTERACTIVE_LIVING_IR_THRESHOLD = 0.78F;
    public static final float FACE_EYE_CENTER_THRESHOLD = 0.3F;
    public static final float CALL_THRESHOLD = 0.5F;
    public static final float GESTURE_RECT_THRESHOLD = 0.3F;
    public static final float GESTURE_LMK_THRESHOLD = 0.6F;
    public static final float GESTURE_LMK_TYPE_THRESHOLD = 0.85F;
    public static final float FACE_UP_FRONT_YAW_MIN = -15F;
    public static final float FACE_UP_FRONT_YAW_MAX = 15F;
    public static final float FACE_UP_FRONT_PITCH_MIN = -15F;
    public static final float FACE_UP_FRONT_PITCH_MAX = 30F;
    public static final float SMALL_EYE_THRESHOLD = 0.3F;

    public static final short USE_PREV_FRAME_TO_DETECT_DMS = 0;
    public static final float DANGER_DRIVE_PITCH_LIMIT = 7F;
    public static final float DANGER_DRIVE_YAW_MIN = -30F;
    public static final float DANGER_DRIVE_YAW_MAX = 30F;

    public static final float ATTENTION_HEAD_RIGHT_ANGLE = -13F;
    public static final float ATTENTION_HEAD_LEFT_ANGLE = 15F;
    public static final float ATTENTION_HEAD_DOWN_ANGLE = 0F;
    public static final float ATTENTION_HEAD_UP_ANGLE = 30F;

    public static final float SHAKE_PEAK_ANGLE = 5F;
    public static final float NOD_PEAK_ANGLE = 5.5F;
    public static final short SHAKE_PEAK_ANGLE_NUM = 4;
    public static final short NOD_PEAK_ANGLE_NUM = 3;
    public static final float SHAKE_PEAK_LMK = 15F;
    public static final float NOD_PEAK_LMK = 12F;
    public static final short SHAKE_PEAK_LMK_NUM = 3;
    public static final short NOD_PEAK_LMK_NUM = 4;

    public static final float WAKING_FACE_ANGLE = 5F;
    public static final float WAKING_EYE_ANGLE_THRESHOLD = 0.12F;
    public static final short WAKING_MODE_HEAD = 0; // 眼神唤醒：头部偏转角
    public static final short WAKING_MODE_HEAD_EYE = 1; // 眼神唤醒：头部偏转角和瞳孔位置（眼球追踪）

    public static final short NO_INTERACT_LIVE_WINDOW_LEN = 5;
    public static final short NO_INTERACT_MIN_LIVE_FRAMES = 3;
    public static final float NO_INTERACT_DEFAULT_DUTY_FACTOR = 0.6F;
    public static final short FACE_QUALITY_COVER_WINDOW_LEN = 1;
    public static final short FACE_QUALITY_COVER_FRAMES = 1;
    public static final float FACE_QUALIRY_DEFAULT_DUTY_FACTOR = 1F;

    public static final short INTERACT_LIVING_ACTION_NONE = 0;
    public static final short INTERACT_LIVING_ACTION_HEAD_LEFT = 1;
    public static final short INTERACT_LIVING_ACTION_HEAD_RIGHT = 2;
    public static final short INTERACT_LIVING_ACTION_SHAKE_HEAD = 3;
    public static final short INTERACT_LIVING_ACTION_CLOSE_EYES = 4;
    public static final short INTERACT_LIVING_ACTION_OPEN_MOUTH = 5;

    public static final float LIVING_START_YAW_UPPER = 10F;
    public static final float LIVING_START_YAW_LOWER = -10F;
    public static final float LIVING_START_PITCH_UPPER = 10F;
    public static final float LIVING_START_PITCH_LOWER = -10F;
    public static final float LIVING_START_ROLL_UPPER = 10F;
    public static final float LIVING_START_ROLL_LOWER = -10F;

    public static final short STRATEGY_TIME_SMOKE = 0;
    public static final short STRATEGY_TIME_DRINK = 0;
    public static final short STRATEGY_TIME_SILENCE = 0;
    public static final short STRATEGY_TIME_OPEN_MOUTH = 0;
    public static final short STRATEGY_TIME_CALL = 0;
    public static final short STRATEGY_TIME_CLOSE_EYE = 0;
    public static final short STRATEGY_TIME_YAWN = 0;
    public static final short STRATEGY_TRIGGER_NEED_TIME_MILLISECOND_ATTENTION = 0;
    public static final short STRATEGY_ACCUMULATIVE_TIME_ATTENTION = 0;

    /**
     * 图像输入格式, 对应 OpenCV types_c.h
     */
    public static final short COLOR_YUV2RGB_NV12 = 90;
    public static final short COLOR_YUV2BGR_NV12 = 91;
    public static final short COLOR_YUV2RGB_NV21 = 92;
    public static final short COLOR_YUV2BGR_NV21 = 93;
    public static final short COLOR_YUV2RGBA_NV12 = 94;
    public static final short COLOR_YUV2BGRA_NV12 = 95;
    public static final short COLOR_YUV2RGBA_NV21 = 96;
    public static final short COLOR_YUV2BGRA_NV21 = 97;
    public static final short COLOR_YUV2BGR_YV12 = 99;

    /**
     * 当前正在依赖 Landmark 的功能
     */
    public static final short RELY_LANDMARK_TYPE_COMMON = 0; // deprecated, use LMK_DETECT_SCENARIO_NORMAL instead
    public static final short RELY_LANDMARK_TYPE_FACEID = 1; // deprecated, use LMK_DETECT_SCENARIO_FACEID instead


    /**
     * 项目编号
     */
    public static final short PRODUCT_NUMBER_MAINLINE = 0;
    public static final short PRODUCT_NUMBER_CHERY_32T = 1;
    public static final short PRODUCT_NUMBER_CHERY_36T2 = 2;

    /**
     * Native 日志等级
     */
    public static final short NATIVE_LOG_LEVEL_NONE = 0;
    public static final short NATIVE_LOG_LEVEL_FATAL = 1;
    public static final short NATIVE_LOG_LEVEL_ERROR = 2;
    public static final short NATIVE_LOG_LEVEL_WARN = 3;
    public static final short NATIVE_LOG_LEVEL_INFO = 4;
    public static final short NATIVE_LOG_LEVEL_DEBUG = 5;
    public static final short NATIVE_LOG_LEVEL_VERBOSE = 6;

    /**
     * 全图裁剪开关
     */
    public static final float IMAGE_ROI_CROP_ON = 1;
    public static final float IMAGE_ROI_CROP_OFF = 0;

    /**
     * 人脸区域限制开关
     */
    public static final float DRIVER_ROI_FILTER_ON = 1;
    public static final float DRIVER_ROI_FILTER_OFF = 0;

    /**
     * 手势检测区域限制开关
     */
    public static final float GESTURE_USE_ROI_ON = 1;
    public static final float GESTURE_USE_ROI_OFF = 0;

    /**
     * 打电话检测左右耳互斥策略
     */
    public static final float CALL_USE_MUTEX_MODE_ON = 1;
    public static final float CALL_USE_MUTEX_MODE_OFF = 0;

    public static final float MAX_NUM_NATIVE_THREADS = 1;

    /**
     * 发版模式
     */
    public static final float RELEASE_MODE_PRODUCT = 0;
    public static final float RELEASE_MODE_BENCHMARK_TEST = 1;
    public static final float RELEASE_MODE_DEMO = 2;

    /**
     * 线程调度模式
     */
    public static final float THREAD_POLICY_NORMAL = 0;
    public static final float THREAD_POLICY_BIG_CORE = 1;
    /**
     * Java层相关配置接口，需要保证和底层C++层的config_params.h保持一致
     */
    public enum Key {
        FRAME_WIDTH,
        FRAME_HEIGHT,
        FRAME_CONVERT_FORMAT,

        FACE_MAX_COUNT,
        GESTURE_MAX_COUNT,
        BODY_MAX_COUNT,
        HEAD_SHOULDER_MAX_COUNT,
        FACE_NEED_CHECK_COUNT,

        CAMERA_FOCAL_LENGTH,
        CAMERA_CCD_WIDTH,
        CAMERA_CCD_HEIGHT,
        CAMERA_LIGHT_TYPE,
        CAMERA_POSITION_X,
        CAMERA_POSITION_Y,

        FACE_RECT_THRESHOLD,
        LANDMARK_DETECT_SCENARIO, // 人脸关键点检测场景
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

        USE_PREV_FRAME_TO_DETECT_DMS,
        DANGER_DRIVE_PITCH_LIMIT,
        DANGER_DRIVE_YAW_MIN,
        DANGER_DRIVE_YAW_MAX,

        ATTENTION_HEAD_RIGHT_ANGLE, // 注意力右转头的角度
        ATTENTION_HEAD_LEFT_ANGLE,  // 注意力左转头的角度
        ATTENTION_HEAD_DOWN_ANGLE,  // 注意力低头的角度
        ATTENTION_HEAD_UP_ANGLE,    // 注意力抬头的角度

        SHAKE_PEAK_ANGLE,
        NOD_PEAK_ANGLE,
        SHAKE_PEAK_ANGLE_NUM,
        NOD_PEAK_ANGLE_NUM,
        SHAKE_PEAK_LMK,
        NOD_PEAK_LMK,
        SHAKE_PEAK_LMK_NUM,
        NOD_PEAK_LMK_NUM,

        WAKING_FACE_ANGLE,
        WAKING_EYE_ANGLE_THRESHOLD,
        WAKING_MODE,
        // 无感活体滑窗长度、最小帧数、占空比
        NO_INTERACT_LIVE_WINDOW_LEN,
        NO_INTERACT_MIN_LIVE_FRAMES,
        NO_INTERACT_DEFAULT_DUTY_FACTOR,

        FACE_QUALITY_COVER_WINDOW_LEN,
        FACE_QUALITY_COVER_FRAMES,
        FACE_QUALITY_DEFAULT_DUTY_FACTOR,

        INTERACT_LIVING_ACTION,
        LIVING_START_YAW_LOWER,     // 有感活体开始条件的 yaw 范围下限
        LIVING_START_YAW_UPPER,     // 有感活体开始条件的 yaw 范围上限
        LIVING_START_PITCH_LOWER,   // 有感活体开始条件的 pitch 范围下限
        LIVING_START_PITCH_UPPER,   // 有感活体开始条件的 pitch 范围上限
        LIVING_START_ROLL_LOWER,    // 有感活体开始条件的 roll 范围下限
        LIVING_START_ROLL_UPPER,    // 有感活体开始条件的 roll 范围上限

        // 策略：检测的时间 单位:ms

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

        DRIVER_ROI_FILTER,     // 主驾驶限定人脸检测区域
        DRIVER_ROI_LT_X,       // 主驾驶 roi left-top 的x坐标
        DRIVER_ROI_LT_Y,       // 主驾驶 roi left-top 的y坐标
        DRIVER_ROI_RB_X,       // 主驾驶 roi right-bottom 的x坐标
        DRIVER_ROI_RB_Y,       // 主驾驶 roi right-bottom 的y坐标
        IMAGE_ROI_CROP,        // 输入图像特定区域裁剪
        IMAGE_ROI_LT_X,        // 裁剪框left-top 的x坐标
        IMAGE_ROI_LT_Y,        // 裁剪框left-top 的y坐标
        IMAGE_ROI_RB_X,        // 裁剪框right-bottom 的x坐标
        IMAGE_ROI_RB_Y,        // 裁剪框right-bottom 的y坐标

        GESTURE_USE_ROI,       // 手势检测使用ROI策略
        CALL_USE_MUTEX_MODE,   // 打电话使用左右耳互斥模式
        THREAD_POLICY,         // 线程策略
        RELEASE_MODE,          // 发版模式
        MAX_NUM_THREADS,       // 允许的并行计算线程数
        PRODUCT_NUMBER,        // 产品车型
        NATIVE_LOG_LEVEL,      // native 层日志等级
        USE_INTERNAL_MEM,      // 是否由 sdk 内部管理内存

        CAMERA_CAMERA_IMAGE_MIRROR,    // 当前摄像头是都存在镜像反
        // 疲劳检测相关的配置开关
        USE_FACE_EYE_CLOSE_ANGLE_LIMIT_SWITCH,
        FACE_EYE_CLOSE_PITCH_LIMIT,           // 闭眼检测的人脸上下pitch角度限制
        FACE_EYE_CLOSE_YAW_MIN,               // 闭眼检测人脸左右yaw角度的下限
        FACE_EYE_CLOSE_YAW_MAX,               // 闭眼检测人脸左右yaw角度的上限

        SCHEDULE_METHOD, // 原子能力执行调度策略
        SCHEDULE_DAG_THREAD_COUNT, // DAG 调度器线程数量

        // 宠物与婴儿检测能力的配置
        EYE_DETECTED_THRESHOLD, // 是否检测到眼睛，置信度配置
        STRATEGY_TRIGGER_NEED_TIME_PLAY_PHONE,
        STRATEGY_TRIGGER_EXPIRE_TIME_PLAY_PHONE,
        BIOLOGY_CATEGORY_COUNT,
        BIOLOGY_CATEGORY_THRESHOLD,
        STRATEGY_TRIGGER_NEED_TIME_CATEGORY,
        FRAME_CONVERT_RGB,

        KEY_LENGTH    // 标记ParamKey的数量

    }

    // Java 层暂无强需求再保存一份，如有必要考虑添加
    private static float[] configs = new float[Key.KEY_LENGTH.ordinal()];

    public static void config(Key key, float value) {
        configs[key.ordinal()] = value;
    }

    public static float config(Key key) {
        return configs[key.ordinal()];
    }

    public static short configAsShort(Key key) {
        return (short) configs[key.ordinal()];
    }

    public static int configAsInt(Key key) {
        return (int) configs[key.ordinal()];
    }

    // TODO 现在配置信息都是从底层获取，所以 runtime 运行时 config 代码是否可以去除
    static {
        configs[Key.FRAME_WIDTH.ordinal()] = FRAME_WIDTH;
        configs[Key.FRAME_HEIGHT.ordinal()] = FRAME_HEIGHT;
        configs[Key.FRAME_CONVERT_FORMAT.ordinal()] = COLOR_YUV2BGR_NV21;

        configs[Key.FACE_MAX_COUNT.ordinal()] = FACE_MAX_COUNT;
        configs[Key.GESTURE_MAX_COUNT.ordinal()] = GESTURE_MAX_COUNT;
        configs[Key.BODY_MAX_COUNT.ordinal()] = BODY_MAX_COUNT;
        configs[Key.FACE_NEED_CHECK_COUNT.ordinal()] = FACE_NEED_CHECK_COUNT;

        configs[Key.CAMERA_FOCAL_LENGTH.ordinal()] = CAMERA_FOCAL_LENGTH;
        configs[Key.CAMERA_CCD_WIDTH.ordinal()] = CAMERA_CCD_WIDTH;
        configs[Key.CAMERA_CCD_HEIGHT.ordinal()] = CAMERA_CCD_HEIGHT;
        configs[Key.CAMERA_LIGHT_TYPE.ordinal()] = CAMERA_LIGHT_TYPE_RGB;
        configs[Key.CAMERA_POSITION_X.ordinal()] = CAMERA_POSITION_X;
        configs[Key.CAMERA_POSITION_Y.ordinal()] = CAMERA_POSITION_Y;

        configs[Key.FACE_RECT_THRESHOLD.ordinal()] = FACE_RECT_THRESHOLD;
        configs[Key.LANDMARK_DETECT_SCENARIO.ordinal()] = LMK_DETECT_SCENARIO_NORMAL;
        configs[Key.FACE_LMK_THRESHOLD_NORMAL.ordinal()] = FACE_LANDMARK_THRESHOLD_NORMAL;
        configs[Key.FACE_LMK_THRESHOLD_NORMAL.ordinal()] = FACE_LANDMARK_THRESHOLD_FACEID;
        configs[Key.FACE_FEATURE_COMPARE_THRESHOLD.ordinal()] = FACE_FEATURE_COMPARE_THRESHOLD;
        configs[Key.FACE_COVER_FEATURE_COMPARE_THRESHOLD.ordinal()] = FACE_COVER_FEATURE_COMPARE_THRESHOLD;
        configs[Key.NO_INTERACT_LIVING_RGB_THRESHOLD.ordinal()] = NO_INTERACTIVE_LIVING_RGB_THRESHOLD;
        configs[Key.NO_INTERACT_LIVING_IR_THRESHOLD.ordinal()] = NO_INTERACTIVE_LIVING_IR_THRESHOLD;
        configs[Key.FACE_EYE_CENTER_THRESHOLD.ordinal()] = FACE_EYE_CENTER_THRESHOLD;
        configs[Key.CALL_THRESHOLD.ordinal()] = CALL_THRESHOLD;
        configs[Key.GESTURE_RECT_THRESHOLD.ordinal()] = GESTURE_RECT_THRESHOLD;
        configs[Key.GESTURE_LMK_THRESHOLD.ordinal()] = GESTURE_LMK_THRESHOLD;
        configs[Key.GESTURE_LMK_TYPE_THRESHOLD.ordinal()] = GESTURE_LMK_TYPE_THRESHOLD;
        configs[Key.FACE_UP_FRONT_YAW_MIN.ordinal()] = FACE_UP_FRONT_YAW_MIN;
        configs[Key.FACE_UP_FRONT_YAW_MAX.ordinal()] = FACE_UP_FRONT_YAW_MAX;
        configs[Key.FACE_UP_FRONT_PITCH_MIN.ordinal()] = FACE_UP_FRONT_PITCH_MIN;
        configs[Key.FACE_UP_FRONT_PITCH_MAX.ordinal()] = FACE_UP_FRONT_PITCH_MAX;
        configs[Key.SMALL_EYE_THRESHOLD.ordinal()] = SMALL_EYE_THRESHOLD;

        configs[Key.USE_PREV_FRAME_TO_DETECT_DMS.ordinal()] = USE_PREV_FRAME_TO_DETECT_DMS;
        configs[Key.DANGER_DRIVE_PITCH_LIMIT.ordinal()] = DANGER_DRIVE_PITCH_LIMIT;
        configs[Key.DANGER_DRIVE_YAW_MIN.ordinal()] = DANGER_DRIVE_YAW_MIN;
        configs[Key.DANGER_DRIVE_YAW_MAX.ordinal()] = DANGER_DRIVE_YAW_MAX;

        configs[Key.ATTENTION_HEAD_RIGHT_ANGLE.ordinal()] = ATTENTION_HEAD_RIGHT_ANGLE;
        configs[Key.ATTENTION_HEAD_LEFT_ANGLE.ordinal()] = ATTENTION_HEAD_LEFT_ANGLE;
        configs[Key.ATTENTION_HEAD_DOWN_ANGLE.ordinal()] = ATTENTION_HEAD_DOWN_ANGLE;
        configs[Key.ATTENTION_HEAD_UP_ANGLE.ordinal()] = ATTENTION_HEAD_UP_ANGLE;

        configs[Key.SHAKE_PEAK_ANGLE.ordinal()] = SHAKE_PEAK_ANGLE;
        configs[Key.NOD_PEAK_ANGLE.ordinal()] = NOD_PEAK_ANGLE;
        configs[Key.SHAKE_PEAK_ANGLE_NUM.ordinal()] = SHAKE_PEAK_ANGLE_NUM;
        configs[Key.NOD_PEAK_ANGLE_NUM.ordinal()] = NOD_PEAK_ANGLE_NUM;
        configs[Key.SHAKE_PEAK_LMK.ordinal()] = SHAKE_PEAK_LMK;
        configs[Key.NOD_PEAK_LMK.ordinal()] = NOD_PEAK_LMK;
        configs[Key.SHAKE_PEAK_LMK_NUM.ordinal()] = SHAKE_PEAK_LMK_NUM;
        configs[Key.NOD_PEAK_LMK_NUM.ordinal()] = NOD_PEAK_LMK_NUM;

        configs[Key.WAKING_FACE_ANGLE.ordinal()] = WAKING_FACE_ANGLE;
        configs[Key.WAKING_EYE_ANGLE_THRESHOLD.ordinal()] = WAKING_EYE_ANGLE_THRESHOLD;
        configs[Key.WAKING_MODE.ordinal()] = WAKING_MODE_HEAD;

        configs[Key.NO_INTERACT_LIVE_WINDOW_LEN.ordinal()] = NO_INTERACT_LIVE_WINDOW_LEN;
        configs[Key.NO_INTERACT_MIN_LIVE_FRAMES.ordinal()] = NO_INTERACT_MIN_LIVE_FRAMES;
        configs[Key.NO_INTERACT_DEFAULT_DUTY_FACTOR.ordinal()] = NO_INTERACT_DEFAULT_DUTY_FACTOR;

        configs[Key.FACE_QUALITY_COVER_WINDOW_LEN.ordinal()] = FACE_QUALITY_COVER_WINDOW_LEN;
        configs[Key.FACE_QUALITY_COVER_FRAMES.ordinal()] = FACE_QUALITY_COVER_FRAMES;
        configs[Key.FACE_QUALITY_DEFAULT_DUTY_FACTOR.ordinal()] = FACE_QUALIRY_DEFAULT_DUTY_FACTOR;

        configs[Key.INTERACT_LIVING_ACTION.ordinal()] = INTERACT_LIVING_ACTION_NONE;
        configs[Key.LIVING_START_YAW_LOWER.ordinal()] = LIVING_START_YAW_LOWER;
        configs[Key.LIVING_START_YAW_UPPER.ordinal()] = LIVING_START_YAW_UPPER;
        configs[Key.LIVING_START_PITCH_LOWER.ordinal()] = LIVING_START_PITCH_LOWER;
        configs[Key.LIVING_START_PITCH_UPPER.ordinal()] = LIVING_START_PITCH_UPPER;
        configs[Key.LIVING_START_ROLL_LOWER.ordinal()] = LIVING_START_ROLL_LOWER;
        configs[Key.LIVING_START_ROLL_UPPER.ordinal()] = LIVING_START_ROLL_UPPER;
        configs[Key.STRATEGY_TRIGGER_NEED_TIME_CALL.ordinal()] = STRATEGY_TIME_CALL;
        configs[Key.STRATEGY_TRIGGER_NEED_TIME_ATTENTION.ordinal()] =
                STRATEGY_TRIGGER_NEED_TIME_MILLISECOND_ATTENTION;
        configs[Key.STRATEGY_TRIGGER_EXPIRE_TIME_SMOKE.ordinal()] = STRATEGY_TIME_SMOKE;
        configs[Key.STRATEGY_TRIGGER_EXPIRE_TIME_DRINK.ordinal()] = STRATEGY_TIME_DRINK;
        configs[Key.STRATEGY_TRIGGER_EXPIRE_TIME_SILENCE.ordinal()] = STRATEGY_TIME_SILENCE;
        configs[Key.STRATEGY_TRIGGER_EXPIRE_TIME_OPEN_MOUTH.ordinal()] = STRATEGY_TIME_OPEN_MOUTH;
        configs[Key.STRATEGY_TRIGGER_EXPIRE_TIME_CALL.ordinal()] = STRATEGY_TIME_CALL;
        configs[Key.STRATEGY_TRIGGER_EXPIRE_TIME_CLOSE_EYE.ordinal()] = STRATEGY_TIME_CLOSE_EYE;
        configs[Key.STRATEGY_TRIGGER_EXPIRE_TIME_YAWN.ordinal()] = STRATEGY_TIME_YAWN;
        configs[Key.STRATEGY_TRIGGER_EXPIRE_TIME_ATTENTION.ordinal()] =
                STRATEGY_ACCUMULATIVE_TIME_ATTENTION;

        configs[Key.DRIVER_ROI_FILTER.ordinal()] = DRIVER_ROI_FILTER_OFF;
        configs[Key.DRIVER_ROI_LT_X.ordinal()] = 0;
        configs[Key.DRIVER_ROI_LT_Y.ordinal()] = 0;
        configs[Key.DRIVER_ROI_RB_X.ordinal()] = FRAME_WIDTH;
        configs[Key.DRIVER_ROI_RB_Y.ordinal()] = FRAME_HEIGHT;

        configs[Key.IMAGE_ROI_CROP.ordinal()] = IMAGE_ROI_CROP_OFF;
        configs[Key.IMAGE_ROI_LT_X.ordinal()] = 0;
        configs[Key.IMAGE_ROI_LT_Y.ordinal()] = 0;
        configs[Key.IMAGE_ROI_RB_X.ordinal()] = FRAME_WIDTH;
        configs[Key.IMAGE_ROI_RB_Y.ordinal()] = FRAME_HEIGHT;

        configs[Key.GESTURE_USE_ROI.ordinal()] = GESTURE_USE_ROI_OFF;
        configs[Key.CALL_USE_MUTEX_MODE.ordinal()] = CALL_USE_MUTEX_MODE_OFF;
        configs[Key.THREAD_POLICY.ordinal()] = THREAD_POLICY_NORMAL;
        configs[Key.RELEASE_MODE.ordinal()] = RELEASE_MODE_PRODUCT;
        configs[Key.MAX_NUM_THREADS.ordinal()] = MAX_NUM_NATIVE_THREADS;
        configs[Key.PRODUCT_NUMBER.ordinal()] = PRODUCT_NUMBER_MAINLINE;
        configs[Key.NATIVE_LOG_LEVEL.ordinal()] = NATIVE_LOG_LEVEL_VERBOSE;
        configs[Key.USE_INTERNAL_MEM.ordinal()] = 1;
    }

}
