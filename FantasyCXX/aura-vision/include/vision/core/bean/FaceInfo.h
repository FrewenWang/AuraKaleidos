#ifndef VISION_FACE_INFO_H
#define VISION_FACE_INFO_H

#include <cstring>
#include "vision/core/common/VConstants.h"
#include "vision/core/common/VStructs.h"
#include "vision/core/common/VTensor.h"

namespace aura::vision {

/**
 * @brief 2d landmark (106 points) definition
 */
enum FaceLandmark : short {
    // face contour: 0~32
    FLM_0_CT_LEFT1 = 0,
    FLM_1_CT_LEFT2 = 1,
    FLM_2_CT_LEFT3 = 2,
    FLM_3_CT_LEFT4 = 3,
    FLM_4_CT_LEFT5 = 4,
    FLM_5_CT_LEFT6 = 5,
    FLM_6_CT_LEFT7 = 6,
    FLM_7_CT_LEFT8 = 7,
    FLM_8_CT_LEFT9 = 8,
    FLM_9_CT_LEFT10 = 9,
    FLM_10_CT_LEFT11 = 10,
    FLM_11_CT_LEFT12 = 11,
    FLM_12_CT_LEFT13 = 12,
    FLM_13_CT_LEFT14 = 13,
    FLM_14_CT_LEFT15 = 14,
    FLM_15_CT_LEFT16 = 15,
    FLM_16_CT_CHIN   = 16,
    FLM_17_CT_RIGHT16 = 17,
    FLM_18_CT_RIGHT15 = 18,
    FLM_19_CT_RIGHT14 = 19,
    FLM_20_CT_RIGHT13 = 20,
    FLM_21_CT_RIGHT12 = 21,
    FLM_22_CT_RIGHT11 = 22,
    FLM_23_CT_RIGHT10 = 23,
    FLM_24_CT_RIGHT9 = 24,
    FLM_25_CT_RIGHT8 = 25,
    FLM_26_CT_RIGHT7 = 26,
    FLM_27_CT_RIGHT6 = 27,
    FLM_28_CT_RIGHT5 = 28,
    FLM_29_CT_RIGHT4 = 29,
    FLM_30_CT_RIGHT3 = 30,
    FLM_31_CT_RIGHT2 = 31,
    FLM_32_CT_RIGHT1 = 32,

    // left eyebrow：33～41
    FLM_33_L_EYEBROW_LEFT_CORNER = 33,
    FLM_34_L_EYEBROW_TOP_LEFT_QUARTER = 34,
    FLM_35_L_EYEBROW_TOP_MIDDLE = 35,
    FLM_36_L_EYEBROW_TOP_RIGHT_QUARTER = 36,
    FLM_37_L_EYEBROW_TOP_RIGHT_CORNER = 37,
    FLM_38_L_EYEBROW_LOWER_LEFT_QUARTER = 38,
    FLM_39_L_EYEBROW_LOWER_LOWER_MIDDLE = 39,
    FLM_40_L_EYEBROW_LOWER_RIGHT_QUARTER = 40,
    FLM_41_L_EYEBROW_LOWER_RIGHT_CORNER = 41,

    // right eyebrow：42～50
    FLM_42_R_EYEBROW_TOP_LEFT_CORNER = 42,
    FLM_43_R_EYEBROW_TOP_LEFT_QUARTER = 43,
    FLM_44_R_EYEBROW_TOP_MIDDLE = 44,
    FLM_45_R_EYEBROW_TOP_RIGHT_CORNER = 45,
    FLM_46_R_EYEBROW_RIGHT_CORNER = 46,
    FLM_47_R_EYEBROW_LOWER_LEFT_CORNER = 47,
    FLM_48_R_EYEBROW_LOWER_LEFT_QUARTER = 48,
    FLM_49_R_EYEBROW_LOWER_MIDDLE = 49,
    FLM_50_R_EYEBROW_LOWER_RIGHT_QUARTER = 50,

    // left eye：51～60
    FLM_51_L_EYE_LEFT_CORNER = 51,
    FLM_52_L_EYE_TOP_LEFT_QUARTER = 52,
    FLM_53_L_EYE_TOP_RIGHT_QUARTER = 53,
    FLM_54_L_EYE_RIGHT_CORNER = 54,
    FLM_55_L_EYE_LOWER_LEFT_QUARTER = 55,
    FLM_56_L_EYE_LOWER_RIGHT_QUARTER = 56,
    FLM_57_L_EYE_TOP = 57,
    FLM_58_L_EYE_BOTTOM = 58,
    FLM_59_L_EYE_PUPIL = 59,
    FLM_60_L_EYE_CENTER = 60,

    // right eye：61～70
    FLM_61_R_EYE_LEFT_CORNER = 61,
    FLM_62_R_EYE_TOP_LEFT_QUARTER = 62,
    FLM_63_R_EYE_TOP_RIGHT_QUARTER = 63,
    FLM_64_R_EYE_RIGHT_CORNER = 64,
    FLM_65_R_EYE_LOWER_LEFT_QUARTER = 65,
    FLM_66_R_EYE_LOWER_RIGHT_QUARTER = 66,
    FLM_67_R_EYE_TOP = 67,
    FLM_68_R_EYE_BOTTOM = 68,
    FLM_69_R_EYE_PUPIL = 69,
    FLM_70_R_EYE_CENTER = 70,

    // nose：71～85
    FLM_71_NOSE_BRIDGE1 = 71,
    FLM_72_NOSE_BRIDGE2 = 72,
    FLM_73_NOSE_BRIDGE3 = 73,
    FLM_74_NOSE_TIP = 74,
    FLM_75_NOSE_LEFT_CT1 = 75,
    FLM_76_NOSE_RIGHT_CT1 = 76,
    FLM_77_NOSE_LEFT_CT2 = 77,
    FLM_78_NOSE_RIGHT_CT2 = 78,
    FLM_79_NOSE_LEFT_CT3 = 79,
    FLM_80_NOSE_RIGHT_CT3 = 80,
    FLM_81_NOSE_LEFT_CT4 = 81,
    FLM_82_NOSE_LEFT_CT5 = 82,
    FLM_83_NOSE_MIDDLE_CT = 83,
    FLM_84_NOSE_RIGHT_CT5 = 84,
    FLM_85_NOSE_RIGHT_CT4 = 85,

    // mouth：86～105
    FLM_86_MOUTH_LEFT_CORNER = 86,
    FLM_87_MOUTH_TOP_LIP_LEFT_CT1 = 87,
    FLM_88_MOUTH_TOP_LIP_TOP = 88,
    FLM_89_MOUTH_TOP_LIP_RIGHT_CT1 = 89,
    FLM_90_MOUTH_RIGHT_CORNER = 90,
    FLM_91_MOUTH_LOWER_LIP_BOTTOM = 91,
    FLM_92_MOUTH_TOP_LIP_LEFT_CT2 = 92,
    FLM_93_MOUTH_TOP_LIP_RIGHT_CT2 = 93,
    FLM_94_MOUTH_LOWER_LIP_RIGHT_CT1 = 94,
    FLM_95_MOUTH_LOWER_LIP_LEFT_CT1 = 95,
    FLM_96_MOUTH_LEFT_INNER_CORNER = 96,
    FLM_97_MOUTH_RIGHT_INNER_CORNER = 97,
    FLM_98_MOUTH_TOP_LIP_BOTTOM_LEFT = 98,
    FLM_99_MOUTH_TOP_LIP_BOTTOM_TOP = 99,
    FLM_100_MOUTH_TOP_LIP_BOTTOM_RIGHT = 100,
    FLM_101_MOUTH_LOWER_LIP_TOP_LEFT = 101,
    FLM_102_MOUTH_LOWER_LIP_TOP_BOTTOM = 102,
    FLM_103_MOUTH_LOWER_LIP_TOP_RIGHT = 103,
    FLM_104_MOUTH_LOWER_LIP_LEFT_CT2 = 104,
    FLM_105_MOUTH_LOWER_LIP_RIGHT_CT2 = 105
};

/**
 * @brief 3d eye landmark (28 points) definition
 */
enum Eye3DLandmark : short {
    EYE_LM_0_CT_LEFT1 = 0,
    EYE_LM_1_CT_LEFT2 = 1,
    EYE_LM_2_CT_LEFT3 = 2,
    EYE_LM_3_CT_LEFT4 = 3,
    EYE_LM_4_CT_LEFT5 = 4,
    EYE_LM_5_CT_LEFT6 = 5,
    EYE_LM_6_CT_LEFT7 = 6,
    EYE_LM_7_CT_LEFT8 = 7,
    EYE_LM_8_CT_LEFT9 = 8,
    EYE_LM_9_CT_LEFT10 = 9,
    EYE_LM_10_CT_LEFT11 = 10,
    EYE_LM_11_CT_LEFT12 = 11,
    EYE_LM_12_CT_LEFT13 = 12,
    EYE_LM_13_CT_LEFT14 = 13,
    EYE_LM_14_CT_LEFT15 = 14,
    EYE_LM_15_CT_LEFT16 = 15,
    EYE_LM_16_CT_CHIN   = 16,
    EYE_LM_17_CT_RIGHT16 = 17,
    EYE_LM_18_CT_RIGHT15 = 18,
    EYE_LM_19_CT_RIGHT14 = 19,
    EYE_LM_20_CT_RIGHT13 = 20,
    EYE_LM_21_CT_RIGHT12 = 21,
    EYE_LM_22_CT_RIGHT11 = 22,
    EYE_LM_23_CT_RIGHT10 = 23,
    EYE_LM_24_CT_RIGHT9 = 24,
    EYE_LM_25_CT_RIGHT8 = 25,
    EYE_LM_26_CT_RIGHT7 = 26,
    EYE_LM_27_CT_RIGHT6 = 27,
};

/** 嘴部关键点数目 0-19,分别对应FaceLandmark的 86-105 */
enum MouthLandmark : short {
    MLM_0_MOUTH_LEFT_CORNER = 0,
    MLM_1_MOUTH_TOP_LIP_LEFT_CT1 = 1,
    MLM_2_MOUTH_TOP_LIP_TOP = 2,
    MLM_3_MOUTH_TOP_LIP_RIGHT_CT1 = 3,
    MLM_4_MOUTH_RIGHT_CORNER = 4,
    MLM_5_MOUTH_LOWER_LIP_BOTTOM = 5,
    MLM_6_MOUTH_TOP_LIP_LEFT_CT2 = 6,
    MLM_7_MOUTH_TOP_LIP_RIGHT_CT2 = 7,
    MLM_8_MOUTH_LOWER_LIP_RIGHT_CT1 = 8,
    MLM_9_MOUTH_LOWER_LIP_LEFT_CT1 = 9,
    MLM_10_MOUTH_LEFT_INNER_CORNER = 10,
    MLM_11_MOUTH_RIGHT_INNER_CORNER = 11,
    MLM_12_MOUTH_TOP_LIP_BOTTOM_LEFT = 12,
    MLM_13_MOUTH_TOP_LIP_BOTTOM_TOP = 13,
    MLM_14_MOUTH_TOP_LIP_BOTTOM_RIGHT = 14,
    MLM_15_MOUTH_LOWER_LIP_TOP_LEFT = 15,
    MLM_16_MOUTH_LOWER_LIP_TOP_BOTTOM = 16,
    MLM_17_MOUTH_LOWER_LIP_TOP_RIGHT = 17,
    MLM_18_MOUTH_LOWER_LIP_LEFT_CT2 = 18,
    MLM_19_MOUTH_LOWER_LIP_RIGHT_CT2 = 19,
};

/**
 * @brief the results of detection whether the driver is making a phone call
 */
enum FaceCallStatus : short {
    F_CALL_NONE = 0,            // the driver is not making a phone call
    F_CALL_CALLING = 1,         // the driver is making a phone call
    F_CALL_GESTURE_NEARBY = 2   // the hand nearby,seem like phone call
};

/**
 * @brief has the face been detected
 */
enum FaceStatus : short {
    F_NONE = 0, // no face
    F_HAVE = 1  // has face
};

/**
 * @brief image cover status for detect
 */
enum ImageCoverStatus : short {
    F_IMAGE_COVER_WINDOW_NOT_FILL = -2, // adapter sliding window not filled
    F_IMAGE_COVER_UNKNOWN = -1,         // image cover status unknown
    F_IMAGE_COVER_GOOD = 0,
    F_IMAGE_COVER_BAD = 1,
};

/**
 * @brief the results of detection whether the driver's behavior is dangerous
 */
enum FaceDangerousStatus : short {
    F_DANGEROUS_NONE = 0,            // normal behavior
    F_DANGEROUS_SMOKE = 1,           // smoking
    F_DANGEROUS_SILENCE = 2,         // making the silence gesture
    F_DANGEROUS_DRINK = 3,           // drinking water
    F_DANGEROUS_OPEN_MOUTH = 4,      // opening mouth
    F_DANGEROUS_MASK_COVER = 5,       // drive with face mask
    F_DANGEROUS_COVER_MOUTH = 6,     // cover mouth with hands
};

/**
 * @brief the results of detection whether smoking
 */
enum SmokingBurningStatus : short {
    F_SMOKE_BURNING_UNKNOWN = -1,            // smoking  burning unknown
    F_SMOKE_BURNING_FALSE = 0,               // smoking  burning false
    F_SMOKE_BURNING_TRUE = 1,                // smoking  burning true
};

/**
 * @brief detection results whether the driver didn't moving head for a while
 */
enum FaceNoMovingStatus : short {
    F_NO_MOVING_UNKNOWN = 0,
    F_NO_MOVING_TRUE = 1,
    F_NO_MOVING_FALSE = 2
};

/**
 * @brief detection results of eye status of the driver
 */
enum FaceEyeStatus : short {
    F_EYE_UNKNOWN = 0,
    F_EYE_OPEN = 1,
    F_EYE_CLOSE = 2
};

/**
 * @brief detect state of eye
 */
enum FaceEyeDetectStatus : short {
    UNKNOWN = 0,
    EYE_UNAVAILABLE = 1,     //未检测到眼睛
    PUPIL_UNAVAILABLE = 2,   //检测到眼睛，但是未检测到瞳孔点
    PUPIL_AVAILABLE = 3,     //检测到眼睛，检测到瞳孔点
    // GLASSES = 4              //人脸属性模型检测，眼睛被遮挡
};

/**
 * @brief detection results whether the driver is blinking eyes
 */
enum FaceEyeBlinkStatus : short {
    F_EYE_BLINKED_UNKNOW = 0,
    F_EYE_BLINKED_TRUE = 1,
    F_EYE_BLINKED_FALSE = 2,
};

/**
 * @brief detection results of detection whether the driver is fatigue
 */
enum FaceFatigueStatus : short {
    F_FATIGUE_NONE = 0,
    F_FATIGUE_YAWN = 1,
    F_FATIGUE_EYECLOSE = 2,
    F_FATIGUE_YAWN_EYECLOSE = 3
};

/**
 * @brief detection results of detection whether the driver is focusing
 */
enum FaceAttentionStatus : short {
    F_ATTENTION_NONE = 0,
    F_ATTENTION_NOT_FOCUS = 1,      // Not focusing
    F_ATTENTION_UNCOORDINATED = 2,  // The driver's behavior is not coordinated, which means the driver's behavior is undefined
    F_ATTENTION_LOOK_FORWARD = 3,   // looking forward (the normal status for driving)
    F_ATTENTION_LOOK_LEFT = 4,      // looking leftward
    F_ATTENTION_LOOK_RIGHT = 5,     // looking rightward
    F_ATTENTION_LOOK_UP = 6,        // looking upward
    F_ATTENTION_LOOK_DOWN = 7       // looking downward
};

/**
 * @brief status definition for eye waking,
 * for example, when the driver is watching the screen, the screen should be waken
 */
enum FaceWakingStatus : short {
    F_EYE_WAKING_NONE = 0,
    F_EYE_WAKING = 1,
};

/**
 * @brief status definition for lip movement ,
 * for example, when the driver is watching the screen, the screen should be waken
 */
enum FaceLipMovementStatus : short {
    F_LIP_MOVEMENT_NONE = 0,
    F_LIP_MOVING = 1,
};

/**
 * @brief head behavior (nod or shake)
 */
enum FaceBehaviorStatus : short {
    F_HEAD_BEHAVIOR_SHAKE = 1,
    F_HEAD_BEHAVIOR_NOD = 2,
    F_HEAD_BEHAVIOR_GOON = 3
};

/**
 * @brief spoof detection without the driver's cooperation
 */
enum FaceNoInteractLivingStatus : short {
    F_NO_INTERACT_LIVING_UNKNOWN = -1,
    F_NO_INTERACT_LIVING_NONE = 0,
    F_NO_INTERACT_LIVING_ATTACK = 1,
    F_NO_INTERACT_LIVING_LIVING = 2
};

/**
 * @brief spoof detection with the driver's cooperation, the driver should behave according to instructions
 */
enum FaceInteractLivingStatus : short {
    // 判断用户层设置的有感活体检测的类型
    DETECT_TYPE_ERROR = 0,
    // 标识用户初始角度的校验失败
    FACE_CHECK_ORIGIN_ANGLE_FAIL = 1,
    // 标识用户初始角度的校验成功
    FACE_CHECK_ORIGIN_ANGLE_SUCCESS = 2,
    // 标志用户有感活体动作错误
    ACTION_INCORRECT = 3,
    // 标记用户开始指定动作的检测
    ACTION_CHECK_START = 4,
    // 标志用户正在有感活体左右转头中
    LEFT_OR_DOWN = 5,
    RIGHT_OR_UP = 6,
    // 标记用户左右转转头过快
    LEFT_OR_DOWN_FAST = 7,
    RIGHT_OR_UP_FAST = 8,
    // 暂未使用
    TIME_OUT = 9,
    // 有感活体验证通过
    DETECT_SUCCESS = 10,
};

/**
 * @brief the definition of face attributes
 */
enum FaceAttributeStatus : short {
    F_ATTR_UNKNOWN = -1,
    // expressions
    F_ATTR_EMOTION_OTHER = 0,
    F_ATTR_EMOTION_HAPPY = 1,
    F_ATTR_EMOTION_SURPRISE = 2,
    F_ATTR_EMOTION_SAD = 3,
    F_ATTR_EMOTION_ANGRY = 4,
    F_ATTR_EMOTION_TYPE_NUM,

    // glasses
    F_ATTR_GLASS_COLOR = 0,
    F_ATTR_GLASS_NO_COLOR = 1,
    F_ATTR_NO_GLASS = 2,

    // gender
    F_ATTR_GENDER_MALE = 0,
    F_ATTR_GENDER_FEMALE = 1,

    // race
    F_ATTR_RACE_BLACK = 0,
    F_ATTR_RACE_WHITE = 1,
    F_ATTR_RACE_YELLOW = 2,

    // ages
    F_ATTR_AGE_BABY = 0,      // 婴幼儿（0-3岁）
    F_ATTR_AGE_CHILDREN = 1,  // 儿童（4-12岁）
    F_ATTR_AGE_TEENAGER = 2,  // 青少年（13-18岁）
    F_ATTR_AGE_YOUTH = 3,     // 青年（19-39岁）
    F_ATTR_AGE_MIDLIFE = 4,   // 中年（40-59岁）
    F_ATTR_AGE_SENIOR = 5,    // 老年（60岁及以上）
};

/**
 * @brief Image quality status, including whether the face is blurred, blocked or noisy;
 * Images with low quality cannot be used for face recognition or dms detection;
 */
enum FaceQualityStatus : short {
    F_QUALITY_UNKNOWN = -1,
    F_QUALITY_NOISE_NORMAL = 0,            //图片噪声正常
    F_QUALITY_NOISE_HIGH = 1,              //图片噪声过高
    F_QUALITY_BLUR_NORMAL = 0,             //图片模糊度正常
    F_QUALITY_BLUR_HIGH = 1,               //图片严重模糊
    F_QUALITY_COVER_NORMAL = 0,            //图片遮挡程度正常
    F_QUALITY_COVER_HIGH = 1,              //图片遮挡严重
    F_QUALITY_COVER_LEFT_EYE_NORMAL = 0,   //图片左眼睛遮挡正常
    F_QUALITY_COVER_LEFT_EYE = 1,          //图片左眼睛遮挡严重
    F_QUALITY_COVER_RIGHT_EYE_NORMAL = 0,  //图片右眼睛遮挡正常
    F_QUALITY_COVER_RIGHT_EYE = 1,         //图片右眼睛遮挡严重
    F_QUALITY_COVER_MOUTH_NORMAL = 0,      //图片嘴部遮挡正常
    F_QUALITY_COVER_MOUTH_HIGH = 1,        //图片嘴部遮挡严重
    F_QUALITY_COVER_OTHER_NORMAL = 0,      //图片其他部位遮挡正常
    F_QUALITY_COVER_OTHER_HIGH = 1         //图片其他部位遮挡严重
};

/**
 * @brief status of tracking face
 */
enum FaceTrackingStatus : short {
    F_TRACKING_UNKNOW = 0,
    F_TRACKING_INIT = 1,
    F_TRACKING_TRACKING = 2,
    F_TRACKING_MISS = 3
};

/**
 * @brief 跟踪时将保留无效人脸的Id，因此人脸不再以Id进行区分，而以人脸来源作区分：
 *        UNKNOWN代表无人脸/无效人脸， DETECT代表模型检测到的人脸，TRACK代表上一帧跟踪遗留的人脸
 */
enum FaceDetectType : short {
    F_TYPE_UNKNOWN = 0,     // 无效人脸
    F_TYPE_DETECT = 1,      // 模型检测到的人脸
    F_TYPE_TRACK = 2,       // 上一帧跟踪遗留人脸
};

/**
 * 图像亮度
 */
enum ImageBrightness : short {
    I_STATE_UNKNOWN = 0,   // 未知
    I_STATE_NORMAL = 1,    // 亮度正常
    I_STATE_OVER_DARK = 2, // 亮度过暗
};

/**
 * @brief Face detection results,
 * including face rect, landmark, feature, face attributes and dms detection results
 */
struct FaceInfo {
    int64_t id;                          /// the detected face index
    FaceDetectType faceType;
    float stateFatigue;               /// fatigue status 1-normal, 2-light fatigue, 3-middle fatigue, 4-heavy fatigue
    float stateAttention;             /// sliding window attention status 1-not focusing, 2-behavior not coordinated, 3-looking forward, 4-looking leftward, 5-looking rightward, 6-looking upward, 7-looking downward
    float stateHeadBehavior;         /// head behavior detection result 1-nod head, 2-shake head, other-normal
    short stateCallSingle;                  /// phone call detction result 0-no phone call,  1-making phone call
    float stateCall;                 /// phone call detction result 0-no phone call,  1-making phone call
    short stateYawnSingle;               /// danger yawn detection result 0-not yawn,  1-making yawn of single frame
    short stateYawn;                     /// danger yawn detection result 0-not yawn,  1-making yawn
    float stateNoInteractLivingSingle;    /// no-interactive living detection result 0-no face detected, 1-attack, 2-living  of single frame
    float stateNoInteractLiving;        /// no-interactive living detection result 0-no face detected, 1-attack, 2-living
    float stateInteractLiving;   /// interactive living detection result
    float stateEmotionSingle;    /// emotions 0-normal, 1-smile(like), 2-dislike, 3-surprise of a single frame(INTERNAL-USE | TEST-ONLY)
    float stateEmotion;           /// emotions 0-normal, 1-smile(like), 2-dislike, 3-surprise
    float stateGlassSingle;      /// glasses 0-color glasses, 1-no color glasses, 2-no glasses
    float stateGlass;             /// glasses 0-sunglasses, 1-noglasses, 2-glasses
    float stateGenderSingle;            /// gender 0-male, 1-female
    float stateGender;            /// gender 0-male, 1-female
    float stateRaceSingle;              /// race 0-black, 1-white, 2-yellow
    float stateRace;              /// race 0-black, 1-white, 2-yellow
    float stateAgeSingle;               /// age 0-baby, 1-teenager, 2-young, 3-midlife, 4-senior
    float stateAge;               /// age 0-baby, 1-teenager, 2-young, 3-midlife, 4-senior
    /** calling detection result of left ear of a single frame (INTERNAL-USE |TEST-ONLY) */
    int stateCallLeftSingle = F_CALL_NONE;
    /** calling detection result of right ear of a single frame (INTERNAL-USE | TEST-ONLY) */
    int stateCallRightSingle = F_CALL_NONE;
    int gestureNearbyLeftEar;     /// calling detection check gesture nearby left ear of a signle frame (INTERNAL-USE | TEST-ONLY)
    int gestureNearbyRightEar;    /// calling detection check gesture nearby right ear of a signle frame (INTERNAL-USE | TEST-ONLY)
    short stateDangerDrive;         /// the dangerous driving detection result
    short stateDangerDriveSingle;   /// the dangerous driving state of a single frame (INTERNAL-USE | TEST-ONLY)
    float dangerDriveConfidence; // 危险驾驶7分类单帧结果对应的得分
    short stateSmokeBurning;         /// the smoke burning state result
    short stateSmokeBurningSingle;   /// the smoke burning state  of a single frame (INTERNAL-USE | TEST-ONLY)
    float eyeCloseConfidence;    /// eye-close confidence judged by landmark (INTERNAL-USE | TEST-ONLY)

    float scoreNoInteractLiving = 0.f;             /// no-interactive living detection score (INTERNAL-USE | TEST-ONLY)
    float scoreCallLeftSingle = 0.f;               /// face call detection score (INTERNAL-USE | TEST-ONLY)
    float scoreCallRightSingle = 0.f;              /// face call detection score (INTERNAL-USE | TEST-ONLY)
    float scoreEmotionSingle = 0.f;                /// face emotion detection score (INTERNAL-USE | TEST-ONLY)
    float scoreCameraCoverSingle = 0.f;            /// camera cover detection score (INTERNAL-USE | TEST-ONLY)
    float scoreDetectEyeLeftSingle = 0.0f;         /// 左眼眼睛检测分数
    float scoreDetectEyeRightSingle = 0.0f;        /// 右眼眼睛检测分数
    float scoreDetectPupilLeftSingle = 0.0f;        /// 左眼瞳孔检测分数
    float scoreDetectPupilRightSingle = 0.0f;       /// 右眼瞳孔检测分数

    float stateFaceNoMoving;     /// status of whether the driver didn't moving head for a while 0-no, 1-yes
    float stateFaceTracking;     /// status of face tracking 0-init, 1-miss, 2-tracking
    float stateFaceDetect;       /// whether face in the frame 0-unknow, 1-nothing, 2-face covered, 3-normal face

    // ============================ 眼部检测相关  =============================================
    bool eyeGazeCalibValid = false;          /// 标记一点标定的视线是否可用的标志变量。默认为false. 只有开启一点标定，才可能为true
    VPoint3 eyeGazeOriginLeft;              /// left eye gaze origin coordinate (in degrees)
    VPoint3 eyeGazeDestinationLeft;         /// right eye gaze direction coordinate (in degrees)
    VPoint3 eyeGazeOriginRight;             /// left eye gaze origin coordinate (in degrees)
    VPoint3 eyeGazeDestinationRight;        /// right eye gaze direction coordinate (in degrees)
    VPoint3 eyeGaze3dVectorLeft;                ///  原始输出的左眼视线归一化的方向向量
    VPoint3 eyeGaze3dVectorRight;               ///  原始输出右眼视线归一化的方向向量
    VPoint3 eyeGaze3dCalibVectorLeft;           ///  输出标定之后的左眼视线归一化的方向向量
    VPoint3 eyeGaze3dCalibVectorRight;          ///  输出标定之后的右眼视线归一化的方向向量
    VPoint3 eyeGaze3dTransVectorLeft;           ///  输出标定之后的左眼视线偏移的归一化的方向向量
    VPoint3 eyeGaze3dTransVectorRight;          ///  输出标定之后的右眼视线偏移的归一化的方向向量

    float eyeEyelidDistanceLeft;                          /// left eyelid opening distance
    float eyeEyelidDistanceRight;                         /// right eyelid opening distance
    float eyeEyelidOpeningDistanceMeanLeft;             /// left eyelid opening distance
    float eyeEyelidOpeningDistanceMeanRight;            /// right eyelid opening distance
    /// left eyelid opening ratio (_eye_eyelid_distance_left/_eye_eyelid_opening_distance_mean_left)
    float eyeEyelidOpenRatioLeft;
    /// right eyelid opening ratio (_eye_eyelid_distance_right/_eye_eyelid_opening_distance_mean_right)
    float eyeEyelidOpenRatioRight;
    float eyeEyelidStatusLeft;  /// eyelid status left, 0-unknown, 1-open, 2-close
    float eyeEyelidStatusRight; /// eyelid status right, 0-unknown, 1-open, 2-close
    float eyeState;               /// eye state, depends on both _eye_eyelid_status
    float eyeCanthusDistanceLeft;    /// left canthus opening distance (INTERNAL-USE | TEST-ONLY)
    float eyeCanthusDistanceRight;   /// right canthus opening distance (INTERNAL-USE | TEST-ONLY)
    short leftEyeDetectSingle;           /// 左眼单帧结果，0-UNKNOWN,1-未检测到眼睛,2-未检测到瞳孔,3-检测到瞳孔
    short rightEyeDetectSingle;          /// 右眼单帧结果，0-UNKNOWN,1-未检测到眼睛,2-未检测到瞳孔,3-检测到瞳孔
    int eyeCloseFrequency;            /// frequency of the driver close eyes in a certain time
    long long eyeStartClosingTime;   /// time of once start closing eyes
    long long eyeEndClosingTime;     /// time of once closing eyes end
    float eyeBlinkState;         /// whether driver is blinking eyes 0-UNKNOW,1-TRUE,2-FALSE
    float eyeBlinkDuration;      /// time interval between _eye_start_closing_time and _eye_end_closing_time
    VPoint eyeTracking[4];        /// eye center coordinates (for eye tracking usage)
    float eyeWaking;              /// whether the screen is waken by eye watching 0-No waking, 1-Waking
    VPoint eyeCentroidLeft;      /// eye pupil of left eye
    VPoint eyeCentroidRight;     /// eye pupil of right eye
    VPoint eyeLmk8Left[LM_EYE_2D_8_COUNT];          /// eyelid landmark of left eye
    VPoint eyeLmk8Right[LM_EYE_2D_8_COUNT];         /// eyelid landmark of right eye
    VPoint mouthLmk20[LM_MOUTH_2D_20_COUNT];        /// 20 landmark points of mouth
    VPoint3 eye3dLandmark28Left[LM_EYE_3D_28_COUNT];    ///  左眼的 3D 关键点(28个关键点)
    VPoint3 eye3dLandmark28Right[LM_EYE_3D_28_COUNT];   ///  右眼的 3D 关键点(28个关键点)
    VPoint eye2dLandmark28Left[LM_EYE_2D_28_COUNT];    ///  左眼的 2D 关键点(28个关键点)，目前只在测试使用
    VPoint eye2dLandmark28Right[LM_EYE_2D_28_COUNT];   ///  右眼的 2D 关键点(28个关键点)，目前只在测试使用

    // ============================ 嘴部检测相关  =============================================
    short stateLipMovementSingle;   /// lip movement status 0-not move  1-moving of single frame
    short stateLipMovement;         /// lip movement status 0-not move  1-moving

    // ============================ 头部检测相关  =============================================
    VPoint3 headLocation;         /// face coordinate (in degrees)
    VAngle headDeflection;        /// raw head pose angle for other abilities such as large angle(in degrees)
    VAngle optimizedHeadDeflection;        /// head pose angle (in degrees)
    VAngle headDeflection3D;      /// head pose angle (in degrees by 3D landmark)
    VMatrix headTranslation;      /// head translation and scale matrix

    float feature[FEATURE_COUNT];  /// face feature vector
    float rectConfidence;         /// face rect confidence
    VPoint rectCenter;            /// face rect center point
    VPoint rectLT;                /// the left-top point of face rect
    VPoint rectRB;                /// the right-bottom point of face rect
    VRect faceRect;               ///  the rect of face rect
    float landmarkConfidence;     /// face landmark confidence
    static const short LANDMARK_2D_106_LENGTH = sizeof(VPoint) * LM_2D_106_COUNT;
    VPoint landmark2D106[LM_2D_106_COUNT];    /// face landmark points (106 points)
    VPoint3 landmark3D106[LM_3D_106_COUNT];   /// 3D face landmark points (106 points)
    VPoint landmark2D68[LM_2D_68_COUNT];      ///  (deprecated) face landmark points (68 points)
    VPoint3 landmark3D68[LM_3D_68_COUNT];     ///  (deprecated) 3D face landmark points (68 3D points)
    VPoint landmark2D72[LM_VIS_2D_72_COUNT];     ///  (deprecated) VIS face landmark (72 points)

    // =============================== 滑窗触发状态VState ===========================================
    VState smokeVState;              /// smoke state processed by strategies
    VState drinkVState;              /// drinking state processed by strategies
    VState silenceVState;            /// silence gesture state processed by strategies
    VState openMouthVState;          /// opening mouth state processed by strategies
    VState coverMouthVState;         /// cover mouth with hands
    VState maskMouthVState;          /// is the driver with face mask
    VState closeEyeVState;           /// closing eye state processed by strategies
    VState yawnVState;               /// yawning state processed by strategies
    VState phoneCallVState;          /// phone call state of ear processed by strategies
    VState _left_attention_state;    /// looking left state processed by strategies (SlidingWindow)
    VState _right_attention_state;   /// looking right state processed by strategies (SlidingWindow)
    VState _up_attention_state;      /// looking up state processed by strategies (SlidingWindow)
    VState _down_attention_state;    /// looking down state processed by strategies (SlidingWindow)
    VState _small_eye_state;         /// small eye detection state processed by strategies
    VState _no_face_state;           /// has the face been detected
    VState gestureNearbyEar;         /// dms call model, check gesture state nearby ear processed by strategies

    short stateBlurSingle;              /// 1 FPS image quality status (blur)
    short stateFaceCoverSingle;         /// 1 FPS image quality status (cover)
    short leftEyeCoverSingle;          /// 1 FPS image quality status (left eye cover)
    short rightEyeCoverSingle;          /// 1 FPS image quality status (right eye cover)
    short stateMouthCoverSingle;        /// 1 FPS image quality status (mouth cover)
    short stateNoiseSingle;             /// 1 FPS image quality status (noise)
    short stateBlur;                    /// image quality status check in sliding window (Blur)
    short stateNoise;                   /// image quality status check in sliding window (Picture Noise)
    short stateLeftEyeCover;            /// image quality status check in sliding window (Left Eye Cover)
    short stateRightEyeCover;           /// image quality status check in sliding window (Right Eye Cover)
    short stateMouthCover;              /// image quality status check in sliding window (Mouth Cover)
    short stateFaceCover;               /// image quality status check in sliding window (Face Cover)
    /// DMS camera image cover  sliding window status -1-unknown 0-good 1-bad
    short stateCameraCover = ImageCoverStatus::F_IMAGE_COVER_UNKNOWN;
    /// single DMS image cover status -1-unknown 0-good 1-bad
    short stateCameraCoverSingle = ImageCoverStatus::F_IMAGE_COVER_UNKNOWN;

    short stateBrightnessSingle = ImageBrightness::I_STATE_UNKNOWN;  /// image brightness state

    /**
     * 持久化计算仿射变换参数的Tensor数据.完全相同的仿射变换数据可以共用。
     * 目前通过计算仿射变换完全一致。使用的业务模块有：
     * 1. FaceQuality
     * 2. FaceNoInteractLiving
     * 3. FaceEmotion
     * 4. FaceAttribute
     */
    VTensor tTensorWarped{};
    /**
     * 进行人脸裁剪的时候，可以根据如果原图人脸裁剪区域完全一致的话可以复用
     * 目前通过计算仿射变换完全一致。使用的业务模块有：
     * 1. FaceQuality
     * 2. FaceNoInteractLiving
     * 3. FaceEmotion
     * 4. FaceAttribute
     */
    VTensor tTensorCropped{};

    void copy(const FaceInfo& info);
    void clearAll();
    void clear();
    bool hasFace() const;
    bool noFace() const;
    bool isFaceInRect(VRect &rect, int threshold);
    bool isFaceInRect(VRect &rect, float* angels, int threshold);

    /**
     * 人脸是检测得到的
     * @param face
     * @return
     */
    bool isDetectType() const;

    /**
     * 人脸不是检测得到的，而是无效状态或跟踪遗留得到
     * @param face
     * @return
     */
    bool isNotDetectType() const;


    FaceInfo() noexcept ;
    FaceInfo(const FaceInfo&) noexcept ;
    FaceInfo(FaceInfo&&) noexcept ;
    FaceInfo& operator= (const FaceInfo&) noexcept ;
    FaceInfo& operator= (FaceInfo&&) noexcept ;
    ~FaceInfo() = default;
    void toString(std::stringstream &ss) const;
};

} // namespace vision

#endif //VISION_FACE_INFO_H
