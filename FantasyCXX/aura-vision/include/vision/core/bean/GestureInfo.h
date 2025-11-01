//
// Created by Li,Wendong on 2019-06-01.
//

#ifndef VISION_GESTURE_INFO_H
#define VISION_GESTURE_INFO_H

#include <cstring>
#include "vision/core/common/VConstants.h"
#include "vision/core/common/VStructs.h"

namespace aura::vision {

/**
 * @brief Gesture landmark
 */
enum GestureLandmark : short {
    GLM_0_GESTURE_START = 0, // gesture start

    // thumb 1～4
    GLM_1_THUMB1 = 1,
    GLM_2_THUMB2 = 2,
    GLM_3_THUMB3 = 3,
    GLM_4_THUMB4 = 4,

    // forefinger 5～8
    GLM_5_FOREFINGER1 = 5,
    GLM_6_FOREFINGER2 = 6,
    GLM_7_FOREFINGER3 = 7,
    GLM_8_FOREFINGER4 = 8,

    // middle finger 9～12
    GLM_9_MIDDLE_FINGER1 = 9,
    GLM_10_MIDDLE_FINGER2 = 10,
    GLM_11_MIDDLE_FINGER3 = 11,
    GLM_12_MIDDLE_FINGER4 = 12,

    // ring finger 13～16
    GLM_13_RING_FINGER1 = 13,
    GLM_14_RING_FINGER2 = 14,
    GLM_15_RING_FINGER3 = 15,
    GLM_16_RING_FINGER4 = 16,
    // little finger 17～20
    GLM_17_LITTLE_THUMB1 = 17,
    GLM_18_LITTLE_THUMB2 = 18,
    GLM_19_LITTLE_THUMB3 = 19,
    GLM_20_LITTLE_THUMB4 = 20
};

/**
 * 算法模型GestureLandmark模型输出的原始的单帧手势结果
 * TODO 跟算法同学讨论。看是否让握拳不作为结果0输出  @wangbin
 * @brief Gesture type 原始模型输出的手势类型
 */
enum GestureTypeRaw : short {
    GESTURE_FIST_RAW = 0,
    GESTURE_1_RAW = 1,
    GESTURE_2_RAW = 2,
    GESTURE_3_RAW = 3,
    GESTURE_4_RAW = 4,
    GESTURE_5_RAW = 5,
    GESTURE_OK_RAW = 6,
    GESTURE_THUMB_UP_RAW = 7,
    GESTURE_THUMB_DOWN_RAW = 8,
    GESTURE_THUMB_RIGHT_RAW = 9,
    GESTURE_THUMB_LEFT_RAW = 10,
    GESTURE_HEART_RAW = 11,
    GESTURE_LEFT5_RAW = 12,
    GESTURE_RIGHT5_RAW = 13

};

/**
 * 能力层提供给上层业务层的手势类型定义，为了遵循：
 * 未执行检测手势：-1
 * 检测不到手势：0
 * 则原始模型输出的手势和提供给上层的手势定义区分开.
 * @brief Gesture type
 */
enum GestureType : short {
    GESTURE_NO_DETECT = -1,
    GESTURE_NONE = 0,
    GESTURE_1 = 1,
    GESTURE_2 = 2,
    GESTURE_3 = 3,
    GESTURE_4 = 4,
    GESTURE_5 = 5,
    GESTURE_OK = 6,
    GESTURE_THUMB_UP = 7,
    GESTURE_THUMB_DOWN = 8,
    GESTURE_THUMB_RIGHT = 9,
    GESTURE_THUMB_LEFT = 10,
    GESTURE_HEART = 11,
    GESTURE_FIST = 12,
    GESTURE_LEFT5 = 13,
    GESTURE_RIGHT5 = 14
};

/**
 * @brief Dynamic gesture type
 */
enum GestureDynamicType : short {
    GESTURE_DYNAMIC_NO_DETECT = -1,
    GESTURE_DYNAMIC_NONE = 0,
    GESTURE_DYNAMIC_PINCH = 1,
    GESTURE_DYNAMIC_GRASP = 2,
    GESTURE_DYNAMIC_LEFT_WAVE = 3,
    GESTURE_DYNAMIC_RIGHT_WAVE = 4,
};
/**
 * Gesture infer
 */
enum GesturePlayPhoneStatus : short {
    G_PLAY_PHONE_STATUS_NONE = 0,
    G_PLAY_PHONE_STATUS_PLAYING = 1,
};

enum RectType : short {
    G_RECT_TYPE_UNKNOWN = 0,
    G_RECT_TYPE_GESTURE = 1,
    G_RECT_TYPE_PLAY_PHONE = 2,
};

/**
 * @brief Gesture Info, including gesture rect, landmark and gesture types
 */
struct GestureInfo {
    int id;                                 /// the detected gesture index
    float rectConfidence;                   /// the gesture rect confidence
    VPoint rectCenter;                      /// rect center point
    VPoint rectLT;                          /// the left-top point of gesture rect
    VPoint rectRB;                          /// the right-bottom point of gesture rect
    short rectType;                         /// the rect type: 0-UNKNOW、1-GESTURE、2-PLAYPHONE
    float landmarkConfidence;               /// the gesture landmark confidence
    VPoint landmark21[GESTURE_LM_21_COUNT]; /// the gesture landmark
    int staticTypeSingle;                   /// gesture type of single frame
    float typeConfidence;                   /// the gesture type confidence
    int staticType;                         /// the static gesture type of sliding window
    int dynamicType;                        /// the dynamic gesture type of sliding window
    short statePlayPhoneSingle;             /// 1FPS the gesture inference play phone status 0- NONE 1-Playing
    short statePlayPhone;                   /// sliding window result, the inference play phone status 0- NONE 1-Playing
    VState playPhoneVState;                 /// state playing phone result during sliding window
    int dynamicTypeSingle;                  /// the dynamic gesture type of single frame

    void clear_all();
    void clear();
    void copy(const GestureInfo& info);
    float rectHeight();
    float rectWidth();

    /** 判断是否有手势 */
    bool hasGesture() const { return id > 0; }

    GestureInfo() noexcept;
    GestureInfo(const GestureInfo&) noexcept;
    GestureInfo(GestureInfo&&) noexcept;
    GestureInfo& operator= (const GestureInfo&) noexcept;
    GestureInfo& operator= (GestureInfo&&) noexcept;
    ~GestureInfo() = default;
    void toString(std::stringstream &ss) const;
};

} // namespace vision

#endif //VISION_GESTURE_INFO_H
