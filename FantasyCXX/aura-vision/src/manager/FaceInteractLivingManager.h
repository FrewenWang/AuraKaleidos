
#ifndef VISION_FACE_INTERACTIVE_LIVING_MANAGER_H
#define VISION_FACE_INTERACTIVE_LIVING_MANAGER_H
#include <cmath>
#include "vision/manager/AbsVisionManager.h"
#include "AbsVisionStrategy.h"
#include "vision/core/request/FaceRequest.h"
#include "vision/core/result/FaceResult.h"

namespace aura::vision {
/**
 * @brief 有感活体检测逻辑
 * 向左转转头：
 * 向右转转头:
 * 眨眨眼：计算用户眨眼次数超过10次。
 * 张张嘴：计算用户张嘴次数超过10次
 **/
class FaceInteractLivingStrategy : public AbsVisionStrategy<FaceInfo>,
                                   public ObjectPool<FaceInteractLivingStrategy> {
public:
    explicit FaceInteractLivingStrategy(RtConfig *cfg);

    ~FaceInteractLivingStrategy();

    /**
     * 根据单人得数据执行逻辑处理
     * @param face 人脸数据信息
     */
    void execute(FaceInfo *face) override;

    void clear() override;

    /**
     * 没有人脸时处理
     * @param request   输入数据
     * @param face      人脸数据
     */
    void onNoFace(VisionRequest *request, FaceInfo *face) override;

private:
    /**
     * @brief 判断是否是第一次执行该检测
     * @param detect_type 检测的类型
     * */
    void is_first_detect(int detect_type);

    /**
     * @brief 通过类型检测
     * @param detectType 指定要检测的类型
     * @param head_pos    头部偏转数据
     * @return 返回检测结果
     * */
    int detect(int detectType, FaceInfo *face);

    /**
     * @brief 检测头部左转或右转
     * @param detectType 检测类型
     * @param headPos    头部偏转数据
     * @return 返回检测结果
     * */
    int detectLeftOrRightRotate(int detectType, VAngle &headPos);

    /**
     * @brief 检测摇头
     * @param headPos 头部偏转数据
     * @return 返回检测结果
     * */
    int detectShakeHead(VAngle &headPos);

    /**
     * @brief 检测关闭眼睛
     * @param eyeThreshold 闭眼的置信度
     * @return 返回检测结果
     * */
    int detectCloseEye(float eyeThreshold, VAngle &headPos);

    /**
     * @brief 检测张开嘴巴
     * @param headPos 头部偏转数据
     * @return 返回检测结果
     * */
    int detectOpenMouth(float lipDistance, VAngle &headPos);

    /**
     * @brief  有感活体转头的执行状态
     * @return 返回转头状态结果
     */
    int headRotateLivenessStatus();

    /**
     * 检测初始角度是否在指定范围内
     */
    bool checkOriginAngle(VAngle &head_pos);

private:
    int status;
    long timeout;

    float lastDistance;
    int lastStatus;
    bool finished;
    long long startTime;
    bool isLarger;

    /**  @brief 判断是不是初次开始检测 */
    bool isFirst = true;
    float firstLipDistance = 0.0f;
    short openMouthCount = 0;

    // 摇头检测使用的参数
    float lastStatusPitch;
    float lastStatusYaw;
    float nowStatusPitch;
    float nowStatusYaw;
    float changPitchValue;
    float changYawValue;

    long long leftTime;
    long long rightTime;
    int leftFineValue;
    int rightFineValue;
    int downFineValue;

    int eyeCloseCount;

    float distance;
    float fineValue;
    float thresholdLow;
    float thresholdHigh;

    int lastDetectType;

    const int HEAD_BEHAVIOR_SHAKE_MAX_TIME = 5000; // ms
    const int HEAD_BEHAVIOR_SHAKE_LEFT_DEFLECT = -3; // 摇头左侧偏转角度
    const int HEAD_BEHAVIOR_SHAKE_RIGHT_DEFLECT = 3; // 摇头右侧偏转角度
    const int HEAD_BEHAVIOR_NOD_DEFLECT_MIN = 8;  // 点头偏转角度
    // const int _k_face_detect_util_head_yaw_high = 10;
    // const int _k_face_detect_util_head_yaw_low = -10;
    // const int _k_face_detect_util_head_pitch_high = 10;
    // const int _k_face_detect_util_head_pitch_low = -10;
    // const int _k_face_detect_util_head_roll_high = 10;
    // const int _k_face_detect_util_head_roll_low = -10;

    const float INTERACT_LIVING_HEAD_LEFT_FINE_VALUE = 12.f;
    const float INTERACT_LIVING_HEAD_LEFT_THRESHOLD_LOW = 5.f;
    const float INTERACT_LIVING_HEAD_LEFT_THRESHOLD_HIGH = 15.f;
    // 有感活体向右转头的单帧运动的最优角度距离
    const float INTERACT_LIVING_HEAD_RIGHT_FINE_VALUE = -12.f;
    // 有感活体向右转头单帧运动的最小角度距离
    const float INTERACT_LIVING_HEAD_RIGHT_THRESHOLD_LOW = 5.f;
    // 有感活体向右转头的单帧运动的最大角度距离
    const float INTERACT_LIVING_HEAD_RIGHT_THRESHOLD_HIGH = 15.f;

    const float INTERACT_LIVING_EYE_THRESHOLD = 0.6f;
    const int INTERACT_LIVING_MOUTH_THRESHOLD = 9;
    // 有感活体检测睁闭眼和张嘴的次数的为10次
    const int INTERACT_LIVING_EYE_OR_MOUTH_DETECT_COUNT = 10;
    const int INTERACT_LIVING_DETECT_OUTPUT = 30000;
};

/**
 * @brief 人脸危险驾驶管理器
 * */
class FaceInteractLivingManager : public AbsVisionManager {
public:
    FaceInteractLivingManager() = default;

    ~FaceInteractLivingManager() override;

    void clear() override;

    void init(RtConfig* cfg) override;

    void deinit() override;

    /**
     * 没有人脸时处理逻辑
     * @param request   请求数据
     * @param result    检测结果
     * @param face      人脸数据
     */
    void onNoFace(VisionRequest *request, VisionResult *result, FaceInfo *face) override;

private:
    bool preDetect(VisionRequest *request, VisionResult *result) override;

    void doDetect(VisionRequest *request, VisionResult *result) override;

private:
    std::map<int, FaceInteractLivingStrategy *> interactLivingMap;
};
} // namespace vision

#endif //VISION_FACE_INTERACTIVE_LIVING_MANAGER_H
