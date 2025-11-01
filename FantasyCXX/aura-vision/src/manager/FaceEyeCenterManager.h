#ifndef VISION_FACE_EYE_CENTER_MANAGER_H
#define VISION_FACE_EYE_CENTER_MANAGER_H

#include <vector>

#include "vision/manager/AbsVisionManager.h"
#include "AbsVisionStrategy.h"
#include "detector/FaceEyeCenterDetector.h"
#include "util/sliding_window.h"
#include "vision/core/request/FaceRequest.h"
#include "vision/core/result/FaceResult.h"

namespace aura::vision {

/**
 * @brief 眼球质点策略处理
 * */
class FaceEyeCenterStrategy : public AbsVisionStrategy<FaceInfo>, public ObjectPool<FaceEyeCenterStrategy> {
public:
    explicit FaceEyeCenterStrategy(RtConfig* cfg);

    ~FaceEyeCenterStrategy();

    /**
     * 根据单人得数据执行逻辑处理
     * @param face 人脸数据信息
     */
    void execute(FaceInfo* face) override;

    /**
     * 清空数据
     */
    void clear() override;

    /**
     * 没有人脸时处理
     * @param request   输入数据
     * @param face      人脸数据
     */
    void onNoFace(VisionRequest* request, FaceInfo* face) override;

    void setupSlidingWindow() override;

private:
    const int EYE_CENTER_WINDOW_LENGTH = 30;

    /**
     * @brief 左眼上下眼皮睁闭眼距离的阈值
     */
    float _left_eyelid_close_distance_threshold = 5.f;
    /**
     * @brief 右眼眼皮睁闭眼距离的阈值
     */
    float _right_eyelid_close_distance_threshold = 5.f;
    /**
     * @brief 左眼上下眼皮距离的滑窗
     */
    CustomSateDutyFactorWindow _left_eyelid_distance_window;
    /**
     * @brief 右眼上下眼皮距离滑窗
     */
    CustomSateDutyFactorWindow _right_eyelid_distance_window;
    /**
     * 老版本的填充单帧结果的滑窗逻辑。已经废弃
     * 已经按照算法最新的策略。替换成V2版本的滑窗填充逻辑
     */
    [[deprecated("has deprecated method v1")]] void executeCloseEyeWindowV1(FaceInfo *face);
    /**
     * 算法最新的策略。替换成V2版本的滑窗填充逻辑.直接使用算法脚本中的单帧睁闭眼结果填充滑窗
     */
    void executeCloseEyeWindowV2(FaceInfo *face);

    /**
     * 老版本的无人脸填充单帧结果的滑窗逻辑。已经废弃
     * 已经按照算法最新的策略。替换成V2版本的滑窗填充逻辑
     */
    [[deprecated("has deprecated method v1")]] void onNoFaceV1(VisionRequest* request, FaceInfo* face);
    /**
     * 算法最新的策略。替换成V2版本的无人脸滑窗填充逻辑.直接使用算法脚本中的单帧睁闭眼结果填充滑窗
     */
    void onNoFaceV2(VisionRequest* request, FaceInfo* face);
    /**
     * 判断是否使用V2的标志变量的，判断是否使用算法最新的V2的版本的策略
     */
    const bool executeV2 = true;

    /**
     * @brief 左眼闭眼状态统计滑窗
     */
    CustomSateDutyFactorWindow leftEyeCloseWindow;
    /**
     * @brief 右眼闭眼状态统计滑窗
     */
    CustomSateDutyFactorWindow rightEyeCloseWindow;
     /**
     * @brief 整体眼睛闭眼状态统计滑窗
     */
    CustomSateDutyFactorWindow totalEyeCloseWindow;
    bool localLeftEyeCloseSingle = false;
    bool localRightEyeCloseSingle = false;
    bool localEyeCloseSingle = false;
};

/**
 * @brief 眼球检测模型管理器
 * */
class FaceEyeCenterManager : public AbsVisionManager {
public:
    FaceEyeCenterManager();

    ~FaceEyeCenterManager() override;

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
    std::shared_ptr<FaceEyeCenterDetector> _detector;

    std::map<int, FaceEyeCenterStrategy*> _eye_center_map;

};

} // namespace aura::vision

#endif //VISION_FACE_EYE_CENTER_MANAGER_H
