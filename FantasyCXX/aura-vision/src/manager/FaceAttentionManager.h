
#ifndef VISION_FACE_ATTENTION_MANAGER_H
#define VISION_FACE_ATTENTION_MANAGER_H

#include "vision/manager/AbsVisionManager.h"
#include "AbsVisionStrategy.h"
#include "vision/core/request/FaceRequest.h"
#include "vision/core/result/FaceResult.h"
#include "util/sliding_window.h"

namespace aura::vision {

/**
 * @brief
 * */
class FaceAttentionStrategy : public AbsVisionStrategy<FaceInfo>, public ObjectPool<FaceAttentionStrategy> {
public:
    explicit FaceAttentionStrategy(RtConfig* cfg);

    ~FaceAttentionStrategy();

    /**
     * 根据单人得数据执行逻辑处理
     * @param face 人脸数据信息
     */
    void execute(FaceInfo* face) override;

    /**
     * 没有人脸时处理
     * @param request   输入数据
     * @param face      人脸数据
     */
    void onNoFace(VisionRequest* request, FaceInfo* face) override;

    /**
     * 根据 Config 字段更新相关属性
     * @param key Config的字段标识
     * @param value 更新的数值
     */
    void onConfigUpdated(int key, float value) override;

    /**
     * 清空数据
     */
    void clear() override;

    void setupSlidingWindow() override;

private:
    /**
     * 检测单帧人脸朝向.根据人脸原始的yaw和pitch角度结合摄像头的硬件成像参数
     * 计算人脸朝向(计算出的结果:左正右负、上正下负)
     * @return 人脸朝向
     */
    int get_head_state(VAngle &angle);

    /**
     * 校准人脸角度
     * @param face_info 人脸信息
     */
    VAngle calibrate_head_pose(FaceInfo *face_info);

private:
    // 判断摄像头是否是镜像的
    bool cameraMirror = true;

    CustomSateDutyFactorWindow _left_window;
    CustomSateDutyFactorWindow _right_window;
    CustomSateDutyFactorWindow _up_window;
    CustomSateDutyFactorWindow _down_window;
};

/**
 * @brief 人脸注意力管理器
 * */
class FaceAttentionManager : public AbsVisionManager {
public:
    FaceAttentionManager();

    ~FaceAttentionManager() override;

    /**
     * 没有人脸时处理逻辑
     * @param request   请求数据
     * @param result    检测结果
     * @param face      人脸数据
     */
    void onNoFace(VisionRequest *request, VisionResult *result, FaceInfo *face) override;

    void onConfigUpdated(int key, float value) override;

    void clear() override;

    void deinit() override;

private:
    bool preDetect(VisionRequest *request, VisionResult *result) override;

    void doDetect(VisionRequest *request, VisionResult *result) override;

private:
    std::map<int, FaceAttentionStrategy*> _attention_strategy_map;
};
} // namespace aura::vision

#endif //VISION_FACE_ATTENTION_MANAGER_H
