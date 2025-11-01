#ifndef VISION_PLAY_PHONE_MANAGER_H
#define VISION_PLAY_PHONE_MANAGER_H

#include <vector>

#include "AbsVisionStrategy.h"
#include "util/sliding_window.h"
#include "detector/GestureRectDetector.h"
#include "vision/core/request/GestureRequest.h"
#include "vision/core/result/GestureResult.h"
#include "vision/manager/AbsVisionManager.h"

namespace aura::vision {
/**
 * @brief
 * */
class PlayPhoneStrategy : public AbsVisionStrategy<GestureInfo>, public ObjectPool<PlayPhoneStrategy> {
public:
    explicit PlayPhoneStrategy(RtConfig *cfg);

    ~PlayPhoneStrategy() override;

    /**
     * 根据单帧数据执行逻辑处理
     * @param gesture 手势框检测模型输出信息
     */
    void execute(GestureInfo *gesture) override;

    /**
     * 执行手势框划船GV策略
     * @param pFaceInfo
     * @param pGestureInfo
     */
    void execute(FaceInfo *pFaceInfo, GestureInfo *pGestureInfo);

    /**
     * 清空数据
     */
    void clear() override;

    void setupSlidingWindow() override;

private:
    // 玩手机检测滑窗
    CustomSateDutyFactorWindow playPhoneWindow;
};

class PlayPhoneManager : public AbsVisionManager {
public:
    PlayPhoneManager();

    ~PlayPhoneManager() override;

    void init(RtConfig* cfg) override;

    void deinit() override;

    void clear() override;

private:
    bool preDetect(VisionRequest *request, VisionResult *result) override;

    void doDetect(VisionRequest *request, VisionResult *result) override;

    PlayPhoneStrategy *playPhoneStrategy = nullptr;
};
} // namespace vision

#endif // VISION_PLAY_PHONE_MANAGER_H
