
#ifndef VISION_FACE_SMILE_MANAGER_H
#define VISION_FACE_SMILE_MANAGER_H

#include <vector>

#include "vision/manager/AbsVisionManager.h"
#include "AbsVisionStrategy.h"
#include "vision/core/request/VisionRequest.h"
#include "vision/core/result/VisionResult.h"

namespace aura::vision {

/**
 * @brief
 * */
class FaceSmileStrategy : public AbsVisionStrategy<FaceInfo>, public ObjectPool<FaceSmileStrategy> {
public:
    explicit FaceSmileStrategy(RtConfig* cfg);

    ~FaceSmileStrategy();

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
     * 清空数据
     */
    void clear() override;

private:
    SimpleSize calculate_mouth_size(FaceInfo *info); // 计算嘴部的大小

    bool face_smile_detect(const FaceInfo *fi, float mouth_value);

    void reset_all_data();

private:
    std::vector<float> _mouth_arr; // 临时存储微笑数据的集合

    const float _k_smile_def_threshold = 0.15f;
    const float _k_smile_threshold_rate = 0.85f;
};

class FaceSmileManager : public AbsVisionManager {
public:
    FaceSmileManager() = default;

    ~FaceSmileManager() override;

    /**
     * 没有人脸时处理逻辑
     * @param request   请求数据
     * @param result    检测结果
     * @param face      人脸数据
     */
    void onNoFace(VisionRequest *request, VisionResult *result, FaceInfo *face) override;

    void clear() override;

    void deinit() override;

private:
    bool preDetect(VisionRequest *request, VisionResult *result) override;

    void doDetect(VisionRequest *request, VisionResult *result) override;

private:
    std::map<int, FaceSmileStrategy *> _face_smile_map;
};
} // namespace vision

#endif //VISION_FACE_ATTRIBUTE_MANAGER_H
