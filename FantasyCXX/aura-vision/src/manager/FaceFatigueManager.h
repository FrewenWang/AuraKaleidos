

#ifndef VISION_FACE_FATIGUE_MANAGER_H
#define VISION_FACE_FATIGUE_MANAGER_H

#include <list>
#include <vector>

#include "vision/manager/AbsVisionManager.h"
#include "AbsVisionStrategy.h"
#include "detector/FaceDangerousDriveDetector.h"
#include "util/sliding_window.h"

namespace aura::vision {
/**
 * @brief 人脸疲劳管理器。疲劳检测没有Detector的模型检测逻辑
 * 打哈欠检测：依赖于危险驾驶七分类模型输出的张嘴类型，输出打哈欠的滑窗策略结果
 * 闭眼检测：依赖于人脸关键点模型输出的睁闭眼阈值，输出闭眼的滑窗策略结果
 * 眨眼检测：依赖于眼球中心点模型输出的眼球睁闭眼状态，输出单位时间内（1分钟)眨眼的次数
 * */
class FaceFatigueStrategy : public AbsVisionStrategy<FaceInfo>, public ObjectPool<FaceFatigueStrategy> {
public:
    explicit FaceFatigueStrategy(RtConfig *cfg);

    ~FaceFatigueStrategy() = default;

    /**
     * 根据单人得数据执行逻辑处理
     * @param face 人脸数据信息
     */
    void execute(FaceInfo *face) override;

    /**
     * 没有人脸时处理
     * @param request   输入数据
     * @param face      人脸数据
     */
    void onNoFace(VisionRequest *request, FaceInfo *face) override;

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

    void clear_longtime_close_eye_data();

    void setupSlidingWindow() override;

public:

    void init();

    char fatigue_detection(FaceInfo *face_info);

    /**
     * 打哈欠的判断，根据人脸关键点进行判断嘴部的开合状态
     * @param face_info  FaceInfo
     * @return
     */
    bool yawn_detection(FaceInfo &face_info); // 判断是否是打哈欠

    /**
     * 根据pitch角度更新张嘴阈值
     */
    void updateOpenMouthThreshold(FaceInfo &faceInfo);

    /**
     * 根据上下嘴唇和左右嘴角判断是否张嘴的逻辑
     * @param landmarks
     * @return  true 张嘴  false 闭嘴
     */
    bool checkOpenMouthStatus(FaceInfo &faceInfo);

    /**
     * 是否长时间闭眼
     * @param face_info 人脸信息
     * @return true 长时间闭眼  false 没有长时间闭眼
     */
    bool closeEyeDetection(FaceInfo &face_info);

    /**
     * 是否长时间闭眼
     * @param face_info 人脸信息
     * @return true 长时间闭眼  false 没有长时间闭眼
     */
    bool closeEyeDetectionUseEyeCenter(FaceInfo &face_info);

    /**
     * 更新眼睛的阈值和模型输出的值
     * @param face_info
     */
    void update_eye_threshold_and_value(FaceInfo &face_info);

    /**
     * 判断是否闭眼
     */
    bool is_eye_close_by_threshold();

    /**
     * 判断是否是小眼睛
     */
    bool is_small_eye(float eye_threshold);
    /**
     * @brief 疲劳眨眼检测
     * @param face_info
     */
    void eye_blink_detection(FaceInfo &face_info);
    /**
     * @brief 头部静止检测
     * @param face_info
     */
    void face_no_moving_detection(FaceInfo &face_info);

private:
    // 模型输出的眼睛状态的均值集合
    std::vector<float> _eye_mean_arr{};
    /** 判断张闭嘴的阈值。将上下嘴唇关键点相对人脸框的映射距离阈值 */
    float _k_mouth_status_threshold = 90.f;
    const float _k_eye_mean_limit_ratio = 0.8f;
    const int _k_fatigue_eye_mean_limit = 10;
    const float _k_fatigue_eye_mean_threshold = 1.f;
    /** 算法模型输出当前人眼判断正闭眼的阈值，大于0.5睁眼，小于0.5闭眼 */
    const float DEFAULT_EYE_CLOSE_THRESHOLD = 0.5f;
    const float _eye_frequency_window_len = 30;
    const float _no_moving_thresh = 80;
    /** @brief 眨眼检测滑窗记录的时间窗口长度 */
    const long long _eye_blink_frequency_time_interval = 1 * 60 * 1000;

    float eyeCloseConf;
    float _eye_close_threshold;
    /** 记录闭眼检测的单帧结果 */
    bool stateCloseSingle = false;
    // 眨眼检测判断
    bool isEyeBlinkOpen = false;
    /** 使用 EyeCenter模型作为依据*/
    const bool useEyeCenter = true;
    /**
     * 判断嘴部张开的相对距离
     * 计算方法：上下嘴唇之间映射到人脸框的距离之后的相关距离
     */
    float mouthOpeningDistance = 0.f;

    CustomSateDutyFactorWindow _eye_status_window;
    CustomSateDutyFactorWindow _fatigue_yawn_window;
    CustomSateDutyFactorWindow _small_eye_window;
    /**
     * @brief 眨眼检测记录窗口
     *
     */
    std::vector<std::pair<int, long long>> _eye_blink_window{};
    /**
     * @brief 眨眼频率记录窗口
     */
    std::vector<long long> _eye_blink_frequency_window{};
    /**
     * @brief
     */
    std::vector<VPoint3> _face_location_window{};
    /**
     * @brief 检测眨眼频率的时候，是否需要进行角度限制的开关
     */
    const bool _k_face_eye_close_angle_limit_switch = true;
    float _k_face_eye_close_pitch_limit = 0.f;
    float _k_face_eye_close_yaw_min = 0.f;
    float _k_face_eye_close_yaw_max = 0.f;
    bool localLeftEyeCloseSingle = false;
    bool localRightEyeCloseSingle = false;
};

class FaceFatigueManager : public AbsVisionManager {
public:
    FaceFatigueManager();

    ~FaceFatigueManager() override;

    void clear() override;

    /**
     * 没有人脸时处理逻辑
     * @param request   请求数据
     * @param result    检测结果
     * @param face      人脸数据
     */
    void onNoFace(VisionRequest *request, VisionResult *result, FaceInfo *face) override;

    void onConfigUpdated(int key, float value) override;

    void init(RtConfig* cfg) override;

    void deinit() override;

private:
    bool preDetect(VisionRequest *request, VisionResult *result) override;

    void doDetect(VisionRequest *request, VisionResult *result) override;

private:
    std::shared_ptr<FaceDangerousDriveDetector> _detector;

    std::map<int, FaceFatigueStrategy *> _face_fatigue_map;
};

} // namespace vision

#endif //VISION_FACE_FATIGUE_MANAGER_H
