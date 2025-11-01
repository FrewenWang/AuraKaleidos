
#ifndef VISION_FACE_HEAD_BEHAVIOR_MANAGER_H
#define VISION_FACE_HEAD_BEHAVIOR_MANAGER_H

#include "vision/manager/AbsVisionManager.h"
#include "AbsVisionStrategy.h"
#include "vision/core/request/FaceRequest.h"
#include "vision/core/result/FaceResult.h"
#include <list>
#include <vector>
#include <deque>
#include <map>

namespace aura::vision {

/**
 * @brief 人脸多模管理器
 * */
class FaceHeadBehaviorStrategy : public AbsVisionStrategy<FaceInfo>, public ObjectPool<FaceHeadBehaviorStrategy> {
public:
    explicit FaceHeadBehaviorStrategy(RtConfig* cfg);

    ~FaceHeadBehaviorStrategy() override;

    /**
     * 根据单人得数据执行逻辑处理
     * @param face 人脸数据信息
     */
    void execute(FaceInfo *face) override;

    /**
     * @brief 重置检测中用到的相关参数
     **/
    void clear() override;

    /**
     * 没有人脸时处理
     * @param request   输入数据
     * @param face      人脸数据
     */
    void onNoFace(VisionRequest* request, FaceInfo* face) override;

private:
    /**
     * @brief 过滤角度原始数据
     * */
    void filter_original_angle_data(const VAngle &head_pose, VAngle &output);

    /**
     * @brief 过滤关键点原始数据
     * */
    void filter_original_landmark_data(const VPoint &nose_point, VPoint &output);

    /**
     * @brief 开始使用角度执行检测
     * @param head_pose 头部偏转角相关信息
     * @param result_value 检测的结果
     **/
    void start_angle_detection(VAngle &head_pose, float &result_value);

    /**
     * @brief 开始使用关键点执行检测
     * @param nose_point 头部偏转角相关信息
     * @param result_value 检测的结果
     **/
    void start_landmark_detection(VPoint &nose_point, float &result_value);

    /**
     * @brief 初始化缓存角度相关数据
     **/
    void init_cache_angle_data();

    /**
     * @brief 初始化缓存关键点相关数据
     **/
    void init_cache_landmark_data();

    /**
     * @brief 寻找滑窗中的极值
     **/
    bool
    find_window_extremum(std::vector<float>::iterator begin, std::vector<float>::iterator end, float &last_extremum,
                         float &extremum, float extremum_limit);

    /**
     *  @brief 通过帧率重新计算相关参数
     * */
    void by_fps_reset_compute_param(int fps);

    /**
     * @brief 转正关键点
     * */
    void rotate_landmark_point(VPoint *landmark2d_106, VPoint &out_point);

private:
    // 使用角度判断点摇头的数据
    std::deque<float> _yaw_filtrate_list;
    std::deque<float> _pitch_filtrate_list;
    std::vector<float> _yaw_mean_list;
    std::vector<float> _pitch_mean_list;
    std::vector<float> _yaw_extremum_list;
    std::vector<float> _pitch_extremum_list;
    short _pitch_extremum_number;
    short _yaw_extremum_number;
    float _pitch_last_extremum;
    float _yaw_last_extremum;

    // 使用关键点判断点摇头的数据
    std::deque<float> _nod_filtrate_landmark_list;
    std::deque<float> _shake_filtrate_landmark_list;
    std::vector<float> _nod_mean_landmark_list;
    std::vector<float> _shake_mean_landmark_list;
    std::vector<float> _nod_extremum_landmark_list;
    std::vector<float> _shake_extremum_landmark_list;
    short _shake_landmark_extremum_number;
    short _nod_landmark_extremum_number;
    float _shake_landmark_last_extremum;
    float _nod_landmark_last_extremum;

    // 公共参数
    short _step;
    short _mean_window_size;
    short _filrate_window_size;
    int _frame_cnt;

#if 1
    // 公共参数
    static const int DEF_LASE_EXTREMUM = -1000;     // 默认最后一个极值
    static const int FILRATE_WINDOW_SIZE = 4;       // 均值滑窗大小
    static const int DEF_FPS = 10;                  // 帧率默认值
    static const int DEF_STEP = 4;                  // 步长默认值
    static const int EXPECTED_RECOGNITION_TIME = 2; // 期望识别时间(s)
    static const int IGNORE_FRAME_CNT = 5;          // 忽略前N帧
    const bool USE_DYNAMIC_PARAMETER = true;
#else
    // 公共参数
    static const int DEF_LASE_EXTREMUM = -1000;     // 默认最后一个极值
    static const int FILRATE_WINDOW_SIZE = 4;       // 均值滑窗大小
    static const int DEF_FPS = 40;                  // 帧率默认值
    static const int DEF_STEP = 11;                  // 步长默认值
    const float EXPECTED_RECOGNITION_TIME = 1.5f; // 期望识别时间(s)
    static const int IGNORE_FRAME_CNT = 5; // 忽略前N帧
    const bool USE_DYNAMIC_PARAMETER = false;
#endif

    std::map<int, int> _fps_step_lut = {{8,  3},
                                        {12, 4},
                                        {15, 5},
                                        {20, 7}};  // 帧率与步长的关联表
    std::map<int, int> _fps_filrate_window_lut = {{8,  2},
                                                  {50, 4}};                        // 帧率与过滤滑窗大小的关联表
    std::map<int, int> _fps_mean_window_lut = {{8,  15},
                                               {12, 25},
                                               {15, 30},
                                               {20, 35}};
    const bool _k_use_angle_detect = false;
    const bool _k_rotate_landmark_point = false;
};

/**
 * @brief 人脸危险驾驶管理器
 * */
class FaceHeadBehaviorManager : public AbsVisionManager {
public:
    FaceHeadBehaviorManager() = default;

    ~FaceHeadBehaviorManager() override;

    void clear() override;

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
    std::map<int, FaceHeadBehaviorStrategy *> _head_behavior_map;
};

} // namespace vision

#endif //VISION_FACE_MULTI_MODE_MANAGER_H
