

#include "FaceHeadBehaviorManager.h"

#include <algorithm>
#include <math.h>
#include <numeric>

#include "vision/config/runtime_config/RtConfig.h"
#include "vision/manager/VisionManagerRegistry.h"

#include "opencv2/opencv.hpp"
#include "util/FpsUtil.h"

namespace aura::vision {
FaceHeadBehaviorStrategy::FaceHeadBehaviorStrategy(RtConfig* cfg) {
    this->rtConfig = cfg;
    clear();
}

FaceHeadBehaviorStrategy::~FaceHeadBehaviorStrategy() {
    clear();
}

void FaceHeadBehaviorStrategy::execute(FaceInfo *face) {
#if 0
    if (_frame_cnt <= IGNORE_FRAME_CNT) {
        _frame_cnt++;
        fi->_state_head_behavior = F_HEAD_BEHAVIOR_GOON;
        return;
    }
#endif

    // 通过整理重新计算相关参数
    if (USE_DYNAMIC_PARAMETER) {
        by_fps_reset_compute_param(FpsUtil::instance(rtConfig->sourceId)->getSmoothedFps());
    }

    if (_k_use_angle_detect) {
        // 使用平滑过滤原始的数据
        VAngle filtrate_angle;
        filter_original_angle_data(face->headDeflection, filtrate_angle);
        // 填充数数据并检测是否点摇头
        start_angle_detection(filtrate_angle, face->stateHeadBehavior);
    } else {
        VPoint nose_point;
        if (_k_rotate_landmark_point) {
            rotate_landmark_point(face->landmark2D106, nose_point);
        } else {
            nose_point.copy(face->landmark2D106[FLM_74_NOSE_TIP]);
        }

        // 使用平滑过滤原始的数据
        VPoint filtrate_point;
        filter_original_landmark_data(nose_point, filtrate_point);
        // 填充数数据并检测是否点摇头
        start_landmark_detection(filtrate_point, face->stateHeadBehavior);
    }

    // 如果检测到点头或摇头清空所有数据
    if (face->stateHeadBehavior == F_HEAD_BEHAVIOR_SHAKE
        || face->stateHeadBehavior == F_HEAD_BEHAVIOR_NOD) {
        clear();
    }
}

void FaceHeadBehaviorStrategy::filter_original_angle_data(const VAngle &head_pose, VAngle &output) {
    // 向过滤滑窗中增加数据
    _pitch_filtrate_list.push_back(head_pose.pitch);
    _yaw_filtrate_list.push_back(head_pose.yaw);

    // 过滤滑窗是否满足条件
    if (V_TO_INT(_pitch_filtrate_list.size()) < _filrate_window_size ||
        V_TO_INT(_yaw_filtrate_list.size()) < _filrate_window_size) {
        output.yaw = _yaw_filtrate_list.back();
        output.pitch = _pitch_filtrate_list.back();
    } else {
        unsigned int offset = _yaw_filtrate_list.size() - _filrate_window_size;
        output.yaw = static_cast<float>(
                std::accumulate(_yaw_filtrate_list.begin() + offset, _yaw_filtrate_list.end(), 0.0) /
                        _filrate_window_size);

        output.pitch = static_cast<float>(
                std::accumulate(_pitch_filtrate_list.begin() + offset, _pitch_filtrate_list.end(), 0.0) /
                        _filrate_window_size);

        _yaw_filtrate_list.pop_front();
        _pitch_filtrate_list.pop_front();
    }
}

void FaceHeadBehaviorStrategy::filter_original_landmark_data(const VPoint &nose_point, VPoint &output) {
    // 向过滤滑窗中增加数据
    _shake_filtrate_landmark_list.push_back(nose_point.x);
    _nod_filtrate_landmark_list.push_back(nose_point.y);

    // 过滤滑窗是否满足条件
    if (V_TO_INT(_shake_filtrate_landmark_list.size()) < _filrate_window_size ||
        V_TO_INT(_nod_filtrate_landmark_list.size()) < _filrate_window_size) {
        output.x = _shake_filtrate_landmark_list.back();
        output.y = _nod_filtrate_landmark_list.back();
    } else {
        unsigned int offset = _nod_filtrate_landmark_list.size() - _filrate_window_size;
        output.y = static_cast<float>(
                std::accumulate(_nod_filtrate_landmark_list.begin() + offset, _nod_filtrate_landmark_list.end(),
                                0.0) / _filrate_window_size);

        output.x = static_cast<float>(
                std::accumulate(_shake_filtrate_landmark_list.begin() + offset, _shake_filtrate_landmark_list.end(),
                                0.0) / _filrate_window_size);

        _nod_filtrate_landmark_list.pop_front();
        _shake_filtrate_landmark_list.pop_front();
    }
}

void FaceHeadBehaviorStrategy::start_angle_detection(vision::VAngle &head_pose, float &result_value) {
    // 增加数据
    _yaw_mean_list.push_back(head_pose.yaw);
    _pitch_mean_list.push_back(head_pose.pitch);

    // 检测均值滑窗是否满足条件
    if (V_TO_INT(_pitch_mean_list.size()) < _mean_window_size ||
        V_TO_INT(_yaw_mean_list.size()) < _mean_window_size) {
        return;
    }

    // 重新初始化缓存数据
    init_cache_angle_data();

    unsigned int offset = _pitch_mean_list.size() - _mean_window_size;
    auto pitch_begin = _pitch_mean_list.begin() + offset;
    auto yaw_begin = _yaw_mean_list.begin() + offset;

    // 计算点摇头峰值的次数
    for (int i = 0; i < _mean_window_size - _step + 1; i += (_step - 2)) {
        float pitch_extremum = 0.0f;
        bool is_pitch_extremum = find_window_extremum(pitch_begin + i,
                                                      pitch_begin + i + _step,
                                                      _pitch_last_extremum, pitch_extremum,
                                                      rtConfig->nodExtremumDistanceAngle);

        float yaw_extremum = 0.0f;
        bool is_yaw_extremum = find_window_extremum(yaw_begin + i,
                                                    yaw_begin + i + _step,
                                                    _yaw_last_extremum, yaw_extremum,
                                                    rtConfig->shakeExtremumDistanceAngle);

        if (is_yaw_extremum && abs(yaw_extremum) >= abs(pitch_extremum)) {
            ++_yaw_extremum_number;
        } else if (is_pitch_extremum && abs(pitch_extremum) >= abs(yaw_extremum)) {
            ++_pitch_extremum_number;
        }

        if (is_yaw_extremum) {
            _yaw_extremum_list.emplace_back(abs(yaw_extremum));
        }

        if (is_pitch_extremum) {
            _pitch_extremum_list.emplace_back(abs(pitch_extremum));
        }
    }

    float mean_yaw_extremum = 0.f;
    float mean_pitch_extremum = 0.f;
    bool use_head_movement = false;
#if 1
    use_head_movement = true;
    if (_yaw_extremum_list.size() > 0) {
        mean_yaw_extremum = std::accumulate(_yaw_extremum_list.begin(), _yaw_extremum_list.end(), 0.f) /
                            _yaw_extremum_list.size();
    }

    if (_pitch_extremum_list.size() > 0) {
        mean_pitch_extremum = std::accumulate(_pitch_extremum_list.begin(), _pitch_extremum_list.end(), 0.f) /
                              _pitch_extremum_list.size();
    }
#endif

    if (_yaw_extremum_number >= rtConfig->shakeExtremumNumberAngle && _yaw_extremum_number >= _pitch_extremum_number) {
        if (use_head_movement) {
            result_value = (mean_yaw_extremum >= mean_pitch_extremum) ? F_HEAD_BEHAVIOR_SHAKE
                                                                      : F_HEAD_BEHAVIOR_GOON;
        } else {
            result_value = F_HEAD_BEHAVIOR_SHAKE;
        }
    } else if (_pitch_extremum_number >= rtConfig->nodExtremumNumberAngle
               && _pitch_extremum_number >= _yaw_extremum_number) {
        if (use_head_movement) {
            result_value = (mean_pitch_extremum >= mean_yaw_extremum) ? F_HEAD_BEHAVIOR_NOD
                                                                      : F_HEAD_BEHAVIOR_GOON;
        } else {
            result_value = F_HEAD_BEHAVIOR_NOD;
        }
    } else {
        result_value = F_HEAD_BEHAVIOR_GOON;
    }

    // 删除数据
    _pitch_mean_list.erase(_pitch_mean_list.begin());
    _yaw_mean_list.erase(_yaw_mean_list.begin());
}

void FaceHeadBehaviorStrategy::start_landmark_detection(vision::VPoint &nose_point, float &result_value) {
    // 增加数据
    _shake_mean_landmark_list.push_back(nose_point.x);
    _nod_mean_landmark_list.push_back(nose_point.y);

    // 检测均值滑窗是否满足条件
    if (V_TO_INT(_shake_mean_landmark_list.size()) < _mean_window_size ||
        V_TO_INT(_nod_mean_landmark_list.size()) < _mean_window_size) {
        return;
    }

    // 重新初始化缓存数据
    init_cache_landmark_data();

    unsigned int offset = _shake_mean_landmark_list.size() - _mean_window_size;
    auto shake_begin = _shake_mean_landmark_list.begin() + offset;
    auto nod_begin = _nod_mean_landmark_list.begin() + offset;

    // 计算点摇头峰值的次数
    for (int i = 0; i < _mean_window_size - _step + 1; i += (_step - 2)) {
        float shake_extremum = 0.0f;
        bool is_shake_extremum = find_window_extremum(shake_begin + i,
                                                      shake_begin + i + _step,
                                                      _shake_landmark_last_extremum, shake_extremum,
                                                      rtConfig->shakeExtremumDistanceLandmark);

        float nod_extremum = 0.0f;
        bool is_nod_extremum = find_window_extremum(nod_begin + i,
                                                    nod_begin + i + _step,
                                                    _nod_landmark_last_extremum, nod_extremum,
                                                    rtConfig->nodExtremumDistanceLandmark);

        if (is_nod_extremum && abs(nod_extremum) >= abs(shake_extremum)) {
            ++_nod_landmark_extremum_number;
        } else if (is_shake_extremum && abs(shake_extremum) >= abs(nod_extremum)) {
            ++_shake_landmark_extremum_number;
        }

        if (is_nod_extremum) {
            _nod_extremum_landmark_list.emplace_back(abs(nod_extremum));
        }

        if (is_shake_extremum) {
            _shake_extremum_landmark_list.emplace_back(abs(shake_extremum));
        }
    }

    float mean_nod_extremum = 0.f;
    float mean_shake_extremum = 0.f;
    bool use_head_movement = false;
#if 1
    use_head_movement = true;
    if (_nod_extremum_landmark_list.size() > 0) {
        mean_nod_extremum =
                std::accumulate(_nod_extremum_landmark_list.begin(), _nod_extremum_landmark_list.end(), 0.f) /
                _nod_extremum_landmark_list.size();
    }

    if (_shake_extremum_landmark_list.size() > 0) {
        mean_shake_extremum =
                std::accumulate(_shake_extremum_landmark_list.begin(), _shake_extremum_landmark_list.end(), 0.f) /
                _shake_extremum_landmark_list.size();
    }
#endif

    // 判断是否点头或摇头
    if (_nod_landmark_extremum_number >= rtConfig->nodExtremumNumberLandmark &&
        _nod_landmark_extremum_number >= _shake_landmark_extremum_number) {
        if (use_head_movement) {
            result_value = (mean_nod_extremum >= mean_shake_extremum) ? F_HEAD_BEHAVIOR_NOD
                                                                      : F_HEAD_BEHAVIOR_GOON;
        } else {
            result_value = F_HEAD_BEHAVIOR_NOD;
        }
    } else if (_shake_landmark_extremum_number >= rtConfig->shakeExtremumNumberLandmark &&
               _shake_landmark_extremum_number >= _nod_landmark_extremum_number) {
        if (use_head_movement) {
            result_value = (mean_shake_extremum >= mean_nod_extremum) ? F_HEAD_BEHAVIOR_SHAKE
                                                                      : F_HEAD_BEHAVIOR_GOON;
        } else {
            result_value = F_HEAD_BEHAVIOR_SHAKE;
        }
    } else {
        result_value = F_HEAD_BEHAVIOR_GOON;
    }

    // 删除数据
    _shake_mean_landmark_list.erase(_shake_mean_landmark_list.begin());
    _nod_mean_landmark_list.erase(_nod_mean_landmark_list.begin());
}

bool
FaceHeadBehaviorStrategy::find_window_extremum(std::vector<float>::iterator begin, std::vector<float>::iterator end,
                                               float &last_extremum, float &extremum,
                                               float extremum_limit) {

    auto max_iter = std::max_element(begin, end);
    int max_index = static_cast<int>(std::distance(begin, max_iter));

    auto min_iter = std::min_element(begin, end);
    int min_index = static_cast<int>(std::distance(begin, min_iter));

    float max_value = *max_iter;
    float min_value = *min_iter;

    float pitch_last_extremum = *begin;
    if (last_extremum > DEF_LASE_EXTREMUM) {
        pitch_last_extremum = last_extremum;
    }

    if (max_index > 0 && max_index < _step - 1 &&
        abs(max_value - pitch_last_extremum) >= extremum_limit) {
        extremum = max_value - pitch_last_extremum;
        last_extremum = max_value;
        return true;
    } else if (min_index > 0 && min_index < _step - 1 &&
               abs(min_value - pitch_last_extremum) >= extremum_limit) {
        extremum = min_value - pitch_last_extremum;
        last_extremum = min_value;
        return true;
    } else {
        extremum = 0;
        return false;
    }
}

void FaceHeadBehaviorStrategy::by_fps_reset_compute_param(int fps) {
    if (fps <= 0) {
        fps = DEF_FPS;
    }

    // 设置默认预值
    _step = DEF_STEP;
    _filrate_window_size = FILRATE_WINDOW_SIZE;

    // 根据帧率获取相应的过滤滑窗大小
    for (auto iter = _fps_filrate_window_lut.begin(); iter != _fps_filrate_window_lut.end(); ++iter) {
        if (fps < iter->first) {
            _filrate_window_size = iter->second;
            break;
        }
    }

    // 根据帧率获取相应的步长值
    for (auto iter = _fps_step_lut.begin(); iter != _fps_step_lut.end(); ++iter) {
        if (fps < iter->first) {
            _step = iter->second;
            break;
        }
    }

    for (auto iter = _fps_mean_window_lut.begin(); iter != _fps_mean_window_lut.end(); ++iter) {
        if (fps < iter->first) {
            _mean_window_size = iter->second;
            break;
        }
    }
//        _mean_window_size = fps * EXPECTED_RECOGNITION_TIME;

    if (_mean_window_size < 15) {
        _mean_window_size = 15;
    }

//        LOGD("nodshake", "fps=%d, step=%d, _filrate_win_size=%d, mean_win_size=%d",
//             fps, _step, _filrate_window_size, _mean_window_size);
}

void FaceHeadBehaviorStrategy::clear() {
    _step = DEF_STEP;
    _mean_window_size = DEF_FPS * EXPECTED_RECOGNITION_TIME;
    _filrate_window_size = FILRATE_WINDOW_SIZE;
    _frame_cnt = 0;

    _yaw_filtrate_list.clear();
    _pitch_filtrate_list.clear();
    _pitch_mean_list.clear();
    _yaw_mean_list.clear();

    _nod_filtrate_landmark_list.clear();
    _shake_filtrate_landmark_list.clear();
    _shake_mean_landmark_list.clear();
    _nod_mean_landmark_list.clear();

    init_cache_angle_data();
    init_cache_landmark_data();
}

void FaceHeadBehaviorStrategy::init_cache_angle_data() {
    _pitch_extremum_number = 0;
    _yaw_extremum_number = 0;
    _pitch_last_extremum = DEF_LASE_EXTREMUM;
    _yaw_last_extremum = DEF_LASE_EXTREMUM;
    _yaw_extremum_list.clear();
    _pitch_extremum_list.clear();
}

void FaceHeadBehaviorStrategy::init_cache_landmark_data() {
    _shake_landmark_extremum_number = 0;
    _nod_landmark_extremum_number = 0;
    _shake_landmark_last_extremum = DEF_LASE_EXTREMUM;
    _nod_landmark_last_extremum = DEF_LASE_EXTREMUM;
    _nod_extremum_landmark_list.clear();
    _shake_extremum_landmark_list.clear();
}

void FaceHeadBehaviorStrategy::onNoFace(VisionRequest* request, FaceInfo* face) {
    // 如果检测到点头或摇头清空所有数据
    if (face->stateHeadBehavior == F_HEAD_BEHAVIOR_SHAKE || face->stateHeadBehavior == F_HEAD_BEHAVIOR_NOD) {
        clear();
    }
}

static std::vector<VPoint> head_behavior_base_landmarks{
        VPoint(2, 23),
        VPoint(3, 32),
        VPoint(4, 40),
        VPoint(6, 49),
        VPoint(8, 58),
        VPoint(11, 66),
        VPoint(14, 74),
        VPoint(17, 80),
        VPoint(21, 86),
        VPoint(25, 90),
        VPoint(29, 95),
        VPoint(34, 98),
        VPoint(38, 101),
        VPoint(42, 103),
        VPoint(46, 105),
        VPoint(52, 106),
        VPoint(56, 107),
        VPoint(109, 23),
        VPoint(108, 32),
        VPoint(107, 40),
        VPoint(105, 49),
        VPoint(103, 58),
        VPoint(100, 66),
        VPoint(97, 74),
        VPoint(94, 80),
        VPoint(90, 86),
        VPoint(86, 90),
        VPoint(82, 95),
        VPoint(77, 98),
        VPoint(73, 101),
        VPoint(69, 103),
        VPoint(65, 105),
        VPoint(59, 106),
        VPoint(7, 9),
        VPoint(14, 3),
        VPoint(23, 2),
        VPoint(33, 4),
        VPoint(42, 8),
        VPoint(15, 10),
        VPoint(24, 10),
        VPoint(33, 10),
        VPoint(42, 10),
        VPoint(104, 9),
        VPoint(97, 3),
        VPoint(88, 2),
        VPoint(78, 4),
        VPoint(69, 8),
        VPoint(96, 10),
        VPoint(87, 10),
        VPoint(78, 10),
        VPoint(69, 10),
        VPoint(18, 23),
        VPoint(24, 21),
        VPoint(31, 21),
        VPoint(38, 25),
        VPoint(31, 26),
        VPoint(24, 26),
        VPoint(28, 21),
        VPoint(27, 26),
        VPoint(28, 23),
        VPoint(28, 24),
        VPoint(93, 23),
        VPoint(87, 21),
        VPoint(80, 21),
        VPoint(73, 25),
        VPoint(80, 26),
        VPoint(87, 26),
        VPoint(83, 21),
        VPoint(84, 26),
        VPoint(83, 23),
        VPoint(83, 24),
        VPoint(56, 22),
        VPoint(56, 34),
        VPoint(56, 46),
        VPoint(56, 57),
        VPoint(48, 25),
        VPoint(64, 24),
        VPoint(43, 50),
        VPoint(69, 49),
        VPoint(37, 55),
        VPoint(75, 55),
        VPoint(45, 60),
        VPoint(49, 63),
        VPoint(56, 65),
        VPoint(63, 63),
        VPoint(67, 60),
        VPoint(38, 81),
        VPoint(51, 83),
        VPoint(56, 82),
        VPoint(61, 83),
        VPoint(74, 81),
        VPoint(56, 91),
        VPoint(44, 82),
        VPoint(68, 82),
        VPoint(68, 86),
        VPoint(44, 86),
        VPoint(39, 81),
        VPoint(73, 81),
        VPoint(49, 80),
        VPoint(56, 80),
        VPoint(63, 80),
        VPoint(49, 83),
        VPoint(56, 84),
        VPoint(63, 83),
        VPoint(52, 90),
        VPoint(60, 90)
};

void FaceHeadBehaviorStrategy::rotate_landmark_point(VPoint *landmark2d_106, VPoint &out_point) {
    float nose_point[3] = {landmark2d_106[FLM_74_NOSE_TIP].x, landmark2d_106[FLM_74_NOSE_TIP].y, 1};

    float lefteyex = 0;
    float lefteyey = 0;
    float righteyex = 0;
    float righteyey = 0;
    for (int i = FLM_61_R_EYE_LEFT_CORNER; i < FLM_71_NOSE_BRIDGE1; i++) {
        lefteyex += landmark2d_106[i].x;
        lefteyey += landmark2d_106[i].y;
    }
    for (int i = FLM_51_L_EYE_LEFT_CORNER; i < FLM_61_R_EYE_LEFT_CORNER; i++) {
        righteyex += landmark2d_106[i].x;
        righteyey += landmark2d_106[i].y;
    }
    lefteyex /= 10.0f;
    lefteyey /= 10.0f;
    righteyex /= 10.0f;
    righteyey /= 10.0f;
    float dx = righteyex - lefteyex;
    float dy = righteyey - lefteyey;
    float src_degrees = static_cast<float>((std::atan2(dy, dx) * ANGEL_180 / M_PI) - ANGEL_180);

    /**
     *   计算 Base 的 x,y 均值
     */
    float base_mean_x = 0.0f;
    float base_mean_y = 0.0f;
    for (int j = 0; j < LM_2D_106_COUNT; ++j) {
        base_mean_x += head_behavior_base_landmarks[j].x;
        base_mean_y += head_behavior_base_landmarks[j].y;
    }
    base_mean_x = base_mean_x / LM_2D_106_COUNT;
    base_mean_y = base_mean_y / LM_2D_106_COUNT;

    /**
     *  计算输入 landmark 的 x 均值
     */
    float input_mean_x = 0.0f;
    float input_mean_y = 0.0f;
    for (int k = 0; k < LM_2D_106_COUNT; ++k) {
        input_mean_x += landmark2d_106[k].x;
        input_mean_y += landmark2d_106[k].y;
    }
    input_mean_x = input_mean_x / LM_2D_106_COUNT;
    input_mean_y = input_mean_y / LM_2D_106_COUNT;

    /**
     * 计算 base 根号平方和
     */
    float sum_base_x = 0.0f;
    float sum_base_y = 0.0f;
    for (int l = 0; l < LM_2D_106_COUNT; ++l) {
        sum_base_x += std::pow(head_behavior_base_landmarks[l].x - base_mean_x, 2);
        sum_base_y += std::pow(head_behavior_base_landmarks[l].y - base_mean_y, 2);
    }
    float base_sign = std::sqrt(sum_base_x + sum_base_y);

    /**
     * 计算输入的根号平方和
     */
    float sum_input_x = 0.0f;
    float sum_input_y = 0.0f;
    for (int l = 0; l < LM_2D_106_COUNT; ++l) {
        sum_input_x += std::pow(landmark2d_106[l].x - input_mean_x, 2);
        sum_input_y += std::pow(landmark2d_106[l].y - input_mean_y, 2);
    }
    float input_sign = std::sqrt(sum_input_x + sum_input_y);

    float scale = (base_sign / LM_2D_106_COUNT) / (input_sign / LM_2D_106_COUNT);
    cv::Mat mat = cv::getRotationMatrix2D(cv::Point(0, 0), src_degrees, scale);
    mat.at<double>(0, 2) = base_mean_x - mat.at<double>(0, 0) * input_mean_x - mat.at<double>(0, 1) * input_mean_y;
    mat.at<double>(1, 2) = base_mean_y - mat.at<double>(1, 0) * input_mean_x - mat.at<double>(1, 1) * input_mean_y;

    double *mat_a = (double *) mat.data;
    out_point.x = (nose_point[0] * mat_a[0] + nose_point[1] * mat_a[1] + nose_point[2] * mat_a[2]) / scale;
    out_point.y = (nose_point[0] * mat_a[3] + nose_point[1] * mat_a[4] + nose_point[2] * mat_a[5]) / scale;
}

FaceHeadBehaviorManager::~FaceHeadBehaviorManager() {
    clear();
}

void FaceHeadBehaviorManager::clear() {
    for(auto& info : _head_behavior_map) {
        if (info.second) {
            info.second->clear();
            FaceHeadBehaviorStrategy::recycle(info.second);
        }
    }
    _head_behavior_map.clear();
}

void FaceHeadBehaviorManager::onNoFace(VisionRequest *request, VisionResult *result, FaceInfo *face) {
    if (face != nullptr && (!_head_behavior_map.empty())) {
        auto iter = _head_behavior_map.find(face->id);
        if (iter != _head_behavior_map.end()) {
            iter->second->onNoFace(request, face);
        }
    }
}

bool FaceHeadBehaviorManager::preDetect(VisionRequest *request, VisionResult *result) {
    VA_CHECK_DETECTED(ABILITY_FACE_HEAD_BEHAVIOR);
}

void FaceHeadBehaviorManager::doDetect(VisionRequest *request, VisionResult *result) {
//    PERF_TICK(result->get_perf_util(), "face_multi_mode");
    // 执行多人脸策略
    execute_face_strategy<FaceHeadBehaviorStrategy>(result, _head_behavior_map, mRtConfig);
//    PERF_TOCK(result->get_perf_util(), "face_multi_mode");
}

// NOLINTNEXTLINE
REGISTER_VISION_MANAGER("FaceHeadBehaviorManager", ABILITY_FACE_HEAD_BEHAVIOR,[]() {
    return std::dynamic_pointer_cast<AbsVisionManager>(std::make_shared<FaceHeadBehaviorManager>());
});
} // namespace vision
