

#include "FaceFatigueManager.h"
#include <cmath>
#include <algorithm>
#include <numeric>

#include "FaceLandmarkManager.h"
#include "vision/core/request/FaceRequest.h"
#include "vision/core/result/FaceResult.h"
#include "vision/manager/VisionManagerRegistry.h"
#include "vision/util/log.h"
#include "vision/util/MouthUtil.h"

namespace aura::vision {

static const char *TAG = "FaceFatigueManager";

FaceFatigueStrategy::FaceFatigueStrategy(RtConfig *cfg)
    : eyeCloseConf(0.f),
      _eye_close_threshold(DEFAULT_EYE_CLOSE_THRESHOLD),
      _eye_status_window(cfg->sourceId, AbsVisionManager::DEFAULT_WINDOW_LENGTH,
                         AbsVisionManager::DEFAULT_TRIGGER_DUTY_FACTOR, AbsVisionManager::DEFAULT_END_DUTY_FACTOR,
                         true),
      _fatigue_yawn_window(cfg->sourceId, AbsVisionManager::DEFAULT_WINDOW_LENGTH,
                           AbsVisionManager::DEFAULT_TRIGGER_DUTY_FACTOR, AbsVisionManager::DEFAULT_END_DUTY_FACTOR,
                           true),
      _small_eye_window(cfg->sourceId, AbsVisionManager::DEFAULT_WINDOW_LENGTH,
                        AbsVisionManager::DEFAULT_TRIGGER_DUTY_FACTOR, AbsVisionManager::DEFAULT_END_DUTY_FACTOR,
                        true),
      _k_face_eye_close_pitch_limit(cfg->eyeClosePitchUpper),
      _k_face_eye_close_yaw_min(cfg->eyeCloseYawLower),
      _k_face_eye_close_yaw_max(cfg->eyeCloseYawUpper) {
    this->rtConfig = cfg;
    init();
    setupSlidingWindow();
    isEyeBlinkOpen = V_F_TO_BOOL(cfg->openEyeBlink);
}

void FaceFatigueStrategy::init() {
    _eye_mean_arr.reserve(_k_fatigue_eye_mean_limit);
    clear_longtime_close_eye_data();
    // 初始化眨眼记录窗口的前两个值
    _eye_blink_window.push_back({false, 0}); // 睁眼
    _eye_blink_window.push_back({false, 0}); // 闭眼
}

void FaceFatigueStrategy::setupSlidingWindow() {
    auto stageParas = std::make_shared<StageParameters>();
    for (int i = AbsVisionManager::WINDOW_LOWER_FPS; i <= AbsVisionManager::WINDOW_UPPER_FPS; ++i) {
        stageParas->stage_routine[i] = {static_cast<int>(round(i * AbsVisionManager::DEFAULT_W_LENGTH_RATIO_1_5)),
                                        AbsVisionManager::DEFAULT_W_DUTY_FACTOR};
    }

    _eye_status_window.set_trigger_expire_time(rtConfig->closeEyeTriggerExpireTime);
    _fatigue_yawn_window.set_trigger_expire_time(rtConfig->yawnTriggerExpireTime);

    _eye_status_window.set_fps_stage_parameters(stageParas);
    _fatigue_yawn_window.set_fps_stage_parameters(stageParas);
}

void FaceFatigueStrategy::execute(FaceInfo *face_info) {
    // 获取上一帧的滑窗状态如果已经是END。 则清空滑窗重新填充
    if (_fatigue_yawn_window.getPreVSlidingState() == VSlidingState::END) {
        _fatigue_yawn_window.clear();
    }
    if (_eye_status_window.getPreVSlidingState() == VSlidingState::END) {
        _eye_status_window.clear();
    }

    // 返回是否打哈欠
    face_info->stateYawn = yawn_detection(*face_info);
    // 是否长时间闭眼
    bool is_eye_close = useEyeCenter ? closeEyeDetectionUseEyeCenter(*face_info) : closeEyeDetection(*face_info);
    // 检测是否小眼睛.
    // 主线2.0 去除小眼睛策略
    // bool small_eye_state = _small_eye_window.update(is_small_eye(_eye_close_threshold), &face_info->_small_eye_state);
    // if (small_eye_state) {
    //     is_eye_close = false;
    //     _eye_status_window.clear();
    //     face_info->_close_eye_state.clear();
    // }
    // 进行眨眼频率检测. 后续可以根据业务需要增加配置开关，决定是否启用。此逻辑依赖眼球中心点模型
    if (isEyeBlinkOpen) {
        eye_blink_detection(*face_info);
    }
    // 判断是否打哈欠,连续3帧是张嘴闭眼，判断为打哈欠
    if (face_info->stateYawn && is_eye_close) {
        face_info->stateFatigue = F_FATIGUE_YAWN_EYECLOSE;
    } else if (is_eye_close) {
        face_info->stateFatigue = F_FATIGUE_EYECLOSE;
    } else if (face_info->stateYawn) {
        face_info->stateFatigue = F_FATIGUE_YAWN;
    } else {
        face_info->stateFatigue = F_FATIGUE_NONE;
    }
    // todo @swm _fatigue_yawn_window.size 暂时保留观察滑窗情况，稳定后后去掉
    VLOGI(TAG, "face_fatigue[%ld] mouthOpeningDistance=[%f], threshold[%f], stateYawnSingle=[%d], stateYawn=[%d], "
               "yawnVState=[%d,%d,%d], eyeCloseSingle=[%d], eyeCloseState=[%d], closeEyeVState=[%d,%d,%d] window[%d]",
          face_info->id, mouthOpeningDistance, _k_mouth_status_threshold,
          face_info->stateYawnSingle, face_info->stateYawn,
          face_info->yawnVState.state, face_info->yawnVState.continue_time, face_info->yawnVState.trigger_count,
          stateCloseSingle, is_eye_close, face_info->closeEyeVState.state, face_info->closeEyeVState.continue_time,
          face_info->closeEyeVState.trigger_count, _fatigue_yawn_window.size());
}

/**
 * 根据pitch角度更新张嘴阈值
 */
void FaceFatigueStrategy::updateOpenMouthThreshold(FaceInfo &faceInfo) {
    switch (rtConfig->sourceId) {
        case SOURCE_1:
            /*
             * 在 0 <= abs(pitch) < 10区间段：观察DMS下Emotion数据，嘴部开合度大于86的数据占20%，Yawn数据不大于86的比例占30%，DMS的打哈欠阈值设置为90
             * 在 10 <= abs(pitch) < 20和 20 <= abs(pitch) < 30两个区间段：
             * 观察DMS下Emotion数据，嘴部开合度大于98的数据占20%，Yawn数据不大于100的比例占30%，因此DMS下的打哈欠阈值设置为100
             * 考虑Emotion数据基本保持一致，因此打哈欠的打哈欠阈值不做区分处理
             * 在 30 <= abs(pitch) < 40及以上区间段：为避免emotion误检为yawn，DMS的打哈欠阈值设置为100
             */
            _k_mouth_status_threshold = 90;
//            if (abs(pitch) < 10) {
//                _k_mouth_status_threshold = 90;
//            } else {
//                _k_mouth_status_threshold = 100;
//            }
            break;

        case SOURCE_2:
            if (std::abs(faceInfo.optimizedHeadDeflection.yaw) > 50) {
                _k_mouth_status_threshold = 65;
            } else if (std::abs(faceInfo.optimizedHeadDeflection.yaw) > 30) {
                _k_mouth_status_threshold = 60;
            } else if (std::abs(faceInfo.optimizedHeadDeflection.yaw) > 20) {
                _k_mouth_status_threshold = 55;
            } else {
                _k_mouth_status_threshold = 50;
            }
            break;

        default:
            break;
    }
}

bool FaceFatigueStrategy::checkOpenMouthStatus(FaceInfo &face) {
    // 上下唇距离取100（上嘴唇下）和103（下嘴唇上）之间的距离(像素绝对距离)
    float lipsDistance = MouthUtil::getLipDistanceWithMouthLandmark(&face);
    // 真实上下嘴唇距离占比人脸框的比例乘以500。求出映射的相对距离
    mouthOpeningDistance = MouthUtil::getLipDistanceRefRect(lipsDistance, std::abs(face.rectLT.y - face.rectRB.y));
    // 根据pitch角度更新张嘴阈值
    updateOpenMouthThreshold(face);

    return mouthOpeningDistance > _k_mouth_status_threshold;
}

bool FaceFatigueStrategy::yawn_detection(FaceInfo &face_info) {
    // 原有逻辑根据危险驾驶七分类模型输出张嘴状态作为打哈欠的单帧结果
    // stateYawnSingle = false;
    // if (face_info.stateDangerDriveSingle == F_DANGEROUS_OPEN_MOUTH) {
    //     stateYawnSingle = true;
    // }
    face_info.stateYawnSingle = checkOpenMouthStatus(face_info);
    bool stateYawn = _fatigue_yawn_window.update(face_info.stateYawnSingle, &face_info.yawnVState);
    return stateYawn;
}

bool FaceFatigueStrategy::closeEyeDetection(FaceInfo &face_info) {
    if (_k_face_eye_close_angle_limit_switch) {
        if (face_info.headDeflection.pitch > _k_face_eye_close_pitch_limit ||
            face_info.headDeflection.yaw > _k_face_eye_close_yaw_max ||
            face_info.headDeflection.yaw < _k_face_eye_close_yaw_min) {
            clear_longtime_close_eye_data();
            return _eye_status_window.update(false, &face_info.closeEyeVState);
        }
    }
    // 更新闭眼阈值
    update_eye_threshold_and_value(face_info);
    // 获取单帧的闭眼结果
    stateCloseSingle = is_eye_close_by_threshold();

    VLOGI(TAG, "face_fatigue[%ld] closeSingle=[%d],curEyeCloseConf=[%f],closeWinThreshold=[%f]",
          face_info.id, stateCloseSingle, eyeCloseConf, _eye_close_threshold);

    // 更新滑窗
    return _eye_status_window.update(stateCloseSingle, &face_info.closeEyeVState);
}

bool FaceFatigueStrategy::closeEyeDetectionUseEyeCenter(FaceInfo &face) {
    localLeftEyeCloseSingle = false;
    localRightEyeCloseSingle = false;
    stateCloseSingle = false;
    // 如果左眼检测不到眼睛。但是右眼检测到闭眼.则只记录右眼闭眼
    if (face.leftEyeDetectSingle == FaceEyeDetectStatus::EYE_UNAVAILABLE
        && face.rightEyeDetectSingle == FaceEyeDetectStatus::PUPIL_UNAVAILABLE) {
        localRightEyeCloseSingle = true;
    }
    // 如果有眼检测不到眼睛，但是左眼检测到闭眼。则只记录左眼闭眼
    if (face.rightEyeDetectSingle == FaceEyeDetectStatus::EYE_UNAVAILABLE
        && face.leftEyeDetectSingle == FaceEyeDetectStatus::PUPIL_UNAVAILABLE) {
        localLeftEyeCloseSingle = true;
    }
    // 根据算法策略计算出来的单帧睁闭眼的结果。如果角度比较正的时候，左右眼都检测闭眼
    // 如果角度比较偏的话。单个眼睛检测到闭眼。都按照闭眼来进行计算。
    if ((face.leftEyeDetectSingle == FaceEyeDetectStatus::PUPIL_UNAVAILABLE
         && face.rightEyeDetectSingle == FaceEyeDetectStatus::PUPIL_UNAVAILABLE)
        || localLeftEyeCloseSingle || localRightEyeCloseSingle) {
        stateCloseSingle = true;
    }
    // 更新滑窗
    return _eye_status_window.update(stateCloseSingle, &face.closeEyeVState);
}

void FaceFatigueStrategy::update_eye_threshold_and_value(FaceInfo &face_info) {
    // 获取当前结果的眼睛鼻炎阈值 (0~1) 越接近0闭眼 约接近1睁眼
    eyeCloseConf = face_info.eyeCloseConfidence;

    // 单帧结果是否超过闭眼的阈值, dynamic setting
    if (eyeCloseConf >= _eye_close_threshold) {
        _eye_mean_arr.push_back(eyeCloseConf);
        float mean = std::accumulate(_eye_mean_arr.begin(), _eye_mean_arr.end(), 0.0) / _eye_mean_arr.size();
        _eye_close_threshold = mean * _k_eye_mean_limit_ratio;
    }

    if ((int)_eye_mean_arr.size() >= _k_fatigue_eye_mean_limit) {
        _eye_mean_arr.erase(_eye_mean_arr.begin());
    }
}

bool FaceFatigueStrategy::is_eye_close_by_threshold() {
    // 如果当前关键点输出的睁闭眼大约默认的睁眼阈值，则认为睁眼，返回false
    if (eyeCloseConf > DEFAULT_EYE_CLOSE_THRESHOLD) {
        return false;
    }
    return eyeCloseConf <= _eye_close_threshold;
}

bool FaceFatigueStrategy::is_small_eye(float eye_threshold) {
    return eye_threshold <= rtConfig->smallEyeThreshold;
}

void FaceFatigueStrategy::face_no_moving_detection(FaceInfo &face_info) {
    VPoint3 curr_pose = face_info.headLocation;
    _face_location_window.push_back(curr_pose);
    if (_face_location_window.size() == 10) {
        // 判断
        VPoint3 last_pose = _face_location_window[0];
        float distance = sqrt((curr_pose.x - last_pose.x) * (curr_pose.x - last_pose.x) +
                              (curr_pose.y - last_pose.y) * (curr_pose.y - last_pose.y) +
                              (curr_pose.z - last_pose.z) * (curr_pose.z - last_pose.z));
        VLOGD(TAG, "face no moving distance %f", distance);
        if (distance > _no_moving_thresh) {
            face_info.stateFaceNoMoving = FaceNoMovingStatus::F_NO_MOVING_FALSE;
        } else {
            face_info.stateFaceNoMoving = FaceNoMovingStatus::F_NO_MOVING_TRUE;
        }
        _face_location_window.erase(_face_location_window.begin());
    }
}

/**
 * @brief 执行眨眼检测
 * @param face_info
 */
void FaceFatigueStrategy::eye_blink_detection(FaceInfo &face_info) {
    bool do_eye_blink_detection = true;
    if ((int)(face_info.eyeState) == FaceEyeStatus::F_EYE_UNKNOWN) {
        do_eye_blink_detection = false;
    }
    // // 判断是否开启眨眼检测角度限定开关
    if (_k_face_eye_close_angle_limit_switch) {
        if (face_info.headDeflection.pitch > _k_face_eye_close_pitch_limit ||
            face_info.headDeflection.yaw > _k_face_eye_close_yaw_max ||
            face_info.headDeflection.yaw < _k_face_eye_close_yaw_min) {
            do_eye_blink_detection = false;
        }
    }
    // 这个会使得检测过程中。如果每次发现未检测到眼睛状态，或者角度太大.则不再检测眨眼，直接吧当前眨眼频率输出
    if (!do_eye_blink_detection) {
        // _eye_blink_window[0] = {false, 0};
        // _eye_blink_window[1] = {false, 0};
        // _eye_blink_frequency_window.clear();
        // 眨眼频率：眨眼窗口的长度（只保留最近一分钟眨眼次数）
        VLOGW(TAG, "not detect eye blink because eye_state: %s", (int) (face_info.eyeState)
                                                                 == FaceEyeStatus::F_EYE_OPEN ? "open" : "close");
        int blink_window_size = _eye_blink_frequency_window.size();
        face_info.eyeCloseFrequency = blink_window_size;
        return;
    }

    timeval tv;
    gettimeofday(&tv, nullptr);
    long long ts = static_cast<long long>(tv.tv_sec) * 1000 + static_cast<long long>(tv.tv_usec) / 1000;

    face_info.eyeBlinkState = FaceEyeBlinkStatus::F_EYE_BLINKED_FALSE;

    if ((int)(face_info.eyeState) == FaceEyeStatus::F_EYE_CLOSE) {
        // 睁 已发生, 当前为闭, 则更新最后闭眼时间
        // 如果当前眼睛的状态是闭眼，则记录窗口第二帧的为true。并且记录时间
        _eye_blink_window[1] = {true, ts};
    } else {
        // 如果当前眼睛是睁眼
        // 判断如果窗口数据一（睁眼）为true. 窗口数据二(闭眼)为true
        if (_eye_blink_window[0].first && _eye_blink_window[1].first) {
            // 睁 - 闭 已发生, 当前为睁, 则记录完成一次眨眼
            face_info.eyeBlinkState = FaceEyeBlinkStatus::F_EYE_BLINKED_TRUE;
            // 开始闭眼的时间为窗口一的记录时间
            face_info.eyeStartClosingTime = _eye_blink_window[0].second;
            face_info.eyeEndClosingTime = ts;
            // 眨眼时长：
            face_info.eyeBlinkDuration = face_info.eyeEndClosingTime - face_info.eyeStartClosingTime;
            // VLOGD(TAG, "eye blink, duration:%f", face_info._eye_blink_duration);
            // 眨眼频率的时间
            _eye_blink_frequency_window.push_back(ts);
            VLOGD(TAG, "eye blink, sliding window begin:%ld , end:%ld , put blink time: %ld",
                  _eye_blink_frequency_window[0],
                  _eye_blink_frequency_window[_eye_blink_frequency_window.size() - 1], ts);
            //移除队列中所有1分钟之前的眨眼行为数据。
            while (_eye_blink_frequency_window[_eye_blink_frequency_window.size() - 1]
                   - _eye_blink_frequency_window[0] > _eye_blink_frequency_time_interval) {
                _eye_blink_frequency_window.erase(_eye_blink_frequency_window.begin());
            }
        }
        // 重置眨眼记录窗口
        _eye_blink_window[0] = {true, ts};
        _eye_blink_window[1] = {false, 0};
    }
    int blink_window_size = _eye_blink_frequency_window.size();
    // 眨眼频率：眨眼窗口的长度（只保留最近一分钟眨眼次数）
    face_info.eyeCloseFrequency = blink_window_size;
}

void FaceFatigueStrategy::clear_longtime_close_eye_data() {
    _eye_mean_arr.clear();
    eyeCloseConf = _k_fatigue_eye_mean_threshold;
    _eye_close_threshold = DEFAULT_EYE_CLOSE_THRESHOLD;
}

void FaceFatigueStrategy::clear() {
    _fatigue_yawn_window.clear();
    _eye_status_window.clear();
    _small_eye_window.clear();
    clear_longtime_close_eye_data();
    _eye_blink_window.clear();
    _eye_blink_frequency_window.clear();
    _face_location_window.clear();
}

void FaceFatigueStrategy::onNoFace(VisionRequest *request, FaceInfo *face) {
    // 获取上一帧的滑窗状态如果已经是END。 则清空滑窗重新填充
    if (_fatigue_yawn_window.getPreVSlidingState() == VSlidingState::END) {
        _fatigue_yawn_window.clear();
    }
    if (_eye_status_window.getPreVSlidingState() == VSlidingState::END) {
        _eye_status_window.clear();
    }

    // 返回是否打哈欠
    // 真实上下嘴唇距离占比人脸框的比例乘以500。求出映射的相对距离
    mouthOpeningDistance = 0;
    face->stateYawnSingle = false;
    face->stateYawn = _fatigue_yawn_window.update(F_FATIGUE_NONE, &face->yawnVState);

    // 是否长时间闭眼
    stateCloseSingle = F_FATIGUE_NONE;
    bool is_eye_close = _eye_status_window.update(stateCloseSingle, &face->closeEyeVState);

    // 进行眨眼频率检测. 后续可以根据业务需要增加配置开关，决定是否启用。此逻辑依赖眼球中心点模
    VLOGD(TAG, "[when no face] not detect eye blink because eye_state: %s", (int) (face->eyeState) ==
                                                                          FaceEyeStatus::F_EYE_OPEN ? "open" : "close");
    face->eyeCloseFrequency = _eye_blink_frequency_window.size();

    // 判断是否打哈欠,连续3帧是张嘴闭眼，判断为打哈欠
    if (face->stateYawn && is_eye_close) {
        face->stateFatigue = F_FATIGUE_YAWN_EYECLOSE;
    } else if (is_eye_close) {
        face->stateFatigue = F_FATIGUE_EYECLOSE;
    } else if (face->stateYawn) {
        face->stateFatigue = F_FATIGUE_YAWN;
    } else {
        face->stateFatigue = F_FATIGUE_NONE;
    }

    VLOGI(TAG, "[when no face] face_fatigue[%ld] mouthOpeningDistance=[%f], stateYawnSingle=[%d], stateYawn=[%d], "
               "yawnVState=[%d,%d,%d], eyeCloseSingle=[%d], eyeCloseState=[%d], closeEyeVState=[%d,%d,%d]",
          face->id, mouthOpeningDistance, face->stateYawnSingle, face->stateYawn, face->yawnVState.state,
          face->yawnVState.continue_time, face->yawnVState.trigger_count, stateCloseSingle, is_eye_close,
          face->closeEyeVState.state, face->closeEyeVState.continue_time, face->closeEyeVState.trigger_count);
}

void FaceFatigueStrategy::onConfigUpdated(int key, float value) {
    switch (key) {
        case STRATEGY_TRIGGER_EXPIRE_TIME_CLOSE_EYE:
            _eye_status_window.set_trigger_expire_time(rtConfig->closeEyeTriggerExpireTime);
            break;
        case STRATEGY_TRIGGER_EXPIRE_TIME_YAWN:
            _fatigue_yawn_window.set_trigger_expire_time(rtConfig->yawnTriggerExpireTime);
            break;
        default:
            break;
    }
}

FaceFatigueManager::FaceFatigueManager() {}

FaceFatigueManager::~FaceFatigueManager() {
    clear();
}

void FaceFatigueManager::clear() {
    for (auto &info : _face_fatigue_map) {
        if (info.second) {
            info.second->clear();
            FaceFatigueStrategy::recycle(info.second);
        }
    }
    _face_fatigue_map.clear();
}

void FaceFatigueManager::onNoFace(VisionRequest *request, VisionResult *result, FaceInfo *face) {
    if (face != nullptr && (!_face_fatigue_map.empty())) {
        auto iter = _face_fatigue_map.find(face->id);
        if (iter != _face_fatigue_map.end()) {
            iter->second->onNoFace(request, face);
        }
    }
}

void FaceFatigueManager::onConfigUpdated(int key, float value) {
    for (auto info : _face_fatigue_map) {
        info.second->onConfigUpdated(key, value);
    }
}

bool FaceFatigueManager::preDetect(VisionRequest *request, VisionResult *result) {
    VA_CHECK_DETECTED(ABILITY_FACE_FATIGUE);
}

void FaceFatigueManager::doDetect(VisionRequest *request, VisionResult *result) {
    VA_SET_DETECTED(ABILITY_FACE_FATIGUE);
    // 执行多人脸策略
    execute_face_strategy<FaceFatigueStrategy>(result, _face_fatigue_map, mRtConfig);
}

void FaceFatigueManager::init(RtConfig *cfg) {
    mRtConfig = cfg;
}

void FaceFatigueManager::deinit() {
    AbsVisionManager::deinit();
    if (_detector != nullptr) {
        _detector->deinit();
        _detector = nullptr;
    }
}

REGISTER_VISION_MANAGER("FaceFatigueManager", ABILITY_FACE_FATIGUE, []() {
    return std::dynamic_pointer_cast<AbsVisionManager>(std::make_shared<FaceFatigueManager>());
});

} // namespace vision
