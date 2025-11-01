//
// Created by wangyan67 on 2019-10-27.
//

#include "sliding_window.h"
#include "FpsUtil.h"

using namespace aura::vision;

static const char *TAG = "SlidingWindow";

AdaptiveSlidingWindow::AdaptiveSlidingWindow(int source, int window_len, float duty_factor, int target,
                                             long trigger_duration, long trigger_expired_period)
        : sourceId(source),
          _default_window_len(window_len),
          _default_duty_factor(duty_factor),
          _window_len(window_len),
          _duty_factor(duty_factor),
          _target_value(target),
          _trigger_duration(trigger_duration),
          _trigger_expired_period(trigger_expired_period),
          _strategy(WindowAdjustStrategy::StageRoutine),
          _start_time(0),
          _prev_checked(false),
          _head_offset(0),
          _prev_sliding_state(NON) {
    _stage_parameter = std::make_shared<StageParameters>();
    _stage_parameter->generate_default_parameters();
    // 初始化不同帧率阶段的滑窗长度列表和占空比系数的list
    _stage_parameter->generate_look_up_table(fpsStageList, dutyFactorStageList);
    _lower_fps_bound = _stage_parameter->get_lower_fps_bound();
    _upper_fps_bound = _stage_parameter->get_upper_fps_bound();
    maxWindowLen = _stage_parameter->get_max_window_len();
    _min_window_len = _stage_parameter->get_min_window_len();
}

AdaptiveSlidingWindow::~AdaptiveSlidingWindow() {
    clear();
}

int AdaptiveSlidingWindow::update(float value, VState *vstate) {
    // 根据帧率调整滑窗参数
    update_window_length();
    if (_window_len <= 0) {
        return false;
    }
    // 插入新的数据。并判断插入数据之后，是否满足占空比
    int trigger = insert_slice(value);
    // 获取当前滑窗的触发状态
    if (vstate) {
        get_trigger_state(static_cast<bool>(trigger), vstate);
    }
    return trigger;
}

int AdaptiveSlidingWindow::update(float value) {
    return update(value, nullptr);
}

int AdaptiveSlidingWindow::insert_slice(float value) {
    int checked = 0;
    if ((int) _data.size() < _window_len) {
        _data.emplace_back(value);
        return checked;
    }
    while ((int) _data.size() >= maxWindowLen) {
        _data.pop_front();
    }
    while ((int) _data.size() >= _window_len) {
        _data.pop_front();
    }
    _data.emplace_back(value);

    // 占空比计算。
    int data_len = (int) _data.size();
    int max_len = std::min(maxWindowLen, data_len);
    _head_offset = max_len - _window_len;
    if (_head_offset < 0) {
        _head_offset = 0;
    } else if (_head_offset > data_len - 1) {
        _head_offset = data_len - 1;
    }

    // 获取占空比个数
    int target_cnt = std::count_if(_data.begin() + _head_offset, _data.end(), [&](const SliceInfo &info) {
        return info.value == _target_value;
    });

    if (static_cast<float>(target_cnt) >= static_cast<float>(_window_len) * _duty_factor) {
        checked = 1;
    }

    return checked;
}

float AdaptiveSlidingWindow::get_window_mean_value() {
    if (_data.size() < 1) {
        return 0;
    }
    float total = 0.;
    for (auto it = _data.begin(); it != _data.end(); it++) {
        total += it->value;
    }
    return total / _data.size();
}
void AdaptiveSlidingWindow::setSourceId(int source) {
    sourceId = source;
}
/**
 * AdaptiveSlidingWindow获取滑窗的的触发状态
 * 算法思想：
 * @param checked
 * @param vState
 */
void AdaptiveSlidingWindow::get_trigger_state(bool checked, vision::VState *vstate) {
    VSlidingState cur_sliding_state;
    int duration = 0;
    int trigger_cnt = 0;
    // 删除过期的触发记录
    long long cur_ts = _data.back().timestamp;
    // 如果用户设置了触发失效时间
    if (_trigger_expired_period > 0) {
        while (!_trigger_records.empty() &&
               cur_ts - _trigger_records.front() >= _trigger_expired_period) {
            _trigger_records.pop_front();
        }
    }
    // 如果当前滑窗的已经触发满足占空比的逻辑
    if (checked) {
        // 如果前一帧没有触发满足占空比。此帧发生触发。则将当前帧作为事件触发。并记录开始时间
        if (!_prev_checked) {
            _start_time = (_data.begin() + _head_offset)->timestamp;
        }
        // 记录动作持续时间
        duration = static_cast<int>(cur_ts - _start_time);
        // 触发状态
        if (_trigger_duration <= 0 || duration >= _trigger_duration) {
            // 如果上一帧的滑窗状态不是START或者ONGOING状态。则标记此帧为开始状态
            if (_prev_sliding_state != VSlidingState::START &&
                _prev_sliding_state != VSlidingState::ONGOING) {
                cur_sliding_state = VSlidingState::START;
                _trigger_records.emplace_back(cur_ts);
            } else {
                cur_sliding_state = VSlidingState::ONGOING;
            }
        } else {
            cur_sliding_state = VSlidingState::NON;
        }
        // 动作触发次数
        if (cur_sliding_state == VSlidingState::START ||
            cur_sliding_state == VSlidingState::ONGOING) {
            trigger_cnt = _trigger_records.size();
        }

    } else {
        _start_time = 0;
        // 触发状态
        if (_prev_sliding_state != VSlidingState::NON &&
            _prev_sliding_state != VSlidingState::END) {
            cur_sliding_state = VSlidingState::END;
        } else {
            cur_sliding_state = VSlidingState::NON;
        }
    }

    vstate->state = cur_sliding_state;
    vstate->continue_time = duration;
    vstate->trigger_count = trigger_cnt;

    _prev_checked = checked;
    _prev_sliding_state = cur_sliding_state;
}

void AdaptiveSlidingWindow::clear() {
    _start_time = 0;
    _prev_checked = false;
    _prev_sliding_state = VSlidingState::NON;
    _data.clear();
    _trigger_records.clear();
}

void AdaptiveSlidingWindow::clear_buffer() {
    _start_time = 0;
    _prev_checked = false;
    _prev_sliding_state = VSlidingState::NON;
    _data.clear();
}

void AdaptiveSlidingWindow::set_fps_stage_parameters(std::shared_ptr<StageParameters> &parameter) {
    _stage_parameter = parameter;
    _stage_parameter->generate_look_up_table(fpsStageList, dutyFactorStageList);
    _lower_fps_bound = _stage_parameter->get_lower_fps_bound();
    _upper_fps_bound = _stage_parameter->get_upper_fps_bound();
    maxWindowLen = _stage_parameter->get_max_window_len();
    _min_window_len = _stage_parameter->get_min_window_len();
}

VSlidingState AdaptiveSlidingWindow::getPreVSlidingState() {
    return _prev_sliding_state;
}

void AdaptiveSlidingWindow::update_window_length() {
    if (_strategy == WindowAdjustStrategy::Fixed) {
        _window_len = _default_window_len;
        _duty_factor = _default_duty_factor;
        return;
    }

    if (fpsUtil == nullptr) {
        fpsUtil = FpsUtil::instance(sourceId);
    }
    // 获取实时帧率
    int fps = static_cast<int>(fpsUtil->getSmoothedFps());
    // 如果帧率为0，或者帧率和之前保持一致，则返回。不需要重新调整滑窗参数
    if (fps < 1e-6 || fps == curFps) {
        return;
    }
    // 根据帧率调整滑窗参数
    adjust_window_by_stage_strategy(fps);
}

void AdaptiveSlidingWindow::adjust_window_by_stage_strategy(const int &fps) {
    curFps = static_cast<int>(fps);
    // 如果当前帧率小于帧率下限，则默认设置为下限帧率，一般为5帧，如果大于上限，则设置为上限帧率，一般为30帧，如果不设置则帧率可能为0
    if (curFps < _lower_fps_bound) {
        curFps = _lower_fps_bound;
    } else if (curFps > _upper_fps_bound) {
        curFps = _upper_fps_bound;
    }

    // 根据当前帧率更新滑窗长度.
    if (fpsStageList.find(curFps) != fpsStageList.end()) {
        // 还原针对滑窗抖动的修改
        _window_len = fpsStageList[curFps];
    } else {
        _window_len = _default_window_len;
        VLOGE(TAG, "could not find valid window length by curFps:%d,set default:%d", curFps, _window_len);
    }

    // 更新占空比
    if (dutyFactorStageList.find(_window_len) != dutyFactorStageList.end()) {
        _duty_factor = dutyFactorStageList[_window_len];
    } else {
        _duty_factor = _default_duty_factor;
        VLOGE(TAG, "could not find valid duty factor by curFps:%d,set default:%f", curFps, _duty_factor);
    }
}

void AdaptiveSlidingWindow::clear_trigger_accumulative() {
    _trigger_records.clear();
}

MultiValueSlidingWindow::MultiValueSlidingWindow(int source, int window_len, float duty_factor)
        : AdaptiveSlidingWindow(source, window_len, duty_factor, 0) {
}

MultiValueSlidingWindow::~MultiValueSlidingWindow() {
    clear();
}

int MultiValueSlidingWindow::update(float value, VState *vstate) {
    return update(value);
}

int MultiValueSlidingWindow::update(float value) {
    // 根据帧率调整滑窗参数
    update_window_length();
    if (_window_len <= 0) {
        return -1;
    }
    // 插入新的数据
    int trigger = insert_slice(value);
    return trigger;
}

int MultiValueSlidingWindow::insert_slice(float value) {
    // 插入数据(多值滑窗主要供目标分类模型使用，分类结果一般从0开始。故此处默认值为-1)
    int checked = -1;

    if ((int) _data.size() < _window_len) {
        _data.emplace_back(value);
        return checked;
    }
    while ((int) _data.size() >= maxWindowLen) {
        _data.pop_front();
    }
    while ((int) _data.size() >= _window_len) {
        _data.pop_front();
    }
    _data.emplace_back(value);

    _multi_count.clear();

    int data_len = (int) _data.size();
    int max_len = std::min(maxWindowLen, data_len);
    _head_offset = max_len - _window_len;
    if (_head_offset < 0) {
        _head_offset = 0;
    } else if (_head_offset > data_len - 1) {
        _head_offset = data_len - 1;
    }
    for (auto iter = _data.begin() + _head_offset; iter != _data.end(); ++iter) {
        _multi_count[iter->value]++;
    }

    auto iter = std::max_element(_multi_count.begin(), _multi_count.end(),
                                 [](const std::pair<int, int> &p1, const std::pair<int, int> &p2) {
                                     return p1.second < p2.second;
                                 });

    if (iter->second >= static_cast<float>(_window_len) * _duty_factor) {
        checked = iter->first;
    }
    return checked;
}

DangerDriveSlidingWindow::DangerDriveSlidingWindow(int source, int window_len, float trigger_duty_factor,
                                                   float end_duty_factor, int duration_limit, int target)
        : CustomSateDutyFactorWindow(source, window_len, trigger_duty_factor, end_duty_factor, target),
          _duration_limit(duration_limit),
          _is_state_start(false) {
}

DangerDriveSlidingWindow::~DangerDriveSlidingWindow() {
    clear();
}

void DangerDriveSlidingWindow::get_trigger_state(bool checked, vision::VState *vstate) {
    VSlidingState cur_sliding_state;
    int duration = 0;
    int trigger_cnt = 0;
    // 删除过期的触发记录
    long long cur_ts = _data.back().timestamp;
    if (_trigger_expired_period > 0) {
        while (!_trigger_records.empty() &&
               cur_ts - _trigger_records.front() >= _trigger_expired_period) {
            _trigger_records.pop_front();
        }
    }
    if (checked) {
        if (!_prev_checked) {
            _start_time = (_data.begin() + _head_offset)->timestamp;
        }
    }
    // 危险驾驶行为检测需要动作结束后才做预警
    if (_prev_checked && !checked) {
        duration = static_cast<int>(cur_ts - _start_time);
        VLOGI(TAG, "DangerDrive Target:%d, Duration: %d", _target_value, duration);
        if ((_trigger_duration <= 0 || duration >= _trigger_duration) && duration < _duration_limit) {
            cur_sliding_state = VSlidingState::START;
            _trigger_records.emplace_back(cur_ts);
            trigger_cnt = _trigger_records.size();
            _is_state_start = true;
        } else {
            cur_sliding_state = VSlidingState::NON;
            duration = 0;
            VLOGE(TAG, "DangerDrive Target:%d, Timeout，duration: %d", _target_value, duration);
        }
    } else {
        if (_is_state_start) {
            cur_sliding_state = VSlidingState::END;
        } else {
            cur_sliding_state = VSlidingState::NON;
        }
        _is_state_start = false;
    }
    if (!checked) {
        _start_time = 0;
    }
    vstate->state = cur_sliding_state;
    vstate->continue_time = duration;
    vstate->trigger_count = trigger_cnt;
    _prev_checked = checked;
    _prev_sliding_state = cur_sliding_state;
}

void DangerDriveSlidingWindow::clear() {
    CustomSateDutyFactorWindow::clear();
    _is_state_start = false;
}

void DangerDriveSlidingWindow::clear_buffer() {
    CustomSateDutyFactorWindow::clear_buffer();
    _is_state_start = false;
}

CustomSateDutyFactorWindow::CustomSateDutyFactorWindow(int source, int window_len, float trigger_duty_factor,
                                                       float end_duty_factor, int target)
        : AdaptiveSlidingWindow(source, window_len, trigger_duty_factor, target),
          _end_duty_factor(end_duty_factor),
          _checked_state(false),
          _is_state_end(false) {
}

CustomSateDutyFactorWindow::~CustomSateDutyFactorWindow() {
    clear();
}

int CustomSateDutyFactorWindow::update(float value, VState *vstate) {
    // 根据帧率调整滑窗参数
    update_window_length();
    if (_window_len <= 0) {
        return false;
    }
    // 插入新的数据
    _is_state_end = false;
    int trigger = insert_slice(value);
    if (trigger) {
        _checked_state = true;
    } else {
        if (_is_state_end) {
            _checked_state = false;
        }
    }
    // 更新触发状态
    if (vstate) {
        get_trigger_state(static_cast<bool>(_checked_state), vstate);
    }
    return _checked_state;
}

int CustomSateDutyFactorWindow::insert_slice(float value) {
    int checked = 0;

    // 插入数据。如果当前滑窗数据小于最大滑窗长度。则直接插入
    if ((int) _data.size() < _window_len) {
        _data.emplace_back(value);
        return checked;
    }
    while ((int) _data.size() >= maxWindowLen) {
        _data.pop_front();
    }
    while ((int) _data.size() >= _window_len) {
        _data.pop_front();
    }
    _data.emplace_back(value);

    // 占空比计算
    int data_len = (int) _data.size();
    int max_len = std::min(maxWindowLen, data_len);
    _head_offset = max_len - _window_len;
    if (_head_offset < 0) {
        _head_offset = 0;
    } else if (_head_offset > data_len - 1) {
        _head_offset = data_len - 1;
    }
    int target_cnt = std::count_if(_data.begin() + _head_offset, _data.end(), [&](const SliceInfo &info) {
        return info.value == _target_value;
    });
    if (static_cast<float>(target_cnt) >= static_cast<float>(_window_len) * _duty_factor) {
        checked = 1;
    }
    if (static_cast<float>(target_cnt) < static_cast<float>(_window_len) * _end_duty_factor) {
        _is_state_end = true;
    }
    return checked;
}

void CustomSateDutyFactorWindow::clear() {
    AdaptiveSlidingWindow::clear();
    _checked_state = false;
}

void CustomSateDutyFactorWindow::clear_buffer() {
    AdaptiveSlidingWindow::clear_buffer();
    _checked_state = false;
}

