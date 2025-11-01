
#ifndef VISION_SLIDING_WINDOW_H
#define VISION_SLIDING_WINDOW_H

#include <algorithm>
#include <deque>
#include <memory>
#include <sys/time.h>
#include <unordered_map>
#include <vector>

#include "vision/core/common/VStructs.h"
#include "vision/core/common/VConstants.h"
#include "FpsUtil.h"

namespace aura::vision {
struct SliceInfo {
    int value;
    long long timestamp;

    explicit SliceInfo(float v) : value(v) {
        struct timeval tv;
        gettimeofday(&tv, nullptr);
        long long ts = static_cast<long long>(tv.tv_sec) * 1000 + static_cast<long long>(tv.tv_usec) / 1000;
        timestamp = ts;
    }

    SliceInfo(float v, long long ts) : value(v), timestamp(ts) { }
};

enum class WindowAdjustStrategy : short {
    Fixed,
    StageRoutine
};

struct WindowPara {
    int window_length;
    float duty_factor;
};

struct StageParameters {
    std::unordered_map<int, WindowPara> stage_routine;

    // 没有定义的情况下，可生成默认参数
    void generate_default_parameters() {
        stage_routine.clear();
        stage_routine[5] = {5, 0.6f};
        stage_routine[15] = {10, 0.7f};
        stage_routine[25] = {20, 0.7f};
        stage_routine[30] = {30, 0.8f};
    }

    /**
     * 获取fps策略中的fps下限
     * @return
     */
    int get_lower_fps_bound() {
        if (stage_routine.empty()) {
            return 0;
        }
        auto iter = std::min_element(stage_routine.begin(), stage_routine.end(),
                                     [](const std::pair<int, WindowPara> &p1, const std::pair<int, WindowPara> &p2) {
                                         return p1.first < p2.first;
                                     });
        return iter->first;
    }

    int get_upper_fps_bound() {
        if (stage_routine.empty()) {
            return 0;
        }
        auto iter = std::max_element(stage_routine.begin(), stage_routine.end(),
                                     [](const std::pair<int, WindowPara> &p1, const std::pair<int, WindowPara> &p2) {
                                         return p1.first < p2.first;
                                     });
        return iter->first;
    }

    /**
     * 获取滑窗的最大滑窗长度，
     * @return
     */
    int get_max_window_len() {
        if (stage_routine.empty()) {
            return 0;
        }
        auto iter = std::max_element(stage_routine.begin(), stage_routine.end(),
                                     [](const std::pair<int, WindowPara> &p1, const std::pair<int, WindowPara> &p2) {
                                         return p1.second.window_length < p2.second.window_length;
                                     });
        return iter->second.window_length;
    }

    int get_min_window_len() {
        if (stage_routine.empty()) {
            return 0;
        }
        auto iter = std::min_element(stage_routine.begin(), stage_routine.end(),
                                     [](const std::pair<int, WindowPara> &p1, const std::pair<int, WindowPara> &p2) {
                                         return p1.second.window_length < p2.second.window_length;
                                     });
        return iter->second.window_length;
    }

    /**
     * 生成滑窗占空比的映射表格
     * @param winLenStageList  滑窗长度映射表
     * @param dutyStageList     占空比映射表
     */
    void generate_look_up_table(std::unordered_map<int, int> &winLenStageList,
                                std::unordered_map<int, float> &dutyStageList) {
        winLenStageList.clear();
        dutyStageList.clear();

        if (stage_routine.empty()) {
            return;
        }

        std::vector<int> stages;
        // 依次获取设置的阶梯滑窗数据的帧率设置。
        for (auto iter = stage_routine.begin(); iter != stage_routine.end(); ++iter) {
            stages.emplace_back(iter->first);
        }
        // 进行滑窗窗口数据排序
        std::sort(stages.begin(), stages.end());
        // 计算滑窗窗口的帧率设置的上限和下限
        int lower_fps_bound = stages.front();
        int upper_fps_bound = stages.back();
        // 根据设置的阶梯滑窗数据来计算不同阶梯状态的滑窗长度
        int idx = 0;
        for (int i = lower_fps_bound; i <= upper_fps_bound;) {
            if (i <= stages[idx]) {
                // 依次遍历帧率上限和帧率下限。然后帧率逐渐上升。会有限选择最近的上限帧率的滑窗长度。设置当前的滑窗长度
                winLenStageList[i] = stage_routine[stages[idx]].window_length;
                if (i == stages[idx]) {
                    idx++;
                }
                ++i;
            } else {
                idx++;
            }
        }
        // 存储对应滑窗长度下的占空比
        for (auto &pair: stage_routine) {
            dutyStageList[pair.second.window_length] = pair.second.duty_factor;
        }
    }
};

class SlidingWindow {
public:
    virtual int update(float value, VState *vstate) = 0;

    virtual int update(float value) = 0;

    virtual void clear() = 0;

    virtual void clear_buffer() = 0;

protected:
    virtual int insert_slice(float value) = 0;

    virtual void get_trigger_state(bool checked, VState *vstate) = 0;
};

class AdaptiveSlidingWindow : public SlidingWindow {
public:
    /**
     * AdaptiveSlidingWindow 自适应滑窗构造函数
     * @param source
     * @param window_len
     * @param duty_factor
     * @param target
     * @param trigger_duration
     * @param trigger_expired_period
     */
    AdaptiveSlidingWindow(int source, int window_len, float duty_factor, int target,
                          long trigger_duration = 0, long trigger_expired_period = 0);

    ~AdaptiveSlidingWindow();

    void set_window_adjust_strategy(WindowAdjustStrategy strategy) { _strategy = strategy; };

    int update(float value, VState *vstate) override;

    int update(float value) override;

    void clear() override;

    void clear_trigger_accumulative();

    void clear_buffer() override;

    void set_fps_stage_parameters(std::shared_ptr<StageParameters> &parameter);

    void set_trigger_need_time(long trigger_need_time) { _trigger_duration = trigger_need_time; }

    void set_trigger_expire_time(long expire_time) { _trigger_expired_period = expire_time; }

    void set_window_length(int length) { _window_len = length; }

    void set_duty_factor(float factor) { _duty_factor = factor; }

    int get_window_length() { return _window_len; }

    /**
     * 获取上一帧的滑窗状态
     * @return 上一帧滑窗状态
     */
    VSlidingState getPreVSlidingState();

    /**
     * 获取滑窗的最大长度
     * @return
     */
    int getMaxWindowLength() { return maxWindowLen; }

    float get_duty_factor() { return _duty_factor; }

    bool get_window_is_fill() { return static_cast<int>(_data.size()) >= _window_len; }

    /**
     * @brief 计算动态滑窗中所有数据的平均值
     * @return 
     */
    float get_window_mean_value();

    int size() {
        return _data.size();
    }

    /**
     *
     * @param source
     */
    void setSourceId(int source);

protected:
    void update_window_length();

    void adjust_window_by_stage_strategy(const int &fps);

    int insert_slice(float value) override;

    void get_trigger_state(bool checked, VState *vstate) override;

    /**  当前滑窗Window归属的Source */
    int sourceId = Source::SOURCE_UNKNOWN;
    // 配置参数
    int _default_window_len;    // 默认滑窗长度
    float _default_duty_factor;   // 默认占空比
    int _window_len;            // 滑窗长度
    float _duty_factor;         // 占空比
    int _target_value;          // 目标检测值
    long _trigger_duration;     // 动作触发需要的时间
    long _trigger_expired_period; // 触发过期时间
    /** fps策略中的fps下限,目前工程中一般是5帧 */
    int _lower_fps_bound;       // fps策略中的fps下限
    /** fps策略中的fps上限,目前工程中一般是30帧 */
    int _upper_fps_bound;       // fps策略中的fps上限
    /** 计算动态滑窗里面的滑窗的最大滑窗长度，一般 */
    int maxWindowLen;        // 最大滑窗长度
    int _min_window_len;        // 最小滑窗长度
    WindowAdjustStrategy _strategy;

    std::shared_ptr<StageParameters> _stage_parameter;
    /** 存储不同帧率阶段的滑窗长度的List */
    std::unordered_map<int, int> fpsStageList;
    /** 存储不同帧率阶段的占空比系数的List */
    std::unordered_map<int, float> dutyFactorStageList;

    // 更新参数
    long long _start_time;   // 动作开始时间
    bool _prev_checked; // 上一帧是否满足占空比
    int _head_offset;   // 滑窗数据起始帧的偏移量
    VSlidingState _prev_sliding_state; // 上一帧触发状态

    std::deque<SliceInfo> _data; // 存储每帧数据
    std::deque<long long> _trigger_records; // 触发记录，存储每次开始触发的时间戳

    std::shared_ptr<FpsUtil> fpsUtil;
    /** 暂存当前的帧率 */
    int curFps = 0;
};

class MultiValueSlidingWindow : public AdaptiveSlidingWindow {
public:
    /**
     * 多值滑窗构造函数
     * @param source
     * @param window_len
     * @param duty_factor
     */
    MultiValueSlidingWindow(int source, int window_len, float duty_factor);

    ~MultiValueSlidingWindow();

    int update(float value, VState *vstate) override;

    int update(float value) override;

    int size() { return _data.size(); }

protected:
    int insert_slice(float value) override;

private:
    std::unordered_map<int, float> _multi_count;

};

/**
 * 自定义开始和结束占空比滑窗
 * 自定义开始占空比滑窗：默认开始占空比，一般是 0.6-0.8  满足占空比即出发滑窗行为开始
 * 自定义结束占空比滑窗：默认结束占空比是0.4。 低于结束占空比滑窗即触发滑窗行为结束
 */
class CustomSateDutyFactorWindow : public AdaptiveSlidingWindow {
public:
    CustomSateDutyFactorWindow(int source, int window_len, float trigger_duty_factor, float end_duty_factor,
                               int target);

    ~CustomSateDutyFactorWindow();

    void clear() override;

    void clear_buffer() override;

    int update(float value, VState *vstate) override;

protected:
    int insert_slice(float value) override;

private:
    /**
     * 行为结束占空比
     */
    float _end_duty_factor;
    bool _checked_state;
    bool _is_state_end;
};

class DangerDriveSlidingWindow : public CustomSateDutyFactorWindow {
public:
    /**
     * 危险驾驶滑窗构造函数
     * @param source                sourceId
     * @param window_len            滑窗长度
     * @param trigger_duty_factor   默认滑窗触发占空比
     * @param end_duty_factor       默认结束的占空比
     * @param duration_limit        默认的时间限制
     * @param target                危险驾驶场景
     */
    DangerDriveSlidingWindow(int source, int window_len, float trigger_duty_factor, float end_duty_factor,
                             int duration_limit, int target);

    ~DangerDriveSlidingWindow();

    void clear() override;

    void clear_buffer() override;

protected:
    void get_trigger_state(bool checked, vision::VState *vstate) override;

private:
    int _duration_limit;
    bool _is_state_start;
};

} // namespace aura::vision

#endif //VISION_SLIDING_WINDOW_H
