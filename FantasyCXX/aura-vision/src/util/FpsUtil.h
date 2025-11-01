
#pragma once

#include <deque>
#include <memory>
#include <string>
#include <array>
#include "vision/core/common/VConstants.h"
#include <mutex>

namespace aura::vision {
/**
 * 平滑策略类型
 */
enum class SmoothType : short {
    LowPassFilter,
    MovingAverage
};

/**
 * 平滑策略基类
 */
class SmoothStrategy {
public:
    virtual ~SmoothStrategy() = default;
    virtual float smooth(const float& data) = 0;
    virtual void reset() = 0;
};

/**
 * 低通滤波平滑
 * data = coefficient * prev_data + (1 - coefficient) * cur_data;
 */
class LowPassFilterStrategy : public SmoothStrategy {
public:
    explicit LowPassFilterStrategy(float coefficient) : coefficient(coefficient), prevData(-1.f), isFirst(true) { }

    ~LowPassFilterStrategy() override = default;

    /**
     * 低通滤波的平滑算法：data = coefficient * prev_data + (1 - coefficient) * cur_data;
     * 算法原理：原有数据帧为coefficient所定义的占比。新数据帧占比(1 - coefficient)
     * @param data
     * @return
     */
    float smooth(const float &data) override;

    void reset() override;

private:
    /**  帧率平滑参数，低通滤波系数  */
    float coefficient;
    /**  之前的帧率数据  */
    float prevData;
    /** 判断是否是第一次执行 */
    bool isFirst;
};

/**
 * 移动平均平滑
 */
class MovingAvgStrategy : public SmoothStrategy {
public:
    explicit MovingAvgStrategy(int period) : period(period) { }

    ~MovingAvgStrategy() override = default;

    /**
     * 移动平均平滑算法：始终计算滑窗内最新10帧的帧率数据的平均值
     * @param data
     * @return
     */
    float smooth(const float &data) override;

    void reset() override;

private:
    int period;
    std::deque<float> dataBuf;
};

/**
 * 帧率统计工具类，可统计实时帧率，并输出平滑后的帧率
 * data = avg(data_buff)
 */
class FpsUtil {
public:
    static std::shared_ptr<FpsUtil> instance(int source);

    static void destroy();

    ~FpsUtil();

    void update(); // 更新帧率
    void reset();  // 重置状态
    /**
     * 设置平滑策略，默认无平滑
     * @param smooth_type
     */
    void setSmoothStrategy(const SmoothType &smooth_type);

    /**
     * 获取真实计算帧率
     * @return
     */
    float getRealtimeFps() { return curFps; };

    /**
     * 获取平滑之后的帧率
     * @return
     */
    float getSmoothedFps() { return curSmoothedFps; }

    FpsUtil();

private:

    static std::array<std::shared_ptr<FpsUtil>, SOURCE_COUNT + 1> fpsUtilArray;
    /**  上一帧的帧率  */
    float preFps;
    /**  当前帧率  */
    float curFps;
    /**  上一帧率平滑之后的帧率  */
    float preSmoothedFps;
    /**  平滑之后的帧率，避免帧率抖动过快  */
    float curSmoothedFps;
    /**  上一帧时间戳  */
    long long prevTimestamp;
    /**  当前帧的时间戳  */
    long long curTimestamp;
    std::shared_ptr<SmoothStrategy> _strategy; // 平滑策略
    /**  帧率平滑参数，低通滤波系数  */
    const float lowPassCoefficient = 0.6f; // 帧率平滑参数，低通滤波系数
    const int _k_moving_avg_period = 10;  // 帧率平滑参数，移动平均周期
    static std::mutex sMutexFps;
};
} // namespace aura::vision
