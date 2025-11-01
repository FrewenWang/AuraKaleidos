#pragma once

#include <deque>
#include <memory>
#include <string>

namespace aura::utils {
/**
 * 平滑策略类型 低通滤波平滑
 * @see LowPassFilter  低通滤波平滑策略
 * @see MovingAverage
 */
enum class SmoothType : short {
    LowPassFilter, MovingAverage
};

/**
 * @brief 平滑策略基类
 */
class SmoothStrategy {
public:
    virtual ~SmoothStrategy() = default;

    /**
     * 纯虚函数。平滑滤波的
     * @param data
     * @return
     */
    virtual float smooth(const float &data) = 0;

    virtual void reset() = 0;
};

/**
 * 文章参考：https://blog.csdn.net/weixin_41563746/article/details/114182380
 * 低通滤波平滑策略
 * data = coefficient * prev_data + (1 - coefficient) * cur_data;
 */
class LowPassFilterStrategy : public SmoothStrategy {
public:
    LowPassFilterStrategy(float coeff) : coefficient(coeff), preData(-1.f), isFirst(true) { }

    virtual ~LowPassFilterStrategy() = default;

    float smooth(const float &data) override;

    void reset() override;

private:
    /**  帧率平滑参数，低通滤波系数  */
    float coefficient;
    /**  之前的帧率数据  */
    float preData;
    /** 判断是否是第一次执行 */
    bool isFirst;
};

/**
 * 移动平均平滑
 */
class MovingAvgStrategy : public SmoothStrategy {
public:
    explicit MovingAvgStrategy(int period) : period(period) { }

    ~MovingAvgStrategy() = default;

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
    ~FpsUtil();

    void update(); // 更新帧率
    void reset();  // 重置状态

    void setSmoothStrategy(const SmoothType &smooth_type); // 设置平滑策略，默认无平滑

    float getRealtimeFps() { return curFps; };

    float getSmoothedFps() { return curSmoothedFps; }

private:
    FpsUtil();

    /**  上一帧的帧率  */
    float preFps{};
    /**  当前帧率  */
    float curFps;
    /**  上一帧率平滑之后的帧率  */
    float preSmoothedFps{};
    /**  平滑之后的帧率，避免帧率抖动过快  */
    float curSmoothedFps;
    /**  上一帧时间戳  */
    long long prevTimestamp;
    /**  当前帧的时间戳  */
    long long curTimestamp;
    std::shared_ptr<SmoothStrategy> smoothStrategy; // 平滑策略
    /**  帧率平滑参数，低通滤波系数  */
    const float lowPassCoefficient = 0.6f; // 帧率平滑参数，低通滤波系数
    /** 帧率平滑参数，移动平均周期 */
    const int kMovingAvgPeriod = 10;  // 帧率平滑参数，移动平均周期
};
} // namespace aura::vision
