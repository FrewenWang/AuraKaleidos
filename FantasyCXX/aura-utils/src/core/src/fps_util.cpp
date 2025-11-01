//
// Created by Frewen.Wang on 2023/9/24.
//
#include <numeric>
#include <mutex>
#include "aura/utils/system_clock.h"
#include "aura/utils/fps_util.h"

namespace aura::utils {

static const char *TAG = "FpsUtil";
static std::mutex sMutexFps;

float LowPassFilterStrategy::smooth(const float &data) {
    if (isFirst) {
        preData = data;
        isFirst = false;
    } else {
        preData = preData * coefficient + (1.f - coefficient) * data;
    }

    return preData;
}

void LowPassFilterStrategy::reset() {
    isFirst = true;
}

float MovingAvgStrategy::smooth(const float &data) {
    int buf_size = static_cast<int>(dataBuf.size());
    if (buf_size >= period) {
        dataBuf.pop_front();
        dataBuf.emplace_back(data);
    } else {
        dataBuf.emplace_back(data);
    }
    return std::accumulate(dataBuf.begin(), dataBuf.end(), 0.f) / buf_size;
}

void MovingAvgStrategy::reset() {
    dataBuf.clear();
}

FpsUtil::FpsUtil() : curFps(0.f), curSmoothedFps(0.f), prevTimestamp(-1), curTimestamp(-1), smoothStrategy(nullptr) {
}

FpsUtil::~FpsUtil() {
    reset();
}

void FpsUtil::setSmoothStrategy(const SmoothType &smooth_type) {
    if (smooth_type == SmoothType::LowPassFilter) {
        smoothStrategy = std::make_shared<LowPassFilterStrategy>(lowPassCoefficient);
    } else if (smooth_type == SmoothType::MovingAverage) {
        smoothStrategy = std::make_shared<MovingAvgStrategy>(kMovingAvgPeriod);
    } else {
        smoothStrategy.reset();
    }
}

/**
 * 每一帧开始检测之前更新当前的动态帧率
 */
void FpsUtil::update() {
    // 获取当前时间戳
    curTimestamp = SystemClock::nowMillis();
    if (prevTimestamp > 0 && prevTimestamp < curTimestamp) {
        // 备份现有帧率，重新计算最细你的帧率
        preFps = curFps;
        preSmoothedFps = curSmoothedFps;
        curFps = 1000.f / (curTimestamp - prevTimestamp);
        if (smoothStrategy) {
            curSmoothedFps = smoothStrategy->smooth(curFps);
        } else {
            curSmoothedFps = curFps;
        }
    } else {
        curFps = 0.f;
        curSmoothedFps = 0.f;
    }
    prevTimestamp = curTimestamp;
}

void FpsUtil::reset() {
    curFps = 0.f;
    curSmoothedFps = 0.f;
    prevTimestamp = -1;
    curTimestamp = -1;

    if (smoothStrategy) {
        smoothStrategy.reset();
    }
}

}
