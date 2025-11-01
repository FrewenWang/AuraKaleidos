//
// Created by wangyan67 on 2019-10-27.
//

#include "FpsUtil.h"
#include <numeric>
#include "util/SystemClock.h"
#include "vision/util/log.h"
#include <memory>

namespace aura::vision {

std::array<std::shared_ptr<FpsUtil>, SOURCE_COUNT + 1> FpsUtil::fpsUtilArray;

const char *TAG = "FpsUtil";

std::mutex FpsUtil::sMutexFps;

float LowPassFilterStrategy::smooth(const float &data) {
    if (isFirst) {
        prevData = data;
        isFirst = false;
    } else {
        prevData = prevData * coefficient + (1.f - coefficient) * data;
    }

    return prevData;
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

std::shared_ptr<FpsUtil> FpsUtil::instance(int source) {
    std::lock_guard<std::mutex> lock(sMutexFps);
    if (source <= SOURCE_UNKNOWN || source > SOURCE_COUNT) {
        VLOGE(TAG, "get FpsUtil failed cause of invalid source:%d", source);
        return nullptr;
    }
    auto fpsUtil = fpsUtilArray[source];
    if (fpsUtil == nullptr) {
        fpsUtil = std::make_shared<FpsUtil>();
        fpsUtilArray[source] = fpsUtil;
    }
    return fpsUtil;
}

void FpsUtil::destroy() {

}

FpsUtil::FpsUtil()
        : curFps(0.f),
          curSmoothedFps(0.f),
          prevTimestamp(-1),
          curTimestamp(-1),
          _strategy(nullptr) {
}

FpsUtil::~FpsUtil() {
    reset();
}

void FpsUtil::setSmoothStrategy(const SmoothType &smooth_type) {
    if (smooth_type == SmoothType::LowPassFilter) {
        _strategy = std::make_shared<LowPassFilterStrategy>(lowPassCoefficient);
    } else if (smooth_type == SmoothType::MovingAverage) {
        _strategy = std::make_shared<MovingAvgStrategy>(_k_moving_avg_period);
    } else {
        _strategy.reset();
    }
}

/**
 * 每一帧开始检测之前更新当前的动态帧率
 */
void FpsUtil::update() {
    // 获取当前时间戳
    curTimestamp = SystemClock::nowMillis();
    if (prevTimestamp > 0.f && prevTimestamp < curTimestamp) {
        // 备份现有帧率，重新计算最细你的帧率
        preFps = curFps;
        preSmoothedFps = curSmoothedFps;
        curFps = 1000.f / (curTimestamp - prevTimestamp);
        if (_strategy) {
            curSmoothedFps = _strategy->smooth(curFps);
        } else {
            curSmoothedFps = curFps;
        }
    } else {
        curFps = 0.f;
        curSmoothedFps = 0.f;
    }
    prevTimestamp = curTimestamp;
    // VLOGD(TAG, "smoothedFps from %f to %f, realFps from from %f to %f", preSmoothedFps, curSmoothedFps, preFps,curFps);
}

void FpsUtil::reset() {
    curFps = 0.f;
    curSmoothedFps = 0.f;
    prevTimestamp = -1;
    curTimestamp = -1;

    if (_strategy) {
        _strategy.reset();
    }
}

}