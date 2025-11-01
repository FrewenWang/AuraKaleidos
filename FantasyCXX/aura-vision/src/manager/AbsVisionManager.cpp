//
// Created by Li,Wendong on 12/19/21.
//

#include <vision/manager/AbsVisionManager.h>
#include "util/SystemClock.h"

namespace aura::vision {

void AbsVisionManager::detect(VisionRequest *request, VisionResult *result) {
    /// 临时规避策略：避免三方库推理异常，catch所有异常进行异常信息输出
    try {
        if (preDetect(request, result)) {
            PERF_AUTO(result->getPerfUtil(), "[Manager]: " + name);
            doDetect(request, result);
        }
    } catch (const std::exception &ex) {
        VLOGE("VisionManager", " %s  detect Error!!!! cause of %s", name.c_str(), ex.what());
    } catch (...) {
        VLOGE("VisionManager", " %s  detect Error!!!!", name.c_str());
    }
}

bool AbsVisionManager::execute() {
    if (pGraph->pRtConfig->get_switch(mId)) {
        detect(pGraph->pRequest, pGraph->pResult);
        return true;
    }
    return true;
}

/**
 * Clear的时候需要检测的时间戳
 */
void AbsVisionManager::clear() {
    preDetectMillis = 0;
    // clear的时候设置强制进行人脸检测标志变量设置为true。
    // 重新启动之后，第一帧就可以进行检测。
    forceDetectCurFrame = true;
}

bool AbsVisionManager::checkFpsFixedDetect(int detectDuration, const bool &forceDetect) {
    // 判断原子能力固定帧率检测是否开启，如果没开启.则不需要按照自定义帧率检测，直接返回true。每帧都进行检测
    return true;
    // auto curMillis = SystemClock::nowMillis();
    // // 如果业务逻辑控制需要强制进行检测。则直接返回true
    // if (forceDetect) {
    //     preDetectMillis = curMillis;
    //     return true;
    // }
    // if (curMillis >= preDetectMillis + detectDuration) {
    //     // 如果当前时间戳相比较上一帧检测已经大于帧率时长的阈值，
    //     // 注意：上帧检测的时间需要回退到固定帧率应该触发检测的时间点
    //     preDetectMillis = curMillis - ((curMillis - preDetectMillis) % detectDuration);
    //     return true;
    // }
    // return false;
}

}