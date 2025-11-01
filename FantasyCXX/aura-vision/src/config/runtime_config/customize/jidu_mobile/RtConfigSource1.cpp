#include "RtConfigSource1.h"
#include "vision/manager/VisionManagerRegistry.h"

namespace aura::vision {

static const char *TAG = "RtConfig";

RtConfigSource1::RtConfigSource1() {
    VLOGD(TAG, "jidu_mobile RtConfig created");
    sourceId = 1;
    // 输入图像参数.
    frameWidth = 720;
    frameHeight = 1280;
    frameFormat = static_cast<float>(FrameFormat::YUV_420_NV21);
    cameraLightType = CAMERA_LIGHT_TYPE_RGB;   // 采光类型
    cameraLightSwapMode = CAMERA_SWAP_MODE_AUTO;   // 采光类型是自动模式还是手动模式切换
    cameraImageMirror = false;                 // 手机摄像头是非镜像的

    // ======================== 手机端人脸检测个数参数设置 ====================================
    faceNeedDetectCount = 1;                    // 集度手机版默认业务检测1个人脸
    gestureNeedDetectCount = 1;                 // 集度手机版默认业务检测1个手势
    bodyNeedDetectCount = 1;                    // 集度手机版默认业务检测1个肢体(头肩)
    livingNeedDetectCount = 1;                  // 集度手机版默认业务检测1个活体

    // ========================  集度手机摄像头相机遮挡阈值设定 ============================
    cameraCoverThresholdIr = 1.5f;
    cameraCoverThresholdRgb = 2.5f;

    // =============== 感知能力计算调度策略 =======================
    scheduleMethod = SchedulerMethod::NAIVE;    // 1: Naive, 2: DAG
    scheduleDagThreadCount = 2;                 // 调度器线程数
    fixedFrameDetectSwitcher = false;                   // 感知能力层设置启用间隔帧检测的策略开关。默认关闭

    // 最小人脸框阈值，小于阈值则表示没有检测到人脸
    faceRectMinPixelThreshold = 100.0f;

};

} // namespace aura::vision