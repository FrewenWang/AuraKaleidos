//
// Created by sunwenming01 on 22-8-23.
//

#include "RtConfigSource2.h"
#include "vision/util/log.h"
#include "vision/core/common/VConstants.h"

namespace aura::vision {

static const char *TAG = "RtConfigSource2";

RtConfigSource2::RtConfigSource2() {
    VLOGD(TAG, "jidu_vehicle RtConfigSource2 created");
    sourceId = 2;
    // 输入图像参数.
    frameWidth = 1920;
    frameHeight = 1280;
    frameFormat = static_cast<float>(FrameFormat::YUV_422_UYVY);

    // ======camera参数(默认我们设置集度的DMS摄像头的硬件信息) ============================
    cameraFocalLength = 5.09f;                // 焦距,
    cameraCcdWidth = 4.0306692f;               // 感光元件宽度
    cameraCcdHeight = 2.4346388f;              // 感光元件高度
    cameraLightType = CAMERA_LIGHT_TYPE_RGB;  // 采光类型
    cameraLightSwapMode = CAMERA_SWAP_MODE_AUTO;   // 采光类型是自动模式还是手动模式切换
    cameraFocalLengthPixelX = 566.351990f;     //
    cameraFocalLengthPixelY = 565.515015f;     //
    cameraOpticalCenterX = 960.495972f;       // 图像的光学中心点(CX 默认一般为图像中心点)
    cameraOpticalCenterY = 547.497009f;       // 图像的光学中心点(CY 默认一般为图像中心点)
    cameraDistortionK1 = 0;                   // 摄像头畸变参数
    cameraDistortionK2 = 0.388549f;           // 摄像头畸变参数
    cameraDistortionK3 = -0.099070f;                   // 摄像头畸变参数
    cameraDistortionP1 = 0.024403f;                   // 摄像头畸变参数
    cameraDistortionP2 = 0;                   // 摄像头畸变参数
    cameraDistortionP3 = 0;                   // 摄像头畸变参数

    // ======================== 人脸检测个数参数设置 ====================================
    faceNeedDetectCount = 2;                    // 集度车机默认业务检测2个人脸
    gestureNeedDetectCount = 2;                 // 集度车机默认业务检测2个手势
    bodyNeedDetectCount = 2;                    // 集度车机默认业务检测2个肢体(头肩)
    livingNeedDetectCount = 1;                  // 集度车机默认业务检测1个活体
    // ========================  集度OMS摄像头相机遮挡阈值设定 ============================
    // OMS IR相机遮挡阈值
    cameraCoverThresholdIr = 0.56f;
    // OMS RGB相机遮挡的比例
    cameraCoverThresholdRgb = 0.25f;

    // ======================== 感知设置 ROI 区域的逻辑 ===========================
    useDriverRoiPositionFilter = 0.0f;
    driverRoiPositionX = 0.f;
    driverRoiPositionY = 0.f;

    // =============== 感知能力计算调度策略 =======================
    faceDetectMethod = FaceDetectMethod::DETECT;
    scheduleMethod = SchedulerMethod::NAIVE; // 1: Naive, 2: DAG
    scheduleDagThreadCount = 2;              // 调度器线程数
    fixedFrameDetectSwitcher = false;        // 感知能力层设置启用间隔帧检测的策略开关。默认关闭

    // 唇动检测滑窗时间，单位ms
    lipMovementWindowTime = 1000;

    // 最小人脸框阈值，小于阈值则表示没有检测到人脸
    faceRectMinPixelThreshold = 110.0f;
}

} // namespace aura::vision
