#include "RtConfigSource1.h"
#include "vision/util/log.h"
#include "vision/core/common/VConstants.h"

namespace aura::vision {

static const char *TAG = "RtConfigSource1";

RtConfigSource1::RtConfigSource1() {
    VLOGD(TAG, "mainline RtConfigSource1 created");
    sourceId = 1;
    // 输入图像参数.
    frameWidth = 1600;
    frameHeight = 1300;
    frameFormat = static_cast<float>(FrameFormat::YUV_422_UYVY);

    // ======camera参数(默认我们设置集度的DMS摄像头的硬件信息) ============================
    cameraFocalLength = 5.09f;                  // 焦距,
    cameraCcdWidth = 4.0306692f;               // 感光元件宽度
    cameraCcdHeight = 2.4346388f;              // 感光元件高度
    cameraLightType = CAMERA_LIGHT_TYPE_IR;    // 采光类型
    cameraLightSwapMode = CAMERA_SWAP_MODE_AUTO;   // 采光类型是自动模式还是手动模式切换
    cameraFocalLengthPixelX = 1698.569946f;     //
    cameraFocalLengthPixelY = 1700.270020f;     //
    cameraOpticalCenterX = 799.038025f;         // 图像的光学中心点(CX 默认一般为图像中心点)
    cameraOpticalCenterY = 651.929016f;          // 图像的光学中心点(CY 默认一般为图像中心点)
    cameraDistortionK1 = 0;                   // 摄像头畸变参数
    cameraDistortionK2 = -0.343834f;          // 摄像头畸变参数
    cameraDistortionK3 = 0.195333f;                   // 摄像头畸变参数
    cameraDistortionP1 = -0.076078f;                   // 摄像头畸变参数
    cameraDistortionP2 = 0;                   // 摄像头畸变参数
    cameraDistortionP3 = 0;                   // 摄像头畸变参数

    // ======================== DMS人脸检测个数参数设置 ====================================
    faceNeedDetectCount = 1;                    // DMS默认业务检测1个人脸
    gestureNeedDetectCount = 1;                 // DMS默认业务检测1个手势
    bodyNeedDetectCount = 1;                    // DMS默认业务检测1个肢体(头肩)
    livingNeedDetectCount = 1;                  // DMS默认业务检测1个活体

    // ========================  集度DMS摄像头相机遮挡阈值设定 ============================
    cameraCoverThresholdIr = 0.71f;
    cameraCoverThresholdRgb = 0.71f;

    // ======================== 感知设置 ROI 区域的逻辑 ===========================
    useDriverRoiPositionFilter = 0.0f;
    driverRoiPositionX = 0.f;
    driverRoiPositionY = 0.f;

    // =============== 感知能力计算调度策略 =========================================
    faceDetectMethod = FaceDetectMethod::DETECT;
    scheduleMethod = SchedulerMethod::NAIVE;      // 1: Naive, 2: DAG
    scheduleDagThreadCount = 2;                   // 调度器线程数
    fixedFrameDetectSwitcher = false;                   // 感知能力层设置启用间隔帧检测的策略开关。默认关闭

    // 唇动检测滑窗时间，单位ms
    lipMovementWindowTime = 1000;

    // 最小人脸框阈值，小于阈值则表示没有检测到人脸
    faceRectMinPixelThreshold = 100.0f;
}

} // namespace aura::vision