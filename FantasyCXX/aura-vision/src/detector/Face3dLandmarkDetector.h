#pragma once

#include "AbsFaceDetector.h"
#include "vision/core/bean/FaceInfo.h"
#include "vision/core/common/VStructs.h"
#include "face_3d_points.h"

namespace aura::vision {

class Face3dLandmarkDetector : public AbsFaceDetector {
public:
    Face3dLandmarkDetector();

    ~Face3dLandmarkDetector() override;

    /**
     * 初始化Manager
     * @param cfg
     * @return
     */
    int init(RtConfig *cfg) override;

    /**
     * 执行3D关键点检测
     * @param frame
     * @param infos
     * @param perf
     * @return
     */
    int doDetect(VisionRequest *request, VisionResult *result) override;
    /**
     * 重置视线标定的逻辑
     * @return
     */
    bool resetEyeGazeCalib();

protected:
    int prepare(VisionRequest *request, FaceInfo **infos, TensorArray& prepared) override;

    int process(VisionRequest *request, TensorArray &inputs, TensorArray &outputs) override;

    int post(VisionRequest *request, TensorArray &infer_results, FaceInfo **infos) override;

  private:
    /**
     * 根据3D关键点的接口定义
     * int &nMaskCali, 一点标定的标识，默认为0,如果不为0,则标定结果不可用
     * 默认我们认为其标定结果可用
     */
    const int DEFAULT_EYE_GAZE_CALIB_VALID_VALUE = 0;
    const int DEFAULT_EYE_GAZE_CALIB_INVALID_VALUE = 1;
    /** 调用算法3d库检测失败时分批打印2d关键点，总共的打印轮数 */
    const int DEFAULT_PRINT_2D_LANDMARK_EPOCH = 6;
    /** 调用算法3d库检测失败时分批打印2d关键点，每批打印的关键点数目 */
    const int DEFAULT_PRINT_2D_LANDMARK_NUMBER = 36;
    /**
     * 3D 人脸关键点生成对象
     */
    cockpitcv::Face3dPts face3dLandmarks;
    std::vector<float> point106;       // 传入的106个关键点坐标,顺序为：x1y1x2y2.....x106y106
    std::vector<float> faceEulerAngle; // 输出头部姿态的三个值
    /**
     * 从V6.0版本开始，3D关键点输出关键点数量为68
     */
    std::vector<float> facePts3d;      // 输出人脸的68个3d关键点
    /**
     * 从106个人脸关键点中提取出左右各10个人眼关键点，并近似出左右个28个人眼关键点（眼皮12个，虹膜8个，瞳孔8个）
     * 28个人眼关键点中，虹膜编号0-7：最左顶点开始顺时针旋转，60度一个点；
     * 28个人眼关键点中，眼皮编号8-19：顺序最左顶点顺时针旋转；
     * 28个人眼关键点中，瞳孔编号20-27：最左顶点为27，顺时针递减
     */
    std::vector<float> leftEye3dPts;  // 输出左眼的28个3d关键点
    std::vector<float> rightEye3dPts; // 输出右眼的28个3d关键点
    std::vector<float> leftEye2dPts;  // 输出左眼的28个2d关键点
    std::vector<float> rightEye2dPts; // 输出右眼的28个2d关键点
    std::vector<float> leftEyeGaze;   // 输出左眼视线
    std::vector<float> rightEyeGaze;  // 输出右眼视线
    std::vector<float> leftEyeGazeCalib;   // 输出左眼视线(标定之后的视线向量)
    std::vector<float> rightEyeGazeCalib;  // 输出右眼视线(标定之后的视线向量)
    std::vector<float> leftEyeGazeTrans;   // 输出左眼视线的偏移量
    std::vector<float> rightEyeGazeTrans;   // 输出右眼视线的偏移量
    // 一点标定的标识，默认为0,
    // 如果为1,标定结果不可用. 能力初始化的时候设置为1. 即标定能力不可用
    int eyeGazeCalibValid = DEFAULT_EYE_GAZE_CALIB_INVALID_VALUE;

    // 默认不开启视线检测
    bool bCalLeftEye = false;  // 是否计算左眼的3D点，若为否，则左眼3d关键点，左眼视线不可取
    bool bCalRightEye = false; // 是否计算右眼的3D点，若为否，则右眼3d关键点，右眼视线不可取
  private:
    // 摄像硬件参数
    // 如果修改相机参数，需在此处设置
    //  相机焦距
    float focalLength_;
    // 相机CCD的宽高
    float ccdWidth_;
    float ccdHeight_;
    // 图像的宽高
    unsigned int imgWidth_;
    unsigned int imgHeight_;
    // 图像的光学中心点(一般情况下)
    float cameraFocalLengthPixX;
    float cameraFocalLengthPixY;

    float opticalCenterX_;
    float opticalCenterY_;
    // 相机畸变相关的参数。目前使用默认参数
    float K1_;
    float K2_;
    float K3_;
    float P1_;
    float P2_;
    float P3_;
    // 相机外参。目前使用matRot，matTrans的默认值，人脸3d关键点与相机坐标系下的值一致
    // 相机外参 长度为9的旋转信息，默认1,0,0,0,1,0,0,0,1
    std::vector<float> matRot_ = {1., 0., 0., 0., 1., 0., 0., 0., 1.};
    // 相机外参 长度为3的平移信息,默认0，0，0
    std::vector<float> matTrans_ = {0, 0, 0};

    // 模型迭代最大次数，默认1000，一般图像运行8-10次即完成。
    // 如果超出次数没有运行成功，则表明运行失败。如果要求时间较少，则可以将次数调小一些。
    int errorMaxIter_ = 100;
    // 模型迭代时，分析误差时使用的参数，默认0.999，
    //  本次误差乘以0.999小于上次误差，如果对精度要求不是很高，可以将此数字调的小一些。
    float errorParamsIter_ = 0.999;
    // 模型迭代时，分析误差满足条件的次数，满足这个数则停止迭代，默认3
    int errorSatisfyNum_ = 3;
    /** 判断camera是否是镜像的标志变量 */
    bool cameraMirror = true;
    /**
     * 获取眼部关键点的平均中心点位置坐标（索引区间为左闭右开）
     * @param points        眼部关键点
     * @param beginIndex    开始索引
     * @param endIndex      结束索引
     * @param result        输出结果
     * @return
     */
    VPoint3 getEyeMeanPoint(std::vector<float> points, int beginIndex, int endIndex, VPoint3 &result);
};

} // namespace vision
