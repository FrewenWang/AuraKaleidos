
#ifdef BUILD_3D_LANDMARK

#include "Face3dLandmarkDetector.h"
#include "util/DebugUtil.h"
#include "util/math_utils.h"
#include <fstream>

namespace aura::vision {

static const char *TAG = "Face3dLandmarkDetector";

Face3dLandmarkDetector::Face3dLandmarkDetector() {
    TAG = "Face3dLandmarkDetector";
    mPerfTag += TAG;
}

Face3dLandmarkDetector::~Face3dLandmarkDetector() = default;

int Face3dLandmarkDetector::init(RtConfig *cfg) {
    mRtConfig = cfg;
    //  相机焦距
    focalLength_ = mRtConfig->cameraFocalLength;
    // 相机CCD的宽高
    ccdWidth_ = mRtConfig->cameraCcdWidth;
    ccdHeight_ = mRtConfig->cameraCcdHeight;
    // 图像的宽高
    imgWidth_ = mRtConfig->frameWidth;
    imgHeight_ = mRtConfig->frameHeight;
    // 计算摄像头的fx、fx
    cameraFocalLengthPixX = mRtConfig->cameraFocalLengthPixelX;
    cameraFocalLengthPixY = mRtConfig->cameraFocalLengthPixelY;
    // 图像的光学中心点cx,xy(一般情况下)
    opticalCenterX_ = mRtConfig->cameraOpticalCenterX;
    opticalCenterY_ = mRtConfig->cameraOpticalCenterY;
    // 相机畸变相关的参数。目前使用默认参数
    K1_ = mRtConfig->cameraDistortionK1;
    K2_ = mRtConfig->cameraDistortionK2;
    K3_ = mRtConfig->cameraDistortionK3;
    P1_ = mRtConfig->cameraDistortionP1;
    P2_ = mRtConfig->cameraDistortionP2;
    P3_ = mRtConfig->cameraDistortionP3;
    bool setResult = face3dLandmarks.SetFace3dPtsParamsFromCamera(cameraFocalLengthPixX, cameraFocalLengthPixY,
                                                                  imgWidth_, imgHeight_,
                                                                  opticalCenterX_, opticalCenterY_,
                                                                  K1_, K2_, K3_,
                                                                  P1_, P2_, P3_,
                                                                  matRot_, matTrans_,
                                                                  mRtConfig->speedThreshold,
                                                                  mRtConfig->steeringWheelAngleThreshold,
                                                                  errorMaxIter_, errorParamsIter_,
                                                                  errorSatisfyNum_);
    // 定义长度为212的vector用于存储3D关键点数据
    point106.resize(LM_3D_106_COUNT * 2);

    if (setResult) {
        V_RET(Error::OK);
    } else {
        V_RET(Error::PREPARE_ERR);
    }
}

int Face3dLandmarkDetector::doDetect(VisionRequest *request, VisionResult *result) {

    FaceInfo **infos = result->getFaceResult()->faceInfos;
    // 如果修改相机参数，需在此处设置
    bool setResult = face3dLandmarks.SetFace3dPtsParamsFromCamera(mRtConfig->cameraFocalLengthPixelX,
                                                                  mRtConfig->cameraFocalLengthPixelY,
                                                                  request->width, request->height,
                                                                  mRtConfig->cameraOpticalCenterX,
                                                                  mRtConfig->cameraOpticalCenterY,
                                                                  mRtConfig->cameraDistortionK1,
                                                                  mRtConfig->cameraDistortionK2,
                                                                  mRtConfig->cameraDistortionK3,
                                                                  mRtConfig->cameraDistortionP1,
                                                                  mRtConfig->cameraDistortionP2,
                                                                  mRtConfig->cameraDistortionP3,
                                                                  matRot_, matTrans_,
                                                                  mRtConfig->speedThreshold,
                                                                  mRtConfig->steeringWheelAngleThreshold,
                                                                  errorMaxIter_, errorParamsIter_,
                                                                  errorSatisfyNum_);
    if (!setResult) {
        VLOGE(TAG, "3d_landmark set params from camera error");
        V_RET(Error::INFER_ERR);
    }

    // 获取摄像头镜像翻转的标志变量
    // 如果是镜像的摄像头，需要进行取反操作，方向遵循原则：
    // https://ku.baidu-int.com/knowledge/HFVrC7hq1Q/pKzJfZczuc/gNDqu2bZJ4/2wtaXFDC3zF8_c
    cameraMirror = V_F_TO_BOOL(mRtConfig->cameraImageMirror);

    for (int i = 0; i < static_cast<int>(mRtConfig->faceNeedDetectCount); ++i) {
        auto *face = infos[i];
        V_CHECK_CONT(face->isNotDetectType()); // 判断当前人脸是否是模型检出
        // 平铺处理106个人脸关键点
        VPoint *point = nullptr;
        for (auto j = 0; j < LM_3D_106_COUNT; ++j) {
            point = &(face->landmark2D106[j]);
            point106[2 * j] = point->x;
            point106[2 * j + 1] = point->y;
        }

        // 检测 3D视线检测逻辑是否开启 add by wangzhijiang 目前如果设置false会出现问题，算法yubin要求设置为true
        // if (mRtConfig->get_switch(ABILITY_FACE_3D_EYE_GAZE)) {
        //     bCalLeftEye = true;
        //     bCalRightEye = true;
        // } else {
        //     bCalLeftEye = false;
        //     bCalRightEye = false;
        // }
        bCalLeftEye = true;
        bCalRightEye = true;
        {
            // 算法提供用于推理结果对其的demo文本读取
            // std::ifstream ifspts("./demo.txt");
            // for (int i = 0; i < 212; i++) {
            //     std::string strline;
            //     ifspts >> strline;
            //     float fnum = atof(strline.c_str());
            //     point106[i] = fnum;
            // }
            
            auto eyeGazeCalibSwitcher = static_cast<bool>(mRtConfig->eyeGazeCalibSwitcher);
            auto vehicle = request->vehicleInfo;
            VLOGD(TAG, "face_3d_landmark[%ld] focalPixel=[%f,%f],image=[%d,%d],center=[%f,%f],K=[%f,%f,%f],P=[%f,%f,%f],"
                       "speedThreshold=[%f],wheelAngleThreshold=[%f],switch=[%d] speed=[%f], turningLamp=[%d],wheelAngle=[%f]",
                  face->id, mRtConfig->cameraFocalLengthPixelX, mRtConfig->cameraFocalLengthPixelY,
                  request->width, request->height, mRtConfig->cameraOpticalCenterX, mRtConfig->cameraOpticalCenterY,
                  mRtConfig->cameraDistortionK1, mRtConfig->cameraDistortionK2, mRtConfig->cameraDistortionK3,
                  mRtConfig->cameraDistortionP1, mRtConfig->cameraDistortionP2, mRtConfig->cameraDistortionP3,
                  mRtConfig->speedThreshold, mRtConfig->steeringWheelAngleThreshold, eyeGazeCalibSwitcher,
                  vehicle->speed, vehicle->turningLamp, vehicle->steeringWheelAngle);
            PERF_AUTO(PerfUtil::global(), "Face3DLandmarkDetector-pro")
            bool cal3dLandmarkResult = face3dLandmarks.Cal3dPoint(point106, faceEulerAngle, facePts3d,
                                                                  leftEye3dPts, rightEye3dPts,
                                                                  leftEyeGaze, rightEyeGaze,
                                                                  leftEyeGazeCalib, rightEyeGazeCalib,
                                                                  leftEyeGazeTrans, rightEyeGazeTrans,
                                                                  vehicle->speed, vehicle->steeringWheelAngle,
                                                                  vehicle->turningLamp,
                                                                  eyeGazeCalibValid,
                                                                  bCalLeftEye, bCalRightEye, eyeGazeCalibSwitcher);
            // 在调用算法3D库检测失败时打印2D关键点信息，以快速定位问题
            if (!cal3dLandmarkResult) {
                std::stringstream ss;
                ss << "Face 3D landmark inference error!!!" << " -> faceId[" << face->id << "]" << " "
                   << "2D_Landmark_Length[" << (int)point106.size() << "]";
                VLOGE(TAG, "%s", ss.str().c_str());

                int end = 0;
                for (auto m = 0; m < DEFAULT_PRINT_2D_LANDMARK_EPOCH; ++m) {
                    std::stringstream ss1;
                    ss1 << "Point106[" << m << "] [";
                    if (m == DEFAULT_PRINT_2D_LANDMARK_EPOCH - 1) {
                        end = LM_3D_106_COUNT * 2;
                    } else {
                        end = (m + 1) * DEFAULT_PRINT_2D_LANDMARK_NUMBER;
                    }
                    for (auto i = m * DEFAULT_PRINT_2D_LANDMARK_NUMBER; i < end; ++i) {
                        ss1 << point106[i] << ",";
                    }
                    ss1 << "]";
                    VLOGE(TAG, "%s", ss1.str().c_str());
                }
                continue;
            }
        }
        // 计算一点标定的标识，默认为0,如果不为0,则标定结果不可用
        face->eyeGazeCalibValid = (eyeGazeCalibValid == DEFAULT_EYE_GAZE_CALIB_VALID_VALUE);
        // 头部姿态偏转角  三向偏转角.算法同学确认，3D模型输出的头部姿态偏转角比2D landmark关键点输出更加精准
        // 故使用3D模型关键点复写headDeflection的值。需要上层控制打开对应能力
        if (mRtConfig->cameraImageMirror) {
            // yaw 和 roll需要考虑镜像反转
            face->headDeflection3D.yaw = faceEulerAngle[1] * RADIAN_2_ANGLE_FACTOR;
            face->headDeflection3D.roll = faceEulerAngle[2] * RADIAN_2_ANGLE_FACTOR;
        } else {
            face->headDeflection3D.yaw = -faceEulerAngle[1] * RADIAN_2_ANGLE_FACTOR;
            face->headDeflection3D.roll = -faceEulerAngle[2] * RADIAN_2_ANGLE_FACTOR;
        }
        face->headDeflection3D.pitch = -faceEulerAngle[0] * RADIAN_2_ANGLE_FACTOR;

        // 转化 3D 人脸关键点
        int index = 0;
        for (int k = 0; k < LM_3D_68_COUNT; k++) {
            VPoint3 &p = face->landmark3D68[k];
            index = 3 * k;
            // 跟算法确认：如果是镜像的摄像头。则X轴进行取反，否则不需要取反
            p.setValue(cameraMirror ? -facePts3d[index] : facePts3d[index], facePts3d[index + 1], facePts3d[index + 2]);
        }

        // 跟算法同学同步：头部位置,目前是采用鼻尖的位置点(68个关键点第31个点)的3D坐标的x,y值， z值使用下巴的位置点(68个关键点第9个点)
        VPoint3 noseTip = face->landmark3D68[FLM68_30_NOSE_TIP];
        auto headLocationZ = face->landmark3D68[FLM68_8_CT_CHIN9].z;
        face->headLocation.setValue(noseTip.x, noseTip.y, headLocationZ);

        // 检测 3D视线检测逻辑是否开启
        if (mRtConfig->get_switch(ABILITY_FACE_3D_EYE_GAZE)) {
            // 获取左眼的 3D 关键点
            for (int m = 0; m < LM_EYE_3D_28_COUNT; m++) {
                index = 3 * m;
                VPoint3 &left = face->eye3dLandmark28Left[m];
                // 跟算法确认：如果是镜像的摄像头。则X轴进行取反，否则不需要取反
                left.setValue(cameraMirror ? -leftEye3dPts[index] : leftEye3dPts[index], leftEye3dPts[index + 1],
                              leftEye3dPts[index + 2]);
            }
            // 调试代码将3D关键点转化为2D关键点
#ifdef DEBUG_SUPPORT
            DBG_PRINT_POINTS(leftEye3dPts, "left_eye_3d_landmarks");
            // 调用眼部关键点3D转2D逻辑
            bool result = cockpitcv::convert3dpointTo2dpoint(leftEye3dPts, leftEye2dPts,
                                                             mRtConfig->cameraFocalLengthPixelX,
                                                             mRtConfig->cameraFocalLengthPixelY,
                                                             mRtConfig->cameraOpticalCenterX,
                                                             mRtConfig->cameraOpticalCenterY);
            for (int m = 0; m < LM_EYE_3D_28_COUNT; m++) {
                index = 2 * m;
                VPoint &left2d = face->eye2dLandmark28Left[m];
                left2d.setValue(leftEye2dPts[index], leftEye2dPts[index + 1]);
            }
            if (result) {
                DBG_PRINT_POINTS(leftEye2dPts, "left_eye_2d_landmarks");
            } else {
                VLOGW(TAG, "convert left eye landmark 3D to 2D failed");
            }
#endif
            // 设置右眼睛的 3D 关键点
            for (int m = 0; m < LM_EYE_3D_28_COUNT; m++) {
                index = 3 * m;
                VPoint3 &right = face->eye3dLandmark28Right[m];
                // 跟算法确认：如果是镜像的摄像头。则X轴进行取反，否则不需要取反
                right.setValue(cameraMirror ? -rightEye3dPts[index] : rightEye3dPts[index], rightEye3dPts[index + 1],
                               rightEye3dPts[index + 2]);
            }
            // 调试代码将3D关键点转化为2D关键点
#ifdef DEBUG_SUPPORT
            DBG_PRINT_POINTS(rightEye3dPts, "right_eye_3d_landmarks");
            // 调用眼部关键点3D转2D逻辑
            result = cockpitcv::convert3dpointTo2dpoint(
                    rightEye3dPts, rightEye2dPts, mRtConfig->cameraFocalLengthPixelX,
                    mRtConfig->cameraFocalLengthPixelY, mRtConfig->cameraOpticalCenterX,
                    mRtConfig->cameraOpticalCenterY);
            for (int m = 0; m < LM_EYE_3D_28_COUNT; m++) {
                index = 2 * m;
                VPoint &right2d = face->eye2dLandmark28Right[m];
                right2d.setValue(rightEye2dPts[index], rightEye2dPts[index + 1]);
            }
            if (result) {
                DBG_PRINT_POINTS(rightEye2dPts, "right_eye_2d_landmarks");
            } else {
                VLOGW(TAG, "convert right eye landmark 3D to 2D failed");
            }
#endif
            // 如果能够检测到瞳孔则输出左眼视线相关的逻辑
            if (face->leftEyeCoverSingle == F_QUALITY_COVER_LEFT_EYE_NORMAL
                && face->leftEyeDetectSingle == PUPIL_AVAILABLE) {
                // 计算3D视线
                // 的起始点三维坐标，使用算法模型输出的左右眼睛的28个关键点中的8-20的进行平均处理（8-20左闭右开）
                getEyeMeanPoint(leftEye3dPts, 8, 20, face->eyeGazeOriginLeft);
                // 跟算法确认：如果是镜像的摄像头。则需要对视线源点X轴进行取反，否则不需要取反
                face->eyeGazeOriginLeft.x = cameraMirror ? -(face->eyeGazeOriginLeft.x) : face->eyeGazeOriginLeft.x;
                // 3D视线结果不需要做弧度转角度，左右眼视线的输出的结果是单位向量
                // 跟算法确认：如果是镜像的摄像头。则X轴进行取反，否则不需要取反
                face->eyeGaze3dVectorLeft.setValue(cameraMirror ? -leftEyeGaze[0] : leftEyeGaze[0], leftEyeGaze[1],
                                                   leftEyeGaze[2]);

                if (!leftEyeGazeCalib.empty()) {
                    face->eyeGaze3dCalibVectorLeft.setValue(cameraMirror ? -leftEyeGazeCalib[0] : leftEyeGazeCalib[0],
                                                            leftEyeGazeCalib[1],leftEyeGazeCalib[2]);
                }
                if (!leftEyeGazeTrans.empty()) {
                    face->eyeGaze3dTransVectorLeft.setValue(cameraMirror ? -leftEyeGazeTrans[0] : leftEyeGazeTrans[0],
                                                            leftEyeGazeTrans[1],leftEyeGazeTrans[2]);
                }
                VLOGI(TAG, "face_3d_landmark[%ld],headDeflection3D=[%f,%f,%f],Left status=[%d],origin=[%f,%f,%f],"
                           "vector=[%f,%f,%f], vectorCalib=[%f,%f,%f],vectorTrans=[%f,%f,%f],gazeCalibValid=[%d]",
                      face->id,
                      face->headDeflection3D.yaw, face->headDeflection3D.pitch, face->headDeflection3D.roll,
                      face->leftEyeDetectSingle,
                      face->eyeGazeOriginLeft.x, face->eyeGazeOriginLeft.y, face->eyeGazeOriginLeft.z,
                      face->eyeGaze3dVectorLeft.x, face->eyeGaze3dVectorLeft.y, face->eyeGaze3dVectorLeft.z,
                      face->eyeGaze3dCalibVectorLeft.x, face->eyeGaze3dCalibVectorLeft.y,
                      face->eyeGaze3dCalibVectorLeft.z,
                      face->eyeGaze3dTransVectorLeft.x, face->eyeGaze3dTransVectorLeft.y,
                      face->eyeGaze3dTransVectorLeft.z, face->eyeGazeCalibValid);

            }
            // 如果能够检测到瞳孔则输出左眼视线相关的逻辑
            if (face->rightEyeCoverSingle == F_QUALITY_COVER_RIGHT_EYE_NORMAL
                && face->rightEyeDetectSingle == PUPIL_AVAILABLE) {
                // 获取眼部关键点的平均中心点位置坐标。作为视线的起点
                getEyeMeanPoint(rightEye3dPts, 8, 20, face->eyeGazeOriginRight);
                // 跟算法确认：如果是镜像的摄像头。则需要对视线源点X轴进行取反，否则不需要取反
                face->eyeGazeOriginRight.x = cameraMirror ? -(face->eyeGazeOriginRight.x) : face->eyeGazeOriginRight.x;
                // 跟算法确认：如果是镜像的摄像头。则视线向量X轴进行取反，否则不需要取反
                face->eyeGaze3dVectorRight.setValue(cameraMirror ? -rightEyeGaze[0] : rightEyeGaze[0], rightEyeGaze[1],
                                                    rightEyeGaze[2]);
                if (!rightEyeGazeCalib.empty()) {
                    face->eyeGaze3dCalibVectorRight.setValue(
                            cameraMirror ? -rightEyeGazeCalib[0] : rightEyeGazeCalib[0],
                            rightEyeGazeCalib[1], rightEyeGazeCalib[2]);
                }
                if (!rightEyeGazeTrans.empty()) {
                    face->eyeGaze3dTransVectorRight.setValue(
                            cameraMirror ? -rightEyeGazeTrans[0] : rightEyeGazeTrans[0],
                            rightEyeGazeTrans[1], rightEyeGazeTrans[2]);
                }
                VLOGI(TAG, "face_3d_landmark[%ld],headDeflection3D=[%f,%f,%f],Right status=[%d],origin=[%f,%f,%f],"
                           "vector=[%f,%f,%f], vectorCalib=[%f,%f,%f],vectorTrans=[%f,%f,%f],gazeCalibValid=[%d]",
                      face->id,
                      face->headDeflection3D.yaw, face->headDeflection3D.pitch, face->headDeflection3D.roll,
                      face->rightEyeDetectSingle,
                      face->eyeGazeOriginRight.x, face->eyeGazeOriginRight.y, face->eyeGazeOriginRight.z,
                      face->eyeGaze3dVectorRight.x, face->eyeGaze3dVectorRight.y, face->eyeGaze3dVectorRight.z,
                      face->eyeGaze3dCalibVectorRight.x, face->eyeGaze3dCalibVectorRight.y,
                      face->eyeGaze3dCalibVectorRight.z,
                      face->eyeGaze3dTransVectorRight.x, face->eyeGaze3dTransVectorRight.y,
                      face->eyeGaze3dTransVectorRight.z, face->eyeGazeCalibValid);
            }
            // 当两只眼睛都未检测到瞳孔，打印眼睛状态日志以方便排查
            if (face->leftEyeDetectSingle != PUPIL_AVAILABLE && face->rightEyeDetectSingle != PUPIL_AVAILABLE) {
                VLOGD(TAG, "pupil[%ld] leftEyeDetectSingle[%d] rightEyeDetectSingle[%d]", face->id,
                      face->leftEyeDetectSingle, face->rightEyeDetectSingle);
            }
        }
        // 调试逻辑：打印所有的3D关键点
        DBG_PRINT_FACE_LMK_3D(face);
    }
    V_RET(Error::OK);
}

bool Face3dLandmarkDetector::resetEyeGazeCalib() {
    eyeGazeCalibValid = DEFAULT_EYE_GAZE_CALIB_INVALID_VALUE;
    return face3dLandmarks.ResetGazevalue();
}

int Face3dLandmarkDetector::prepare(VisionRequest *request, FaceInfo **infos, TensorArray &prepared) {
    V_RET(Error::OK);
}

int Face3dLandmarkDetector::process(VisionRequest *request, TensorArray &inputs, TensorArray &outputs) {
    V_RET(Error::OK);
}

int Face3dLandmarkDetector::post(VisionRequest *request, TensorArray &infer_results, FaceInfo **infos) {
    V_RET(Error::OK);
}

VPoint3 Face3dLandmarkDetector::getEyeMeanPoint(std::vector<float> points, int beginIndex, int endIndex,
                                                VPoint3 &result) {
    if (points.empty() || beginIndex <= 0 || endIndex >= static_cast<int>(points.size())) {
        VLOGW(TAG, "get eye mean point error");
        return result;
    }
    float totalX = 0.;
    float totalY = 0.;
    float totalZ = 0.;
    int index = 0;
    for (int i = beginIndex; i < endIndex; ++i) {
        index = 3 * i;
        totalX += points[index];
        totalY += points[index + 1];
        totalZ += points[index + 2];
    }
    auto totalCount = static_cast<float>(endIndex - beginIndex + 1);
    result.setValue(totalX / totalCount, totalY / totalCount, totalZ / totalCount);
    return result;
}

} // namespace vision

#endif