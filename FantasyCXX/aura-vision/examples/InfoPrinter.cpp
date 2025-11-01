//
// Created by wangzhijiang on 22-9-4.
//

#include "InfoPrinter.h"
#include <fstream>
#include <iostream>
#include <sstream>

namespace vision {

static const char *TAG = "VisionResult";

void InfoPrinter::print(vision::VisionService *service, vision::VisionResult *result) {
    VLOGI(TAG, "=============================== detect result begin======================================");
    FrameInfo *frameInfo = result->getFrameInfo();
    VLOGI(TAG, "FrameInfo : width: %d,height:%d, timestamp: %ld", frameInfo->width, frameInfo->height,
          frameInfo->timestamp);
    printFaceInfo(service, result);
    printGestureInfo(service, result);
    printBodyInfo(service, result);
    printLivingInfo(service, result);
    VLOGI(TAG, "=============================== detect result end======================================");
}

void InfoPrinter::printFaceInfo(VisionService *service, VisionResult *result) {
    FaceResult *faceResult = result->getFaceResult();
    FaceInfo *face = nullptr;
    for (int i = 0; i < faceResult->faceMaxCount; ++i) {
        FaceInfo *face = faceResult->faceInfos[i];
        if (face == nullptr || !face->hasFace()) {
            VLOGD(TAG, "face[%d]无人脸", i);
            continue;
        }
        VLOGD(TAG, "==========================打印第%d个人脸==========================", i + 1);
        // 打印 FaceRect 计算结果
        VLOGI(TAG, "FaceRect : ID: %ld | confidence: %f | RectCenter(%f , %f) RectLT(%f , %f) RectBT(%f , %f)",
              face->id, face->rectConfidence, face->rectCenter.x, face->rectCenter.y,
              face->rectLT.x, face->rectLT.y, face->rectRB.x, face->rectRB.y);
        // 打印 FaceLandmark 计算结果
        if (service->get_switch(ABILITY_FACE_LANDMARK)) {
            VLOGI(TAG, "FaceLandmark : | confidence: %f | 1:(%f , %f), 106:(%f , %f)",
                  face->landmarkConfidence, face->landmark2D106[0].x, face->landmark2D106[0].y,
                  face->landmark2D106[105].x, face->landmark2D106[105].y);
        }
        // 打印头部偏转角计算结果
        if (service->get_switch(ABILITY_FACE_LANDMARK)) {
            VLOGI(TAG, "optimizedHeadDeflection : roll:%f, yaw:%f, pitch:%f)",
                  face->optimizedHeadDeflection.roll, face->optimizedHeadDeflection.yaw, face->optimizedHeadDeflection.pitch);
        }
        // 打印 FaceFeature 计算结果
        if (service->get_switch(ABILITY_FACE_FEATURE)) {
            std::stringstream feature;
            feature << "FaceFeature : (";
            for (int j = 0; j < 10; ++j) {
                feature << face->feature[j] << " , ";
            }
            feature << " ... ";
            feature << face->feature[1021];
            feature << face->feature[1022];
            feature << face->feature[1023];
            feature << " ) ";
            //VLOGI(TAG, feature.str().c_str());
        }

        // 打印 FaceFeature 计算结果
        if (service->get_switch(ABILITY_FACE_QUALITY)) {
            VLOGI(TAG, "Quality State %f: leftEyeCover:%f, rightEyeCover:%f, mouthCover:%f)", face->stateFaceCoverSingle,
                  face->leftEyeCoverSingle, face->rightEyeCoverSingle, face->stateMouthCoverSingle);
        }

        // 打印实现相关的输出日志
        // if (service->get_switch(ABILITY_FACE_EYE_CENTER)) {
        //     VLOGI(TAG, "Face Left EyeCenter[%f, %f],Eyelid distance:%f, Canthus distance:%f",
        //           face->_eye_centroid_left.x, face->_eye_centroid_left.y,
        //           face->_eye_eyelid_distance_left, face->_eye_canthus_distance_left);
        //     VLOGI(TAG, "Face Right EyeCenter[%f, %f],Eyelid distance:%f, Canthus distance:%f",
        //           face->_eye_centroid_right.x, face->_eye_centroid_right.y,
        //           face->_eye_eyelid_distance_right, face->_eye_canthus_distance_right);
        // }
        // 打印3D人脸Landmark和视线相关的输出日志
        if (service->get_switch(ABILITY_FACE_3D_LANDMARK)) {
            VLOGI(TAG, "3DFaceLandmark : | confidence: %f | 1:(%f , %f, %f), 106:(%f , %f, %f)",
                  face->landmarkConfidence, face->landmark3D68[0].x, face->landmark3D68[0].y, face->landmark3D68[0].z,
                  face->landmark3D68[67].x, face->landmark3D68[67].y, face->landmark3D68[67].z);
            // 打印3D关键点输出的头部姿态偏转角
            VLOGI(TAG, "HeadDeflection3D : roll:%f, yaw:%f, pitch:%f)",
                  face->headDeflection3D.roll, face->headDeflection3D.yaw, face->headDeflection3D.pitch);
            // 打印头部位置,就是鼻尖的 3D 坐标
            VLOGI(TAG, "3DHeadLocation : | NoseTip74:(%f , %f, %f)", face->headLocation.x, face->headLocation.y,
                  face->headLocation.z);
        }

        if (service->get_switch(ABILITY_FACE_3D_EYE_GAZE)) {
            VLOGI(TAG, "3DEyeLandmarkLeft : | 1:(%f , %f, %f), 28:(%f , %f, %f)",
                  face->eye3dLandmark28Left[0].x, face->eye3dLandmark28Left[0].y, face->eye3dLandmark28Left[0].z,
                  face->eye3dLandmark28Left[27].x, face->eye3dLandmark28Left[27].y, face->eye3dLandmark28Left[27].z);
            VLOGI(TAG, "3DEyeLandmarkRight : | 1:(%f , %f, %f), 28:(%f , %f, %f)",
                  face->eye3dLandmark28Right[0].x, face->eye3dLandmark28Right[0].y, face->eye3dLandmark28Right[0].z,
                  face->eye3dLandmark28Right[27].x, face->eye3dLandmark28Right[27].y, face->eye3dLandmark28Right[27].z);
            // 打印3D landmark 模型 输出的实现的起点和方向向量
            VLOGI(TAG, "3DEyeGazeOrigin : | left:(%f , %f, %f), right:(%f , %f, %f)",
                  face->eyeGazeOriginLeft.x, face->eyeGazeOriginLeft.y, face->eyeGazeOriginLeft.z,
                  face->eyeGazeOriginRight.x, face->eyeGazeOriginRight.y, face->eyeGazeOriginRight.z);
            VLOGI(TAG, "3DEyeGazeVector : | left:(%f , %f, %f), right:(%f , %f, %f)",
                  face->eyeGaze3dVectorLeft.x, face->eyeGaze3dVectorLeft.y, face->eyeGaze3dVectorLeft.z,
                  face->eyeGaze3dVectorRight.x, face->eyeGaze3dVectorRight.y, face->eyeGaze3dVectorRight.z);
        }
    }

}

void InfoPrinter::printGestureInfo(VisionService *service, VisionResult *result) {
    // 打印 GestureRect 计算结果
    GestureInfo *gesture = result->getGestureResult()->gestureInfos[0];

    if (service->get_switch(ABILITY_GESTURE_RECT)) {
        VLOGI(TAG, "GestureRect : id: %ld | confidence: %f | LT(%f , %f) BT(%f , %f)",
              gesture->id, gesture->rectConfidence, gesture->rectLT.x, gesture->rectLT.y,
              gesture->rectRB.x, gesture->rectRB.y);
    }

    if (service->get_switch(ABILITY_GESTURE_TYPE)) {
        VLOGI(TAG, "Gesture staticTypeSingle: %d | typeConfidence: %f | StaticType %d", gesture->staticTypeSingle,
              gesture->typeConfidence, gesture->staticType);
    }
}

void InfoPrinter::printBodyInfo(VisionService *service, VisionResult *result) {
    BodyResult const *bodyResult = result->getBodyResult();
    for (int j = 0; j < bodyResult->bodyMaxCount; ++j) {
        if (service->get_switch(ABILITY_BODY_HEAD_SHOULDER)) {
            BodyInfo *bodyInfo = bodyResult->pBodyInfos[j];
            if (bodyInfo == nullptr || !bodyInfo->hasBody()) {
                VLOGD(TAG, "Body[%d]无头肩", j);
                continue;
            }
            VLOGI(TAG, "BodyRect : ID: %ld | confidence: %f | RectLT(%f,%f) RectBT(%f,%f)",
                  bodyInfo->id, bodyInfo->rectConfidence,
                  bodyInfo->headShoulderRectLT.x, bodyInfo->headShoulderRectLT.y,
                  bodyInfo->headShoulderRectRB.x, bodyInfo->headShoulderRectRB.y);
        }
    }

}

void InfoPrinter::printLivingInfo(VisionService *service, VisionResult *result) {
    LivingResult const *livingResult = result->getLivingResult();
    for (int j = 0; j < livingResult->livingCount; ++j) {
        if (service->get_switch(ABILITY_LIVING_DETECTION)) {
            LivingInfo *living = livingResult->livingInfos[j];
            VLOGI(TAG, "LivingInfo  typeSingle: %d", living->livingTypeSingle);
        }
    }
}

}