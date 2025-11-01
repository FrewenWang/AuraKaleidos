
#include "vision/config/runtime_config/RtConfig.h"
#include "vision/core/bean/FaceInfo.h"
#include <sstream>

using namespace std;

namespace aura::vision {

FaceInfo::FaceInfo() noexcept { clearAll();
}

FaceInfo::FaceInfo(const FaceInfo& info) noexcept {
    copy(info);
}

FaceInfo::FaceInfo(FaceInfo&& info) noexcept {
    copy(info);
}

FaceInfo& FaceInfo::operator=(FaceInfo&& info) noexcept {
    copy(info);
    return *this;
}

FaceInfo& FaceInfo::operator=(const FaceInfo& info) noexcept {
    if (&info != this) {
        copy(info);
    }
    return *this;
}

void FaceInfo::copy(const FaceInfo& info) {
    id = info.id;

    rectConfidence = info.rectConfidence;
    landmarkConfidence = info.landmarkConfidence;

    rectCenter.copy(info.rectCenter);

    rectLT.copy(info.rectLT);
    rectRB.copy(info.rectRB);
    faceRect.copy(info.faceRect);

    for (int i = 0; i < 68; ++i) {
        landmark2D68[i].copy(info.landmark2D68[i]);
    }

    for (int i = 0; i < 68; ++i) {
        landmark3D68[i].copy(info.landmark3D68[i]);
    }

    for (int i = 0; i < 106; ++i) {
        landmark2D106[i].copy(info.landmark2D106[i]);
    }

    for (int i = 0; i < 106; ++i) {
        landmark3D106[i].copy(info.landmark3D106[i]);
    }

    for (int i = 0; i < 8; ++i) {
        eyeLmk8Left[i].copy(info.eyeLmk8Left[i]);
    }

    for (int i = 0; i < 8; ++i) {
        eyeLmk8Right[i].copy(info.eyeLmk8Right[i]);
    }
    // 拷贝mouth的关键点
    for (int i = 0; i < LM_MOUTH_2D_20_COUNT; ++i) {
        mouthLmk20[i].copy(info.mouthLmk20[i]);
    }

    // 拷贝左右眼的关键点
    for (int i = 0; i < LM_EYE_3D_28_COUNT; ++i) {
        eye3dLandmark28Left[i].copy(info.eye3dLandmark28Left[i]);
    }
    for (int i = 0; i < LM_EYE_3D_28_COUNT; ++i) {
        eye3dLandmark28Right[i].copy(info.eye3dLandmark28Right[i]);
    }
    // 拷贝左右眼睛的视线相关的逻辑
    eyeGaze3dVectorLeft.copy(info.eyeGaze3dVectorLeft);
    eyeGaze3dVectorRight.copy(info.eyeGaze3dVectorRight);
    headDeflection.copy(info.headDeflection);
    eyeGaze3dCalibVectorLeft.copy(info.eyeGaze3dCalibVectorLeft);
    eyeGaze3dCalibVectorRight.copy(info.eyeGaze3dCalibVectorRight);
    eyeGaze3dTransVectorLeft.copy(info.eyeGaze3dTransVectorLeft);
    eyeGaze3dTransVectorRight.copy(info.eyeGaze3dTransVectorRight);

    optimizedHeadDeflection.copy(info.optimizedHeadDeflection);
    headDeflection3D.copy(info.headDeflection3D);
    headTranslation.copy(info.headTranslation);
    headLocation.copy(info.headLocation);

    stateDangerDrive = info.stateDangerDrive;
    stateFatigue = info.stateFatigue;
    stateFatigue = info.stateFatigue;
    stateAttention = info.stateAttention;

    for (int i = 0; i < 4; ++i) {
        eyeTracking[i].copy(info.eyeTracking[i]);
    }

    memcpy(feature, info.feature, FEATURE_COUNT * sizeof(float));

    stateInteractLiving = info.stateInteractLiving;
    stateHeadBehavior = info.stateHeadBehavior;
    stateEmotion = info.stateEmotion;
    stateEmotionSingle = info.stateEmotionSingle;

    stateBlurSingle = info.stateBlurSingle;
    stateFaceCoverSingle = info.stateFaceCoverSingle;
    leftEyeCoverSingle = info.leftEyeCoverSingle;
    rightEyeCoverSingle = info.rightEyeCoverSingle;
    stateMouthCoverSingle = info.stateMouthCoverSingle;
    stateNoiseSingle = info.stateNoiseSingle;
    stateBlur = info.stateBlur;
    stateNoise = info.stateNoise;
    stateLeftEyeCover = info.stateLeftEyeCover;
    stateRightEyeCover = info.stateRightEyeCover;
    stateMouthCover = info.stateMouthCover;
    stateFaceCover = info.stateFaceCover;
    leftEyeDetectSingle = info.leftEyeDetectSingle;
    rightEyeDetectSingle = info.rightEyeDetectSingle;

    // ================================= DMS 打电话相关字段拷贝 ====================================
    stateCallLeftSingle = info.stateCallLeftSingle;
    stateCallRightSingle = info.stateCallRightSingle;
    scoreCallLeftSingle = info.scoreCallLeftSingle;
    scoreCallRightSingle = info.scoreCallRightSingle;
    stateCallSingle = info.stateCallSingle;
    stateCall = info.stateCall;
    phoneCallVState = info.phoneCallVState;

    // ================================= FaceInfo相关检测的分值结果拷贝 ====================================
    scoreNoInteractLiving = info.scoreNoInteractLiving;
    scoreEmotionSingle = info.scoreEmotionSingle;
    scoreCameraCoverSingle = info.scoreCameraCoverSingle;
    dangerDriveConfidence = info.dangerDriveConfidence;
}

void FaceInfo::clearAll() {
    rectCenter.clear();
    rectLT.clear();
    rectRB.clear();
    faceRect.clear();
    landmarkConfidence = 0;
    eyeCloseConfidence = 0;

    memset(eyeTracking, 0, 4 * sizeof(VPoint));
    memset(landmark2D68, 0, 68 * sizeof(VPoint));
    memset(landmark3D68, 0, 68 * sizeof(VPoint3));
    memset(landmark2D106, 0, 106 * sizeof(VPoint));
    memset(landmark3D106, 0, 106 * sizeof(VPoint3));
    memset(eyeLmk8Left, 0, 8 * sizeof(VPoint));
    memset(eyeLmk8Right, 0, 8 * sizeof(VPoint));
    memset(mouthLmk20, 0, 20 * sizeof(VPoint));
    memset(eye3dLandmark28Left, 0, 28 * sizeof(VPoint3));
    memset(eye3dLandmark28Left, 0, 28 * sizeof(VPoint3));
    clear();
}

void FaceInfo::clear() {
    // 每帧检测前都将result中人脸置为无效人脸，避免无人脸时误报上一帧result的有效人脸
    id = 0;
    faceType = FaceDetectType::F_TYPE_UNKNOWN;
    stateDangerDrive = F_DANGEROUS_NONE;
    stateFatigue = 0;
    stateAttention = 0;
    stateHeadBehavior = F_HEAD_BEHAVIOR_GOON;
    stateCall = 0;
    // 无感活体字段重置为UNKNOWN
    stateNoInteractLivingSingle = F_NO_INTERACT_LIVING_UNKNOWN;
    stateNoInteractLiving = F_NO_INTERACT_LIVING_UNKNOWN;
    stateInteractLiving = 0;
    stateEmotion = F_ATTR_UNKNOWN;
    stateEmotionSingle = F_ATTR_UNKNOWN;
    stateGlass = F_ATTR_UNKNOWN;
    stateGender = F_ATTR_UNKNOWN;
    stateRace = F_ATTR_UNKNOWN;
    stateAge = F_ATTR_UNKNOWN;
    stateGlassSingle = F_ATTR_UNKNOWN;
    stateGenderSingle = F_ATTR_UNKNOWN;
    stateRaceSingle = F_ATTR_UNKNOWN;
    stateAgeSingle = F_ATTR_UNKNOWN;
    eyeWaking = 0;
    rectConfidence = 0;
    // 无感活体分值清空
    scoreNoInteractLiving = 0.f;
    scoreEmotionSingle = 0.f;
    scoreCameraCoverSingle = 0.f;
    scoreDetectEyeLeftSingle = 0.0f;         /// 左眼眼睛检测分数
    scoreDetectEyeRightSingle = 0.0f;        /// 右眼眼睛检测分数
    scoreDetectPupilLeftSingle = 0.0f;        /// 左眼瞳孔检测分数
    scoreDetectPupilRightSingle = 0.0f;       /// 右眼瞳孔检测分数

    // ================================= DMS 打电话相关字段清除 ====================================
    //  单帧检测左右耳朵结果数据不能清除, 交替检测左右耳朵的时候依赖上一阵数据
    //  stateCallLeftSingle = F_CALL_NONE;
    //  stateCallRightSingle = F_CALL_NONE;
    scoreCallLeftSingle = 0.f;
    scoreCallRightSingle = 0.f;
    stateCallSingle = F_CALL_NONE;;
    stateCall = F_CALL_NONE;
    phoneCallVState.clear();

    stateBlurSingle = F_QUALITY_UNKNOWN;
    stateFaceCoverSingle = F_QUALITY_UNKNOWN;
    leftEyeCoverSingle = F_QUALITY_UNKNOWN;
    rightEyeCoverSingle = F_QUALITY_UNKNOWN;
    stateMouthCoverSingle = F_QUALITY_UNKNOWN;
    stateNoiseSingle = F_QUALITY_UNKNOWN;
    stateBlur = F_QUALITY_UNKNOWN;
    stateNoise = F_QUALITY_UNKNOWN;
    stateLeftEyeCover = F_QUALITY_UNKNOWN;
    stateRightEyeCover = F_QUALITY_UNKNOWN;
    stateMouthCover = F_QUALITY_UNKNOWN;
    stateFaceCover = F_QUALITY_UNKNOWN;
    stateFaceTracking = F_TRACKING_UNKNOW;

    stateDangerDriveSingle = F_DANGEROUS_NONE;
    dangerDriveConfidence = 0.f;
    stateSmokeBurningSingle = SmokingBurningStatus::F_SMOKE_BURNING_UNKNOWN;
    stateSmokeBurning = SmokingBurningStatus::F_SMOKE_BURNING_UNKNOWN;

    smokeVState.clear();
    drinkVState.clear();
    silenceVState.clear();
    openMouthVState.clear();
    closeEyeVState.clear();
    yawnVState.clear();
    _left_attention_state.clear();
    _right_attention_state.clear();
    maskMouthVState.clear();
    coverMouthVState.clear();

    eyeCentroidLeft.clear(); // 左眼瞳孔质点
    eyeCentroidRight.clear(); // 右眼瞳孔质点

    phoneCallVState.clear(); // 清除打电话策略状态
    gestureNearbyEar.clear();   // 清除策略状态，打电话模型检测手

    _up_attention_state.clear();  // 向上注意力偏移状态
    _down_attention_state.clear(); // 向下注意力偏移状态

    _small_eye_state.clear();

    _no_face_state.clear();

    // 图像遮挡相关变量。每一帧检测之后不能清除camera遮挡状态。
    // 人脸遮挡的情况下，无人脸的时候FaceInfo会清空数据。导致图像遮挡状态回到默认值
    // _state_image_cover_single = ImageCoverStatus::F_IMAGE_COVER_UNKNOWN;
    // _state_image_cover = ImageCoverStatus::F_IMAGE_COVER_UNKNOWN;

    stateFaceNoMoving = 0;
    stateLipMovementSingle = FaceLipMovementStatus::F_LIP_MOVEMENT_NONE;
    stateLipMovement = FaceLipMovementStatus::F_LIP_MOVEMENT_NONE;

    headLocation.clear();
    optimizedHeadDeflection.clear();

    headDeflection.clear();
    headDeflection3D.clear();
    headTranslation.clear();

    eyeGazeCalibValid = false;
    eyeCloseConfidence = 0;
    eyeBlinkState = 0;
    eyeState = FaceEyeStatus::F_EYE_UNKNOWN;
    eyeEyelidStatusLeft = FaceEyeStatus::F_EYE_UNKNOWN;
    eyeEyelidStatusRight = FaceEyeStatus::F_EYE_UNKNOWN;
    eyeEyelidDistanceLeft = 0.f;
    eyeEyelidDistanceRight = 0.f;
    eyeEyelidOpeningDistanceMeanLeft = 0.f;
    eyeEyelidOpeningDistanceMeanRight = 0.f;
    eyeEyelidOpenRatioLeft = 0.f;
    eyeEyelidOpenRatioRight = 0.f;
    leftEyeDetectSingle = FaceEyeDetectStatus::UNKNOWN;
    rightEyeDetectSingle = FaceEyeDetectStatus::UNKNOWN;
    eyeCanthusDistanceLeft = 0.f;
    eyeCanthusDistanceRight = 0.f;

    eyeBlinkDuration = 0;
    eyeCloseFrequency = 0;
    eyeStartClosingTime = 0;
    eyeEndClosingTime = 0;

    eyeGazeOriginLeft.clear();
    eyeGazeDestinationLeft.clear();
    eyeGazeOriginRight.clear();
    eyeGazeDestinationRight.clear();
    eyeTracking->clear();

    eyeLmk8Left->clear();
    eyeLmk8Right->clear();
    mouthLmk20->clear();
    eye3dLandmark28Left->clear();
    eye3dLandmark28Right->clear();
    eye2dLandmark28Left->clear();
    eye2dLandmark28Right->clear();
    eyeGaze3dVectorLeft.clear();
    eyeGaze3dVectorRight.clear();
    eyeGaze3dCalibVectorLeft.clear();
    eyeGaze3dCalibVectorRight.clear();
    eyeGaze3dTransVectorLeft.clear();
    eyeGaze3dTransVectorRight.clear();

    memset(feature, 0, FEATURE_COUNT * sizeof(float));
    gestureNearbyLeftEar = 0;
    gestureNearbyRightEar = 0;


    // 如下是生成的FaceInfo的中间临时结果的清除
    tTensorWarped.release();
    tTensorCropped.release();

}

bool FaceInfo::hasFace() const {
    // 在人脸跟踪时会保留失效的人脸Id，但检测类型faceType会重置为UNKNOWN，因此改而使用faceType来判断有无人脸
    return faceType != FaceDetectType::F_TYPE_UNKNOWN;
}

bool FaceInfo::noFace() const {
    // 在人脸跟踪时会保留失效的人脸Id，但检测类型faceType会重置为UNKNOWN，因此改而使用faceType来判断有无人脸
    return faceType == FaceDetectType::F_TYPE_UNKNOWN;
}

bool FaceInfo::isDetectType() const {
     if (faceType == FaceDetectType::F_TYPE_DETECT) {
         return true;
     }
    return false;
}

bool FaceInfo::isNotDetectType() const {
    return !isDetectType();
}

bool FaceInfo::isFaceInRect(VRect &rect, int threshold) {
    float width = (rect.right - rect.left) / 2;
    rect.set(rect.left - width, 0, rect.right + width, 720);
    int outCount = 0;
    int len = LANDMARK_2D_106_LENGTH / 2;

    for (int i = 0; i < len; ++i) {
        if (!rect.contains(landmark2D106->x, landmark2D106->y)) {
            ++outCount;
        }
    }

    return outCount < threshold;
}

bool FaceInfo::isFaceInRect(VRect &rect, float *angels, int threshold) {
    return false;
}

void FaceInfo::toString(stringstream &ss) const {
    ss << "[FaceInfo] ------------------------------" << endl;
    ss << "id : " << id << endl;

    ss << "stateFatigue : " << stateFatigue << endl;
    ss << "stateAttention : " << stateAttention << endl;
    ss << "stateHeadBehavior : " << stateHeadBehavior << endl;
    ss << "stateCallSingle : " << stateCallSingle << endl;
    ss << "stateCall : " << stateCall << endl;
    ss << "stateNoInteractLivingSingle : " << stateNoInteractLivingSingle << endl;
    ss << "stateNoInteractLiving : " << stateNoInteractLiving << endl;
    ss << "stateInteractLiving : " << stateInteractLiving << endl;
    ss << "stateEmotionSingle : " << stateEmotionSingle << endl;
    ss << "stateEmotion : " << stateEmotion << endl;
    ss << "stateGlassSingle : " << stateGlassSingle << endl;
    ss << "stateCall : " << stateGlass << endl;
    ss << "stateGenderSingle : " << stateGenderSingle << endl;
    ss << "stateGender : " << stateGender << endl;
    ss << "stateRaceSingle : " << stateRaceSingle << endl;
    ss << "stateRace : " << stateRace << endl;
    ss << "stateAgeSingle : " << stateAgeSingle << endl;
    ss << "stateAge : " << stateAge << endl;

    ss << "leftEarCall : " << stateCallLeftSingle << endl;
    ss << "rightEarCall : " << stateCallRightSingle << endl;
    ss << "gestureNearbyLeftEar : " << gestureNearbyLeftEar << endl;
    ss << "gestureNearbyRightEar : " << gestureNearbyRightEar << endl;
    ss << "stateDangerDrive : " << stateDangerDrive << endl;
    ss << "stateDangerDriveSingle : " << stateDangerDriveSingle << endl;
    ss << "stateSmokeBurning : " << stateSmokeBurning << endl;
    ss << "stateSmokeBurningSingle : " << stateSmokeBurningSingle << endl;
    ss << "eyeCloseConfidence : " << eyeCloseConfidence << endl;
    ss << "scoreNoInteractLiving : " << scoreNoInteractLiving << endl;

    ss << "stateFaceNoMoving : " << stateFaceNoMoving << endl;
    ss << "stateFaceTracking : " << stateFaceTracking << endl;
    ss << "stateFaceDetect : " << stateFaceDetect << endl;

    // ============================ 眼部检测相关  =============================================
    ss << "eyeGazeOriginLeft : "; eyeGazeOriginLeft.toString(ss); ss << endl;
    ss << "eyeGazeDestinationLeft : "; eyeGazeDestinationLeft.toString(ss); ss << endl;
    ss << "eyeGazeOriginRight : "; eyeGazeOriginRight.toString(ss); ss << endl;
    ss << "eyeGazeDestinationRight : "; eyeGazeDestinationRight.toString(ss); ss << endl;
    ss << "eyeGaze3dVectorLeft : "; eyeGaze3dVectorLeft.toString(ss); ss << endl;
    ss << "eyeGaze3dVectorRight : "; eyeGaze3dVectorRight.toString(ss); ss << endl;

    ss << "eyeEyelidDistanceLeft : " << eyeEyelidDistanceLeft << endl;
    ss << "eyeEyelidDistanceRight : " << eyeEyelidDistanceRight << endl;
    ss << "eyeEyelidOpeningDistanceMeanLeft : " << eyeEyelidOpeningDistanceMeanLeft << endl;
    ss << "eyeEyelidOpeningDistanceMeanRight : " << eyeEyelidOpeningDistanceMeanRight << endl;
    ss << "eyeEyelidOpenRatioLeft : " << eyeEyelidOpenRatioLeft << endl;
    ss << "eyeEyelidOpenRatioRight : " << eyeEyelidOpenRatioRight << endl;
    ss << "eyeEyelidStatusLeft : " << eyeEyelidStatusLeft << endl;
    ss << "eyeEyelidStatusRight : " << eyeEyelidStatusRight << endl;
    ss << "eyeState : " << eyeState << endl;
    ss << "eyeCanthusDistanceLeft : " << eyeCanthusDistanceLeft << endl;
    ss << "eyeCanthusDistanceRight : " << eyeCanthusDistanceRight << endl;
    ss << "leftEyeDetectSingle : " << leftEyeDetectSingle << endl;
    ss << "rightEyeDetectSingle : " << rightEyeDetectSingle << endl;
    ss << "eyeCloseFrequency : " << eyeCloseFrequency << endl;
    ss << "eyeStartClosingTime : " << eyeStartClosingTime << endl;
    ss << "eyeEndClosingTime : " << eyeEndClosingTime << endl;
    ss << "eyeBlinkState : " << eyeBlinkState << endl;
    ss << "eyeBlinkDuration : " << eyeBlinkDuration << endl;
    ss << "eyeTracking[0] : "; eyeTracking[0].toString(ss); ss << endl;
    ss << "eyeWaking : " << eyeWaking << endl;
    ss << "eyeCentroidLeft : "; eyeCentroidLeft.toString(ss); ss << endl;
    ss << "eyeCentroidRight : "; eyeCentroidRight.toString(ss); ss << endl;
    ss << "eyeLmk8Left[0] : "; eyeLmk8Left[0].toString(ss); ss << endl;
    ss << "eyeLmk8Right[0] : "; eyeLmk8Right[0].toString(ss); ss << endl;
    ss << "mouthLmk20[0] : "; mouthLmk20[0].toString(ss); ss << endl;
    ss << "eye3dLandmark28Left[0] : "; eye3dLandmark28Left[0].toString(ss); ss << endl;
    ss << "eye3dLandmark28Right[0] : "; eye3dLandmark28Right[0].toString(ss); ss << endl;

    // ============================ 嘴部检测相关  =============================================
    ss << "stateLipMovementSingle : " << stateLipMovementSingle << endl;
    ss << "stateLipMovement : " << stateLipMovement << endl;

    // ============================ 头部检测相关  =============================================
    ss << "headLocation : "; headLocation.toString(ss); ss << endl;
    ss << "optimizedHeadDeflection : ";
    optimizedHeadDeflection.toString(ss); ss << endl;
    ss << "headDeflection3D : "; headDeflection3D.toString(ss); ss << endl;
    ss << "headTranslation : "; headTranslation.toString(ss); ss << endl;

    ss << "feature[0] : " << feature[0] << endl;
    ss << "rectConfidence : " << rectConfidence << endl;
    ss << "rectCenter : "; rectCenter.toString(ss); ss << endl;
    ss << "rectLT : "; rectLT.toString(ss); ss << endl;
    ss << "rectRB : "; rectRB.toString(ss); ss << endl;
    ss << "faceRect : "; faceRect.toString(ss); ss << endl;
    ss << "landmarkConfidence : " << landmarkConfidence << endl;
    ss << "landmark2D106[0] : "; landmark2D106[0].toString(ss); ss << endl;
    ss << "landmark2D106[10] : "; landmark2D106[10].toString(ss); ss << endl;
    ss << "landmark2D106[105] : "; landmark2D106[105].toString(ss); ss << endl;
    ss << "landmark3D106[0] : "; landmark3D106[0].toString(ss); ss << endl;

    // =============================== 滑窗触发状态VState ===========================================
    ss << "smokeVState : "; smokeVState.toString(ss); ss << endl;
    ss << "drinkVState : "; drinkVState.toString(ss); ss << endl;
    ss << "silenceVState : "; silenceVState.toString(ss); ss << endl;
    ss << "openMouthVState : "; openMouthVState.toString(ss); ss << endl;
    ss << "coverMouthVState : "; coverMouthVState.toString(ss); ss << endl;
    ss << "maskMouthVState : "; maskMouthVState.toString(ss); ss << endl;
    ss << "closeEyeVState : "; closeEyeVState.toString(ss); ss << endl;
    ss << "yawnVState : "; yawnVState.toString(ss); ss << endl;
    ss << "phoneCallVState : "; phoneCallVState.toString(ss); ss << endl;
    ss << "_left_attention_state : "; _left_attention_state.toString(ss); ss << endl;
    ss << "_right_attention_state : "; _right_attention_state.toString(ss); ss << endl;
    ss << "_up_attention_state : "; _up_attention_state.toString(ss); ss << endl;
    ss << "_down_attention_state : "; _down_attention_state.toString(ss); ss << endl;
    ss << "_small_eye_state : "; _small_eye_state.toString(ss); ss << endl;
    ss << "_no_face_state : "; _no_face_state.toString(ss); ss << endl;
    ss << "gestureNearbyEar : "; gestureNearbyEar.toString(ss); ss << endl;

    ss << "stateBlurSingle : " << stateBlurSingle << endl;
    ss << "stateFaceCoverSingle : " << stateFaceCoverSingle << endl;
    ss << "leftEyeCoverSingle : " << leftEyeCoverSingle << endl;
    ss << "leftEyeCoverSingle : " << rightEyeCoverSingle << endl;
    ss << "stateMouthCoverSingle : " << stateMouthCoverSingle << endl;
    ss << "stateNoiseSingle : " << stateNoiseSingle << endl;
    ss << "stateBlur : " << stateBlur << endl;
    ss << "stateNoise : " << stateNoise << endl;
    ss << "stateLeftEyeCover : " << stateLeftEyeCover << endl;
    ss << "stateRightEyeCover : " << stateRightEyeCover << endl;
    ss << "stateMouthCover : " << stateMouthCover << endl;
    ss << "stateFaceCover : " << stateFaceCover << endl;
    ss << "stateCameraCover : " << stateCameraCover << endl;
    ss << "stateCameraCoverSingle : " << stateCameraCoverSingle << endl;
}

} // namespace aura::vision
