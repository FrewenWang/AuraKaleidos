
#include "com_baidu_iov_vision_ability_core_bean_FaceInfo.h"

#include "vision/core/bean/FaceInfo.h"

#define TAG "FaceInfoNative"

#ifdef __cplusplus
extern "C" {
#endif

using namespace vision;
using namespace std;

static jclass cInfo = 0;
static jfieldID fNativeBuffer = 0;

static FaceInfo *getBuffer(JNIEnv *env, jobject thiz) {
    jobject buffer = env->GetObjectField(thiz, fNativeBuffer);
    return reinterpret_cast<FaceInfo*>(env->GetDirectBufferAddress(buffer));
}

JNIEXPORT void JNICALL Java_com_baidu_iov_vision_ability_core_bean_FaceInfo_init
        (JNIEnv *env, jobject thiz, jobject buffer) {
    cInfo = env->GetObjectClass(thiz);
    fNativeBuffer = env->GetFieldID(cInfo, "mNativeBuffer", "Ljava/nio/FloatBuffer;");
}

/*
 * Class:     com_baidu_iov_vision_ability_core_bean_FaceInfo
 * Method:    getBufferSize
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_com_baidu_iov_vision_ability_core_bean_FaceInfo_getBufferSize
        (JNIEnv *env, jobject thiz) {
    return sizeof(FaceInfo);
}

/*
 * Class:     com_baidu_iov_vision_ability_core_bean_FaceInfo
 * Method:    id
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_com_baidu_iov_vision_ability_core_bean_FaceInfo_id__
        (JNIEnv *env, jobject thiz) {
    return (jint) getBuffer(env, thiz)->id;
}

/*
 * Class:     com_baidu_iov_vision_ability_core_bean_FaceInfo
 * Method:    rectConfidence
 * Signature: ()F
 */
JNIEXPORT jfloat JNICALL Java_com_baidu_iov_vision_ability_core_bean_FaceInfo_rectConfidence__
        (JNIEnv *env, jobject thiz) {
    return getBuffer(env, thiz)->rectConfidence;
}

/*
 * Class:     com_baidu_iov_vision_ability_core_bean_FaceInfo
 * Method:    singleCallState
 * Signature: ()I
 */
JNIEXPORT jboolean JNICALL Java_com_baidu_iov_vision_ability_core_bean_FaceInfo_singleCallState
        (JNIEnv *env, jobject thiz) {
    return (jboolean)(getBuffer(env, thiz)->leftEarCall || getBuffer(env, thiz)->rightEarCall);
}

/*
 * Class:     com_baidu_iov_vision_ability_core_bean_FaceInfo
 * Method:    singleDangerDriveState
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_com_baidu_iov_vision_ability_core_bean_FaceInfo_singleDangerDriveState
        (JNIEnv *env, jobject thiz) {
    return (jint) (getBuffer(env, thiz)->stateDangerDriveSingle);
}

/*
 * Class:     com_baidu_iov_vision_ability_core_bean_FaceInfo
 * Method:    rect
 * Signature: ()[F
 */
JNIEXPORT jfloatArray JNICALL Java_com_baidu_iov_vision_ability_core_bean_FaceInfo_rect
        (JNIEnv *env, jobject thiz) {
    jsize ltlen = 2; // sizeof(getBuffer(env, thiz)->_rect_lt);
    jsize rblen = 2; // sizeof(getBuffer(env, thiz)->_rect_rb);
    jfloatArray result = env->NewFloatArray(ltlen + rblen);
    if (result != nullptr) {
        env->SetFloatArrayRegion(result, 0, ltlen, (jfloat *) &getBuffer(env, thiz)->rectLT);
        env->SetFloatArrayRegion(result, ltlen, rblen, (jfloat *) &getBuffer(env, thiz)->rectRB);
    }
    return result;
}

/*
 * Class:     com_baidu_iov_vision_ability_core_bean_FaceInfo
 * Method:    landmarkConfidence
 * Signature: ()F
 */
JNIEXPORT jfloat JNICALL Java_com_baidu_iov_vision_ability_core_bean_FaceInfo_landmarkConfidence
        (JNIEnv *env, jobject thiz) {
    return getBuffer(env, thiz)->landmarkConfidence;
}

/*
 * Class:     com_baidu_iov_vision_ability_core_bean_FaceInfo
 * Method:    landmark2D106
 * Signature: ()[F
 */
JNIEXPORT jfloatArray JNICALL Java_com_baidu_iov_vision_ability_core_bean_FaceInfo_landmark2D106
        (JNIEnv *env, jobject thiz) {
    jsize len = 212; // sizeof(getBuffer(env, thiz)->_landmark_2d_106);
    jfloatArray result = env->NewFloatArray(len);
    if (result != nullptr) {
        env->SetFloatArrayRegion(result, 0, len, (jfloat *) &getBuffer(env, thiz)->landmark2D106);
    }
    return result;
}

/*
 * Class:     com_baidu_iov_vision_ability_core_bean_FaceInfo
 * Method:    landmark2D68
 * Signature: ()[F
 */
JNIEXPORT jfloatArray JNICALL Java_com_baidu_iov_vision_ability_core_bean_FaceInfo_landmark2D68
        (JNIEnv *env, jobject thiz) {
    jsize len = 136; // sizeof(getBuffer(env, thiz)->_landmark_2d_68);
    jfloatArray result = env->NewFloatArray(len);
    if (result != nullptr) {
        env->SetFloatArrayRegion(result, 0, len, (jfloat *) &getBuffer(env, thiz)->landmark2D68);
    }
    return result;
}

/*
 * Class:     com_baidu_iov_vision_ability_core_bean_FaceInfo
 * Method:    landmark3D68
 * Signature: ()[F
 */
JNIEXPORT jfloatArray JNICALL Java_com_baidu_iov_vision_ability_core_bean_FaceInfo_landmark3D68
        (JNIEnv *env, jobject thiz) {
    jsize len = 204; // sizeof(getBuffer(env, thiz)->_landmark_3d_68);
    jfloatArray result = env->NewFloatArray(len);
    if (result != nullptr) {
        env->SetFloatArrayRegion(result, 0, len, (jfloat *) &getBuffer(env, thiz)->landmark3D68);
    }
    return result;
}

/*
 * Class:     com_baidu_iov_vision_ability_core_bean_FaceInfo
 * Method:    feature
 * Signature: ()[F
 */
JNIEXPORT jfloatArray JNICALL Java_com_baidu_iov_vision_ability_core_bean_FaceInfo_feature
        (JNIEnv *env, jobject thiz) {
    jsize len = 1024; // sizeof(getBuffer(env, thiz)->_feature);
    jfloatArray result = env->NewFloatArray(len);
    if (result != nullptr) {
        env->SetFloatArrayRegion(result, 0, len, (jfloat *) &getBuffer(env, thiz)->feature);
    }
    return result;
}

/*
 * Class:     com_baidu_iov_vision_ability_core_bean_FaceInfo
 * Method:    noInteractLivingState
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_com_baidu_iov_vision_ability_core_bean_FaceInfo_noInteractLivingState
        (JNIEnv *env, jobject thiz) {
    return (jint) getBuffer(env, thiz)->stateNoInteractLiving;
}

/*
 * Class:     com_baidu_iov_vision_ability_core_bean_FaceInfo
 * Method:    noInteractLivingSingleState
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_com_baidu_iov_vision_ability_core_bean_FaceInfo_noInteractLivingSingleState
        (JNIEnv *env, jobject thiz) {
    return (jint) getBuffer(env, thiz)->stateNoInteractLivingSingle;
}

/*
 * Class:     com_baidu_iov_vision_ability_core_bean_FaceInfo
 * Method:    interactLivingState
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_com_baidu_iov_vision_ability_core_bean_FaceInfo_interactLivingState
        (JNIEnv *env, jobject thiz) {
    return (jint) getBuffer(env, thiz)->stateInteractLiving;
}

/*
 * Class:     com_baidu_iov_vision_ability_core_bean_FaceInfo
 * Method:    dangerousState
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_com_baidu_iov_vision_ability_core_bean_FaceInfo_dangerousState
        (JNIEnv *env, jobject thiz) {
    return (jint) getBuffer(env, thiz)->stateDangerDrive;
}

/*
 * Class:     com_baidu_iov_vision_ability_core_bean_FaceInfo
 * Method:    dangerousSmokeState
 * Signature: ()[F
 */
JNIEXPORT jintArray JNICALL Java_com_baidu_iov_vision_ability_core_bean_FaceInfo_dangerousSmokeState
        (JNIEnv *env, jobject thiz) {
    jsize len = 3; // sizeof(getBuffer(env, thiz)->_smoke_state);
    jintArray result = env->NewIntArray(len);
    if (result != nullptr) {
        env->SetIntArrayRegion(result, 0, len, (jint *) &getBuffer(env, thiz)->smokeVState);
    }
    return result;
}

/*
 * Class:     com_baidu_iov_vision_ability_core_bean_FaceInfo
 * Method:    dangerousDrinkState
 * Signature: ()[F
 */
JNIEXPORT jintArray JNICALL Java_com_baidu_iov_vision_ability_core_bean_FaceInfo_dangerousDrinkState
        (JNIEnv *env, jobject thiz) {
    jsize len = 3; // sizeof(getBuffer(env, thiz)->_drink_state);
    jintArray result = env->NewIntArray(len);
    if (result != nullptr) {
        env->SetIntArrayRegion(result, 0, len, (jint *) &getBuffer(env, thiz)->drinkVState);
    }
    return result;
}

/*
 * Class:     com_baidu_iov_vision_ability_core_bean_FaceInfo
 * Method:    dangerousMouthState
 * Signature: ()[F
 */
JNIEXPORT jintArray JNICALL Java_com_baidu_iov_vision_ability_core_bean_FaceInfo_dangerousMouthState
        (JNIEnv *env, jobject thiz) {
    jsize len = 3; // sizeof(getBuffer(env, thiz)->_open_mouth_state);
    jintArray result = env->NewIntArray(len);
    if (result != nullptr) {
        env->SetIntArrayRegion(result, 0, len, (jint *) &getBuffer(env, thiz)->openMouthVState);
    }
    return result;
}

/*
 * Class:     com_baidu_iov_vision_ability_core_bean_FaceInfo
 * Method:    dangerousCallState
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_com_baidu_iov_vision_ability_core_bean_FaceInfo_dangerousCallState
        (JNIEnv *env, jobject thiz) {
    return (jint) getBuffer(env, thiz)->_state_call;
}

/*
 * Class:     com_baidu_iov_vision_ability_core_bean_FaceInfo
 * Method:    dangerousPhoneCallState
 * Signature: ()[F
 */
JNIEXPORT jintArray JNICALL Java_com_baidu_iov_vision_ability_core_bean_FaceInfo_dangerousPhoneCallState
        (JNIEnv *env, jobject thiz) {
    jsize len = 3; // sizeof(getBuffer(env, thiz)->_phone_call_state);
    jintArray result = env->NewIntArray(len);
    if (result != nullptr) {
        env->SetIntArrayRegion(result, 0, len, (jint *) &getBuffer(env, thiz)->phoneCallVState);
    }
    return result;
}

/*
 * Class:     com_baidu_iov_vision_ability_core_bean_FaceInfo
 * Method:    fatigueState
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_com_baidu_iov_vision_ability_core_bean_FaceInfo_fatigueState
        (JNIEnv *env, jobject thiz) {
    return (jint) getBuffer(env, thiz)->_state_fatigue;
}

/*
 * Class:     com_baidu_iov_vision_ability_core_bean_FaceInfo
 * Method:    fatigueEyeCloseState
 * Signature: ()[F
 */
JNIEXPORT jintArray JNICALL Java_com_baidu_iov_vision_ability_core_bean_FaceInfo_fatigueEyeCloseState
        (JNIEnv *env, jobject thiz) {
    jsize len = 3; // sizeof(getBuffer(env, thiz)->_close_eye_state);
    jintArray result = env->NewIntArray(len);
    if (result != nullptr) {
        env->SetIntArrayRegion(result, 0, len, (jint *) &getBuffer(env, thiz)->closeEyeVState);
    }
    return result;
}

/*
 * Class:     com_baidu_iov_vision_ability_core_bean_FaceInfo
 * Method:    fatigueYawnState
 * Signature: ()[F
 */
JNIEXPORT jintArray JNICALL Java_com_baidu_iov_vision_ability_core_bean_FaceInfo_fatigueYawnState
        (JNIEnv *env, jobject thiz) {
    jsize len = 3; // sizeof(getBuffer(env, thiz)->_yawn_state);
    jintArray result = env->NewIntArray(len);
    if (result != nullptr) {
        env->SetIntArrayRegion(result, 0, len, (jint *) &getBuffer(env, thiz)->yawnVState);
    }
    return result;
}

/*
 * Class:     com_baidu_iov_vision_ability_core_bean_FaceInfo
 * Method:    attentionState
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_com_baidu_iov_vision_ability_core_bean_FaceInfo_attentionState
        (JNIEnv *env, jobject thiz) {
    return (jint) getBuffer(env, thiz)->_state_attention;
}

/*
 * Class:     com_baidu_iov_vision_ability_core_bean_FaceInfo
 * Method:    attentionLookLeftState
 * Signature: ()[F
 */
JNIEXPORT jintArray JNICALL Java_com_baidu_iov_vision_ability_core_bean_FaceInfo_attentionLookLeftState
        (JNIEnv *env, jobject thiz) {
    jsize len = 3; // sizeof(getBuffer(env, thiz)->_left_attention_state);
    jintArray result = env->NewIntArray(len);
    if (result != nullptr) {
        env->SetIntArrayRegion(result, 0, len, (jint *) &getBuffer(env, thiz)->leftAttentionState);
    }
    return result;
}

/*
 * Class:     com_baidu_iov_vision_ability_core_bean_FaceInfo
 * Method:    attentionLookRightState
 * Signature: ()[F
 */
JNIEXPORT jintArray JNICALL Java_com_baidu_iov_vision_ability_core_bean_FaceInfo_attentionLookRightState
        (JNIEnv *env, jobject thiz) {
    jsize len = 3; // sizeof(getBuffer(env, thiz)->_right_attention_state);
    jintArray result = env->NewIntArray(len);
    if (result != nullptr) {
        env->SetIntArrayRegion(result, 0, len, (jint *) &getBuffer(env, thiz)->rightAttentionState);
    }
    return result;
}

/*
 * Class:     com_baidu_iov_vision_ability_core_bean_FaceInfo
 * Method:    attentionLookUpState
 * Signature: ()[F
 */
JNIEXPORT jintArray JNICALL Java_com_baidu_iov_vision_ability_core_bean_FaceInfo_attentionLookUpState
        (JNIEnv *env, jobject thiz) {
    jsize len = 3; // sizeof(getBuffer(env, thiz)->_up_attention_state);
    jintArray result = env->NewIntArray(len);
    if (result != nullptr) {
        env->SetIntArrayRegion(result, 0, len, (jint *) &getBuffer(env, thiz)->upAttentionState);
    }
    return result;
}

/*
 * Class:     com_baidu_iov_vision_ability_core_bean_FaceInfo
 * Method:    attentionLookDownState
 * Signature: ()[F
 */
JNIEXPORT jintArray JNICALL Java_com_baidu_iov_vision_ability_core_bean_FaceInfo_attentionLookDownState
        (JNIEnv *env, jobject thiz) {
    jsize len = 3; // sizeof(getBuffer(env, thiz)->_down_attention_state);
    jintArray result = env->NewIntArray(len);
    if (result != nullptr) {
        env->SetIntArrayRegion(result, 0, len, (jint *) &getBuffer(env, thiz)->downAttentionState);
    }
    return result;
}

/*
 * Class:     com_baidu_iov_vision_ability_core_bean_FaceInfo
 * Method:    smallEyeState
 * Signature: ()[F
 */
JNIEXPORT jintArray JNICALL Java_com_baidu_iov_vision_ability_core_bean_FaceInfo_smallEyeState
        (JNIEnv *env, jobject thiz) {
    jsize len = 3; // sizeof(getBuffer(env, thiz)->_small_eye_state);
    jintArray result = env->NewIntArray(len);
    if (result != nullptr) {
        env->SetIntArrayRegion(result, 0, len, (jint *) &getBuffer(env, thiz)->smallEyeState);
    }
    return result;
}

/*
 * Class:     com_baidu_iov_vision_ability_core_bean_FaceInfo
 * Method:    attributeEmotionState
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_com_baidu_iov_vision_ability_core_bean_FaceInfo_attributeEmotionState
        (JNIEnv *env, jobject thiz) {
    return (jint) getBuffer(env, thiz)->stateEmotion;
}

/*
 * Class:     com_baidu_iov_vision_ability_core_bean_FaceInfo
 * Method:    attributeGlassState
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_com_baidu_iov_vision_ability_core_bean_FaceInfo_attributeGlassState
        (JNIEnv *env, jobject thiz) {
    return (jint) getBuffer(env, thiz)->stateGlass;
}

/*
 * Class:     com_baidu_iov_vision_ability_core_bean_FaceInfo
 * Method:    attributeGenderState
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_com_baidu_iov_vision_ability_core_bean_FaceInfo_attributeGenderState
        (JNIEnv *env, jobject thiz) {
    return (jint) getBuffer(env, thiz)->stateGender;
}

/*
 * Class:     com_baidu_iov_vision_ability_core_bean_FaceInfo
 * Method:    attributeRaceState
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_com_baidu_iov_vision_ability_core_bean_FaceInfo_attributeRaceState
        (JNIEnv *env, jobject thiz) {
    return (jint) getBuffer(env, thiz)->stateRace;
}

/*
 * Class:     com_baidu_iov_vision_ability_core_bean_FaceInfo
 * Method:    attributeAgeState
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_com_baidu_iov_vision_ability_core_bean_FaceInfo_attributeAgeState
        (JNIEnv *env, jobject thiz) {
    return (jint) getBuffer(env, thiz)->stateAge;
}

/*
 * Class:     com_baidu_iov_vision_ability_core_bean_FaceInfo
 * Method:    eyeCloseConfidence
 * Signature: ()F
 */
JNIEXPORT jfloat JNICALL Java_com_baidu_iov_vision_ability_core_bean_FaceInfo_eyeCloseConfidence
        (JNIEnv *env, jobject thiz) {
    return (jint) getBuffer(env, thiz)->eyeCloseConfidence;
}

/*
 * Class:     com_baidu_iov_vision_ability_core_bean_FaceInfo
 * Method:    eyeCentroid
 * Signature: ()[F
 */
JNIEXPORT jfloatArray JNICALL Java_com_baidu_iov_vision_ability_core_bean_FaceInfo_eyeCentroid
        (JNIEnv *env, jobject thiz) {
    jsize llen = 2; // sizeof(getBuffer(env, thiz)->_eye_centroid_left);
    jsize rlen = 2; // sizeof(getBuffer(env, thiz)->_eye_centroid_right);
    jfloatArray result = env->NewFloatArray(llen + rlen);
    if (result != nullptr) {
        env->SetFloatArrayRegion(result, 0, llen, (jfloat *) &getBuffer(env, thiz)->eyeCentroidLeft);
        env->SetFloatArrayRegion(result, llen, rlen, (jfloat *) &getBuffer(env, thiz)->eyeCentroidRight);
    }
    return result;
}

/*
 * Class:     com_baidu_iov_vision_ability_core_bean_FaceInfo
 * Method:    eyeTracking
 * Signature: ()[F
 */
JNIEXPORT jfloatArray JNICALL Java_com_baidu_iov_vision_ability_core_bean_FaceInfo_eyeTracking
        (JNIEnv *env, jobject thiz) {
    jsize len = 8; // sizeof(getBuffer(env, thiz)->_eye_tracking);
    jfloatArray result = env->NewFloatArray(len);
    if (result != nullptr) {
        env->SetFloatArrayRegion(result, 0, len, (jfloat *) &getBuffer(env, thiz)->eyeTracking);
    }
    return result;
}

/*
 * Class:     com_baidu_iov_vision_ability_core_bean_FaceInfo
 * Method:    eyeWakingState
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_com_baidu_iov_vision_ability_core_bean_FaceInfo_eyeWakingState
        (JNIEnv *env, jobject thiz) {
    return (jint) getBuffer(env, thiz)->eyeWaking;
}

/*
 * Class:     com_baidu_iov_vision_ability_core_bean_FaceInfo
 * Method:    headBehaviorState
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_com_baidu_iov_vision_ability_core_bean_FaceInfo_headBehaviorState
        (JNIEnv *env, jobject thiz) {
    return (jint) getBuffer(env, thiz)->stateHeadBehavior;
}

/*
 * Class:     com_baidu_iov_vision_ability_core_bean_FaceInfo
 * Method:    headDeflection
 * Signature: ()[F
 */
JNIEXPORT jfloatArray JNICALL Java_com_baidu_iov_vision_ability_core_bean_FaceInfo_headDeflection
        (JNIEnv *env, jobject thiz) {
    jsize len = 3; // sizeof(getBuffer(env, thiz)->_head_deflection);
    jfloatArray result = env->NewFloatArray(len);
    if (result != nullptr) {
        env->SetFloatArrayRegion(result, 0, len, (jfloat *) &getBuffer(env, thiz)->headDeflection);
    }
    return result;
}

/*
 * Class:     com_baidu_iov_vision_ability_core_bean_FaceInfo
 * Method:    headPosition
 * Signature: ()[F
 */
JNIEXPORT jfloatArray JNICALL Java_com_baidu_iov_vision_ability_core_bean_FaceInfo_headPosition
        (JNIEnv *env, jobject thiz) {
    jsize len = 3; // sizeof(getBuffer(env, thiz)->_head_deflection);
    jfloatArray result = env->NewFloatArray(len);
    if (result != nullptr) {
        env->SetFloatArrayRegion(result, 0, len, (jfloat *) &getBuffer(env, thiz)->headLocation);
    }
    return result;
}

/*
 * Class:     com_baidu_iov_vision_ability_core_bean_FaceInfo
 * Method:    beQuiteState
 * Signature: ()[F
 */
JNIEXPORT jintArray JNICALL Java_com_baidu_iov_vision_ability_core_bean_FaceInfo_beQuiteState
        (JNIEnv *env, jobject thiz) {
    jsize len = 3; // sizeof(getBuffer(env, thiz)->_silence_state);
    jintArray result = env->NewIntArray(len);
    if (result != nullptr) {
        env->SetIntArrayRegion(result, 0, len, (jint *) &getBuffer(env, thiz)->silenceVState);
    }
    return result;
}

/*
 * Class:     com_baidu_iov_vision_ability_core_bean_FaceInfo
 * Method:    isBlur
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_com_baidu_iov_vision_ability_core_bean_FaceInfo_isBlur
        (JNIEnv *env, jobject thiz) {
    return (jint) getBuffer(env, thiz)->stateBlur;
}

/*
 * Class:     com_baidu_iov_vision_ability_core_bean_FaceInfo
 * Method:    isBlocked
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_com_baidu_iov_vision_ability_core_bean_FaceInfo_isFaceBlocked
        (JNIEnv *env, jobject thiz) {
    return (jint) getBuffer(env, thiz)->stateFaceCover;
}

/*
 * Class:     com_baidu_iov_vision_ability_core_bean_FaceInfo
 * Method:    isBlocked
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_com_baidu_iov_vision_ability_core_bean_FaceInfo_isSingleBlocked
(JNIEnv *env, jobject thiz) {
return (jint) getBuffer(env, thiz)->stateFaceCoverSingle;
}

/*
 * Class:     com_baidu_iov_vision_ability_core_bean_FaceInfo
 * Method:    isNoisy
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_com_baidu_iov_vision_ability_core_bean_FaceInfo_isNoisy
        (JNIEnv *env, jobject thiz) {
    return (jint) getBuffer(env, thiz)->stateNoise;
}

/*
 * Class:     com_baidu_iov_vision_ability_core_bean_FaceInfo
 * Method:    isCameraBlocked
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_com_baidu_iov_vision_ability_core_bean_FaceInfo_isCameraBlocked
        (JNIEnv *env, jobject thiz) {
    return (jint) getBuffer(env, thiz)->stateCameraCover;
}

/*
 * Class:     com_baidu_iov_vision_ability_core_bean_FaceInfo
 * Method:    noMoving
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_com_baidu_iov_vision_ability_core_bean_FaceInfo_noMoving
        (JNIEnv *env, jobject thiz){
    return (jint) getBuffer(env, thiz)->stateFaceNomoving;
}

/*
 * Class:     com_baidu_iov_vision_ability_core_bean_FaceInfo
 * Method:    trackState
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_com_baidu_iov_vision_ability_core_bean_FaceInfo_trackState
        (JNIEnv *env, jobject thiz) {
    return (jint) getBuffer(env, thiz)->stateFaceTracking;
}

/*
 * Class:     com_baidu_iov_vision_ability_core_bean_FaceInfo
 * Method:    searchState
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_com_baidu_iov_vision_ability_core_bean_FaceInfo_searchState
        (JNIEnv *env, jobject thiz) {
    return (jint) getBuffer(env, thiz)->stateFaceDetect;
}

/*
 * Class:     com_baidu_iov_vision_ability_core_bean_FaceInfo
 * Method:    leftEyeGazeOrigin
 * Signature: ()[F
 */
JNIEXPORT jfloatArray JNICALL Java_com_baidu_iov_vision_ability_core_bean_FaceInfo_leftEyeGazeOrigin
        (JNIEnv *env, jobject thiz) {
    jsize len = 3; // sizeof(getBuffer(env, thiz)->_eye_gaze_origin_left);
    jfloatArray result = env->NewFloatArray(len);
    if (result != nullptr) {
        env->SetFloatArrayRegion(result, 0, len, (jfloat *) &getBuffer(env, thiz)->eyeGazeOriginLeft);
    }
    return result;
}

/*
 * Class:     com_baidu_iov_vision_ability_core_bean_FaceInfo
 * Method:    leftEyeGazeDirection
 * Signature: ()[F
 */
JNIEXPORT jfloatArray JNICALL Java_com_baidu_iov_vision_ability_core_bean_FaceInfo_leftEyeGazeDirection
        (JNIEnv *env, jobject thiz) {
    jsize len = 3; // sizeof(getBuffer(env, thiz)->_eye_gaze_destination_left);
    jfloatArray result = env->NewFloatArray(len);
    if (result != nullptr) {
        env->SetFloatArrayRegion(result, 0, len, (jfloat *) &getBuffer(env, thiz)->eyeGazeDestinationLeft);
    }
    return result;
}

/*
 * Class:     com_baidu_iov_vision_ability_core_bean_FaceInfo
 * Method:    rightEyeGazeOrigin
 * Signature: ()[F
 */
JNIEXPORT jfloatArray JNICALL Java_com_baidu_iov_vision_ability_core_bean_FaceInfo_rightEyeGazeOrigin
        (JNIEnv *env, jobject thiz) {
    jsize len = 3; // sizeof(getBuffer(env, thiz)->_eye_gaze_origin_right);
    jfloatArray result = env->NewFloatArray(len);
    if (result != nullptr) {
        env->SetFloatArrayRegion(result, 0, len, (jfloat *) &getBuffer(env, thiz)->eyeGazeOriginRight);
    }
    return result;
}

/*
 * Class:     com_baidu_iov_vision_ability_core_bean_FaceInfo
 * Method:    rightEyeGazeDirection
 * Signature: ()[F
 */
JNIEXPORT jfloatArray JNICALL Java_com_baidu_iov_vision_ability_core_bean_FaceInfo_rightEyeGazeDirection
        (JNIEnv *env, jobject thiz) {
    jsize len = 3; // sizeof(getBuffer(env, thiz)->_eye_gaze_destination_right);
    jfloatArray result = env->NewFloatArray(len);
    if (result != nullptr) {
        env->SetFloatArrayRegion(result, 0, len, (jfloat *) &getBuffer(env, thiz)->eyeGazeDestinationRight);
    }
    return result;
}

/*
 * Class:     com_baidu_iov_vision_ability_core_bean_FaceInfo
 * Method:    leftEyelidOpening
 * Signature: ()F
 */
JNIEXPORT jfloat JNICALL Java_com_baidu_iov_vision_ability_core_bean_FaceInfo_leftEyelidOpening
        (JNIEnv *env, jobject thiz) {
    return getBuffer(env, thiz)->eyeEyelidDistanceLeft;
}

/*
 * Class:     com_baidu_iov_vision_ability_core_bean_FaceInfo
 * Method:    rightEyelidOpening
 * Signature: ()F
 */
JNIEXPORT jfloat JNICALL Java_com_baidu_iov_vision_ability_core_bean_FaceInfo_rightEyelidOpening
        (JNIEnv *env, jobject thiz) {
    return getBuffer(env, thiz)->eyeEyelidDistanceRight;
}

/*
 * Class:     com_baidu_iov_vision_ability_core_bean_FaceInfo
 * Method:    leftEyelidState
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_com_baidu_iov_vision_ability_core_bean_FaceInfo_leftEyelidState
        (JNIEnv *env, jobject thiz) {
    return getBuffer(env, thiz)->eyeEyelidStatusLeft;
}

/*
 * Class:     com_baidu_iov_vision_ability_core_bean_FaceInfo
 * Method:    rightEyelidState
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_com_baidu_iov_vision_ability_core_bean_FaceInfo_rightEyelidState
        (JNIEnv *env, jobject thiz) {
    return getBuffer(env, thiz)->eyeEyelidStatusRight;
}

/*
 * Class:     com_baidu_iov_vision_ability_core_bean_FaceInfo
 * Method:    eyeState
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_com_baidu_iov_vision_ability_core_bean_FaceInfo_eyeState
        (JNIEnv *env, jobject thiz) {
    return getBuffer(env, thiz)->eyeState;
}

/*
 * Class:     com_baidu_iov_vision_ability_core_bean_FaceInfo
 * Method:    leftEyeState
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_com_baidu_iov_vision_ability_core_bean_FaceInfo_leftEyeState
        (JNIEnv *env, jobject thiz) {
    return getBuffer(env, thiz)->leftEyeDetectSingle;
}

/*
 * Class:     com_baidu_iov_vision_ability_core_bean_FaceInfo
 * Method:    rightEyeState
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_com_baidu_iov_vision_ability_core_bean_FaceInfo_rightEyeState
        (JNIEnv *env, jobject thiz) {
    return getBuffer(env, thiz)->rightEyeDetectSingle;
}

/*
 * Class:     com_baidu_iov_vision_ability_core_bean_FaceInfo
 * Method:    eyeBlinkCloseSpeed
 * Signature: ()F
 */
JNIEXPORT jfloat JNICALL Java_com_baidu_iov_vision_ability_core_bean_FaceInfo_eyeBlinkCloseSpeed
        (JNIEnv *env, jobject thiz){
    return getBuffer(env, thiz)->eyeCloseFrequency;
}

/*
 * Class:     com_baidu_iov_vision_ability_core_bean_FaceInfo
 * Method:    eyeBlinkCloseStartTime
 * Signature: ()F
 */
JNIEXPORT jlong JNICALL Java_com_baidu_iov_vision_ability_core_bean_FaceInfo_eyeBlinkCloseStartTime
        (JNIEnv *env, jobject thiz){
    return getBuffer(env, thiz)->eyeStartClosingTime;
}
/*
 * Class:     com_baidu_iov_vision_ability_core_bean_FaceInfo
 * Method:    eyeBlinkCloseStopTime
 * Signature: ()F
 */
JNIEXPORT jlong JNICALL Java_com_baidu_iov_vision_ability_core_bean_FaceInfo_eyeBlinkCloseStopTime
        (JNIEnv *env, jobject thiz){
    return getBuffer(env, thiz)->eyeEndClosingTime;
}

/*
 * Class:     com_baidu_iov_vision_ability_core_bean_FaceInfo
 * Method:    eyeBlinkState
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_com_baidu_iov_vision_ability_core_bean_FaceInfo_eyeBlinkState
        (JNIEnv *env, jobject thiz) {
    return getBuffer(env, thiz)->eyeBlinkState;
}

/*
 * Class:     com_baidu_iov_vision_ability_core_bean_FaceInfo
 * Method:    eyeBlinkDuration
 * Signature: ()F
 */
JNIEXPORT jfloat JNICALL Java_com_baidu_iov_vision_ability_core_bean_FaceInfo_eyeBlinkDuration
        (JNIEnv *env, jobject thiz) {
    return getBuffer(env, thiz)->eyeBlinkDuration;
}

/*
 * Class:     com_baidu_iov_vision_ability_core_bean_FaceInfo
 * Method:    noFaceState
 * Signature: ()[I
 */
JNIEXPORT jintArray JNICALL Java_com_baidu_iov_vision_ability_core_bean_FaceInfo_noFaceState
        (JNIEnv *env, jobject thiz) {
    jsize len = 3;
    jintArray result = env->NewIntArray(len);
    if (result != nullptr) {
        env->SetIntArrayRegion(result, 0, len, (jint *) &getBuffer(env, thiz)->noFaceState);
    }
    return result;
}

#ifdef __cplusplus
}
#endif