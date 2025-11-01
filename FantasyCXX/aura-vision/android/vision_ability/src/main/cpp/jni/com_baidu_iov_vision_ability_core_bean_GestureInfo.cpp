
#include "com_baidu_iov_vision_ability_core_bean_GestureInfo.h"

#include "vision/core/bean/GestureInfo.h"

#define TAG "GestureInfoNative"

#ifdef __cplusplus
extern "C" {
#endif

using namespace vision;
using namespace std;

static jclass cInfo = 0;
static jfieldID fNativeBuffer = 0;

static GestureInfo *getBuffer(JNIEnv *env, jobject thiz) {
    jobject buffer = env->GetObjectField(thiz, fNativeBuffer);
    return (GestureInfo *) env->GetDirectBufferAddress(buffer);
}

/*
 * Class:     com_baidu_iov_vision_ability_core_bean_GestureInfo
 * Method:    init
 * Signature: (Ljava/nio/FloatBuffer;)V
 */
JNIEXPORT void JNICALL Java_com_baidu_iov_vision_ability_core_bean_GestureInfo_init
        (JNIEnv *env, jobject thiz, jobject buffer) {
    cInfo = env->GetObjectClass(thiz);
    fNativeBuffer = env->GetFieldID(cInfo, "mNativeBuffer", "Ljava/nio/FloatBuffer;");
}

/*
 * Class:     com_baidu_iov_vision_ability_core_bean_GestureInfo
 * Method:    getBufferSize
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_com_baidu_iov_vision_ability_core_bean_GestureInfo_getBufferSize
        (JNIEnv *env, jobject thiz) {
    return (jint) sizeof(GestureInfo);
}

/*
 * Class:     com_baidu_iov_vision_ability_core_bean_GestureInfo
 * Method:    id
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_com_baidu_iov_vision_ability_core_bean_GestureInfo_id__
        (JNIEnv *env, jobject thiz) {
    return (jint) getBuffer(env, thiz)->id;
}

/*
 * Class:     com_baidu_iov_vision_ability_core_bean_GestureInfo
 * Method:    rectConfidence
 * Signature: ()F
 */
JNIEXPORT jfloat JNICALL Java_com_baidu_iov_vision_ability_core_bean_GestureInfo_rectConfidence__
        (JNIEnv *env, jobject thiz) {
    return getBuffer(env, thiz)->rectConfidence;
}

/*
 * Class:     com_baidu_iov_vision_ability_core_bean_GestureInfo
 * Method:    rect
 * Signature: ()[F
 */
JNIEXPORT jfloatArray JNICALL Java_com_baidu_iov_vision_ability_core_bean_GestureInfo_rect
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
 * Class:     com_baidu_iov_vision_ability_core_bean_GestureInfo
 * Method:    landmarkConfidence
 * Signature: ()F
 */
JNIEXPORT jfloat JNICALL Java_com_baidu_iov_vision_ability_core_bean_GestureInfo_landmarkConfidence
        (JNIEnv *env, jobject thiz) {
    return getBuffer(env, thiz)->landmarkConfidence;
}

/*
 * Class:     com_baidu_iov_vision_ability_core_bean_GestureInfo
 * Method:    landmark
 * Signature: ()[F
 */
JNIEXPORT jfloatArray JNICALL Java_com_baidu_iov_vision_ability_core_bean_GestureInfo_landmark
        (JNIEnv *env, jobject thiz) {
    jsize len = 42; // sizeof(getBuffer(env, thiz)->_landmark_21);
    jfloatArray result = env->NewFloatArray(len);
    if (result != nullptr) {
        env->SetFloatArrayRegion(result, 0, len, (jfloat *) &getBuffer(env, thiz)->landmark21);
    }
    return result;
}

/*
 * Class:     com_baidu_iov_vision_ability_core_bean_GestureInfo
 * Method:    type
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_com_baidu_iov_vision_ability_core_bean_GestureInfo_type
        (JNIEnv *env, jobject thiz) {
    return (jint) getBuffer(env, thiz)->type;
}

/*
 * Class:     com_baidu_iov_vision_ability_core_bean_GestureInfo
 * Method:    singleType
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_com_baidu_iov_vision_ability_core_bean_GestureInfo_singleType
        (JNIEnv *env, jobject thiz) {
    return (jint) getBuffer(env, thiz)->type;
}

/*
 * Class:     com_baidu_iov_vision_ability_core_bean_GestureInfo
 * Method:    staticType
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_com_baidu_iov_vision_ability_core_bean_GestureInfo_staticType
        (JNIEnv *env, jobject thiz) {
    return (jint) getBuffer(env, thiz)->staticType;
}

/*
 * Class:     com_baidu_iov_vision_ability_core_bean_GestureInfo
 * Method:    dynamicType
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_com_baidu_iov_vision_ability_core_bean_GestureInfo_dynamicType
        (JNIEnv *env, jobject thiz) {
    return (jint) getBuffer(env, thiz)->dynamicType;
}

#ifdef __cplusplus
}
#endif