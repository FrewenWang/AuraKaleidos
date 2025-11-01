
#include "com_baidu_iov_vision_ability_core_bean_FrameInfo.h"

#include "vision/core/bean/FrameInfo.h"

#define TAG "FrameInfoNative"

#ifdef __cplusplus
extern "C" {
#endif

using namespace vision;
using namespace std;

static jclass cInfo = 0;
static jfieldID fNativeBuffer = 0;

static FrameInfo *getBuffer(JNIEnv *env, jobject thiz) {
    jobject buffer = env->GetObjectField(thiz, fNativeBuffer);
    return (FrameInfo *) env->GetDirectBufferAddress(buffer);
}

/*
 * Class:     com_baidu_iov_vision_ability_core_bean_FrameInfo
 * Method:    init
 * Signature: (Ljava/nio/FloatBuffer;)V
 */
JNIEXPORT void JNICALL Java_com_baidu_iov_vision_ability_core_bean_FrameInfo_init
        (JNIEnv *env, jobject thiz, jobject buffer) {
    cInfo = env->GetObjectClass(thiz);
    fNativeBuffer = env->GetFieldID(cInfo, "mNativeBuffer", "Ljava/nio/FloatBuffer;");
}

/*
 * Class:     com_baidu_iov_vision_ability_core_bean_FrameInfo
 * Method:    getBufferSize
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_com_baidu_iov_vision_ability_core_bean_FrameInfo_getBufferSize
        (JNIEnv *env, jobject thiz) {
    return (jint) sizeof(FrameInfo);
}

/*
 * Class:     com_baidu_iov_vision_ability_core_bean_FrameInfo
 * Method:    brightness
 * Signature: ()F
 */
JNIEXPORT jfloat JNICALL Java_com_baidu_iov_vision_ability_core_bean_FrameInfo_brightness
        (JNIEnv *env, jobject thiz) {
    return getBuffer(env, thiz)->brightness;
}

/*
 * Class:     com_baidu_iov_vision_ability_core_bean_FrameInfo
 * Method:    imageState
 * Signature: ()F
 */
JNIEXPORT jfloat JNICALL Java_com_baidu_iov_vision_ability_core_bean_FrameInfo_imageState
        (JNIEnv *env, jobject thiz) {
    return getBuffer(env, thiz)->stateFrameLightness;
}

/*
 * Class:     com_baidu_iov_vision_ability_core_bean_FrameInfo
 * Method:    blockage
 * Signature: ()F
 */
JNIEXPORT jfloat JNICALL Java_com_baidu_iov_vision_ability_core_bean_FrameInfo_blockage
        (JNIEnv *env, jobject thiz){
    return getBuffer(env, thiz)->stateFrameOcclusion;
}

/*
 * Class:     com_baidu_iov_vision_ability_core_bean_FrameInfo
 * Method:    spoofed
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_com_baidu_iov_vision_ability_core_bean_FrameInfo_spoofed
        (JNIEnv *env, jobject thiz) {

    return (jint) getBuffer(env, thiz)->stateSpoof;
}

/*
 * Class:     com_baidu_iov_vision_ability_core_bean_FrameInfo
 * Method:    timestamp
 * Signature: ()I
 */
JNIEXPORT jlong JNICALL Java_com_baidu_iov_vision_ability_core_bean_FrameInfo_timestamp
        (JNIEnv *env, jobject thiz) {

    return (jlong) getBuffer(env, thiz)->timestamp;
}
#ifdef __cplusplus
}
#endif