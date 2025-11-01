#include "NaiveScheduler.h"

#include "vision/manager/VisionManagerRegistry.h"

namespace aura::vision {

NaiveScheduler::NaiveScheduler() {
    faceTrackSubScheduler.reset(new FaceTrackSubScheduler);
}

void NaiveScheduler::run(VisionRequest *request, VisionResult *result) {
    /**
     * TODO zhushisong sunwenming 当FaceId与dms/oms通用原子能力同时执行时，可能存在某一帧仅执行runDetectionForcibly而不执行后续
     * 其他检测情况，待商讨具体修改策略
     */
    // detect only one ability with the ability-id in the request
    if (request->specific_detection()) {
        runDetectionForcibly(request->specific_ability(), request, result);
        return;
    }
    // FaceId特征值获取是在result的FaceInfo结果之上直接执行feature检测，不能在visionService中直接将人脸置为无效
    // 当不是FaceId的FaceFeature这种特殊检测，在检测的开始处清除之前VisionResult的结果数据
    result->clear();

    // result清除时会清空VisionService中赋值的frame和timestamp，此处为其再一次赋值
    result->getFrameInfo()->frame = request->frame;
    result->getFrameInfo()->timestamp = request->timestamp;

    // detect all the abilities enabled in the config
    run_face_detections(request, result); // FaceID & DMS
    run_gesture_detections(request, result); // MMI abilities
    runLivingDetections(request, result);   // Living abilities
    runBodyDetections(request, result);   // pose abilities
}

/**
 * @param request
 * @param result
 */
void NaiveScheduler::run_face_detections(VisionRequest *request, VisionResult *result) {
    runDetection(ABILITY_IMAGE_BRIGHTNESS, request, result);
    // Basic abilities of face detection
    runDetection(ABILITY_CAMERA_COVER, request, result);

    faceTrackSubScheduler->run(request, result);

    // 如果当前帧检测无人脸，则直接停止检测，与检测模式无关。
    auto detectFaceCount = 0;
    for (int i = 0; i < mRtConfig->faceNeedDetectCount; ++i) {
        auto face = result->getFaceResult()->faceInfos[i];
        if (face->faceType == FaceDetectType::F_TYPE_TRACK) {
            for (auto &manager: getManagerRegistry()->getManagers()) {
                manager->onNoFace(request, result, face);
            }
        } else if (face->faceType == FaceDetectType::F_TYPE_DETECT) {
            // 如果当前人脸是检测出来，则检测出来的人脸数
            ++detectFaceCount;
        }
    }
    // 如果无检测出来的人脸数为0。则停止后面的所有模型检测
    if (detectFaceCount == 0) {
        return;
    }

    // FaceID abilities
    runDetection(ABILITY_FACE_QUALITY, request, result);
    runDetection(ABILITY_FACE_INTERACTIVE_LIVING, request, result);
    runDetection(ABILITY_FACE_NO_INTERACTIVE_LIVING, request, result);
    runDetection(ABILITY_FACE_FEATURE, request, result);
    // 2D 转 3D 模型没有启动，暂时注释掉其调用
    // run_detection(ABILITY_FACE_2DTO3D, request, result);
    runDetection(ABILITY_FACE_HEAD_BEHAVIOR, request, result);
    runDetection(ABILITY_FACE_ATTRIBUTE, request, result);
    runDetection(ABILITY_FACE_EMOTION, request, result);

    // DMS abilities
    runDetection(ABILITY_FACE_DANGEROUS_DRIVING, request, result);
    runDetection(ABILITY_FACE_CALL, request, result);
    runDetection(ABILITY_FACE_ATTENTION, request, result);
    // 唇动：业务逻辑上依赖遮挡模型、DMS七分类模型 & mouth landmark模型
    runDetection(ABILITY_FACE_MOUTH_LANDMARK, request, result);
    // EyeCenter模型会反向修改Landmark的关键点，所以需要在其他检测之后，所有依赖EyeCenter的逻辑跟随其后
    runDetection(ABILITY_FACE_EYE_CENTER, request, result);
    // fatigue relies on 2D_landmark or eyeCenter model
    runDetection(ABILITY_FACE_FATIGUE, request, result);
    runDetection(ABILITY_FACE_EYE_GAZE, request, result);
    runDetection(ABILITY_FACE_EYE_TRACKING, request, result);
    // 3D 人脸关键点的检测：3D人脸关键点依赖： Landmark关键点模型、EyeCenter眼球中心点模型(如果需要进行视线检测)
    runDetection(ABILITY_FACE_3D_LANDMARK, request, result);
    // 3D视线检测：3D视线检测依赖3D人脸关键点的检测
    // run_detection(ABILITY_FACE_3D_EYE_GAZE, request, result);
    // 唇动去掉对3DLandMark依赖，此处位置可不动
    runDetection(ABILITY_FACE_LIP_MOVEMENT, request, result);

    // Experimental abilities
#ifdef WITH_EXPERIMENTAL
    run_detection(ABILITY_FACE_RECONSTRUCT, request, result);
#endif
}


void NaiveScheduler::run_gesture_detections(VisionRequest *request, VisionResult *result) {

    runDetection(ABILITY_GESTURE_RECT, request, result);
    // 执行玩手机的检测. 玩手机的检测依赖手势框模型
    // add by wangzhijiang 为规避算法模型(玩手机和打电话的误识别)玩手机需要打电话之后调用。
    runDetection(ABILITY_PLAY_PHONE_DETECT, request, result);

    if (!result->hasGesture() && mRtConfig->releaseMode != ReleaseMode::BENCHMARK_TEST) {
        for (auto &manager: getManagerRegistry()->getManagers()) {
            manager->onNoGesture(request, result);
        }
        return;
    }
    runDetectionIF(ABILITY_GESTURE_LANDMARK, result->hasGesture(), request, result);
    runDetectionIF(ABILITY_GESTURE_TYPE, result->hasGesture(), request, result);
    runDetectionIF(ABILITY_GESTURE_DYNAMIC, result->hasGesture(), request, result);
}

void NaiveScheduler::runLivingDetections(VisionRequest *request, VisionResult *result) {
    // Living abilities
    runDetection(ABILITY_LIVING_DETECTION, request, result);
}

void NaiveScheduler::runBodyDetections(VisionRequest *request, VisionResult *result) {
    // pose abilities
    runDetection(ABILITY_BODY_HEAD_SHOULDER, request, result);
    if (!result->hasBody() && mRtConfig->releaseMode != ReleaseMode::BENCHMARK_TEST) {
        for (auto &manager: getManagerRegistry()->getManagers()) {
            if (manager != nullptr) {
                manager->onNoBody(request, result);
            }
        }
        return;
    }
    runDetectionIF(ABILITY_BODY_LANDMARK, result->hasBody(), request, result);
}

void NaiveScheduler::injectManagerRegistry(const std::shared_ptr<VisionManagerRegistry> &registry) {
    AbsScheduler::injectManagerRegistry(registry);
    faceTrackSubScheduler->injectManagerRegistry(registry);
}

void NaiveScheduler::initManagers(RtConfig *cfg) {
    mRtConfig = cfg;
    faceTrackSubScheduler->set_config(cfg);
    for (auto &manager: pMgrRegistry->getManagers()) {
        manager->init(cfg);
    }
}

} // namespace vision