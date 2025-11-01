#include "DagScheduler.h"
#include <scheduler/dag/Node.h>

namespace aura::vision {

#define INIT_NODE(id, graph) \
    pMgrRegistry->getManager(id)->initNode(id, graph);

DagScheduler::DagScheduler() {
}

DagScheduler::~DagScheduler() {
    for (auto &graph : mGraphs) {
        delete graph;
        graph = nullptr;
    }
    mGraphs.clear();
}

void DagScheduler::run(VisionRequest *request, VisionResult *result) {
    // detect only one ability with the ability-id in the request
    if (request->specific_detection()) {
        runDetectionForcibly(request->specific_ability(), request, result);
        return;
    }
    // FaceId特征值获取是在result的FaceInfo结果之上直接执行feature检测，不能在visionService中直接将人脸置为无效
    // 当不是FaceId的FaceFeature这种特殊检测，在检测的开始处清除之前VisionResult的结果数据
    result->clear();

    /// 按照构建的图，执行图的运行，我们现在只有一个图需要运行
    for (auto graph : mGraphs) {
        graph->run(request, result);
    }
}

void DagScheduler::injectManagerRegistry(const std::shared_ptr<VisionManagerRegistry> &registry) {
    AbsScheduler::injectManagerRegistry(registry);
}

void DagScheduler::initManagers(RtConfig *cfg) {
    AbsScheduler::initManagers(cfg);
    initGraph();
}
/**
 * 需要支持动态配置调度策略
 */
void DagScheduler::initGraph() {
    auto *gDms = new Graph();

    Node *nImageBrightness =
            pMgrRegistry->getManager(ABILITY_IMAGE_BRIGHTNESS)->initNode(ABILITY_IMAGE_BRIGHTNESS, gDms);
    Node *nCameraCover = pMgrRegistry->getManager(ABILITY_CAMERA_COVER)->initNode(ABILITY_CAMERA_COVER, gDms);
    Node *nFaceRect = pMgrRegistry->getManager(ABILITY_FACE_RECT)->initNode(ABILITY_FACE_RECT, gDms);
    nFaceRect->pre = [](VisionRequest* req, VisionResult* res) -> bool {
        return !res->hasFace();
    };
    nFaceRect->post = [](VisionRequest* req, VisionResult* res) -> bool {
        return true;
    };

    Node *nFaceLmk = pMgrRegistry->getManager(ABILITY_FACE_LANDMARK)->initNode(ABILITY_FACE_LANDMARK, gDms);
    nFaceLmk->post = [this](VisionRequest* req, VisionResult* res) -> bool {
        if (!res->hasFace() && mRtConfig->releaseMode != ReleaseMode::BENCHMARK_TEST) {
            for (auto& manager : getManagerRegistry()->getManagers()) {
                manager->onNoFace(req, res);
            }
            return false;
        }
        return true;
    };

    Node *nFaceQuality = pMgrRegistry->getManager(ABILITY_FACE_QUALITY)->initNode(ABILITY_FACE_QUALITY, gDms);
    Node *nFaceIntLiving = pMgrRegistry->getManager(ABILITY_FACE_INTERACTIVE_LIVING)->initNode(ABILITY_FACE_INTERACTIVE_LIVING, gDms);
    Node *nFaceNoIntLiving = pMgrRegistry->getManager(ABILITY_FACE_NO_INTERACTIVE_LIVING)->initNode(ABILITY_FACE_NO_INTERACTIVE_LIVING, gDms);
    Node *nFaceFeature = pMgrRegistry->getManager(ABILITY_FACE_FEATURE)->initNode(ABILITY_FACE_FEATURE, gDms);
    nFaceFeature->pre = [](VisionRequest* req, VisionResult* res) -> bool {
//        return res->face_live();
        return true;
    };
    nFaceFeature->post = [](VisionRequest* req, VisionResult* res) -> bool {
        return true;
    };

    Node *nFace2dTo3d = pMgrRegistry->getManager(ABILITY_FACE_2DTO3D)->initNode(ABILITY_FACE_2DTO3D, gDms);
    Node *nFaceEyeCenter = pMgrRegistry->getManager(ABILITY_FACE_EYE_CENTER)->initNode(ABILITY_FACE_EYE_CENTER, gDms);
    Node *nFaceMouthLandmark = pMgrRegistry->getManager(ABILITY_FACE_MOUTH_LANDMARK)->initNode(ABILITY_FACE_MOUTH_LANDMARK, gDms);
    Node *nFaceEyeGaze = pMgrRegistry->getManager(ABILITY_FACE_EYE_GAZE)->initNode(ABILITY_FACE_EYE_GAZE, gDms);
    Node *nFaceHeadBehavior = pMgrRegistry->getManager(ABILITY_FACE_HEAD_BEHAVIOR)->initNode(ABILITY_FACE_HEAD_BEHAVIOR, gDms);
//    Node *nFaceEyeTracking = pMgrRegistry->getManager(ABILITY_FACE_EYE_TRACKING)->initNode(ABILITY_FACE_EYE_TRACKING, gDms);
    Node *nFaceAttribute = pMgrRegistry->getManager(ABILITY_FACE_ATTRIBUTE)->initNode(ABILITY_FACE_ATTRIBUTE, gDms);
    Node *nFaceEmotion = pMgrRegistry->getManager(ABILITY_FACE_EMOTION)->initNode(ABILITY_FACE_EMOTION, gDms);

    // DMS abilities
    Node *nFaceDanger = pMgrRegistry->getManager(ABILITY_FACE_DANGEROUS_DRIVING)->initNode(ABILITY_FACE_DANGEROUS_DRIVING, gDms);
    Node *nFaceCall = pMgrRegistry->getManager(ABILITY_FACE_CALL)->initNode(ABILITY_FACE_CALL, gDms);
    Node *nFaceAttention = pMgrRegistry->getManager(ABILITY_FACE_ATTENTION)->initNode(ABILITY_FACE_ATTENTION, gDms);
    Node *nFaceFatigue = pMgrRegistry->getManager(ABILITY_FACE_FATIGUE)->initNode(ABILITY_FACE_FATIGUE, gDms);

    Node *nGestureRect = pMgrRegistry->getManager(ABILITY_GESTURE_RECT)->initNode(ABILITY_GESTURE_RECT, gDms);
    nGestureRect->post = [this](VisionRequest* req, VisionResult* res) -> bool {
        if (!res->hasGesture() && mRtConfig->releaseMode != ReleaseMode::BENCHMARK_TEST) {
            for (auto& manager : getManagerRegistry()->getManagers()) {
                manager->onNoGesture(req, res);
            }
            return false;
        }
        return true;
    };

    // Node *nGestureLmk = pMgrRegistry->getManager(ABILITY_GESTURE_LANDMARK)->initNode(ABILITY_GESTURE_LANDMARK, gDms);
    // Node *nGestureStatic = pMgrRegistry->getManager(ABILITY_GESTURE_TYPE)->initNode(ABILITY_GESTURE_TYPE, gDms);
    // Node *nGestureDynamic = pMgrRegistry->getManager(ABILITY_GESTURE_DYNAMIC)->initNode(ABILITY_GESTURE_DYNAMIC,gDms);
    // 执行玩手机的检测. 玩手机的检测依赖手势框模型
    // TODO add by wangzhijiang 为规避算法模型(玩手机和打电话的误识别)玩手机需要打电话之后调用。
    // TODO Dag调度需要进行设计玩手机在打电话之后执行,需要好好设计DAG调度
    Node *nPlayPhone = pMgrRegistry->getManager(ABILITY_PLAY_PHONE_DETECT)->initNode(ABILITY_PLAY_PHONE_DETECT, gDms);


    Node *nBiology = pMgrRegistry->getManager(ABILITY_LIVING_DETECTION)->initNode(ABILITY_LIVING_DETECTION, gDms);
    Node *nShoulder = pMgrRegistry->getManager(ABILITY_BODY_HEAD_SHOULDER)->initNode(ABILITY_BODY_HEAD_SHOULDER, gDms);
    nShoulder->post = [this](VisionRequest* req, VisionResult* res) -> bool {
        if (!res->hasBody() && mRtConfig->releaseMode != ReleaseMode::BENCHMARK_TEST) {
            for (auto& manager : getManagerRegistry()->getManagers()) {
                manager->onNoBody(req, res);
            }
            return false;
        }
        return true;
    };

    Node *nBodyLmk = pMgrRegistry->getManager(ABILITY_BODY_LANDMARK)->initNode(ABILITY_BODY_LANDMARK, gDms);
    // 建立 Node 关系
    nFaceRect->addNode(nFaceLmk);
    nFaceLmk->addNode(nFaceQuality)
        ->addNode(nFaceEmotion)
        ->addNode(nFaceAttribute)
        ->addNode(nFaceCall)
        ->addNode(nFaceDanger)
        ->addNode(nFaceEyeCenter)
        ->addNode(nFaceMouthLandmark)
        ->addNode(nFaceHeadBehavior)
        ->addNode(nFaceAttention)
        ->addNode(nFace2dTo3d)
        ->addNode(nFaceFeature);
    nFaceEyeCenter->addNode(nFaceFatigue)
        ->addNode(nFaceEyeGaze);
    nFaceQuality->addNode(nFaceNoIntLiving);

    // 建立手势相关的 Node 的节点关系
    nGestureRect->addNode(nPlayPhone);

    // BodyLandmark follow BodyRect bode
    nShoulder->addNode(nBodyLmk);
    
    // 添加 Graph 根节点, OMS使用策略, DMS使用模型
    gDms->addNode(nImageBrightness);
    gDms->addNode(nCameraCover);
    gDms->addNode(nFaceRect);
    gDms->addNode(nGestureRect);
    gDms->addNode(nBiology);
    gDms->addNode(nShoulder);
    mGraphs.push_back(gDms);

    for (auto graph : mGraphs) {
        graph->pRtConfig = mRtConfig;
    }
}

} // namespace vision
