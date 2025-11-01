

#ifndef VISION_ABS_VISION_STRATEGY_H
#define VISION_ABS_VISION_STRATEGY_H

#include "vision/config/runtime_config/RtConfig.h"
#include "vision/core/bean/FaceInfo.h"
#include "vision/core/request/VisionRequest.h"
#include "vision/core/result/VisionResult.h"
#include <map>
#include <set>

namespace aura::vision {

class RtConfig;

template<typename DataType>
class AbsVisionStrategy {
public :
    /** VisionService运行时配置 */
    RtConfig *rtConfig;

    explicit AbsVisionStrategy() = default;

    virtual ~AbsVisionStrategy() = default;

    virtual void onNoFace(VisionRequest *request, DataType *face) {};

    virtual void onConfigUpdated(int key, float value) {};

    virtual void execute(DataType *face) = 0;

    virtual void clear() {};

    /**
      * 设置不同帧率下动态滑窗长度和占空比，每个manager自定义滑窗长度因子和占空比因子：
      *     滑窗长度因子：实时帧率下，对应滑窗长度与该实时帧率的比值
      *     占空比因子：实时帧率下，对应滑窗的占空比
      */
    virtual void setupSlidingWindow() {};

};

template<typename T>
void execute_face_strategy(VisionResult *result, std::map<int, T*>& strategy_map, RtConfig* cfg) {
    // 判断 T 是否继承了 AbsVisionStrategy<FaceInfo>
    static_assert(std::is_base_of<AbsVisionStrategy<FaceInfo>, T>::value, "Typename T is not inheritance of AbsVisionStrategy<FaceInfo> !!!");

    int face_check_count = static_cast<int>(cfg->faceNeedDetectCount);

    // 执行多人练策略
    std::set<int> face_id_list;
    auto *face_result = result->getFaceResult();
    for (int i = 0; i < face_check_count; ++i) {
        // 检测是否有人脸
        FaceInfo *face = face_result->faceInfos[i];

        // 只有检测的和跟踪延续得到的人脸才作为有效人脸，记录人脸 id
        if (face->faceType != FaceDetectType::F_TYPE_UNKNOWN) {
            face_id_list.insert(face->id);
        }
        // 如果是跟踪延续和无效人脸都不执行execute
        V_CHECK_CONT(face->isNotDetectType()); // 判断当前人脸是否是模型检出
        // 创建或获取属于该 faceid 的策略处理对象，如果存在则执行execute，否则则重新创建执行execute
        auto iter = strategy_map.find(face->id);
        if (iter == strategy_map.end()) {
            T* driving_logic = T::obtain(cfg);
            driving_logic->clear();
            driving_logic->rtConfig = cfg;
            driving_logic->execute(face);
            strategy_map.insert({face->id, driving_logic});
        } else {
            iter->second->execute(face);
        }
    }
    // 依次遍历现有StrategyMap的集合，确认当前帧检测出来是否存在对应ID的人脸，无此ID则回收对应的Strategy
    for (auto iter = strategy_map.begin(); iter != strategy_map.end();) {
        if (!face_id_list.count(iter->first)) {
            iter->second->clear();
            T::recycle(iter->second);
            iter = strategy_map.erase(iter);
        } else {
            ++iter;
        }
    }
}

template<typename T>
void execute_gesture_strategy(VisionResult *result, std::map<int, T*>& strategy_map, RtConfig* cfg) {
    // 判断 T 是否继承了 AbsVisionStrategy<FaceInfo>
    static_assert(std::is_base_of<AbsVisionStrategy<GestureInfo>, T>::value, "Typename T is not inheritance of "
                                                                   "AbsVisionStrategy<GestureInfo> !!!");
    // 判断需要检测手势框的数量
    int checkCount = static_cast<int>(cfg->gestureNeedDetectCount);
    // 执行多手势检测
    std::set<int> gestureIdList;
    auto *gesture_result = result->getGestureResult();
    for (int i = 0; i < checkCount; ++i) {
        // 检测是否有人脸
        GestureInfo *gesture = gesture_result->gestureInfos[i];
        if (!gesture->hasGesture()) {
            continue;
        }
        // 记录存在的人脸 id
        gestureIdList.insert(gesture->id);
        // 判断该手势框是否存在对应的策略处理类
        auto iter = strategy_map.find(gesture->id);
        if (iter == strategy_map.end()) {
            T*strategyLogic = T::obtain(cfg);
            strategyLogic->clear();
            strategyLogic->rtConfig = cfg;
            strategyLogic->execute(gesture);
            strategy_map.insert({gesture->id, strategyLogic});
        } else {
            iter->second->execute(gesture);
        }
    }

    for (auto iter = strategy_map.begin(); iter != strategy_map.end();) {
        if (!gestureIdList.count(iter->first)) {
            iter->second->clear();
            T::recycle(iter->second);
            iter = strategy_map.erase(iter);
        } else {
            ++iter;
        }
    }
}

template<typename T>
void execute_body_strategy(VisionResult *result, std::map<int, T*>& strategy_map, RtConfig* cfg) {
    // 判断 T 是否继承了 AbsVisionStrategy<FaceInfo>
    static_assert(std::is_base_of<AbsVisionStrategy<BodyInfo>, T>::value, "Typename T is not inheritance of "
                                                                             "AbsVisionStrategy<BodyInfo> !!!");
    // 判断需要检测手势框的数量
    int checkCount = static_cast<int>(cfg->bodyNeedDetectCount);
    // 执行多手势检测
    std::set<int> gestureIdList;
    auto *body_result = result->getBodyResult();
    for (int i = 0; i < checkCount; ++i) {
        // 检测是否有人脸
        BodyInfo *body = body_result->pBodyInfos[i];
        if (!body->hasBody()) {
            continue;
        }
        // 记录存在的肢体id
        gestureIdList.insert(body->id);
        // 判断该手势框是否存在对应的策略处理类
        auto iter = strategy_map.find(body->id);
        if (iter == strategy_map.end()) {
            T *strategyLogic = T::obtain(cfg);
            strategyLogic->clear();
            strategyLogic->rtConfig = cfg;
            strategyLogic->execute(body);
            strategy_map.insert({body->id, strategyLogic});
        } else {
            iter->second->execute(body);
        }
    }

    for (auto iter = strategy_map.begin(); iter != strategy_map.end();) {
        if (!gestureIdList.count(iter->first)) {
            iter->second->clear();
            T::recycle(iter->second);
            iter = strategy_map.erase(iter);
        } else {
            ++iter;
        }
    }
}

template<typename T>
void execute_living_strategy(VisionResult *result, std::map<int, T*>& strategy_map, RtConfig* cfg) {
    // 判断 T 是否继承了 AbsVisionStrategy<FaceInfo>
    static_assert(std::is_base_of<AbsVisionStrategy<LivingInfo>, T>::value, "Typename T is not inheritance of "
                                                                          "AbsVisionStrategy<LivingInfo> !!!");
    // 判断需要检测手势框的数量
    int checkCount = static_cast<int>(cfg->livingNeedDetectCount);
    // 执行多活体检测
    std::set<int> idList;
    auto *living_result = result->getLivingResult();
    for (int i = 0; i < checkCount; ++i) {
        // 检测是否有人脸
        LivingInfo *living = living_result->livingInfos[i];
        if (!living->hasLiving()) {
            continue;
        }
        // 记录存在的肢体id
        idList.insert(living->id);
        // 判断该手势框是否存在对应的策略处理类
        auto iter = strategy_map.find(living->id);
        if (iter == strategy_map.end()) {
            T *strategyLogic = T::obtain(cfg);
            strategyLogic->clear();
            strategyLogic->rtConfig = cfg;
            strategyLogic->execute(living);
            strategy_map.insert({living->id, strategyLogic});
        } else {
            iter->second->execute(living);
        }
    }

    for (auto iter = strategy_map.begin(); iter != strategy_map.end();) {
        if (!idList.count(iter->first)) {
            iter->second->clear();
            T::recycle(iter->second);
            iter = strategy_map.erase(iter);
        } else {
            ++iter;
        }
    }
}

}

#endif //VISION_ABS_VISION_STRATEGY_H
