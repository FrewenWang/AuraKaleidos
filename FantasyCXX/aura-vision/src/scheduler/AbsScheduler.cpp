#include <util/InferenceConverter.hpp>
#include "AbsScheduler.h"

#include "vision/core/common/VMacro.h"
#include "vision/manager/VisionManagerRegistry.h"

namespace aura::vision {

static const char *TAG = "AbsScheduler";

void AbsScheduler::registerManagers() {
    if (!pMgrRegistry) {
        VLOGE(TAG, "VisionManagerRegistry is nullptr!");
        return;
    }
    if(pMgrRegistry->CreateFromCfg() != 0) {
        VLOGE(TAG, "register managers from config FAILED!");
        return;
    }
}

void AbsScheduler::runDetection(AbilityId id, VisionRequest* request, VisionResult* result) {
    if (!pMgrRegistry) {
        VLOGE(TAG, "VisionManagerRegistry is nullptr!");
        return;
    }

    if (!mRtConfig) {
        VLOGE(TAG, "Config is not initialized!");
        return;
    }

    if (mRtConfig->get_switch(id)) {
        auto manager = pMgrRegistry->getManager(id);
        if (!manager) {
            result->errorCode = V_TO_SHORT(Error::MANAGER_ERR);
            return;
        }
        manager->detect(request, result);
        return;
    }
}

void AbsScheduler::runDetectionIF(AbilityId id, bool cond, VisionRequest* request, VisionResult* result) {
    if (!pMgrRegistry) {
        VLOGE(TAG, "VisionManagerRegistry is nullptr!");
        return;
    }

    if (!mRtConfig) {
        VLOGE(TAG, "Config is not initialized!");
        return;
    }

    if (cond && mRtConfig->get_switch(id)) {
        auto manager = pMgrRegistry->getManager(id);
        if (!manager) {
            result->errorCode = V_TO_SHORT(Error::MANAGER_ERR);
            return;
        }
        manager->detect(request, result);
        return;
    }
    result->errorCode = V_TO_SHORT(Error::ERR_DISABLED);
}

void AbsScheduler::runDetectionForcibly(AbilityId id, VisionRequest* request, VisionResult* result) {
    if (!pMgrRegistry) {
        VLOGE(TAG, "VisionManagerRegistry is nullptr!");
        return;
    }

    auto manager = pMgrRegistry->getManager(id);
    if (!manager) {
        result->errorCode = V_TO_SHORT(Error::MANAGER_ERR);
        return;
    }
    manager->detect(request, result);
}

void AbsScheduler::runDetectionForciblyIF(AbilityId id, bool cond, VisionRequest* request, VisionResult* result) {
    if (!cond) {
        return;
    }

    if (!pMgrRegistry) {
        VLOGE(TAG, "VisionManagerRegistry is nullptr!");
        return;
    }

    auto manager = pMgrRegistry->getManager(id);
    if (!manager) {
        result->errorCode = V_TO_SHORT(Error::MANAGER_ERR);
        return;
    }
    manager->detect(request, result);
}

void AbsScheduler::runDetectionAndForciblyRely(AbilityId id, AbilityId rely_id, VisionRequest* request, VisionResult* result) {
    if (!mRtConfig) {
        VLOGE(TAG, "Config is not initialized!");
        return;
    }

    if (!mRtConfig->get_switch(id)) {
        result->errorCode = V_TO_SHORT(Error::ERR_DISABLED);
        return;
    }

    if (!VA_GET_DETECTED(rely_id)) {
        runDetectionForcibly(rely_id, request, result);
    }

    if (!pMgrRegistry) {
        VLOGE(TAG, "VisionManagerRegistry is nullptr!");
        return;
    }

    auto manager = pMgrRegistry->getManager(id);
    if (!manager) {
        result->errorCode = V_TO_SHORT(Error::MANAGER_ERR);
        return;
    }

    manager->detect(request, result);
}

bool AbsScheduler::checkFunctionSwitchAndState(AbilityId id, bool state) {
    if (!mRtConfig) {
        VLOGE(TAG, "Config is not initialized!");
        return false;
    }
    if (mRtConfig->get_switch(id)) {
        return state;
    }
    return true;
}

void AbsScheduler::injectManagerRegistry(const std::shared_ptr<VisionManagerRegistry>& registry) {
    pMgrRegistry = registry;
}

void AbsScheduler::initManagers(RtConfig* cfg) {
    mRtConfig = cfg;
    for (auto& manager : pMgrRegistry->getManagers()) {
        manager->init(cfg);
    }
}

void AbsScheduler::deinit() {
    if (pMgrRegistry != nullptr) {
        for (auto& manager : pMgrRegistry->getManagers()) {
            manager->deinit();
        }
        pMgrRegistry->clear();
    }
}

std::shared_ptr<VisionManagerRegistry> AbsScheduler::getManagerRegistry() {
    return pMgrRegistry;
}

} // namespace vision