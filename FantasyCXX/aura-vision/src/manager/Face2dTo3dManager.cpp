//
// Created by v_bigaoxian on 2020-09-28.
//

#ifdef BUILD_EXPERIMENTAL

#include <algorithm>

#include "Face2dTo3dManager.h"
#include "vision/manager/VisionManagerRegistry.h"

namespace aura::vision {
Face2dto3dManager::Face2dto3dManager() : is_init_model(false) {
    _detector = std::make_shared<Face2Dto3DDetector>();
}

Face2dto3dManager::~Face2dto3dManager() {
    clear();
}

bool Face2dto3dManager::preDetect(VisionRequest *request, VisionResult *result) {
    if (!is_init_model && _detector->init_params() == static_cast<int>(Error::OK)) {
        is_init_model = true;
    }

    VA_CHECK_DETECTED(ABILITY_FACE_2DTO3D);
}

void Face2dto3dManager::doDetect(VisionRequest *request, VisionResult *result) {
    VA_SET_DETECTED(ABILITY_FACE_2DTO3D);

    auto* face_result = result->getFaceResult();
    if (!is_init_model || face_result->noFace() || !_detector->_pdm_model_init_state) {
        return;
    }

    _detector->detect(request, result);
//    _detector->detect(request->get_frame(), result->get_face_result()->_face_infos, result->get_perf_util());
}

void Face2dto3dManager::clear() {

}

void Face2dto3dManager::init(RtConfig* cfg) {
    mRtConfig = cfg;
    _detector->init(cfg);
}

void Face2dto3dManager::deinit() {
    AbsVisionManager::deinit();
    if (detector != nullptr) {
        detector->deinit();
        detector = nullptr;
    }
}

// NOLINTNEXTLINE
REGISTER_VISION_MANAGER("Face2dto3dManager", ABILITY_FACE_2DTO3D,[]() {
    return std::dynamic_pointer_cast<AbsVisionManager>(std::make_shared<Face2dto3dManager>());
});

} // namespace vision

#endif