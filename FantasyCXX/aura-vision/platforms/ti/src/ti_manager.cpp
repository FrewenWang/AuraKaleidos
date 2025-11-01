//
// Created by Li,Wendong on 2019-05-11.
//

#include "ti_manager.h"

namespace vision {

    TiManager *TiManager::_s_instance = NULL;

    TiManager *TiManager::instance() {
        if (_s_instance == NULL) {
            _s_instance = new TiManager();
        }
        return _s_instance;
    }

    TiManager::TiManager() {
    }

    TiManager::~TiManager() {
    }

    void TiManager::detect(VisionRequest *request, VisionResult *result) {

    }

} // namespace vision {
