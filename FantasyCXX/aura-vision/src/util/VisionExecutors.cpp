//
// Created by Li,Wendong on 12/18/21.
//

#include "vision/util/VisionExecutors.h"

namespace aura::vision {

int VisionExecutors::maxThreadCount = 2;

VisionExecutors *VisionExecutors::instance() {
    static auto *sInstance = new VisionExecutors();
    return sInstance;
}

VisionExecutors::VisionExecutors() {
    pTaskExecutor = std::make_shared<ThreadPool>(maxThreadCount);
    pTaskExecutor->start();
}

VisionExecutors::~VisionExecutors() {
    if (pTaskExecutor != nullptr) {
        if (pTaskExecutor->isStarted()) {
            pTaskExecutor->stop();
        }
        pTaskExecutor = nullptr;
    }
}

void VisionExecutors::execTask(Runnable *task) {
    pTaskExecutor->execute(task);
}

} // namespace aura::vision

