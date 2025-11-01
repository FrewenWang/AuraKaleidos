//
// Created by Li,Wendong on 12/18/21.
//

#pragma once

#include "ThreadPool.h"

namespace aura::vision {

class VisionExecutors {

public:
    static int maxThreadCount;
    static VisionExecutors *instance();

    VisionExecutors();

    ~VisionExecutors();

    void execTask(Runnable *task);

private:
    std::shared_ptr<ThreadPool> pTaskExecutor = nullptr;

};

} // namespace vision