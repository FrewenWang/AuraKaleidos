#include "Graph.h"
#include <vision/util/VisionExecutors.h>

namespace aura::vision {

static const char *TAG = "DagGraph";

Graph::Graph() {
    mNodeRoots.clear();
    pRequest = nullptr;
    pResult = nullptr;
    pRtConfig = nullptr;
}

Graph::~Graph() {
    // TODO: Stop nodes

    mNodeRoots.clear();
    pRequest = nullptr;
    pResult = nullptr;
    pRtConfig = nullptr;
}

bool Graph::addNode(Node *root) {
    auto it = find(mNodeRoots.begin(), mNodeRoots.end(), root);
    if (it != mNodeRoots.end()) {
        return false;
    }
    mNodeRoots.push_back(root);
    return true;
}

bool Graph::removeNode(Node *root) {
    auto it = find(mNodeRoots.begin(), mNodeRoots.end(), root);
    if (it != mNodeRoots.end()) {
        mNodeRoots.erase(it);
    }
    return true;
}

void Graph::run(VisionRequest *request, VisionResult *result) {
    pRequest = request;
    pResult = result;

    if (!pRtConfig) {
        VLOGE(TAG, "Config is not initialized!");
        return;
    }

    /// 遍历所有的根据点
    for (const auto it : mNodeRoots) {
        // 每当开始执行一个耗时任务的时候。我们就进行执行计数器的countUp
        mCounter.countUp();
        VisionExecutors::instance()->execTask(it);
    }
    mCounter.wait();
}

} // namespace vision

