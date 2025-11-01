#include "Graph.h"

namespace aura::utils {

static const char *TAG = "DagGraph";

Graph::Graph() {
    mNodeRoots.clear();
    pRequest = nullptr;
    pResult = nullptr;
}

Graph::~Graph() {
    mNodeRoots.clear();
    pRequest = nullptr;
    pResult = nullptr;
}

bool Graph::addNode(Node *root) {
    const auto it = find(mNodeRoots.begin(), mNodeRoots.end(), root);
    if (it != mNodeRoots.end()) {
        return false;
    }
    mNodeRoots.push_back(root);
    return true;
}

bool Graph::removeNode(const Node *root) {
    const auto it = find(mNodeRoots.begin(), mNodeRoots.end(), root);
    if (it != mNodeRoots.end()) {
        mNodeRoots.erase(it);
    }
    return true;
}

void Graph::run(NodeReq *request, NodeRes *result) {
    pRequest = request;
    pResult = result;
    for (auto it : mNodeRoots) {
        mCounter.countUp();
        Executors::instance()->execTask(it);
    }
    mCounter.wait();
}

} // namespace vision

