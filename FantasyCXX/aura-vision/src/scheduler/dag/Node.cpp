//
// Created by Li,Wendong on 12/19/21.
//

#include "Node.h"
#include <algorithm>
#include <vision/util/VisionExecutors.h>

using namespace std;

namespace aura::vision {

static const char *TAG = "Node";

Node::Node() : Node(ABILITY_UNKNOWN, nullptr, nullptr) {
}

Node::Node(AbilityId id, Graph *graph) : Node(id, graph, nullptr) {
}

Node::Node(AbilityId id, Graph *graph, Node *prev) {
    initNode(id, graph, prev);
}

Node::~Node() {
    mId = ABILITY_UNKNOWN;
    pGraph = nullptr;
    mNodes.clear();
}

Node *Node::initNode(AbilityId id, Graph *graph) {
    mId = id;
    pGraph = graph;
    mNodes.clear();
    return this;
}

Node *Node::initNode(AbilityId id, Graph *graph, Node *prev) {
    mId = id;
    pGraph = graph;
    mNodes.clear();
    if (prev != nullptr) {
        prev->addNode(this);
    }
    return this;
}

Node *Node::addNode(Node *node) {
    auto it = find(mNodes.begin(), mNodes.end(), node);
    if (it != mNodes.end()) {
        return node;
    }
    mNodes.push_back(node);
    return this;
}

bool Node::removeNode(Node *node) {
    auto it = find(mNodes.begin(), mNodes.end(), node);
    if (it != mNodes.end()) {
        mNodes.erase(it);
    }
    return true;
}

void Node::run() {
    bool doExe = true;
    if (pre != nullptr) {
        doExe = pre(pGraph->pRequest, pGraph->pResult);
    }
    if (doExe) {
        doExe = execute();
    }
    if (post != nullptr) {
        doExe = post(pGraph->pRequest, pGraph->pResult);
    }

    // 每个节点的逻辑
    if (doExe) {
        for (auto node : mNodes) {
            // 执行mCounter的countUp
            pGraph->mCounter.countUp();
            VisionExecutors::instance()->execTask(node);
        }
    }
    pGraph->mCounter.countDown();
}

//bool Node::execute() {
//    return true;
//}

} // namespace vision

