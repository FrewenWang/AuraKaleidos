//
// Created by Li,Wendong on 12/19/21.
//

#include "Node.h"
#include <algorithm>

using namespace std;

namespace aura::utils {

static const char *TAG = "Node";

Node::Node() : Node(0, nullptr, nullptr) {
}

Node::Node(int id, Graph *graph) : Node(id, graph, nullptr) {
}

Node::Node(int id, Graph *graph, Node *prev) {
    initNode(id, graph, prev);
}

Node::~Node() {
    mId = 0;
    pGraph = nullptr;
    mNodes.clear();
}

Node *Node::initNode(int id, Graph *graph) {
    mId = id;
    pGraph = graph;
    mNodes.clear();
    return this;
}

Node *Node::initNode(int id, Graph *graph, Node *prev) {
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
    if (doExe) {
        for (auto node : mNodes) {
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

