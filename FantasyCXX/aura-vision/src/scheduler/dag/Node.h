//
// Created by Li,Wendong on 12/19/21.
//

#pragma once

#include <vision/util/Runnable.h>
#include <vector>
#include <functional>
#include <vision/core/common/VConstants.h>
#include "Graph.h"
#include "vision/core/request/VisionRequest.h"
#include "vision/core/result/VisionResult.h"

namespace aura::vision {

class Graph;

class Node : public Runnable {
public:
    Node();

    Node(AbilityId id, Graph *graph);

    Node(AbilityId id, Graph *graph, Node *prev);

    ~Node();

    Node *initNode(AbilityId id, Graph *graph);

    Node *initNode(AbilityId id, Graph *graph, Node *prev);

    Node *addNode(Node *node);

    bool removeNode(Node *node);

    void run() override;

    std::function<bool(VisionRequest* request, VisionResult* result)> pre;
    std::function<bool(VisionRequest* request, VisionResult* result)> post;

protected:
    virtual bool execute() = 0;

    Graph *pGraph;
    AbilityId mId;
    std::vector<Node *> mNodes;
};

} // namespace vision