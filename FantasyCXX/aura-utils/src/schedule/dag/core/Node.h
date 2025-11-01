#pragma once

#include <functional>
#include "Runnable.h"
#include "NodeReq.h"
#include "NodeRes.h"

namespace aura::utils {

class Graph;

class Node : public Runnable {
public:
    Node();

    Node(int id, Graph *graph);

    Node(int id, Graph *graph, Node *prev);

    ~Node();

    Node *initNode(int id, Graph *graph);

    Node *initNode(int id, Graph *graph, Node *prev);

    Node *addNode(Node *node);

    bool removeNode(Node *node);

    void run() override;

    std::function<bool(NodeReq* request, NodeReq* result)> pre;
    std::function<bool(NodeReq* request, NodeReq* result)> post;

protected:
    virtual bool execute() = 0;

    Graph *pGraph;
    int mId;
    std::vector<Node *> mNodes;
};

}