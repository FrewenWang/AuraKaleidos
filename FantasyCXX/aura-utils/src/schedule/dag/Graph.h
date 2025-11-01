#pragma once

#include <vector>
#include <algorithm>
#include <util/CountLatch.h>
#include "Node.h"
#include "core/NodeReq.h"
#include "core/NodeRes.h"

namespace aura::utils{
class Node;

class Graph {
public:
  Graph();

  ~Graph();

  bool addNode(Node *root);

  bool removeNode(const Node *root);

  void run(NodeReq *req, NodeRes *res);

  NodeReq *pRequest;
  NodeRes *pResult;
  /**
   * 技术锁存器
   */
  CountLatch mCounter;

private:
  std::vector<Node *> mNodeRoots;
};
} // namespace vision
