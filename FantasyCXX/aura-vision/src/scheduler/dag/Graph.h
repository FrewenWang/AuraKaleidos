#pragma once

#include <vector>
#include <algorithm>
#include <util/CountLatch.h>
#include "Node.h"
//#include "vision/manager/vision_manager_registry.h"
#include "vision/core/request/VisionRequest.h"
#include "vision/core/result/VisionResult.h"

namespace aura::vision{
class Node;

class Graph {
public:
  Graph();

  ~Graph();

  bool addNode(Node *root);

  bool removeNode(Node *root);

  void run(VisionRequest *req, VisionResult *res);

  RtConfig *pRtConfig;
  VisionRequest *pRequest;
  VisionResult *pResult;
  /**
   * 技术锁存器
   */
  CountLatch mCounter;

private:
  std::vector<Node *> mNodeRoots;
};
} // namespace vision
