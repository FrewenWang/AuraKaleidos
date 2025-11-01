#ifndef VISION_DAGSCHEDULER_H
#define VISION_DAGSCHEDULER_H

#include "AbsScheduler.h"
#include <scheduler/dag/Graph.h>
#include <vector>

namespace aura::vision {

class DagScheduler : public AbsScheduler {
public:
    DagScheduler();

    ~DagScheduler() override;

    void run(VisionRequest *request, VisionResult *result) override;

    void injectManagerRegistry(const std::shared_ptr<VisionManagerRegistry> &registry) override;

    void initManagers(RtConfig *cfg) override;

    void initGraph();

private:
    std::vector<Graph *> mGraphs;
};

template<>
inline std::shared_ptr<AbsScheduler> make_scheduler<SCHED_DAG>() {
    return std::dynamic_pointer_cast<AbsScheduler>(std::make_shared<DagScheduler>());
}

}  // namespace vision

#endif //VISION_DAGSCHEDULER_H
