#ifndef AURA_ALGOS_CORE_ALGO_HPP__
#define AURA_ALGOS_CORE_ALGO_HPP__

#include "aura/algos/core/graph.hpp"
#include "aura/ops/core.h"

namespace aura
{

class AURA_EXPORTS AlgoImpl : public OpImpl
{
public:
    AlgoImpl(Context *ctx, const std::string &name) : OpImpl(ctx, name, OpTarget()), m_graph(ctx)
    {}

    OpTarget GetOpTarget() const = delete;

    Graph* GetGraph()
    {
        return &m_graph;
    }

    const Graph* GetGraph() const
    {
        return &m_graph;
    }

protected:
    Graph m_graph;
};

class AURA_EXPORTS Algo : public Op
{
public:
    Algo(Context *ctx) : Op(ctx)
    {}

    OpTarget GetOpTarget() const = delete;

    Graph* GetGraph()
    {
        if (m_ctx && m_impl)
        {
            return dynamic_cast<AlgoImpl*>(m_impl.get())->GetGraph();
        }
        else
        {
            return MI_NULL;
        }
    }

    const Graph* GetGraph() const
    {
        if (m_ctx && m_impl)
        {
            return dynamic_cast<AlgoImpl*>(m_impl.get())->GetGraph();
        }
        else
        {
            return MI_NULL;
        }
    }
};

} // namespace aura

#endif // AURA_ALGOS_CORE_ALGO_HPP__