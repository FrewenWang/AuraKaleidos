#ifndef AURA_RUNTIME_NN_NP_EXECUTOR_IMPL_HPP__
#define AURA_RUNTIME_NN_NP_EXECUTOR_IMPL_HPP__

#include "np_model.hpp"
#include "nn_executor_impl.hpp"

namespace aura
{

struct NpExecutorConfig
{
    NpExecutorConfig(const NNConfig &config) : perf_level(NNPerfLevel::PERF_DEFAULT)
    {
        perf_level = GetNNPerfLevel(config);
    }

    NNPerfLevel perf_level;
};

class NpExecutor : public NNExecutorInterface
{
public:
    NpExecutor(Context *ctx, const std::shared_ptr<NpModel> model, const NNConfig &config);
};

class NpExecutorImpl : public NNExecutorImpl
{
public:
    NpExecutorImpl(Context *ctx, const std::shared_ptr<NpModel> &model, const NNConfig &config)
                   : NNExecutorImpl(ctx), m_model(model), m_config(config)
    {}

protected:
    std::shared_ptr<NpModel> m_model;
    NpExecutorConfig m_config;
};

using NpExecutorImplCreator = NpExecutorImpl*(*)(Context*, const std::shared_ptr<NpModel>&, const NNConfig&);

template<typename Tp, typename std::enable_if<std::is_base_of<NpExecutorImpl, Tp>::value>::type* = DT_NULL>
NpExecutorImpl* NpExecutorImplHelper(Context *ctx, const std::shared_ptr<NpModel> &model, const NNConfig &config)
{
    return new Tp(ctx, model, config);
}

class NpExecutorImplRegister
{
public:
    NpExecutorImplRegister(const std::string &name, NpExecutorImplCreator creator);

    Status Register();
};

} // namespace aura

#endif // AURA_RUNTIME_NN_NP_EXECUTOR_IMPL_HPP__