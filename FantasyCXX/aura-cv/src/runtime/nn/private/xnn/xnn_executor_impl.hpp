/** @brief      : xnn_executor_impl header for aura
 *  @file       : xnn_exrcutol_impl.hpp
 *  @author     : jiyingyu@xiaomi.com
 *  @version    : 1.0.0
 *  @date       : Mar. 29, 2024
 *  @Copyright  : Copyright 2024 Xiaomi Mobile Software Co., Ltd. All Rights reserved.
 */

#ifndef AURA_RUNTIME_NN_XNN_EXECUTOR_IMPL_HPP__
#define AURA_RUNTIME_NN_XNN_EXECUTOR_IMPL_HPP__

#include "xnn_model.hpp"
#include "nn_executor_impl.hpp"

namespace aura
{

struct XnnExecutorConfig
{
    XnnExecutorConfig(const NNConfig &config) : perf_level(NNPerfLevel::PERF_HIGH),
                                                log_level(NNLogLevel::LOG_ERROR)
    {
        perf_level = GetNNPerfLevel(config);
        log_level  = GetNNLogLevel(config);
    }

    NNPerfLevel perf_level;
    NNLogLevel log_level;
};

class XnnExecutor : public NNExecutorInterface
{
public:
    XnnExecutor(Context *ctx, const std::shared_ptr<XnnModel> model, const NNConfig &config);
};

class XnnExecutorImpl : public NNExecutorImpl
{
public:
    XnnExecutorImpl(Context *ctx, const std::shared_ptr<XnnModel> &model, const NNConfig &config)
                     : NNExecutorImpl(ctx), m_model(model), m_config(config)
    {}

protected:
    std::shared_ptr<XnnModel> m_model;
    XnnExecutorConfig m_config;
};

using XnnExecutorImplCreator = XnnExecutorImpl*(*)(Context*, const std::shared_ptr<XnnModel>&, const NNConfig&);

template<typename Tp, typename std::enable_if<std::is_base_of<XnnExecutorImpl, Tp>::value>::type* = DT_NULL>
XnnExecutorImpl* XnnExecutorImplHelper(Context *ctx, const std::shared_ptr<XnnModel> &model, const NNConfig &config)
{
    return new Tp(ctx, model, config);
}

class XnnExecutorImplRegister
{
public:
    XnnExecutorImplRegister(const std::string &name, XnnExecutorImplCreator creator);

    Status Register();
};

} // namespace aura

#endif // AURA_RUNTIME_NN_XNN_EXECUTOR_IMPL_HPP__