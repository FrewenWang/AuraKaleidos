/** @brief      : snpe_executor_impl header for aura
 *  @file       : snpe_exrcutol_impl.hpp
 *  @author     : liuguangxin1@xiaomi.com
 *  @version    : 1.0.0
 *  @date       : Jan. 15, 2024
 *  @Copyright  : Copyright 2024 Xiaomi Mobile Software Co., Ltd. All Rights reserved.
 */

#ifndef AURA_RUNTIME_NN_SNPE_EXECUTOR_IMPL_HPP__
#define AURA_RUNTIME_NN_SNPE_EXECUTOR_IMPL_HPP__

#include "snpe_model.hpp"
#include "nn_executor_impl.hpp"

namespace aura
{

AURA_INLINE DT_BOOL GetSnpeUnsignedPd(const NNConfig &config)
{
    DT_BOOL snpe_unsigned_pd = DT_TRUE;

    if (config.count("snpe_unsigned_pd"))
    {
        std::string value = config.at("snpe_unsigned_pd");

        NNStringToLower(value);

        if (value == std::string("false"))
        {
            snpe_unsigned_pd = DT_FALSE;
        }
    }

    return snpe_unsigned_pd;
}

struct SnpeExecutorConfig
{
    SnpeExecutorConfig(const NNConfig &config) : unsigned_pd(DT_TRUE),
                                                 perf_level(NNPerfLevel::PERF_HIGH),
                                                 profiling_level(NNProfilingLevel::PROFILING_OFF),
                                                 log_level(NNLogLevel::LOG_ERROR),
                                                 async_call(DT_TRUE),
                                                 mem_step_size(0),
                                                 profiling_path(std::string())
    {
        perf_level      = GetNNPerfLevel(config);
        profiling_level = GetNNProfilingLevel(config);
        log_level       = GetNNLogLevel(config);
        async_call      = GetHtpAsyncCall(config);
        profiling_path  = GetNNProfilingPath(config);
        backend         = GetNNBackend(config);
        mem_step_size   = GetHtpMemStepSize(config);
        unsigned_pd     = GetSnpeUnsignedPd(config);
    }

    DT_BOOL unsigned_pd;
    NNPerfLevel perf_level;
    NNProfilingLevel profiling_level;
    NNLogLevel log_level;
    DT_BOOL async_call;
    DT_S32 mem_step_size;
    std::string profiling_path;
    NNBackend backend;
};

class SnpeExecutor : public NNExecutorInterface
{
public:
    SnpeExecutor(Context *ctx, const std::shared_ptr<SnpeModel> model, const NNConfig &config);
};

class SnpeExecutorImpl : public NNExecutorImpl
{
public:
    SnpeExecutorImpl(Context *ctx, const std::shared_ptr<SnpeModel> &model, const NNConfig &config)
                     : NNExecutorImpl(ctx), m_model(model), m_config(config)
    {}

protected:
    std::shared_ptr<SnpeModel> m_model;
    SnpeExecutorConfig m_config;
};

using SnpeExecutorImplCreator = SnpeExecutorImpl*(*)(Context*, const std::shared_ptr<SnpeModel>&, const NNConfig&);

template<typename Tp, typename std::enable_if<std::is_base_of<SnpeExecutorImpl, Tp>::value>::type* = DT_NULL>
SnpeExecutorImpl* SnpeExecutorImplHelper(Context *ctx, const std::shared_ptr<SnpeModel> &model, const NNConfig &config)
{
    return new Tp(ctx, model, config);
}

class SnpeExecutorImplRegister
{
public:
    SnpeExecutorImplRegister(const std::string &name, SnpeExecutorImplCreator creator);

    Status Register();
};

} // namespace aura

#endif // AURA_RUNTIME_NN_SNPE_EXECUTOR_IMPL_HPP__