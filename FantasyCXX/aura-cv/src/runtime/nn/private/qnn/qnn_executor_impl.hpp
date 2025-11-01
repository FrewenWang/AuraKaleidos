
#ifndef AURA_RUNTIME_NN_QNN_EXECUTOR_IMPL_HPP__
#define AURA_RUNTIME_NN_QNN_EXECUTOR_IMPL_HPP__

#include "qnn_model.hpp"
#include "nn_executor_impl.hpp"

namespace aura
{

AURA_INLINE MI_S32 GetQnnBudget(const NNConfig &config)
{
    if (config.count("qnn_budget"))
    {
        std::string value = config.at("qnn_budget");

        return std::stoi(value);
    }

    return 0;
}

AURA_INLINE std::vector<MI_S8> GetQnnGraphIds(const NNConfig &config)
{
    std::vector<MI_S8> qnn_graph_ids;

    if (config.count("qnn_graph_ids"))
    {
        std::string value = config.at("qnn_graph_ids");

        std::vector<std::string> graph_ids_strs = NNSplit(value, ';');

        for (auto &graph_id : graph_ids_strs)
        {
            qnn_graph_ids.push_back(std::stoi(graph_id));
        }
    }

    if (qnn_graph_ids.size() == 0)
    {
        qnn_graph_ids.push_back(-1);
    }

    return qnn_graph_ids;
}

AURA_INLINE std::string GetQnnUdoPath(const NNConfig &config)
{
    std::string qnn_udo_path = std::string();

    if (config.count("qnn_udo_path"))
    {
        qnn_udo_path = config.at("qnn_udo_path");
    }

    return qnn_udo_path;
}

AURA_INLINE NNPerfLevel GetQnnHmxPerfLevel(const NNConfig &config)
{
    NNPerfLevel hmx_perf_level = NNPerfLevel::PERF_HIGH;

    if (config.count("qnn_hmx_perf_level"))
    {
        std::string value = config.at("qnn_hmx_perf_level");

        NNStringToLower(value);

        if (std::string("perf_high") == value)
        {
            hmx_perf_level = NNPerfLevel::PERF_HIGH;
        }
        else if (std::string("perf_normal") == value)
        {
            hmx_perf_level = NNPerfLevel::PERF_NORMAL;
        }
        else if (std::string("perf_low") == value)
        {
            hmx_perf_level = NNPerfLevel::PERF_LOW;
        }
    }

    return hmx_perf_level;
}

AURA_INLINE Status GetQnnHmxPerfLevel(const Context *ctx, const AnyParams &params, NNPerfLevel &hmx_perf_level)
{
    hmx_perf_level = NNPerfLevel::PERF_HIGH;

    std::string value;
    Status status = params.Get("qnn_hmx_perf_level", value);
    if (status != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "AnyParams get failed");
        return status;
    }

    NNStringToLower(value);

    if (std::string("perf_high") == value)
    {
        hmx_perf_level = NNPerfLevel::PERF_HIGH;
    }
    else if (std::string("perf_normal") == value)
    {
        hmx_perf_level = NNPerfLevel::PERF_NORMAL;
    }
    else if (std::string("perf_low") == value)
    {
        hmx_perf_level = NNPerfLevel::PERF_LOW;
    }
    else
    {
        std::string info = "qnn_hmx_perf_level set wrong value " + value;
        AURA_ADD_ERROR_STRING(ctx, info.c_str());
        return Status::ERROR;
    }

    return Status::OK;
}

struct QnnExecutorConfig
{
    QnnExecutorConfig(const NNConfig &config) : htp_perf_level(NNPerfLevel::PERF_HIGH),
                                                hmx_perf_level(NNPerfLevel::PERF_HIGH),
                                                profiling_level(NNProfilingLevel::PROFILING_OFF),
                                                log_level(NNLogLevel::LOG_ERROR),
                                                async_call(MI_TRUE),
                                                profiling_path(std::string()),
                                                mem_step_size(0),
                                                budget(0),
                                                graph_ids(std::vector<MI_S8>(1, -1)),
                                                udo_path(std::string())
    {
        htp_perf_level  = GetNNPerfLevel(config);
        hmx_perf_level  = GetQnnHmxPerfLevel(config);
        profiling_level = GetNNProfilingLevel(config);
        log_level       = GetNNLogLevel(config);
        async_call      = GetHtpAsyncCall(config);
        profiling_path  = GetNNProfilingPath(config);
        mem_step_size   = GetHtpMemStepSize(config);
        budget          = GetQnnBudget(config);
        graph_ids       = GetQnnGraphIds(config);
        udo_path        = GetQnnUdoPath(config);
    }

    NNPerfLevel htp_perf_level;
    NNPerfLevel hmx_perf_level;
    NNProfilingLevel profiling_level;
    NNLogLevel log_level;
    MI_BOOL async_call;
    std::string profiling_path;
    MI_S32 mem_step_size;
    MI_S32 budget;
    std::vector<MI_S8> graph_ids;
    std::string udo_path;
};

class QnnExecutor : public NNExecutorInterface
{
public:
    QnnExecutor(Context *ctx, const std::shared_ptr<QnnModel> model, const NNConfig &config);
};

class QnnExecutorImpl : public NNExecutorImpl
{
public:
    QnnExecutorImpl(Context *ctx, const std::shared_ptr<QnnModel> &model, const NNConfig &config)
                    : NNExecutorImpl(ctx), m_model(model), m_config(config)
    {}

protected:
    std::shared_ptr<QnnModel> m_model;
    QnnExecutorConfig m_config;
};

using QnnExecutorImplCreator = QnnExecutorImpl* (*)(Context*, const std::shared_ptr<QnnModel>&, const NNConfig&);

template<typename Tp, typename std::enable_if<std::is_base_of<QnnExecutorImpl, Tp>::value>::type* = MI_NULL>
QnnExecutorImpl* QnnExecutorImplHelper(Context *ctx, const std::shared_ptr<QnnModel> &model, const NNConfig &config)
{
    return new Tp(ctx, model, config);
}

class QnnExecutorImplRegister
{
public:
    QnnExecutorImplRegister(const std::string &name, QnnExecutorImplCreator creator);

    Status Register();
};

} // namespace aura

#endif // AURA_RUNTIME_NN_QNN_EXECUTOR_IMPL_HPP__