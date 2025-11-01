#ifndef AURA_RUNTIME_NN_MNN_EXECUTOR_IMPL_HPP__
#define AURA_RUNTIME_NN_MNN_EXECUTOR_IMPL_HPP__

#include "mnn_model.hpp"
#include "nn_executor_impl.hpp"

namespace aura
{

AURA_INLINE MI_S32 GetMnnDumplayer(const NNConfig &config)
{
    MI_S32 mnn_dump_layers = MI_FALSE;

    if (config.count("mnn_dump_layers"))
    {
        std::string value = config.at("mnn_dump_layers");

        NNStringToLower(value);

        if (value == std::string("true"))
        {
            mnn_dump_layers = MI_TRUE;
        }
    }

    return mnn_dump_layers;
}

AURA_INLINE MnnPrecision GetMnnPrecision(const NNConfig &config)
{
    MnnPrecision mnn_precision = MnnPrecision::PRECISION_NORMAL;

    if (config.count("mnn_precision"))
    {
        std::string value = config.at("mnn_precision");

        NNStringToLower(value);

        if (value == std::string("precision_normal"))
        {
            mnn_precision = MnnPrecision::PRECISION_NORMAL;
        }
        else if (value == std::string("precision_high"))
        {
            mnn_precision = MnnPrecision::PRECISION_HIGH;
        }
        else if (value == std::string("precision_low"))
        {
            mnn_precision = MnnPrecision::PRECISION_LOW;
        }
    }

    return mnn_precision;
}

AURA_INLINE MnnMemory GetMnnMemory(const NNConfig &config)
{
    MnnMemory mnn_memory = MnnMemory::MEMORY_NORMAL;

    if (config.count("mnn_memory"))
    {
        std::string value = config.at("mnn_memory");

        NNStringToLower(value);

        if (value == std::string("memory_normal"))
        {
            mnn_memory = MnnMemory::MEMORY_NORMAL;
        }
        else if (value == std::string("memory_high"))
        {
            mnn_memory = MnnMemory::MEMORY_HIGH;
        }
        else if (value == std::string("memory_low"))
        {
            mnn_memory = MnnMemory::MEMORY_LOW;
        }
    }

    return mnn_memory;
}

AURA_INLINE MnnTuning GetMnnTuning(const NNConfig &config)
{
    MnnTuning mnn_tuning = MnnTuning::GPU_TUNING_NONE;

    if (config.count("mnn_tuning"))
    {
        std::string value = config.at("mnn_tuning");

        NNStringToLower(value);

        if (value == std::string("gpu_tuning_none"))
        {
            mnn_tuning = MnnTuning::GPU_TUNING_NONE;
        }
        else if (value == std::string("gpu_tuning_heavy"))
        {
            mnn_tuning = MnnTuning::GPU_TUNING_HEAVY;
        }
        else if (value == std::string("gpu_tuning_wide"))
        {
            mnn_tuning = MnnTuning::GPU_TUNING_WIDE;
        }
        else if (value == std::string("gpu_tuning_normal"))
        {
            mnn_tuning = MnnTuning::GPU_TUNING_NORMAL;
        }
        else if (value == std::string("gpu_tuning_fast"))
        {
            mnn_tuning = MnnTuning::GPU_TUNING_FAST;
        }
    }

    return mnn_tuning;
}

AURA_INLINE MnnCLMem GetMnnClmem(const NNConfig &config)
{
    MnnCLMem mnn_clmem = MnnCLMem::GPU_MEMORY_NONE;

    if (config.count("mnn_clmem"))
    {
        std::string value = config.at("mnn_clmem");

        NNStringToLower(value);

        if (value == std::string("gpu_memory_none"))
        {
            mnn_clmem = MnnCLMem::GPU_MEMORY_NONE;
        }
        else if (value == std::string("gpu_memory_buffer"))
        {
            mnn_clmem = MnnCLMem::GPU_MEMORY_BUFFER;
        }
        else if (value == std::string("gpu_memory_iaura"))
        {
            mnn_clmem = MnnCLMem::GPU_MEMORY_IAURA;
        }
    }

    return mnn_clmem;
}

struct MnnExecutorConfig
{
    MnnExecutorConfig(const NNConfig &config) : backend(NNBackend::CPU),
                                                profiling_path(std::string()),
                                                dump_layers(MI_FALSE),
                                                precision(MnnPrecision::PRECISION_NORMAL),
                                                memory(MnnMemory::MEMORY_NORMAL),
                                                tuning(MnnTuning::GPU_TUNING_NONE),
                                                cl_mem(MnnCLMem::GPU_MEMORY_NONE)
    {
        profiling_path = GetNNProfilingPath(config);
        dump_layers    = GetMnnDumplayer(config);
        precision      = GetMnnPrecision(config);
        memory         = GetMnnMemory(config);
        tuning         = GetMnnTuning(config);
        cl_mem         = GetMnnClmem(config);

        backend = GetNNBackend(config);
        if (NNBackend::NPU == backend)
        {
            backend = NNBackend::CPU;
        }
    }

    NNBackend backend;
    std::string profiling_path;
    MI_S32 dump_layers;
    MnnPrecision precision;
    MnnMemory memory;
    MnnTuning tuning;
    MnnCLMem cl_mem;
};

class MnnExecutor : public NNExecutorInterface
{
public:
    MnnExecutor(Context *ctx, std::shared_ptr<MnnModel> mnn_model, const NNConfig &config);
};

class MnnExecutorImpl : public NNExecutorImpl
{
public:
    MnnExecutorImpl(Context *ctx, const std::shared_ptr<MnnModel> &model, const NNConfig &config)
                    : NNExecutorImpl(ctx), m_model(model), m_config(config)
    {}

protected:
    std::shared_ptr<MnnModel> m_model;
    MnnExecutorConfig m_config;
};

using MnnExecutorImplCreator = MnnExecutorImpl*(*)(Context*, const std::shared_ptr<MnnModel>&, const NNConfig&);

template<typename Tp, typename std::enable_if<std::is_base_of<MnnExecutorImpl, Tp>::value>::type* = MI_NULL>
MnnExecutorImpl* MnnExecutorImplHelper(Context *ctx, const std::shared_ptr<MnnModel> &model, const NNConfig &config)
{
    return new Tp(ctx, model, config);
}

class MnnExecutorImplRegister
{
public:
    MnnExecutorImplRegister(const std::string &name, MnnExecutorImplCreator creator);

    Status Register();
};

} // namespace aura

#endif // AURA_RUNTIME_NN_MNN_EXECUTOR_IMPL_HPP__