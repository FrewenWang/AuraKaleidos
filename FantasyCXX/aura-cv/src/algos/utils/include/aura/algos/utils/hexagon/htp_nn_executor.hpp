#ifndef AURA_ALGOS_UTILS_HTP_NN_EXECUTOR_HPP__
#define AURA_ALGOS_UTILS_HTP_NN_EXECUTOR_HPP__

#include "aura/runtime/mat.h"
#include "aura/runtime/nn/nn_engine.hpp"

namespace aura
{

class AURA_EXPORTS HtpNNExecutor
{
public:
    HtpNNExecutor(Context *ctx);

    ~HtpNNExecutor();

    Status Initialize(const Buffer &minn_buffer, const std::string &decrypt_key, const NNConfig &config = NNConfig());

    Status DeInitialize();

    std::shared_ptr<NNExecutor> GetNNExecutor();

    std::vector<TensorDescMap> GetInputs();

    std::vector<TensorDescMap> GetOutputs();

    std::string GetVersion();

private:
    Context *m_ctx;
    std::shared_ptr<NNExecutor> m_nn_executor;
};

} // namespace aura

#endif // AURA_ALGOS_UTILS_HTP_NN_EXECUTOR_HPP__