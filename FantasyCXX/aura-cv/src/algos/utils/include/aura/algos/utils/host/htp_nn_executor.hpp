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

    Status Initialize(const std::string &minn_file, const std::string &decrypt_key, const NNConfig &config = NNConfig());

    Status DeInitialize();

    MI_U32 GetDeviceAddr();

    std::vector<TensorDescMap> GetInputs();

    std::vector<TensorDescMap> GetOutputs();

    std::string GetVersion();
private:
    Buffer ReadMinnFile(const std::string &minn_file);

private:
    Context *m_ctx;
    MI_U32 m_device_addr;
    std::vector<TensorDescMap> m_input_tensor_desc;
    std::vector<TensorDescMap> m_output_tensor_desc;
    std::string m_version;
};

} // namespace aura

#endif // AURA_ALGOS_UTILS_HTP_NN_EXECUTOR_HPP__