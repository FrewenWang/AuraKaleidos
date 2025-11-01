#include "aura/algos/htp_nn_executor.h"
#include "htp_nn_executor_impl.hpp"

namespace aura
{

#if defined(AURA_ENABLE_NN)
HtpNNExecutor::HtpNNExecutor(Context *ctx) : m_ctx(ctx)
{}

HtpNNExecutor::~HtpNNExecutor()
{
    DeInitialize();
}

Status HtpNNExecutor::Initialize(const Buffer &minn_buffer, const std::string &decrypt_key, const NNConfig &config)
{
    Status ret = Status::ERROR;

    NNEngine *nn_engine = m_ctx->GetNNEngine();
    if (MI_NULL == nn_engine)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "nn_engine is null");
        return ret;
    }

    m_nn_executor = nn_engine->CreateNNExecutor(static_cast<const MI_U8*>(minn_buffer.m_data), minn_buffer.m_size, decrypt_key, config);
    if (MI_NULL == m_nn_executor)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "nn_executor is null\n");
        return ret;
    }

    ret = m_nn_executor->Initialize();
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "nn_executor Initialize failed\n");
    }

    return ret;
}

Status HtpNNExecutor::DeInitialize()
{
    if (m_nn_executor)
    {
        m_nn_executor.reset();
    }

    return Status::OK;
}

std::shared_ptr<NNExecutor> HtpNNExecutor::GetNNExecutor()
{
    return m_nn_executor;
}

std::vector<TensorDescMap> HtpNNExecutor::GetInputs()
{
    std::vector<TensorDescMap> tensor_desc;
    if (m_nn_executor)
    {
        tensor_desc = m_nn_executor->GetInputs();
    }
    return tensor_desc;
}

std::vector<TensorDescMap> HtpNNExecutor::GetOutputs()
{
    std::vector<TensorDescMap> tensor_desc;
    if (m_nn_executor)
    {
        tensor_desc = m_nn_executor->GetOutputs();
    }
    return tensor_desc;
}

std::string HtpNNExecutor::GetVersion()
{
    std::string version;
    if (m_nn_executor)
    {
        version = m_nn_executor->GetVersion();
    }
    return version;
}

template <typename Tp, typename std::enable_if<std::is_same<Tp, TensorDesc>::value>::type* = MI_NULL>
Status Serialize(Context *ctx, HexagonRpcParam *rpc_param, const Tp &tensor_desc)
{
    Status ret = rpc_param->Set(tensor_desc.elem_type, tensor_desc.sizes, tensor_desc.scale, tensor_desc.zero_point, tensor_desc.graph_id);
    AURA_RETURN(ctx, ret);
}

static Status HtpNNExecutorInitializeRpc(Context *ctx, HexagonRpcParam &rpc_param)
{
    Status ret = Status::ERROR;

    Buffer minn_buffer;
    std::string decrypt_key;
    NNConfig config;

    ret = rpc_param.Get(minn_buffer, decrypt_key, config);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "get param failed");
        return ret;
    }

    HtpNNExecutor *htp_nn_excutor = new HtpNNExecutor(ctx);
    if (MI_NULL == htp_nn_excutor)
    {
        AURA_ADD_ERROR_STRING(ctx, "htp_nn_excutor new failed");
        return ret;
    }

    ret = htp_nn_excutor->Initialize(minn_buffer, decrypt_key, config);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "htp_nn_excutor Initialize failed");
        return ret;
    }

    MI_U32 device_addr = reinterpret_cast<MI_U32>(htp_nn_excutor);

    std::vector<TensorDescMap> input_tensor_desc = htp_nn_excutor->GetInputs();
    std::vector<TensorDescMap> output_tensor_desc = htp_nn_excutor->GetOutputs();
    std::string version = htp_nn_excutor->GetVersion();

    rpc_param.ResetBuffer();
    ret = rpc_param.Set(device_addr, input_tensor_desc, output_tensor_desc, version);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "rpc_param Set failed");
    }

    return ret;
}

static Status HtpNNExecutorDeInitializeRpc(Context *ctx, HexagonRpcParam &rpc_param)
{
    Status ret = Status::ERROR;

    MI_U32 device_addr;
    HtpNNExecutor *htp_nn_excutor = MI_NULL;

    ret = rpc_param.Get(device_addr);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "rpc_param Get failed");
        return ret;
    }

    htp_nn_excutor = reinterpret_cast<HtpNNExecutor*>(device_addr);
    if (MI_NULL == htp_nn_excutor)
    {
        return ret;
    }

    delete htp_nn_excutor;
    return ret;
}

static Status HtpNNExecutorTestRunRpc(Context *ctx, HexagonRpcParam &rpc_param)
{
    Status ret = Status::ERROR;

    MI_U32 device_addr;
    HtpNNExecutor *htp_nn_excutor = MI_NULL;
    std::unordered_map<std::string, Mat> input_map;
    std::unordered_map<std::string, Mat> output_map;

    ret = rpc_param.Get(input_map, output_map, device_addr);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "rpc_param Get failed");
        return ret;
    }

    htp_nn_excutor = reinterpret_cast<HtpNNExecutor*>(device_addr);
    if (MI_NULL == htp_nn_excutor)
    {
        AURA_ADD_ERROR_STRING(ctx, "htp_nn_excutor is null");
        return ret;
    }

    std::shared_ptr<NNExecutor> nn_executor = htp_nn_excutor->GetNNExecutor();
    if (MI_NULL == nn_executor)
    {
        AURA_ADD_ERROR_STRING(ctx, "nn_executor is null");
        return ret;
    }

    MatMap input;
    MatMap output;
    for (auto iter = input_map.begin(); iter != input_map.end(); iter++)
    {
        input.insert(std::make_pair(iter->first, &(iter->second)));
    }

    for (auto iter = output_map.begin(); iter != output_map.end(); iter++)
    {
        output.insert(std::make_pair(iter->first, &(iter->second)));
    }

    ret = nn_executor->Forward(input, output);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "Forward failed");
    }

    return ret;
}

AURA_HEXAGON_RPC_FUNC_REGISTER(AURA_HTP_NN_EXCUTOR_PACKAGE_NAME, AURA_HTP_NN_EXCUTOR_INITIALIZE, HtpNNExecutorInitializeRpc);
AURA_HEXAGON_RPC_FUNC_REGISTER(AURA_HTP_NN_EXCUTOR_PACKAGE_NAME, AURA_HTP_NN_EXCUTOR_DEINITIALIZE, HtpNNExecutorDeInitializeRpc);
AURA_HEXAGON_RPC_FUNC_REGISTER(AURA_HTP_NN_EXCUTOR_PACKAGE_NAME, AURA_HTP_NN_EXCUTOR_TEST_RUN, HtpNNExecutorTestRunRpc);

#endif // AURA_ENABLE_NN

} // namespace aura