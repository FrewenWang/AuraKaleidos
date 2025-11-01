#include "aura/algos/htp_nn_executor.h"
#include "htp_nn_executor_impl.hpp"

namespace aura
{

#if (defined(AURA_ENABLE_NN) && defined(AURA_ENABLE_HEXAGON))
HtpNNExecutor::HtpNNExecutor(Context *ctx) : m_ctx(ctx), m_device_addr(0)
{}

HtpNNExecutor::~HtpNNExecutor()
{
    DeInitialize();
}

Buffer HtpNNExecutor::ReadMinnFile(const std::string &minn_file)
{
    Buffer minn_buffer;
    if (minn_file.empty())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "minn_file is empty");
        return minn_buffer;
    }

    MI_U64 file_length = 0;
    size_t bytes = 0;

    FILE *fp = fopen(minn_file.c_str(), "rb");
    if (MI_NULL == fp)
    {
        std::string info = "open model: " + minn_file + " failed";
        AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
        goto EXIT;
    }

    fseek(fp, 0, SEEK_END);
    file_length = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    minn_buffer = m_ctx->GetMemPool()->GetBuffer(AURA_ALLOC_PARAM(m_ctx, AURA_MEM_DEFAULT, file_length, 0));
    if (!minn_buffer.IsValid())
    {
        std::string info = "alloc " + std::to_string(file_length) + " failed";
        AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
        goto EXIT;
    }

    bytes = fread(minn_buffer.m_data, 1, file_length, fp);
    if (bytes != file_length)
    {
        std::string info = "minn_file " +  minn_file + " need fread " + std::to_string(file_length) + ", but actual only fread " + std::to_string(bytes);
        AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
        goto EXIT;
    }

EXIT:
    if (fp)
    {
        fclose(fp);
    }
    return minn_buffer;
}

template <typename Tp, typename std::enable_if<std::is_same<Tp, TensorDesc>::value>::type* = MI_NULL>
Status Deserialize(Context *ctx, HexagonRpcParam *rpc_param, Tp &tensor_desc)
{
    Status ret = rpc_param->Get(tensor_desc.elem_type, tensor_desc.sizes, tensor_desc.scale, tensor_desc.zero_point, tensor_desc.graph_id);
    AURA_RETURN(ctx, ret);
}

Status HtpNNExecutor::Initialize(const std::string &minn_file, const std::string &decrypt_key, const NNConfig &config)
{
    Status ret = Status::ERROR;

    Buffer minn_buffer = ReadMinnFile(minn_file);
    if (!minn_buffer.IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "ReadMinnFile failed");
        return ret;
    }

    HexagonEngine *engine = MI_NULL;
    HexagonRpcParam rpc_param(m_ctx, 4096);

    rpc_param.ResetBuffer();
    ret = rpc_param.Set(minn_buffer, decrypt_key, config);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Set failed");
        goto EXIT;
    }

    engine = m_ctx->GetHexagonEngine();
    if (MI_NULL == engine)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "GetHexagonEngine failed");
        goto EXIT;
    }

    ret = engine->Run(AURA_HTP_NN_EXCUTOR_PACKAGE_NAME, AURA_HTP_NN_EXCUTOR_INITIALIZE, rpc_param);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Run failed");
        goto EXIT;
    }

    ret = rpc_param.Get(m_device_addr, m_input_tensor_desc, m_output_tensor_desc, m_version);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Get failed");
    }

EXIT:
    AURA_FREE(m_ctx, minn_buffer.m_data);
    return ret;
}

Status HtpNNExecutor::DeInitialize()
{
    Status ret = Status::ERROR;

    if (0 == m_device_addr)
    {
        return Status::OK;
    }

    HexagonEngine *engine = m_ctx->GetHexagonEngine();
    if (MI_NULL == engine)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "GetHexagonEngine failed");
        return ret;
    }

    HexagonRpcParam rpc_param(m_ctx);
    rpc_param.ResetBuffer();
    if (rpc_param.Set(m_device_addr) != Status::OK)
    {
        AURA_LOGE(m_ctx, AURA_TAG, "rpc_param Set fail\n");
        goto EXIT;
    }

    ret = engine->Run(AURA_HTP_NN_EXCUTOR_PACKAGE_NAME, AURA_HTP_NN_EXCUTOR_DEINITIALIZE, rpc_param);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Run failed");
    }

    m_device_addr = 0;

EXIT:
    return ret;
}

MI_U32 HtpNNExecutor::GetDeviceAddr()
{
    return m_device_addr;
}

std::vector<TensorDescMap> HtpNNExecutor::GetInputs()
{
    return m_input_tensor_desc;
}

std::vector<TensorDescMap> HtpNNExecutor::GetOutputs()
{
    return m_output_tensor_desc;
}

std::string HtpNNExecutor::GetVersion()
{
    return m_version;
}
#endif // AURA_ENABLE_NN && AURA_ENABLE_HEXAGON

} // namespace aura