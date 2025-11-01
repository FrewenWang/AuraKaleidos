#include "aura/runtime/nn/nn_engine.hpp"
#include "mnn/mnn_executor_impl.hpp"
#include "np/np_executor_impl.hpp"
#include "qnn/qnn_executor_impl.hpp"
#include "snpe/snpe_executor_impl.hpp"
#include "xnn/xnn_executor_impl.hpp"

namespace aura
{

// TODO : this function should be encoded with python script, here temporarily add it
AURA_INLINE Status NNExecutorImplRegister()
{
    // extern SnpeExecutorImplRegister g_snpe_executor_impl_dummy_register;
    // g_snpe_executor_impl_dummy_register.Register();
#if defined(AURA_BUILD_HOST)
    /* snpe executor regist */
    extern SnpeExecutorImplRegister g_snpe_executor_impl_v213_register;
    g_snpe_executor_impl_v213_register.Register();

    extern SnpeExecutorImplRegister g_snpe_executor_impl_v217_register;
    g_snpe_executor_impl_v217_register.Register();

    extern SnpeExecutorImplRegister g_snpe_executor_impl_v219_register;
    g_snpe_executor_impl_v219_register.Register();

    extern SnpeExecutorImplRegister g_snpe_executor_impl_v220_register;
    g_snpe_executor_impl_v220_register.Register();

    extern SnpeExecutorImplRegister g_snpe_executor_impl_v221_register;
    g_snpe_executor_impl_v221_register.Register();

    extern SnpeExecutorImplRegister g_snpe_executor_impl_v222_register;
    g_snpe_executor_impl_v222_register.Register();

    extern SnpeExecutorImplRegister g_snpe_executor_impl_v223_register;
    g_snpe_executor_impl_v223_register.Register();

    extern SnpeExecutorImplRegister g_snpe_executor_impl_v224_register;
    g_snpe_executor_impl_v224_register.Register();

    extern SnpeExecutorImplRegister g_snpe_executor_impl_v225_register;
    g_snpe_executor_impl_v225_register.Register();

    extern SnpeExecutorImplRegister g_snpe_executor_impl_v226_register;
    g_snpe_executor_impl_v226_register.Register();

    extern SnpeExecutorImplRegister g_snpe_executor_impl_v227_register;
    g_snpe_executor_impl_v227_register.Register();

    extern SnpeExecutorImplRegister g_snpe_executor_impl_v228_register;
    g_snpe_executor_impl_v228_register.Register();

    extern SnpeExecutorImplRegister g_snpe_executor_impl_v229_register;
    g_snpe_executor_impl_v229_register.Register();

    extern SnpeExecutorImplRegister g_snpe_executor_impl_v230_register;
    g_snpe_executor_impl_v230_register.Register();

    extern SnpeExecutorImplRegister g_snpe_executor_impl_v231_register;
    g_snpe_executor_impl_v231_register.Register();

    extern SnpeExecutorImplRegister g_snpe_executor_impl_v233_register;
    g_snpe_executor_impl_v233_register.Register();

    /* np executor regist */
    extern NpExecutorImplRegister g_np_executor_impl_v7_register;
    g_np_executor_impl_v7_register.Register();

    extern NpExecutorImplRegister g_np_executor_impl_v8_register;
    g_np_executor_impl_v8_register.Register();

    /* mnn executor regist */
    extern MnnExecutorImplRegister g_mnn_executor_impl_v271_register;
    g_mnn_executor_impl_v271_register.Register();

    /* xnn executor regist */
    extern XnnExecutorImplRegister g_xnn_executor_impl_v100_register;
    g_xnn_executor_impl_v100_register.Register();
#endif // AURA_BUILD_HOST

   /* qnn executor regist */
    extern QnnExecutorImplRegister g_qnn_executor_impl_v213_register;
    g_qnn_executor_impl_v213_register.Register();

    extern QnnExecutorImplRegister g_qnn_executor_impl_v214_register;
    g_qnn_executor_impl_v214_register.Register();

    extern QnnExecutorImplRegister g_qnn_executor_impl_v217_register;
    g_qnn_executor_impl_v217_register.Register();

    extern QnnExecutorImplRegister g_qnn_executor_impl_v219_register;
    g_qnn_executor_impl_v219_register.Register();

    extern QnnExecutorImplRegister g_qnn_executor_impl_v220_register;
    g_qnn_executor_impl_v220_register.Register();

    extern QnnExecutorImplRegister g_qnn_executor_impl_v221_register;
    g_qnn_executor_impl_v221_register.Register();

    extern QnnExecutorImplRegister g_qnn_executor_impl_v222_register;
    g_qnn_executor_impl_v222_register.Register();

    extern QnnExecutorImplRegister g_qnn_executor_impl_v223_register;
    g_qnn_executor_impl_v223_register.Register();

    extern QnnExecutorImplRegister g_qnn_executor_impl_v224_register;
    g_qnn_executor_impl_v224_register.Register();

    extern QnnExecutorImplRegister g_qnn_executor_impl_v225_register;
    g_qnn_executor_impl_v225_register.Register();

    extern QnnExecutorImplRegister g_qnn_executor_impl_v226_register;
    g_qnn_executor_impl_v226_register.Register();

    extern QnnExecutorImplRegister g_qnn_executor_impl_v227_register;
    g_qnn_executor_impl_v227_register.Register();

    extern QnnExecutorImplRegister g_qnn_executor_impl_v228_register;
    g_qnn_executor_impl_v228_register.Register();

    extern QnnExecutorImplRegister g_qnn_executor_impl_v229_register;
    g_qnn_executor_impl_v229_register.Register();

    extern QnnExecutorImplRegister g_qnn_executor_impl_v230_register;
    g_qnn_executor_impl_v230_register.Register();

    extern QnnExecutorImplRegister g_qnn_executor_impl_v231_register;
    g_qnn_executor_impl_v231_register.Register();

    extern QnnExecutorImplRegister g_qnn_executor_impl_v233_register;
    g_qnn_executor_impl_v233_register.Register();

    return Status::OK;
}

static MI_S16 GetFrameWork(Context *ctx, const ModelInfo &model_info)
{
    MI_S16 framework = 0;

    if (MI_NULL == ctx)
    {
        AURA_ADD_ERROR_STRING(ctx, "ctx is NULL");
        return framework;
    }

    MI_S64 data_offset = 0;
    MinnHeader header;

    if (NNDeserialize(ctx, model_info.minn_buffer, data_offset, header) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "NNDeserialize failed");
        return framework;
    }

    if (header.magic_num != AURA_NN_MODEL_MAGIC)
    {
        AURA_ADD_ERROR_STRING(ctx, "model header check failed");
        return framework;
    }

    if ((1 == header.version.major) && (header.version.minor <= 2))
    {
        MinnDataV1 data;
        if (NNDeserialize(ctx, model_info.minn_buffer, data_offset, data) != Status::OK)
        {
            AURA_ADD_ERROR_STRING(ctx, "NNDeserialize failed");
            return framework;
        }

        framework = data.framework;
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "unsupported minn version");
        return framework;
    }

    return framework;
}

static std::shared_ptr<NNExecutor> CreateNNExecutorImpl(Context *ctx, const ModelInfo &model_info, const NNConfig &config)
{
    MI_S16 framework = GetFrameWork(ctx, model_info);

    if (1 == framework)
    {
        std::shared_ptr<QnnModel> qnn_model = std::make_shared<QnnModel>(ctx, model_info);
        if ((MI_NULL == qnn_model) || (!qnn_model->IsValid()))
        {
            AURA_ADD_ERROR_STRING(ctx, "create model failed");
        }

        return std::make_shared<QnnExecutor>(ctx, qnn_model, config);
    }
#if defined(AURA_BUILD_HOST)
    else if (2 == framework)
    {
        std::shared_ptr<SnpeModel> snpe_model = std::make_shared<SnpeModel>(ctx, model_info);
        if ((MI_NULL == snpe_model) || (!snpe_model->IsValid()))
        {
            AURA_ADD_ERROR_STRING(ctx, "create model failed");
        }

        return std::make_shared<SnpeExecutor>(ctx, snpe_model, config);
    }
    else if (10 == framework)
    {
        std::shared_ptr<NpModel> np_model = std::make_shared<NpModel>(ctx, model_info);
        if ((MI_NULL == np_model) || (!np_model->IsValid()))
        {
            AURA_ADD_ERROR_STRING(ctx, "create model failed");
        }

        return std::make_shared<NpExecutor>(ctx, np_model, config);
    }
    else if (20 == framework)
    {
        std::shared_ptr<XnnModel> xnn_model = std::make_shared<XnnModel>(ctx, model_info);
        if ((MI_NULL == xnn_model) || (!xnn_model->IsValid()))
        {
            AURA_ADD_ERROR_STRING(ctx, "create model failed");
        }

        return std::make_shared<XnnExecutor>(ctx, xnn_model, config);
    }
    else if (30 == framework)
    {
        std::shared_ptr<MnnModel> mnn_model = std::make_shared<MnnModel>(ctx, model_info);
        if ((MI_NULL == mnn_model) || (!mnn_model->IsValid()))
        {
            AURA_ADD_ERROR_STRING(ctx, "create model failed");
        }

        return std::make_shared<MnnExecutor>(ctx, mnn_model, config);
    }
#endif // AURA_BUILD_HOST
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "unsupported framework type");
    }

    return MI_NULL;
}

NNEngine::NNEngine(Context *ctx, MI_BOOL enable_nn) : m_ctx(ctx), m_enable_nn(enable_nn)
{
    NNExecutorImplRegister();
#if defined(AURA_BUILD_HEXAGON)
    NNLibraryManager::GetInstance().AddRefCount();
#endif
}

NNEngine::~NNEngine()
{
#if defined(AURA_BUILD_HEXAGON)
    NNLibraryManager::GetInstance().Destroy();
#endif
}

std::shared_ptr<NNExecutor> NNEngine::CreateNNExecutor(const std::string &minn_file,
                                                       const std::string &decrypt_key,
                                                       const NNConfig &config)
{
    if (!m_enable_nn)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "nn is not enabled");
        return MI_NULL;
    }

#if defined(AURA_BUILD_HOST)
    Buffer minn_buffer = NNModel::MapModelBufferFromFile(m_ctx, minn_file);
    if (!minn_buffer.IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "MapModelBufferFromFile failed");
        return MI_NULL;
    }
#else
    Buffer minn_buffer = NNModel::CreateModelBufferFromFile(m_ctx, minn_file);
    if (!minn_buffer.IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "CreateModelBufferFromFile failed");
        return MI_NULL;
    }
#endif // AURA_BUILD_HOST

    std::string::size_type pos = (minn_file.find_last_of('\\') + 1) == 0 ? minn_file.find_last_of('/') + 1 : minn_file.find_last_of('\\') + 1;
    std::string file_name = minn_file.substr(pos, minn_file.length() - pos);
    std::string minn_name = file_name.substr(0, file_name.rfind("."));

    return CreateNNExecutorImpl(m_ctx, ModelInfo(minn_buffer, decrypt_key, minn_name), config);
}

std::shared_ptr<NNExecutor> NNEngine::CreateNNExecutor(const MI_U8 *minn_data,
                                                       MI_S64 minn_size,
                                                       const std::string &decrypt_key,
                                                       const NNConfig &config)
{
    if (!m_enable_nn)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "nn is not enabled");
        return MI_NULL;
    }

    if (MI_NULL == minn_data)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "minn_data is null ptr");
        return MI_NULL;
    }

    if (minn_size == 0)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "minn_size is zero, is invalid");
        return MI_NULL;
    }

    Buffer minn_buffer(AURA_MEM_HEAP, minn_size, minn_size, const_cast<MI_U8*>(minn_data), const_cast<MI_U8*>(minn_data), 0);
    if (!minn_buffer.IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "minn_buffer init failed");
        return MI_NULL;
    }

    return CreateNNExecutorImpl(m_ctx, ModelInfo(minn_buffer, decrypt_key), config);
}

std::shared_ptr<NNExecutor> NNEngine::CreateNNExecutor(const std::string &minb_file,
                                                       const std::string &minn_name,
                                                       const std::string &decrypt_key,
                                                       const NNConfig &config)
{
    if (!m_enable_nn)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "nn is not enabled");
        return MI_NULL;
    }

    if (minb_file.empty())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "minb_file is empty");
        return MI_NULL;
    }

    if (minn_name.empty())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "minn_name is empty");
        return MI_NULL;
    }

    Buffer minn_buffer;
    std::shared_ptr<NBModel> nb_model = std::make_shared<NBModel>(m_ctx, minb_file);
    if (!nb_model->IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "create nb model failed");
        return MI_NULL;
    }

    minn_buffer = nb_model->GetModelBuffer(minn_name);
    if (!minn_buffer.IsValid())
    {
        std::string info = "get model buffer " + minn_name + " fail!\n";
        AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
        return MI_NULL;
    }

    return CreateNNExecutorImpl(m_ctx, ModelInfo(minn_buffer, decrypt_key, minn_name), config);
}

} // namespace aura