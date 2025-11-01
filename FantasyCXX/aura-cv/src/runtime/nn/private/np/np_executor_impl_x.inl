
namespace aura
{

AURA_INLINE QoSOptions GetNpPerfLevel(NNPerfLevel perf_level)
{
    QoSOptions options;
    memset(&options, 0, sizeof(options));

    if (NNPerfLevel::PERF_LOW == perf_level)
    {
        options.preference = NEURONRUNTIME_PREFER_POWER;
        options.priority = NEURONRUNTIME_PRIORITY_LOW;
        options.boostValue = 10;
    }
    else if (NNPerfLevel::PERF_NORMAL == perf_level)
    {
        options.preference = NEURONRUNTIME_PREFER_PERFORMANCE;
        options.priority = NEURONRUNTIME_PRIORITY_MED;
        options.boostValue = 50;
    }
    else
    {
        options.preference = NEURONRUNTIME_PREFER_PERFORMANCE;
        options.priority = NEURONRUNTIME_PRIORITY_HIGH;
        options.boostValue = 100;
    }

    return options;
}

static const std::string g_np_lib_name = LIBNEURON_RUNTIME;

class NpLibrary : public NNLibrary
{
public:
    static NpLibrary& Get()
    {
        static NpLibrary np_library;
        return np_library;
    }

    Status Destroy() override
    {
        return UnLoad();
    }

public:
    AURA_API_DEF(NeuronRuntimeV2_createFromBuffer) = MI_S32 (*)(const AURA_VOID*, size_t, size_t, AURA_VOID**, size_t);
    AURA_API_PTR(NeuronRuntimeV2_createFromBuffer);

    AURA_API_DEF(NeuronRuntimeV2_release) = AURA_VOID (*)(AURA_VOID*);
    AURA_API_PTR(NeuronRuntimeV2_release);

    AURA_API_DEF(NeuronRuntimeV2_run) = MI_S32 (*)(AURA_VOID*, SyncInferenceRequest);
    AURA_API_PTR(NeuronRuntimeV2_run);

    AURA_API_DEF(NeuronRuntimeV2_getInputNumber) = MI_S32 (*)(AURA_VOID*, size_t*);
    AURA_API_PTR(NeuronRuntimeV2_getInputNumber);

    AURA_API_DEF(NeuronRuntimeV2_getOutputNumber) = MI_S32 (*)(AURA_VOID*, size_t*);
    AURA_API_PTR(NeuronRuntimeV2_getOutputNumber);

    AURA_API_DEF(NeuronRuntimeV2_getInputSize) = MI_S32 (*)(AURA_VOID*, MI_U64, size_t*);
    AURA_API_PTR(NeuronRuntimeV2_getInputSize);

    AURA_API_DEF(NeuronRuntimeV2_getOutputSize) = MI_S32 (*)(AURA_VOID*, MI_U64, size_t*);
    AURA_API_PTR(NeuronRuntimeV2_getOutputSize);

    AURA_API_DEF(NeuronRuntimeV2_setQoSOption) = MI_S32 (*)(AURA_VOID*, const QoSOptions*);
    AURA_API_PTR(NeuronRuntimeV2_setQoSOption);

    AURA_API_DEF(NeuronRuntime_getVersion) = MI_S32 (*)(NeuronVersion*);
    AURA_API_PTR(NeuronRuntime_getVersion);

private:
    NpLibrary() : NNLibrary(), m_handle(MI_NULL)
    {
        Load();
    }

    ~NpLibrary()
    {
        Destroy();
    }

    AURA_DISABLE_COPY_AND_ASSIGN(NpLibrary);

    Status Load() override
    {
        Status ret = Status::ERROR;

        dlerror();
        m_handle = dlopen(g_np_lib_name.c_str(), RTLD_LAZY | RTLD_LOCAL);
        if (MI_NULL == m_handle)
        {
            std::string info = "dlopen libneuron_runtime.so failed, err : " + std::string(dlerror());
            AURA_PRINTE(AURA_TAG, "%s\n", info.c_str());
            return ret;
        }

        do
        {
            AURA_DLSYM_API(m_handle, NeuronRuntimeV2_createFromBuffer);
            AURA_DLSYM_API(m_handle, NeuronRuntimeV2_release);
            AURA_DLSYM_API(m_handle, NeuronRuntimeV2_run);
            AURA_DLSYM_API(m_handle, NeuronRuntimeV2_getInputNumber);
            AURA_DLSYM_API(m_handle, NeuronRuntimeV2_getOutputNumber);
            AURA_DLSYM_API(m_handle, NeuronRuntimeV2_getInputSize);
            AURA_DLSYM_API(m_handle, NeuronRuntimeV2_getOutputSize);
            AURA_DLSYM_API(m_handle, NeuronRuntimeV2_setQoSOption);
            AURA_DLSYM_API(m_handle, NeuronRuntime_getVersion);

            ret = Status::OK;
        } while (0);

        return ret;
    }

    Status UnLoad() override
    {
        Status ret = Status::OK;
        if (m_handle)
        {
            dlerror();
            if (dlclose(m_handle) != 0)
            {
                std::string info = "dlclose failed, err : " + std::string(dlerror());
                AURA_PRINTE(AURA_TAG, "%s\n", info.c_str());
                ret = Status::ERROR;
            }
            m_handle = MI_NULL;
        }

        return ret;
    }

private:
    AURA_VOID *m_handle;
};

static MI_S32 NeuronRuntimeV2_createFromBuffer(Context *ctx, const AURA_VOID *buffer, size_t len, size_t num_threads,
                                               AURA_VOID **runtime, size_t backlog)
{
    auto func = NpLibrary::Get().NeuronRuntimeV2_createFromBuffer;
    if (func)
    {
        return func(buffer, len, num_threads, runtime, backlog);
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "NeuronRuntimeV2_createFromBuffer is null ptr");
        return NEURONRUNTIME_UNEXPECTED_NULL;
    }
}

static MI_S32 NeuronRuntimeV2_release(Context *ctx, AURA_VOID* runtime)
{
    auto func = NpLibrary::Get().NeuronRuntimeV2_release;
    if (func)
    {
        func(runtime);
        return NEURONRUNTIME_NO_ERROR;
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "NeuronRuntimeV2_release is null ptr");
        return NEURONRUNTIME_UNEXPECTED_NULL;
    }
}

static MI_S32 NeuronRuntimeV2_run(Context *ctx, AURA_VOID *runtime, SyncInferenceRequest request)
{
    auto func = NpLibrary::Get().NeuronRuntimeV2_run;
    if (func)
    {
        return func(runtime, request);
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "NeuronRuntimeV2_run is null ptr");
        return NEURONRUNTIME_UNEXPECTED_NULL;
    }
}

static MI_S32 NeuronRuntimeV2_getInputNumber(Context *ctx, AURA_VOID *runtime, size_t *size)
{
    auto func = NpLibrary::Get().NeuronRuntimeV2_getInputNumber;
    if (func)
    {
        return func(runtime, size);
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "NeuronRuntimeV2_getInputNumber is null ptr");
        return NEURONRUNTIME_UNEXPECTED_NULL;
    }
}

static MI_S32 NeuronRuntimeV2_getOutputNumber(Context *ctx, AURA_VOID *runtime, size_t *size)
{
    auto func = NpLibrary::Get().NeuronRuntimeV2_getOutputNumber;
    if (func)
    {
        return func(runtime, size);
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "NeuronRuntimeV2_getOutputNumber is null ptr");
        return NEURONRUNTIME_UNEXPECTED_NULL;
    }
}

static MI_S32 NeuronRuntimeV2_getInputSize(Context *ctx, AURA_VOID *runtime, MI_U64 idx, size_t *size)
{
    auto func = NpLibrary::Get().NeuronRuntimeV2_getInputSize;
    if (func)
    {
        return func(runtime, idx, size);
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "NeuronRuntimeV2_getInputSize is null ptr");
        return NEURONRUNTIME_UNEXPECTED_NULL;
    }
}

static MI_S32 NeuronRuntimeV2_getOutputSize(Context *ctx, AURA_VOID *runtime, MI_U64 idx, size_t *size)
{
    auto func = NpLibrary::Get().NeuronRuntimeV2_getOutputSize;
    if (func)
    {
        return func(runtime, idx, size);
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "NeuronRuntimeV2_getOutputSize is null ptr");
        return NEURONRUNTIME_UNEXPECTED_NULL;
    }
}

static MI_S32 NeuronRuntimeV2_setQoSOption(Context *ctx, AURA_VOID *runtime, const QoSOptions *options)
{
    auto func = NpLibrary::Get().NeuronRuntimeV2_setQoSOption;
    if (func)
    {
        return func(runtime, options);
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "NeuronRuntimeV2_setQoSOption is null ptr");
        return NEURONRUNTIME_UNEXPECTED_NULL;
    }
}

static MI_S32 NeuronRuntime_getVersion(Context *ctx, NeuronVersion *version)
{
    auto func = NpLibrary::Get().NeuronRuntime_getVersion;
    if (func)
    {
        return func(version);
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "NeuronRuntime_getVersion is null ptr");
        return NEURONRUNTIME_UNEXPECTED_NULL;
    }
}

class NpUtils
{
public:
    static std::string GetLibraryVersion(Context *ctx)
    {
        if (MI_NULL == ctx)
        {
            return std::string();
        }

        NeuronVersion version;
        memset(&version, 0, sizeof(version));

        MI_S32 ret = NeuronRuntime_getVersion(ctx, &version);
        if (ret != NEURONRUNTIME_NO_ERROR)
        {
            std::string info = "NeuronRuntime_getVersion failed, err : " + std::to_string(ret);
            AURA_ADD_ERROR_STRING(ctx, info.c_str());
            return std::string();
        }

        return "v" + std::to_string(version.major) + "." + std::to_string(version.minor) + "." + std::to_string(version.patch);
    }
};

class NpIOBufferMap
{
public:
    NpIOBufferMap(Context *ctx, AURA_VOID *np_handle, const std::vector<std::pair<std::string, NpModel::TensorAttr>> &attrs, MI_BOOL is_input)
                  : m_ctx(ctx), m_is_valid(MI_FALSE), m_is_input(is_input)
    {
        do
        {
            if (MI_NULL == m_ctx || MI_NULL == np_handle)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "null ptr");
                break;
            }

            size_t num = 0;
            if (m_is_input)
            {
                MI_S32 ret = NeuronRuntimeV2_getInputNumber(m_ctx, np_handle, &num);
                if (ret != NEURONRUNTIME_NO_ERROR)
                {
                    std::string info = "NeuronRuntimeV2_getInputNumber failed, err : " + std::to_string(ret);
                    AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
                    break;
                }
            }
            else
            {
                MI_S32 ret = NeuronRuntimeV2_getOutputNumber(m_ctx, np_handle, &num);
                if (ret != NEURONRUNTIME_NO_ERROR)
                {
                    std::string info = "NeuronRuntimeV2_getOutputNumber failed, err : " + std::to_string(ret);
                    AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
                    break;
                }
            }
            m_tensor_attrs = attrs;

            m_io_buffers = std::vector<IOBuffer>(num, IOBuffer(MI_NULL, 0, -1, 0));
            for (size_t i = 0; i < num; i++)
            {
                size_t size = 0;
                if (is_input)
                {
                    MI_S32 ret = NeuronRuntimeV2_getInputSize(m_ctx, np_handle, i, &size);
                    if (ret != NEURONRUNTIME_NO_ERROR)
                    {
                        std::string info = "NeuronRuntimeV2_getInputSize failed, err : " + std::to_string(ret);
                        AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
                        break;
                    }
                }
                else
                {
                    MI_S32 ret = NeuronRuntimeV2_getOutputSize(m_ctx, np_handle, i, &size);
                    if (ret != NEURONRUNTIME_NO_ERROR)
                    {
                        std::string info = "NeuronRuntimeV2_getOutputSize failed, err : " + std::to_string(ret);
                        AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
                        break;
                    }
                }
                m_buffer_sizes.push_back(size);
            }

            m_is_valid = MI_TRUE;
        } while (0);
    }

    ~NpIOBufferMap()
    {
        DeInitialize();
    }

    Status Initialize(const MatMap *mat_map)
    {
        if (!m_is_valid)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "m_is_valid is false");
            return Status::ERROR;
        }

        if (MI_NULL == mat_map || mat_map->size() != m_io_buffers.size())
        {
            AURA_ADD_ERROR_STRING(m_ctx, "size match error");
            return Status::ERROR;
        }

        m_mat_map = *mat_map;

        for (size_t i = 0; i < m_io_buffers.size(); i++)
        {
            std::string tensor_name = m_tensor_attrs[i].first;
            if (m_mat_map.find(tensor_name) == m_mat_map.end())
            {
                std::string info = "mat names " + tensor_name + " not provided";
                AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
                return Status::ERROR;
            }

            if (InitIOBuffers(i, tensor_name) != Status::OK)
            {
                std::string info = "InitIOBuffers " + tensor_name + " failed";
                AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
                return Status::ERROR;
            }
        }

        return Status::OK;
    }

    Status DeInitialize()
    {
        for (auto &mat : m_quant_map)
        {
            Delete<Mat>(m_ctx, &mat.second);
        }

        return Status::OK;
    }

    Status Quant()
    {
        if (!m_is_input)
        {
            return Status::OK;
        }

        for (size_t i = 0; i < m_tensor_attrs.size(); i++)
        {
            std::string name = m_tensor_attrs[i].first;

            if (m_quant_map.count(name))
            {
                MI_S32 zero_point = m_tensor_attrs[i].second.zero_point;
                MI_F32 scale = m_tensor_attrs[i].second.scale;

                Status ret = NNQuantize(m_ctx, *(m_mat_map[name]), *(m_quant_map[name]), zero_point, scale);
                if (ret != Status::OK)
                {
                    AURA_ADD_ERROR_STRING(m_ctx, "NNQuantize failed");
                    return Status::ERROR;
                }
            }
        }

        return Status::OK;
    }

    Status DeQuant()
    {
        if (m_is_input)
        {
            return Status::OK;
        }

        for (size_t i = 0; i < m_tensor_attrs.size(); i++)
        {
            std::string name = m_tensor_attrs[i].first;

            if (m_quant_map.count(name))
            {
                MI_S32 zero_point = m_tensor_attrs[i].second.zero_point;
                MI_F32 scale = m_tensor_attrs[i].second.scale;

                Status ret = NNDeQuantize(m_ctx, *(m_quant_map[name]), *(m_mat_map[name]), zero_point, scale);
                if (ret != Status::OK)
                {
                    AURA_ADD_ERROR_STRING(m_ctx, "NNDeQuantize failed");
                    return Status::ERROR;
                }
            }
        }

        return Status::OK;
    }

    IOBuffer* Get()
    {
        return m_io_buffers.data();
    }

private:
    Status InitIOBuffers(MI_S32 id, const std::string &tensor_name)
    {
        Mat *mat = m_mat_map[tensor_name];

        MI_BOOL is_quant = MI_FALSE;
        ElemType src_elem_type = mat->GetElemType();
        ElemType dst_elem_type = m_tensor_attrs[id].second.elem_type;
        if ((src_elem_type != ElemType::INVALID) && (src_elem_type == dst_elem_type))
        {
            if (!mat->IsContinuous())
            {
                AURA_ADD_ERROR_STRING(m_ctx, "external mat data memory must be continuous");
                return Status::ERROR;
            }
        }
        else if ((ElemType::F32 == src_elem_type) && ((ElemType::U8  == dst_elem_type) ||
                                                      (ElemType::S8  == dst_elem_type) ||
                                                      (ElemType::U16 == dst_elem_type) ||
                                                      (ElemType::S16 == dst_elem_type)))
        {
            is_quant = MI_TRUE;
        }
        else
        {
            std::string info = "tensor dataType " + ElemTypesToString(dst_elem_type) + " is not match to mat element type " + ElemTypesToString(src_elem_type);
            AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
            return Status::ERROR;
        }

        if (is_quant)
        {
            if (m_quant_map.find(tensor_name) == m_quant_map.end())
            {
                m_quant_map[tensor_name] = Create<Mat>(m_ctx, dst_elem_type, mat->GetSizes());
                if (!m_quant_map[tensor_name]->IsValid())
                {
                    AURA_ADD_ERROR_STRING(m_ctx, "mat is invalid");
                    return Status::ERROR;
                }
            }

            mat = m_quant_map[tensor_name];
        }

        MI_S32 elem_counts = static_cast<MI_S32>(m_buffer_sizes[id]) / ElemTypeSize(dst_elem_type);
        if (elem_counts != mat->GetSizes().Total())
        {
            std::string info = "tensor " + std::string(tensor_name) + ": expected " + std::to_string(m_buffer_sizes[id]) + " bytes, "
                               "but got " + std::to_string(mat->GetSizes().Total()) + " bytes";
            AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
            return Status::ERROR;
        }

        m_io_buffers[id] = IOBuffer(mat->GetData(), m_buffer_sizes[id],
                                    mat->GetBuffer().m_type == AURA_MEM_HEAP ? -1 : mat->GetBuffer().m_property,
                                    mat->GetBuffer().GetOffset());

        return Status::OK;
    }

    Context *m_ctx;
    MatMap  m_mat_map;
    MatMap  m_quant_map;
    MI_BOOL m_is_valid;
    MI_BOOL m_is_input;
    std::vector<IOBuffer> m_io_buffers;
    std::vector<std::pair<std::string, NpModel::TensorAttr>> m_tensor_attrs;
    std::vector<MI_U64> m_buffer_sizes;
};

class NpExecutorImplVx : public NpExecutorImpl
{
public:
    NpExecutorImplVx(Context *ctx, const std::shared_ptr<NpModel> &model, const NNConfig &config)
                     : NpExecutorImpl(ctx, model, config), m_np_handle(MI_NULL)
    {
        do
        {
            // get np version : np.v6.3.1
            std::vector<std::string> np_version;
            m_version = NpUtils::GetLibraryVersion(ctx);
            np_version = NNSplit(m_version, '.');

            // framework version
            std::vector<std::string> framework_version;
            framework_version = NNSplit(m_model->GetFrameWorkVersion(), '.');

            if ((4 == framework_version.size()) && (3 == np_version.size()))
            {
                if (framework_version[1] != np_version[0])
                {
                    std::string info = "frame version not match np version, frame verison: " + m_model->GetFrameWorkVersion() + " np version: " + m_version;
                    AURA_ADD_ERROR_STRING(ctx, info.c_str());
                    break;
                }
            }
            else
            {
                AURA_ADD_ERROR_STRING(ctx, "frame version size must be 4 and np version size must be 3");
                break;
            }

            auto init_func = [=]() -> Status
            {
                if (CreateRuntime() != Status::OK)
                {
                    AURA_ADD_ERROR_STRING(m_ctx, "CreateRuntime failed");
                    return Status::ERROR;
                }

                if (SetRuntimeQoSConf() != Status::OK)
                {
                    AURA_ADD_ERROR_STRING(m_ctx, "SetRuntimeQoSConf failed");
                    return Status::ERROR;
                }

                m_is_valid = MI_TRUE;
                return Status::OK;
            };

            if (m_wp)
            {
                m_token = m_wp->AsyncRun(init_func);
            }
            else
            {
                AURA_ADD_ERROR_STRING(ctx, "m_wp null ptr");
            }
        } while (0);
    }

    ~NpExecutorImplVx()
    {
        if (m_token.valid())
        {
            m_token.wait();
        }

        if (m_np_handle != MI_NULL)
        {
            NeuronRuntimeV2_release(m_ctx, m_np_handle);
            m_np_handle = MI_NULL;
        }
    }

    Status Initialize() override
    {
        if (m_token.valid())
        {
            m_token.wait();
        }

        if (!m_is_valid)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "m_is_valid is false");
            return Status::ERROR;
        }

        //create NpIOBufferMap
        m_input_map = std::make_shared<NpIOBufferMap>(m_ctx, m_np_handle, m_model->GetInputAttrs(), MI_TRUE);
        m_output_map = std::make_shared<NpIOBufferMap>(m_ctx, m_np_handle, m_model->GetOutputAttrs(), MI_FALSE);

        return Status::OK;
    }

    Status Update(const std::string &name, AnyParams &params) override
    {
        if ("update_perf" == name)
        {
            return UpdatePerf(params);
        }
        else
        {
            std::string info = "the specified function '" + name + "' does not exist";
            AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
            return Status::ERROR;
        }

        return Status::OK;
    }

    std::string GetVersion() override
    {
        return (m_model->GetVersion() + " device(np." + m_version + ")");
    }

    Status Forward(const MatMap &input, MatMap &output, MI_S32 graph_id) override
    {
        AURA_UNUSED(graph_id);
        Status ret = Status::ERROR;

        MatMap input_mapped = m_model->MapMatNames(input, MI_TRUE);
        MatMap output_mapped = m_model->MapMatNames(output, MI_FALSE);

        if (input_mapped.empty() || output_mapped.empty())
        {
            AURA_ADD_ERROR_STRING(m_ctx, "map names failed");
            return Status::ERROR;
        }

        ret = m_input_map->Initialize(&input_mapped);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "m_input_map Initialize failed");
            return ret;
        }

        ret = m_output_map->Initialize(&output_mapped);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "m_output_map Initialize failed");
            return ret;
        }

        ret = m_input_map->Quant();
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "m_input_map Quant failed");
            return ret;
        }

        SyncInferenceRequest sync_data = {m_input_map->Get(), m_output_map->Get()};
        MI_S32 err_code = NeuronRuntimeV2_run(m_ctx, m_np_handle, sync_data);
        if (err_code != NEURONRUNTIME_NO_ERROR)
        {
            std::string info = "NeuronRuntimeV2_run failed, err : " + std::to_string(err_code);
            AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
            return Status::ERROR;
        }

        ret = m_output_map->DeQuant();
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "m_output_map DeQuant failed");
        }

        return ret;
    }

    std::vector<TensorDescMap> GetInputs() override
    {
        TensorDescMap result;

        const std::vector<std::pair<std::string, NpModel::TensorAttr>> &input_attrs = m_model->GetInputAttrs();
        for (auto &input_attr : input_attrs)
        {
            TensorDesc desc;
            desc.elem_type = input_attr.second.elem_type;
            desc.sizes = input_attr.second.sizes;
            desc.scale = input_attr.second.scale;
            desc.zero_point = input_attr.second.zero_point;
            result[input_attr.first] = desc;
        }

        result = m_model->MapTensorDescNames(result, MI_TRUE);
        return {result};
    }

    std::vector<TensorDescMap> GetOutputs() override
    {
        TensorDescMap result;

        const std::vector<std::pair<std::string, NpModel::TensorAttr>> &output_attrs = m_model->GetOutputAttrs();
        for (auto &output_attr : output_attrs)
        {
            TensorDesc desc;
            desc.elem_type = output_attr.second.elem_type;
            desc.sizes = output_attr.second.sizes;
            desc.scale = output_attr.second.scale;
            desc.zero_point = output_attr.second.zero_point;
            result[output_attr.first] = desc;
        }

        result = m_model->MapTensorDescNames(result, MI_FALSE);
        return {result};
    }

private:
    Status CreateRuntime()
    {
        Status ret = Status::ERROR;
        MI_S32 err_code = NEURONRUNTIME_NO_ERROR;

        Buffer buffer = m_model->GetModelBuffer();
        if (!buffer.IsValid())
        {
            AURA_ADD_ERROR_STRING(m_ctx, "GetModelBuffer failed");
            goto EXIT;
        }

        err_code = NeuronRuntimeV2_createFromBuffer(m_ctx, buffer.m_data, buffer.m_size, 1, &m_np_handle, 2048);
        if (err_code != NEURONRUNTIME_NO_ERROR)
        {
            std::string info = "NeuronRuntimeV2_createFromBuffer failed, err : " + std::to_string(err_code);
            AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
            goto EXIT;
        }

        ret = Status::OK;
EXIT:
        m_model->ReleaseModelBuffer();
        return ret;
    }

    Status SetRuntimeQoSConf()
    {
        QoSOptions qos_option = GetNpPerfLevel(m_config.perf_level);

        MI_S32 ret = NeuronRuntimeV2_setQoSOption(m_ctx, m_np_handle, &qos_option);
        if (ret != NEURONRUNTIME_NO_ERROR)
        {
            std::string info = "NeuronRuntime_setQoSOption failed, err : " + std::to_string(ret);
            AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
            return Status::ERROR;
        }

        return Status::OK;
    }

    Status UpdatePerf(const AnyParams &params)
    {
        NNPerfLevel perf_level = NNPerfLevel::PERF_HIGH;
        Status ret = Status::ERROR;

        if (GetNNPerfLevel(m_ctx, params, perf_level) != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "GetNNPerfLevel failed");
            return Status::ERROR;
        }

        if (perf_level != m_config.perf_level)
        {
            m_config.perf_level = perf_level;

            ret = SetRuntimeQoSConf();
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "SetRuntimeQoSConf failed");
                return Status::ERROR;
            }
        }

        return Status::OK;
    }

    std::string m_version;
    AURA_VOID *m_np_handle;
    std::shared_ptr<NpIOBufferMap> m_input_map;
    std::shared_ptr<NpIOBufferMap> m_output_map;
};

} // namespace aura