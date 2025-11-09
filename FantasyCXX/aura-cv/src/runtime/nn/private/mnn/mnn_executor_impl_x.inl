namespace aura
{

AURA_INLINE MNN::NNBackend GetMnnBackend(const NNBackend &backend)
{
    switch (backend)
    {
        case NNBackend::GPU:
        {
            return MNN::NNBackend::GPU;
        }
        default:
        {
            return MNN::NNBackend::CPU;
        }
    }
}

AURA_INLINE MNN::Precision GetMnnPrecision(const MnnPrecision &precision)
{
    switch (precision)
    {
        case MnnPrecision::PRECISION_HIGH:
        {
            return MNN::Precision::PRECISION_HIGH;
        }
        case MnnPrecision::PRECISION_LOW:
        {
            return MNN::Precision::PRECISION_LOW;
        }
        default:
        {
            return MNN::Precision::PRECISION_NORMAL;
        }
    }
}

AURA_INLINE MNN::Memory GetMnnMemory(const MnnMemory &memory)
{
    switch (memory)
    {
        case MnnMemory::MEMORY_HIGH:
        {
            return MNN::Memory::MEMORY_HIGH;
        }
        case MnnMemory::MEMORY_LOW:
        {
            return MNN::Memory::MEMORY_LOW;
        }
        default:
        {
            return MNN::Memory::MEMORY_NORMAL;
        }
    }
}

AURA_INLINE MNN::Tuning GetMnnTuning(const MnnTuning &tuning)
{
    switch (tuning)
    {
        case MnnTuning::GPU_TUNING_HEAVY:
        {
            return MNN::Tuning::GPU_TUNING_HEAVY;
        }
        case MnnTuning::GPU_TUNING_WIDE:
        {
            return MNN::Tuning::GPU_TUNING_WIDE;
        }
        case MnnTuning::GPU_TUNING_NORMAL:
        {
            return MNN::Tuning::GPU_TUNING_NORMAL;
        }
        case MnnTuning::GPU_TUNING_FAST:
        {
            return MNN::Tuning::GPU_TUNING_FAST;
        }
        default:
        {
            return MNN::Tuning::GPU_TUNING_NONE;
        }
    }
}

AURA_INLINE MNN::CLMem GetMnnCLMem(const MnnCLMem &cl_mem)
{
    switch (cl_mem)
    {
        case MnnCLMem::GPU_MEMORY_IAURA:
        {
            return MNN::CLMem::GPU_MEMORY_IAURA;
        }
        case MnnCLMem::GPU_MEMORY_BUFFER:
        {
            return MNN::CLMem::GPU_MEMORY_BUFFER;
        }
        default:
        {
            return MNN::CLMem::GPU_MEMORY_NONE;
        }
    }
}

AURA_INLINE ElemType GetElemType(MNN::MnnDataType data_type)
{
    switch (data_type)
    {
        case MNN::DATA_TYPE_UINT8:
        {
            return ElemType::U8;
        }

        case MNN::DATA_TYPE_INT8:
        {
            return ElemType::S8;
        }

        case MNN::DATA_TYPE_UINT16:
        {
            return ElemType::U16;
        }

        case MNN::DATA_TYPE_INT16:
        {
            return ElemType::S16;
        }

        case MNN::DATA_TYPE_UINT32:
        {
            return ElemType::U32;
        }

        case MNN::DATA_TYPE_INT32:
        {
            return ElemType::S32;
        }

        case MNN::DATA_TYPE_HALF:
        {
            return ElemType::F16;
        }

        case MNN::DATA_TYPE_FLOAT:
        {
            return ElemType::F32;
        }

        case MNN::DATA_TYPE_DOUBLE:
        {
            return ElemType::F64;
        }

        default:
        {
            return ElemType::INVALID;
        }
    }
}

static const std::string g_mnn_lib_name = LIBMNN_WRAPPER;

class MnnLibrary : public NNLibrary
{
public:
    static MnnLibrary& Get()
    {
        static MnnLibrary mnn_library;
        return mnn_library;
    }

    Status Destroy() override
    {
        return UnLoad();
    }

public:
    AURA_API_DEF(MnnInit) = MNN::Context (*)(MNN::NNBackend, MNN::Precision, MNN::Memory, MNN::Tuning, MNN::CLMem, DT_U32, const DT_VOID*, size_t, const DT_CHAR*, DT_U32, const DT_CHAR*);
    AURA_API_PTR(MnnInit);

    AURA_API_DEF(MnnCopySessionInput) = DT_S32 (*)(MNN::Context, const DT_CHAR*, DT_VOID*);
    AURA_API_PTR(MnnCopySessionInput);

    AURA_API_DEF(MnnCopySessionOutput) = DT_S32 (*)(MNN::Context, const DT_CHAR*, DT_VOID*);
    AURA_API_PTR(MnnCopySessionOutput);

    AURA_API_DEF(MnnRunSession) = DT_S32 (*)(MNN::Context);
    AURA_API_PTR(MnnRunSession);

    AURA_API_DEF(MnnUnit) = DT_S32 (*)(MNN::Context*);
    AURA_API_PTR(MnnUnit);

    AURA_API_DEF(MnnGetVersion) = const DT_CHAR* (*)();
    AURA_API_PTR(MnnGetVersion);

    AURA_API_DEF(MnnGetSessionInputs) = MNN::TensorDesc* (*)(MNN::Context, DT_S32*);
    AURA_API_PTR(MnnGetSessionInputs);

    AURA_API_DEF(MnnGetSessionOutputs) = MNN::TensorDesc* (*)(MNN::Context, DT_S32*);
    AURA_API_PTR(MnnGetSessionOutputs);

    AURA_API_DEF(MnnDeleteTensorDesc) = DT_VOID (*)(MNN::TensorDesc**, DT_S32);
    AURA_API_PTR(MnnDeleteTensorDesc);

private:
    MnnLibrary() : NNLibrary(), m_handle(DT_NULL)
    {
        Load();
    }

    ~MnnLibrary()
    {
        Destroy();
    }

    AURA_DISABLE_COPY_AND_ASSIGN(MnnLibrary);

    Status Load() override
    {
        Status ret = Status::ERROR;

        dlerror();
        m_handle = dlopen(g_mnn_lib_name.c_str(), RTLD_LAZY | RTLD_LOCAL);
        if (DT_NULL == m_handle)
        {
            std::string info = "dlopen libmnn_wrapper.so failed, err : " + std::string(dlerror());
            AURA_PRINTE(AURA_TAG, "%s\n", info.c_str());
            return ret;
        }

        do
        {
            AURA_DLSYM_API(m_handle, MnnInit);
            AURA_DLSYM_API(m_handle, MnnCopySessionInput);
            AURA_DLSYM_API(m_handle, MnnCopySessionOutput);
            AURA_DLSYM_API(m_handle, MnnRunSession);
            AURA_DLSYM_API(m_handle, MnnUnit);
            AURA_DLSYM_API(m_handle, MnnGetVersion);
            AURA_DLSYM_API(m_handle, MnnGetSessionInputs);
            AURA_DLSYM_API(m_handle, MnnGetSessionOutputs);
            AURA_DLSYM_API(m_handle, MnnDeleteTensorDesc);

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
            m_handle = DT_NULL;
        }

        return ret;
    }

private:
    DT_VOID *m_handle;
};

static MNN::Context MnnInit(Context *ctx, MNN::NNBackend backend, MNN::Precision precision, MNN::Memory memory, MNN::Tuning tuning, MNN::CLMem cl_mem,
                            DT_U32 version, const DT_VOID *buffer, size_t size, const DT_CHAR *cache_file, DT_U32 dump_layers, const DT_CHAR *profiling_path)
{
    auto func = MnnLibrary::Get().MnnInit;
    if (func)
    {
        return func(backend, precision, memory, tuning, cl_mem, version, buffer, size, cache_file, dump_layers, profiling_path);
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "MnnInit is null ptr");
        return DT_NULL;
    }
}

static DT_S32 MnnCopySessionInput(Context *ctx, MNN::Context mnn_ctx, const DT_CHAR* name, DT_VOID* data)
{
    auto func = MnnLibrary::Get().MnnCopySessionInput;
    if (func)
    {
        return func(mnn_ctx, name, data);
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "MnnCopySessionInput is null ptr");
        return -1;
    }
}

static DT_S32 MnnCopySessionOutput(Context *ctx, MNN::Context mnn_ctx, const DT_CHAR* name, DT_VOID* data)
{
    auto func = MnnLibrary::Get().MnnCopySessionOutput;
    if (func)
    {
        return func(mnn_ctx, name, data);
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "MnnCopySessionOutput is null ptr");
        return -1;
    }
}

static DT_S32 MnnRunSession(Context *ctx, MNN::Context mnn_ctx)
{
    auto func = MnnLibrary::Get().MnnRunSession;
    if (func)
    {
        return func(mnn_ctx);
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "MnnRunSession is null ptr");
        return -1;
    }
}

static DT_S32 MnnUnit(Context *ctx, MNN::Context *mnn_ctx)
{
    auto func = MnnLibrary::Get().MnnUnit;
    if (func)
    {
        return func(mnn_ctx);
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "MnnUnit is null ptr");
        return -1;
    }
}

static const DT_CHAR* MnnGetVersion(Context *ctx)
{
    auto func = MnnLibrary::Get().MnnGetVersion;
    if (func)
    {
        return func();
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "MnnGetVersion is null ptr");
        return DT_NULL;
    }
}

static MNN::TensorDesc* MnnGetSessionInputs(Context *ctx, MNN::Context mnn_ctx, DT_S32 *num)
{
    auto func = MnnLibrary::Get().MnnGetSessionInputs;
    if (func)
    {
        return func(mnn_ctx, num);
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "MnnGetSessionInputs is null ptr");
        return DT_NULL;
    }
}

static MNN::TensorDesc* MnnGetSessionOutputs(Context *ctx, MNN::Context mnn_ctx, DT_S32 *num)
{
    auto func = MnnLibrary::Get().MnnGetSessionOutputs;
    if (func)
    {
        return func(mnn_ctx, num);
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "MnnGetSessionOutputs is null ptr");
        return DT_NULL;
    }
}

static DT_VOID MnnDeleteTensorDesc(Context *ctx, MNN::TensorDesc **ptr, DT_S32 num)
{
    auto func = MnnLibrary::Get().MnnDeleteTensorDesc;
    if (func)
    {
        func(ptr, num);
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "MnnDeleteTensorDesc is null ptr");
    }
}

class MnnUtils
{
public:
    static std::string GetLibraryVersion(Context *ctx)
    {
        if (DT_NULL == ctx)
        {
            return std::string();
        }

        std::string version = MnnGetVersion(ctx);

        return "v" + version;
    }

    static TensorDescMap GetTensorDescMap(Context *ctx, MNN::Context mnn_ctx, DT_BOOL is_input)
    {
        if (DT_NULL == ctx)
        {
            return TensorDescMap();
        }

        if (DT_NULL == mnn_ctx)
        {
            AURA_ADD_ERROR_STRING(ctx, "null ptr");
            return TensorDescMap();
        }

        DT_S32 tensor_num = 0;
        MNN::TensorDesc *tensor_desc = is_input ? MnnGetSessionInputs(ctx, mnn_ctx, &tensor_num)
                                                : MnnGetSessionOutputs(ctx, mnn_ctx, &tensor_num);
        if (DT_NULL == tensor_desc)
        {
            AURA_ADD_ERROR_STRING(ctx, "get tensor desc failed");
            return TensorDescMap();
        }

        TensorDescMap tensor_desc_map;
        for (DT_S32 i = 0; i < tensor_num; i++)
        {
            TensorDesc desc;
            if (4 == tensor_desc[i].rank)
            {
                desc.sizes.push_back(tensor_desc[i].dims[0]);
                desc.sizes.push_back(tensor_desc[i].dims[2]);
                desc.sizes.push_back(tensor_desc[i].dims[3]);
                desc.sizes.push_back(tensor_desc[i].dims[1]);
            }
            else
            {
                for (DT_S32 j = 0; j < tensor_desc[i].rank; j++)
                {
                    desc.sizes.push_back(tensor_desc[i].dims[j]);
                }
            }
            desc.elem_type  = GetElemType(tensor_desc[i].elem_type);
            desc.scale      = tensor_desc[i].scale;
            desc.zero_point = tensor_desc[i].zero_point;
            tensor_desc_map[tensor_desc[i].name] = desc;
        }

        MnnDeleteTensorDesc(ctx, &tensor_desc, tensor_num);
        return tensor_desc_map;
    }
};

class MnnTensorMap
{
public:
    MnnTensorMap(Context *ctx, MNN::Context mnn_ctx, DT_BOOL is_input)
                 : m_ctx(ctx), m_mnn_ctx(mnn_ctx), m_is_valid(DT_FALSE), m_is_input(is_input)
    {
        do
        {
            if ((DT_NULL == m_ctx) || (DT_NULL == m_mnn_ctx))
            {
                AURA_ADD_ERROR_STRING(m_ctx, "null ptr");
                break;
            }

            m_tensor_desc_map = MnnUtils::GetTensorDescMap(m_ctx, m_mnn_ctx, m_is_input);
            if (m_tensor_desc_map.empty())
            {
                AURA_ADD_ERROR_STRING(m_ctx, "get tensor desc map failed");
                break;
            }

            m_is_valid = DT_TRUE;
        } while (0);
    }

    ~MnnTensorMap()
    {}

    Status Initialize(const MatMap *mat_map)
    {
        if (!m_is_valid)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "m_is_valid is false");
            return Status::ERROR;
        }

        if ((DT_NULL == mat_map) || (mat_map->size() != m_tensor_desc_map.size()))
        {
            AURA_ADD_ERROR_STRING(m_ctx, "size match error");
            return Status::ERROR;
        }

        for (const auto &tensor_desc : m_tensor_desc_map)
        {
            std::string tensor_name = tensor_desc.first;
            if (mat_map->find(tensor_name) == mat_map->end())
            {
                std::string info = "mat names " + tensor_name + " not provided";
                AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
                return Status::ERROR;
            }

            Mat *mat = mat_map->at(tensor_name);
            if (mat->GetElemType() != tensor_desc.second.elem_type)
            {
                std::string info = "tensor " + tensor_name + ": expected " + ElemTypesToString(tensor_desc.second.elem_type) + ", "
                                   "but got " + ElemTypesToString(mat->GetElemType());
                return Status::ERROR;
            }

            if (!mat->IsContinuous())
            {
                AURA_ADD_ERROR_STRING(m_ctx, "external mat data memory must be continuous");
                return Status::ERROR;
            }

            auto get_elem_counts_func = [](const TensorDesc &desc) -> DT_S32
            {
                DT_S32 elem_counts = 1;
                for (auto &size: desc.sizes)
                {
                    elem_counts *= size;
                }
                return elem_counts;
            };

            DT_S32 elem_counts = get_elem_counts_func(tensor_desc.second);
            if (elem_counts != mat->GetSizes().Total())
            {
                std::string info = "tensor " + tensor_name + ": expected " + std::to_string(elem_counts) + " bytes, "
                                   "but got " + std::to_string(mat->GetSizes().Total()) + " bytes";
                AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
                return Status::ERROR;
            }

            if (m_is_input)
            {
                if (MnnCopySessionInput(m_ctx, m_mnn_ctx, tensor_name.c_str(), mat->GetData()))
                {
                    std::string info = "MnnCopySessionInput " + tensor_name + " failed";
                    AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
                    return Status::ERROR;
                }
            }
            else
            {
                if (MnnCopySessionOutput(m_ctx, m_mnn_ctx, tensor_name.c_str(), mat->GetData()))
                {
                    std::string info = "MnnCopySessionOutput " + tensor_name + " failed";
                    AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
                    return Status::ERROR;
                }
            }
        }

        return Status::OK;
    }

private:
    Context *m_ctx;
    MNN::Context m_mnn_ctx;
    DT_BOOL m_is_valid;
    DT_BOOL m_is_input;
    TensorDescMap m_tensor_desc_map;
};

class MnnExecutorImplVx : public MnnExecutorImpl
{
public:
    MnnExecutorImplVx(Context *ctx, const std::shared_ptr<MnnModel> &model, const NNConfig &config)
                      : MnnExecutorImpl(ctx, model, config), m_mnn_ctx(DT_NULL)
    {
        do
        {
            if (("cpu" == m_model->GetBackendType()) && (NNBackend::GPU == m_config.backend))
            {
                AURA_ADD_ERROR_STRING(ctx, "cpu backend is not compatible with gpu backend");
                break;
            }

            if ((m_config.precision != m_model->GetPrecision()) ||
                (m_config.memory    != m_model->GetMemory())    ||
                (m_config.tuning    != m_model->GetTuning())    ||
                (m_config.cl_mem    != m_model->GetCLMem()))
            {
                if (NNBackend::CPU == m_config.backend)
                {
                    AURA_LOGI(m_ctx, AURA_TAG, "input parameters do not match model configuration, making input parameters valid\n");
                }
                else if (NNBackend::GPU == m_config.backend)
                {
                    AURA_LOGI(m_ctx, AURA_TAG, "input parameters do not match model configuration, making model parameters valid\n");
                    m_config.precision = m_model->GetPrecision();
                    m_config.memory    = m_model->GetMemory();
                    m_config.tuning    = m_model->GetTuning();
                    m_config.cl_mem    = m_model->GetCLMem();
                }
            }

            //get mnn version : v2.7.1
            std::vector<std::string> mnn_version;
            m_version = MnnUtils::GetLibraryVersion(ctx);
            mnn_version = NNSplit(m_version, '.');

            // framework version : mnn.v2.7.1
            std::vector<std::string> framework_version;
            framework_version = NNSplit(m_model->GetFrameWorkVersion(), '.');

            if ((4 == framework_version.size()) && (mnn_version.size() >= 3))
            {
                if ((framework_version[1] != mnn_version[0]) || (framework_version[2] != mnn_version[1]) || (framework_version[3] != mnn_version[2]))
                {
                    std::string info = "frame version not match mnn version, frame version: " + m_model->GetFrameWorkVersion() + " mnn version: " + m_version;
                    AURA_ADD_ERROR_STRING(ctx, info.c_str());
                    break;
                }
            }
            else
            {
                AURA_ADD_ERROR_STRING(ctx, "frame version size must be 4 and mnn version size can not be less than 3");
                break;
            }

            auto init_func = [=]() -> Status
            {
                if (CreateRuntime() != Status::OK)
                {
                    AURA_LOGE(m_ctx, AURA_TAG, "CreateRuntime failed\n");
                    return Status::ERROR;
                }

                m_is_valid = DT_TRUE;
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

    ~MnnExecutorImplVx()
    {
        if (m_token.valid())
        {
            m_token.wait();
        }

        if (m_mnn_ctx != NULL)
        {
            MnnUnit(m_ctx, &m_mnn_ctx);
            m_mnn_ctx = NULL;
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

        m_input_map = std::make_shared<MnnTensorMap>(m_ctx, m_mnn_ctx, DT_TRUE);
        m_output_map = std::make_shared<MnnTensorMap>(m_ctx, m_mnn_ctx, DT_FALSE);

        return Status::OK;
    }

    std::string GetVersion() override
    {
        return (m_model->GetVersion() + " device(mnn." + m_version + ")");
    }

    Status Forward(const MatMap &input, MatMap &output, DT_S32 graph_id) override
    {
        AURA_UNUSED(graph_id);
        Status ret = Status::ERROR;

        MatMap input_mapped = m_model->MapMatNames(input, DT_TRUE);
        MatMap output_mapped = m_model->MapMatNames(output, DT_FALSE);

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

        if (MnnRunSession(m_ctx, m_mnn_ctx))
        {
            AURA_ADD_ERROR_STRING(m_ctx, "runSession failed");
            return Status::ERROR;
        }

        ret = m_output_map->Initialize(&output_mapped);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "m_output_map Initialize failed");
            return ret;
        }
        return ret;
    }

    std::vector<TensorDescMap> GetInputs() override
    {
        TensorDescMap tensor_desc_map = MnnUtils::GetTensorDescMap(m_ctx, m_mnn_ctx, DT_TRUE);
        tensor_desc_map = m_model->MapTensorDescNames(tensor_desc_map, DT_TRUE);
        return {tensor_desc_map};
    }

    std::vector<TensorDescMap> GetOutputs() override
    {
        TensorDescMap tensor_desc_map = MnnUtils::GetTensorDescMap(m_ctx, m_mnn_ctx, DT_FALSE);
        tensor_desc_map = m_model->MapTensorDescNames(tensor_desc_map, DT_FALSE);
        return {tensor_desc_map};
    }

private:
    Status CreateRuntime()
    {
        Status ret = Status::ERROR;

        Buffer buffer = m_model->GetModelBuffer();
        if (!buffer.IsValid())
        {
            AURA_ADD_ERROR_STRING(m_ctx, "GetModelBuffer failed");
            goto EXIT;
        }

        m_mnn_ctx = MnnInit(m_ctx, GetMnnBackend(m_config.backend), GetMnnPrecision(m_config.precision),
                            GetMnnMemory(m_config.memory), GetMnnTuning(m_config.tuning), GetMnnCLMem(m_config.cl_mem),
                            m_model->GetMinnVersion(), buffer.m_data, buffer.m_size, NULL, DT_FALSE, m_config.profiling_path.c_str());
        if (DT_NULL == m_mnn_ctx)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "MnnInit failed");
            goto EXIT;
        }

        ret = Status::OK;
EXIT:
        m_model->ReleaseModelBuffer();
        return ret;
    }

    std::string m_version;
    MNN::Context m_mnn_ctx;
    std::shared_ptr<MnnTensorMap> m_input_map;
    std::shared_ptr<MnnTensorMap> m_output_map;
};

} // namespace aura