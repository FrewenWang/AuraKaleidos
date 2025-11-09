/** @brief      : xnn executor impl x inl for aura
 *  @file       : xnn_executor_impl_x.inl
 *  @author     : jiyingyu@xiaomi.com
 *  @version    : 1.0.0
 *  @date       : Mar. 29, 2024
 *  @Copyright  : Copyright 2023 Xiaomi Mobile Software Co., Ltd. All Rights reserved.
 */

#include "aura/runtime/core/status.hpp"

namespace aura
{

class XnnIOBufferMap;

struct RegisterBufferMap
{
    XnnIOBufferMap *input_map;
    XnnIOBufferMap *output_map;
};

struct XnnPerfOptions
{
    XnnPriority  priority;
    XnnPowerMode power_mode;
};

AURA_INLINE XnnPerfOptions GetXnnPerfOption(const NNPerfLevel &perf_level)
{
    XnnPerfOptions options;
    memset(&options, 0, sizeof(options));

    if (NNPerfLevel::PERF_LOW == perf_level)
    {
        options.priority   = XnnPriority::XNN_PRIORITY_LOW;
        options.power_mode = XnnPowerMode::XNN_POWER_LOW;
    }
    else if (NNPerfLevel::PERF_NORMAL == perf_level)
    {
        options.priority   = XnnPriority::XNN_PRIORITY_MID;
        options.power_mode = XnnPowerMode::XNN_POWER_BALANCE;
    }
    else
    {
        options.priority   = XnnPriority::XNN_PRIORITY_HIGH;
        options.power_mode = XnnPowerMode::XNN_POWER_FULL;
    }

    return options;
}

AURA_INLINE XnnLogLevel GetXnnLogLevel(const NNLogLevel &log_level)
{
    switch (log_level)
    {
        case NNLogLevel::LOG_DEBUG:
        {
            return XnnLogLevel::XNN_DEBUG;
        }

        case NNLogLevel::LOG_INFO:
        {
            return XnnLogLevel::XNN_INFO;
        }

        default:
        {
            return XnnLogLevel::XNN_ERROR;
        }
    }
}

AURA_INLINE ElemType GetElemType(XnnPrecisionType elem_type)
{
    switch (elem_type)
    {
        case XnnPrecisionType::kUInt8:
        {
            return ElemType::U8;
        }

        case XnnPrecisionType::kInt8:
        {
            return ElemType::S8;
        }

        case XnnPrecisionType::kUInt16:
        {
            return ElemType::U16;
        }

        case XnnPrecisionType::kInt16:
        {
            return ElemType::S16;
        }

        // case XnnPrecisionType::kUInt32:
        // {
        //     return ElemType::U32;
        // }

        case XnnPrecisionType::kInt32:
        {
            return ElemType::S32;
        }

        case XnnPrecisionType::kFloat:
        {
            return ElemType::F32;
        }

        // case XnnPrecisionType::kFP64:
        // {
        //     return ElemType::F64;
        // }

        // case XnnPrecisionType::kFP16:
        // {
        //     return ElemType::F16;
        // }

        default:
        {
            return ElemType::INVALID;
        }
    }
}

static const std::string g_xnn_lib_name = "libxnn.so";

class XnnLibrary : public NNLibrary
{
public:
    static XnnLibrary& Get()
    {
        static XnnLibrary xnn_library;
        return xnn_library;
    }

    Status Destroy() override
    {
        return UnLoad();
    }
public:
    AURA_API_DEF(XNNPredictor_create) = XNNPredictor_Error_t (*)(XNN_PredictorHandle_t*, XNN_MobileConfig*);
    AURA_API_PTR(XNNPredictor_create);

    AURA_API_DEF(XNNPredictor_getInput) = XNN_TensorHandle_t (*)(XNN_PredictorHandle_t, DT_S32);
    AURA_API_PTR(XNNPredictor_getInput);

    AURA_API_DEF(XNNPredictor_getOutput) = XNN_TensorHandle_t (*)(XNN_PredictorHandle_t, DT_S32);
    AURA_API_PTR(XNNPredictor_getOutput);

    AURA_API_DEF(XNNPredictor_getInputCount) = DT_U32 (*)(XNN_PredictorHandle_t);
    AURA_API_PTR(XNNPredictor_getInputCount);

    AURA_API_DEF(XNNPredictor_getOutputCount) = DT_U32 (*)(XNN_PredictorHandle_t);
    AURA_API_PTR(XNNPredictor_getOutputCount);

    AURA_API_DEF(XNNPredictor_getInputInfo) = XNNPredictor_Error_t (*)(XNN_PredictorHandle_t, XnnIOInfo*);
    AURA_API_PTR(XNNPredictor_getInputInfo);

    AURA_API_DEF(XNNPredictor_getOutputInfo) = XNNPredictor_Error_t (*)(XNN_PredictorHandle_t, XnnIOInfo*);
    AURA_API_PTR(XNNPredictor_getOutputInfo);

    AURA_API_DEF(XNNPredictor_run) = XNNPredictor_Error_t (*)(XNN_PredictorHandle_t);
    AURA_API_PTR(XNNPredictor_run);

    AURA_API_DEF(XNN_getVersion) = XNN_Error_t (*)(DT_CHAR*);
    AURA_API_PTR(XNN_getVersion);

    AURA_API_DEF(XNNPredictor_free) = XNNPredictor_Error_t (*)(XNN_PredictorHandle_t);
    AURA_API_PTR(XNNPredictor_free);

    AURA_API_DEF(XNNTensor_shareExternalMemoryWithOffset) = XNNTensor_Error_t (*)(XNN_TensorHandle_t, XnnTargetType, DT_U32, DT_U8*, DT_S32, DT_U32, DT_U32);
    AURA_API_PTR(XNNTensor_shareExternalMemoryWithOffset);

    AURA_API_DEF(XNNTensor_unmapShareMemory) = XNNTensor_Error_t (*)(XNN_TensorHandle_t, DT_S32);
    AURA_API_PTR(XNNTensor_unmapShareMemory);

    AURA_API_DEF(XNNTensor_cacheEnd) = XNNTensor_Error_t (*)(XNN_TensorHandle_t);
    AURA_API_PTR(XNNTensor_cacheEnd);

private:
    XnnLibrary() : NNLibrary(), m_handle(DT_NULL)
    {
        Load();
    }

    ~XnnLibrary()
    {
        Destroy();
    }

    AURA_DISABLE_COPY_AND_ASSIGN(XnnLibrary);

    Status Load() override
    {
        Status ret = Status::ERROR;

        dlerror();
        m_handle = dlopen(g_xnn_lib_name.c_str(), RTLD_LAZY | RTLD_LOCAL);
        if (DT_NULL == m_handle)
        {
            std::string info = "dlopen libxnn.so failed, err : " + std::string(dlerror());
            AURA_PRINTE(AURA_TAG, "%s\n", info.c_str());
            return ret;
        }

        do
        {
            AURA_DLSYM_API(m_handle, XNNPredictor_create);
            AURA_DLSYM_API(m_handle, XNNPredictor_getInput);
            AURA_DLSYM_API(m_handle, XNNPredictor_getOutput);
            AURA_DLSYM_API(m_handle, XNNPredictor_getInputCount);
            AURA_DLSYM_API(m_handle, XNNPredictor_getOutputCount);
            AURA_DLSYM_API(m_handle, XNNPredictor_getInputInfo);
            AURA_DLSYM_API(m_handle, XNNPredictor_getOutputInfo);
            AURA_DLSYM_API(m_handle, XNNPredictor_run);
            AURA_DLSYM_API(m_handle, XNN_getVersion);
            AURA_DLSYM_API(m_handle, XNNPredictor_free);
            AURA_DLSYM_API(m_handle, XNNTensor_shareExternalMemoryWithOffset);
            AURA_DLSYM_API(m_handle, XNNTensor_unmapShareMemory);
            AURA_DLSYM_API(m_handle, XNNTensor_cacheEnd);

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

static XNNPredictor_Error_t XNNPredictor_create(Context *ctx, XNN_PredictorHandle_t *predictor_handle, XNN_MobileConfig *config)
{
    auto func = XnnLibrary::Get().XNNPredictor_create;
    if (func)
    {
        return func(predictor_handle, config);
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "XNNPredictor_create is null ptr");
        return XNN_PREDICTOR_ERROR_INVALID_HANDLE;
    }
}

static XNN_TensorHandle_t XNNPredictor_getInput(Context *ctx, XNN_PredictorHandle_t predictor_handle, DT_S32 i)
{
    auto func = XnnLibrary::Get().XNNPredictor_getInput;
    if (func)
    {
        return func(predictor_handle, i);
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "XNNPredictor_getInput is null ptr");
        return DT_NULL;
    }
}

static XNN_TensorHandle_t XNNPredictor_getOutput(Context *ctx, XNN_PredictorHandle_t predictor_handle, DT_S32 i)
{
    auto func = XnnLibrary::Get().XNNPredictor_getOutput;
    if (func)
    {
        return func(predictor_handle, i);
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "XNNPredictor_getOutput is null ptr");
        return DT_NULL;
    }
}

static DT_U32 XNNPredictor_getInputCount(Context *ctx, XNN_PredictorHandle_t predictor_handle)
{
    DT_U32 count = 0;
    auto func = XnnLibrary::Get().XNNPredictor_getInputCount;
    if (func)
    {
        count = func(predictor_handle);
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "XNNPredictor_getInputCount is null ptr");
    }
    return count;
}

static DT_U32 XNNPredictor_getOutputCount(Context *ctx, XNN_PredictorHandle_t predictor_handle)
{
    DT_U32 count = 0;
    auto func = XnnLibrary::Get().XNNPredictor_getOutputCount;
    if (func)
    {
        count = func(predictor_handle);
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "XNNPredictor_getOutputCount is null ptr");
    }
    return count;
}

static XNNPredictor_Error_t XNNPredictor_getInputInfo(Context *ctx, XNN_PredictorHandle_t predictor_handle, XnnIOInfo *input_info)
{
    auto func = XnnLibrary::Get().XNNPredictor_getInputInfo;
    if (func)
    {
        return func(predictor_handle, input_info);
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "XNNPredictor_getInputInfo is null ptr");
        return XNN_PREDICTOR_ERROR_INVALID_HANDLE;
    }
}

static XNNPredictor_Error_t XNNPredictor_getOutputInfo(Context *ctx, XNN_PredictorHandle_t predictor_handle, XnnIOInfo *output_info)
{
    auto func = XnnLibrary::Get().XNNPredictor_getOutputInfo;
    if (func)
    {
        return func(predictor_handle, output_info);
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "XNNPredictor_getOutputInfo is null ptr");
        return XNN_PREDICTOR_ERROR_INVALID_HANDLE;
    }
}

static XNNPredictor_Error_t XNNPredictor_run(Context *ctx, XNN_PredictorHandle_t predictor_handle)
{
    auto func = XnnLibrary::Get().XNNPredictor_run;
    if (func)
    {
        return func(predictor_handle);
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "XNNPredictor_run is null ptr");
        return XNN_PREDICTOR_ERROR_INVALID_HANDLE;
    }
}

static XNN_Error_t XNN_getVersion(Context *ctx, DT_CHAR *version)
{
    auto func = XnnLibrary::Get().XNN_getVersion;
    if (func)
    {
        return func(version);
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "XNN_getVersion is null ptr");
        return XNN_COMMON_ERROR_INVALID_PARAM;
    }
}

static XNNPredictor_Error_t XNNPredictor_free(Context *ctx, XNN_PredictorHandle_t predictor_handle)
{
    auto func = XnnLibrary::Get().XNNPredictor_free;
    if (func)
    {
        return func(predictor_handle);
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "XNNPredictor_free is null ptr");
        return XNN_PREDICTOR_ERROR_INVALID_HANDLE;
    }
}

static XNNTensor_Error_t XNNTensor_shareExternalMemoryWithOffset(Context *ctx, XNN_TensorHandle_t tensor_handle,
                                                                 XnnTargetType target, DT_U32 size, DT_U8 *addr,
                                                                 DT_U64 mem_handle, DT_U32 offset, DT_U32 io_size)
{
    auto func = XnnLibrary::Get().XNNTensor_shareExternalMemoryWithOffset;
    if (func)
    {
        return func(tensor_handle, target, size, addr, mem_handle, offset, io_size);
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "XNNTensor_shareExternalMemoryWithOffset is null ptr");
        return XNN_TENSOR_ERROR_EXTERNAL_MEMORY_NULL;
    }
}

static XNNTensor_Error_t XNNTensor_unmapShareMemory(Context *ctx, XNN_TensorHandle_t tensor_handle, DT_U64 mem_handle)
{
    auto func = XnnLibrary::Get().XNNTensor_unmapShareMemory;
    if (func)
    {
        return func(tensor_handle, mem_handle);
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "XNNTensor_unmapShareMemory is null ptr");
        return XNN_TENSOR_ERROR_EXTERNAL_MEMORY_NULL;
    }
}

static XNNTensor_Error_t XNNTensor_cacheEnd(Context *ctx, XNN_TensorHandle_t tensor_handle)
{
    auto func = XnnLibrary::Get().XNNTensor_cacheEnd;
    if (func)
    {
        return func(tensor_handle);
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "XNNTensor_cacheEnd is null ptr");
        return XNN_TENSOR_ERROR_EXTERNAL_MEMORY_NULL;
    }
}

class XnnUtils
{
public:
    static std::string GetLibraryVersion(Context *ctx)
    {
        if (DT_NULL == ctx)
        {
            return std::string();
        }

        DT_CHAR version[512];
        memset(version, 0, sizeof(version));

        XNN_Error_t ret = XNN_getVersion(ctx, version);
        if (ret != XNN_SUCCESS)
        {
            std::string info = "XNN_getVersion failed, err : " + std::to_string(ret);
            AURA_ADD_ERROR_STRING(ctx, info.c_str());
            return std::string();
        }
        return "v" + std::string(version);
    }

    static TensorDescMap GetTensorDescMap(Context *ctx, XNN_PredictorHandle_t xnn_handle, DT_BOOL is_input)
    {
        if (DT_NULL == ctx)
        {
            return TensorDescMap();
        }

        if (DT_NULL == xnn_handle)
        {
            AURA_ADD_ERROR_STRING(ctx, "null ptr");
            return TensorDescMap();
        }

        DT_U32 tensor_num = is_input ? XNNPredictor_getInputCount(ctx, xnn_handle)
                                     : XNNPredictor_getOutputCount(ctx, xnn_handle);
        if (tensor_num <= 0)
        {
            std::string info = "get tensor num failed, get num: " + std::to_string(tensor_num);
            AURA_ADD_ERROR_STRING(ctx, info.c_str());
            return TensorDescMap();
        }

        std::vector<XnnIOInfo> tensor_desc(tensor_num);
        XNNPredictor_Error_t err = is_input ? XNNPredictor_getInputInfo(ctx, xnn_handle, tensor_desc.data())
                                            : XNNPredictor_getOutputInfo(ctx, xnn_handle, tensor_desc.data());
        if (err != XNN_PREDICTOR_NO_ERROR)
        {
            std::string info = "get tensor desc failed, err: " + std::to_string(err);
            AURA_ADD_ERROR_STRING(ctx, info.c_str());
            return TensorDescMap();
        }

        TensorDescMap tensor_desc_map;
        for (DT_U32 i = 0; i < tensor_num; i++)
        {
            TensorDesc desc;
            for (DT_U32 j = tensor_desc[i].dim_count; j > 0; j--)
            {
                desc.sizes.push_back(tensor_desc[i].dim_size[j - 1]);
            }
            desc.elem_type = GetElemType(tensor_desc[i].data_format);
            desc.scale = tensor_desc[i].scale;
            desc.zero_point = tensor_desc[i].zero_point;
            tensor_desc_map[tensor_desc[i].name] = desc;
        }

        return tensor_desc_map;
    }
};

class XnnIOBufferMap
{
public:
    XnnIOBufferMap(Context *ctx, DT_VOID *xnn_handle, DT_BOOL is_input)
                   : m_ctx(ctx), m_is_valid(DT_FALSE), m_is_input(is_input)
    {
        do
        {
            if ((DT_NULL == m_ctx) || (DT_NULL == xnn_handle))
            {
                AURA_ADD_ERROR_STRING(m_ctx, "null ptr");
                break;
            }

            if (m_is_input)
            {
                DT_U32 num = XNNPredictor_getInputCount(m_ctx, xnn_handle);
                if (num <= 0)
                {
                    std::string info = "XNNPredictor_getInputCount failed, input num : " + std::to_string(num);
                    AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
                    break;
                }

                m_tensor_handle.resize(num);
                for (DT_U32 i = 0; i < num; i++)
                {
                    XNN_TensorHandle_t input_tensor = XNNPredictor_getInput(m_ctx, xnn_handle, i);
                    if (DT_NULL == input_tensor)
                    {
                        std::string info = "XNNPredictor_getInput failed, index : " + std::to_string(i);
                        AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
                        break;
                    }
                    m_tensor_handle[i] = input_tensor;
                }

                m_io_info.resize(num);
                XNNPredictor_Error_t err = XNNPredictor_getInputInfo(m_ctx, xnn_handle, m_io_info.data());
                if (err != XNN_PREDICTOR_NO_ERROR)
                {
                    std::string info = "XNNPredictor_getInputInfo failed, err : " + std::to_string(err);
                    AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
                }
            }
            else
            {
                DT_U32 num = XNNPredictor_getOutputCount(m_ctx, xnn_handle);
                if (num <= 0)
                {
                    std::string info = "XNNPredictor_getOutputCount failed, output num : " + std::to_string(num);
                    AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
                    break;
                }

                m_tensor_handle.resize(num);
                for (DT_U32 i = 0; i < num; i++)
                {
                    XNN_TensorHandle_t output_tensor = XNNPredictor_getOutput(m_ctx, xnn_handle, i);
                    if (DT_NULL == output_tensor)
                    {
                        std::string info = "XNNPredictor_getOutput failed, index : " + std::to_string(i);
                        AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
                        break;
                    }
                    m_tensor_handle[i] = output_tensor;
                }

                m_io_info.resize(num);
                XNNPredictor_Error_t err = XNNPredictor_getOutputInfo(m_ctx, xnn_handle, m_io_info.data());
                if (err != XNN_PREDICTOR_NO_ERROR)
                {
                    std::string info = "XNNPredictor_getOutputInfo failed, err : " + std::to_string(err);
                    AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
                }
            }

            m_is_valid = DT_TRUE;
        } while (0);
    }

    ~XnnIOBufferMap()
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

        if ((DT_NULL == mat_map) || (mat_map->size() != m_io_info.size()))
        {
            AURA_ADD_ERROR_STRING(m_ctx, "size match error");
            return Status::ERROR;
        }

        m_mat_map = *mat_map;

        for (size_t i = 0; i < m_io_info.size(); i++)
        {
            std::string tensor_name = std::string(m_io_info[i].name);
            if (m_mat_map.find(tensor_name) == m_mat_map.end())
            {
                std::string info = "mat names " + tensor_name + " not provided";
                AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
                return Status::ERROR;
            }

            if (m_mat_map[tensor_name]->GetBuffer().m_type != AURA_MEM_DMA_BUF_HEAP)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "memory type is not valid, only support dmabuf");
                return Status::ERROR;
            }

            if (InitXnnTensor(i, tensor_name) != Status::OK)
            {
                std::string info = "InitXnnTensor " + tensor_name + " failed";
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

        for (size_t i = 0; i < m_io_info.size(); i++)
        {
            std::string name = m_io_info[i].name;

            if (m_quant_map.count(name))
            {
                DT_S32 zero_point = m_io_info[i].zero_point;
                DT_F32 scale = m_io_info[i].scale;

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

        for (size_t i = 0; i < m_io_info.size(); i++)
        {
            std::string name = m_io_info[i].name;

            if (m_quant_map.count(name))
            {
                DT_S32 zero_point = m_io_info[i].zero_point;
                DT_F32 scale = m_io_info[i].scale;

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

    std::vector<Mat>* GetRegisterMat()
    {
        return &m_register_mat;
    }

    Status RegisterMem()
    {
        for (size_t i = 0; i < m_io_info.size(); i++)
        {
            std::string name = std::string(m_io_info[i].name);
            if (m_register_map.find(name) == m_register_map.end())
            {
                std::string info = "cannot find key: " + name;
                AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
                return Status::ERROR;
            }

            DT_U8 *data       = reinterpret_cast<DT_U8*>(m_register_map[name]->GetData());
            DT_U64 fd         = m_register_map[name]->GetBuffer().m_property;
            DT_U32 total_size = m_register_map[name]->GetBuffer().m_capacity;
            DT_U32 offset     = m_register_map[name]->GetBuffer().GetOffset();

            // TODO: note the `total_size` should be the capacity of the buffer.
            //       xnn ask `io_size` aligned to 64B, and `total_size > offset + io_size`,
            //       consider the dmabufheap is aligned to 4K, we remove stride check.
            //       this is a bug of xnn, we should fix it on F3.
            total_size = Max(total_size, static_cast<DT_U32>(m_io_info[i].size));
            XNNTensor_Error_t err = XNNTensor_shareExternalMemoryWithOffset(m_ctx, m_tensor_handle[i], XNN_NPU,
                                                                            total_size, data, fd, offset,
                                                                            m_io_info[i].size);
            if (err != XNN_TENSOR_NO_ERROR)
            {
                std::string info = "XNNTensor_shareExternalMemoryWithOffset failed, err : " + std::to_string(err);
                AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
                return Status::ERROR;
            }
        }

        return Status::OK;
    }

    Status DeRegisterMem()
    {
        for (size_t i = 0; i < m_io_info.size(); i++)
        {
            std::string name = std::string(m_io_info[i].name);
            if (m_register_map.find(name) == m_register_map.end())
            {
                std::string info = "cannot find key: " + name;
                AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
                return Status::ERROR;
            }

            DT_U64 fd = m_register_map[name]->GetBuffer().m_property;

            XNNTensor_Error_t err = XNNTensor_unmapShareMemory(m_ctx, m_tensor_handle[i], fd);
            if (err != XNN_TENSOR_NO_ERROR)
            {
                std::string info = "XNNTensor_unmapShareMemory failed, err : " + std::to_string(err);
                AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
                return Status::ERROR;
            }
        }

        return Status::OK;
    }

    Status Sync()
    {
        if (!m_is_input)
        {
            for (size_t i = 0; i < m_io_info.size(); i++)
            {
                XNNTensor_Error_t err = XNNTensor_cacheEnd(m_ctx, m_tensor_handle[i]);
                if (err != XNN_TENSOR_NO_ERROR)
                {
                    std::string info = "XNNTensor_cacheEnd failed, err : " + std::to_string(err);
                    AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
                    return Status::ERROR;
                }
            }
        }

        return Status::OK;
    }

private:
    Status InitXnnTensor(DT_S32 id, const std::string &tensor_name)
    {
        XnnIOInfo io_info = m_io_info[id];
        Mat *mat = m_mat_map[tensor_name];

        DT_BOOL is_quant = DT_FALSE;
        ElemType src_elem_type = mat->GetElemType();
        ElemType dst_elem_type = GetElemType(io_info.data_format);
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
                                                      (ElemType::S16 == dst_elem_type) ||
                                                      (ElemType::S32 == dst_elem_type)))
        {
            is_quant = DT_TRUE;
        }
        else
        {
            std::string info = "tensor dataType " + ElemTypesToString(dst_elem_type) + " is not match to mat element type " + ElemTypesToString(src_elem_type);
            AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
            return Status::ERROR;
        }

        if (is_quant)
        {
            if (!m_quant_map.count(tensor_name))
            {
                m_quant_map[tensor_name] = Create<Mat>(m_ctx, dst_elem_type, mat->GetSizes(), AURA_MEM_DMA_BUF_HEAP);
                if (!m_quant_map[tensor_name]->IsValid())
                {
                    AURA_ADD_ERROR_STRING(m_ctx, "mat is invalid");
                    return Status::ERROR;
                }
            }

            mat = m_quant_map[tensor_name];
        }

        auto get_elem_counts_func = [](const XnnIOInfo &info) -> DT_S32
        {
            DT_S32 elem_counts = 1;
            for (DT_U32 i = 0; i < info.dim_count; i++)
            {
                elem_counts *= info.dim_size[i];
            }
            return elem_counts;
        };

        DT_S32 elem_counts = get_elem_counts_func(io_info);
        if (elem_counts != mat->GetSizes().Total())
        {
            std::string info = "tensor " + std::string(tensor_name) + ": expected " + std::to_string(elem_counts) + " bytes, "
                               "but got " + std::to_string(mat->GetSizes().Total()) + " bytes";
            AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
            return Status::ERROR;
        }

        m_register_map[tensor_name] = mat;

        return Status::OK;
    }

private:
    Context *m_ctx;
    MatMap  m_mat_map;
    MatMap  m_quant_map;
    MatMap  m_register_map;
    DT_BOOL m_is_valid;
    DT_BOOL m_is_input;
    std::vector<XNN_TensorHandle_t> m_tensor_handle;
    std::vector<XnnIOInfo> m_io_info;
    std::vector<Mat> m_register_mat;
};

class XnnExecutorImplVx : public XnnExecutorImpl
{
public:
    XnnExecutorImplVx(Context *ctx, const std::shared_ptr<XnnModel> &model, const NNConfig &config)
                      : XnnExecutorImpl(ctx, model, config), m_xnn_handle(DT_NULL)
    {
        do
        {
            // get xnn version : xnn.v1.0.0
            std::vector<std::string> xnn_version;
            m_version = XnnUtils::GetLibraryVersion(ctx);
            xnn_version = NNSplit(m_version, '.');

            // framework version
            std::vector<std::string> framework_version;
            framework_version = NNSplit(m_model->GetFrameWorkVersion(), '.');

            if ((framework_version.size() != 4) || (xnn_version.size() < 3))
            {
                AURA_ADD_ERROR_STRING(ctx, "frame version size must be 4 and xnn version size must be 3");
                break;
            }

            auto init_func = [=]() -> Status
            {
                Status ret = CreatePredictor();
                if (ret != Status::OK)
                {
                    AURA_ADD_ERROR_STRING(m_ctx, "CreatePredictor failed");
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

    ~XnnExecutorImplVx()
    {
        if (m_token.valid())
        {
            m_token.wait();
        }

        if (DeInitialize() != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "DeInitialize failed");
        }

        if (m_xnn_handle != DT_NULL)
        {
            XNNPredictor_Error_t err = XNNPredictor_free(m_ctx, m_xnn_handle);
            if (err != XNN_PREDICTOR_NO_ERROR)
            {
                std::string info = "XNNPredictor_free failed, err : " + std::to_string(err);
                AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
            }
            m_xnn_handle = DT_NULL;
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

        // create XnnIOBufferMap
        m_input_map = std::make_shared<XnnIOBufferMap>(m_ctx, m_xnn_handle, DT_TRUE);
        m_output_map = std::make_shared<XnnIOBufferMap>(m_ctx, m_xnn_handle, DT_FALSE);

        return Status::OK;
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

        XnnIOBufferMap *input_map  = NULL;
        XnnIOBufferMap *output_map = NULL;

        RegisterBufferMap *register_map = GetRegisterBufferMap(input_mapped, output_mapped);
        if (register_map != NULL)
        {
            input_map  = register_map->input_map;
            output_map = register_map->output_map;
        }
        else
        {
            input_map  = m_input_map.get();
            output_map = m_output_map.get();

            ret = input_map->Initialize(&input_mapped);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "input_map Initialize failed");
                return ret;
            }

            ret = input_map->RegisterMem();
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "input_map RegisterMem failed");
                return ret;
            }

            ret = output_map->Initialize(&output_mapped);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "output_map Initialize failed");
                return ret;
            }

            ret = output_map->RegisterMem();
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "output_map RegisterMem failed");
                return ret;
            }
        }

        ret = input_map->Quant();
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "m_input_map Quant failed");
            return ret;
        }

        ret = output_map->Sync();
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "m_input_map Sync failed");
            return ret;
        }

        DT_S32 err_code = XNNPredictor_run(m_ctx, m_xnn_handle);
        if (err_code != XNN_PREDICTOR_NO_ERROR)
        {
            std::string info = "XNNPredictor_run failed, err : " + std::to_string(err_code);
            AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
            return Status::ERROR;
        }

        ret = output_map->DeQuant();
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "m_output_map DeQuant failed");
        }

        if (NULL == register_map)
        {
            ret = input_map->DeRegisterMem();
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "input_map DeRegisterMem failed");
                return ret;
            }

            ret = output_map->DeRegisterMem();
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "output_map DeRegisterMem failed");
                return ret;
            }
        }

        return ret;
    }

    std::vector<TensorDescMap> GetInputs() override
    {
        TensorDescMap tensor_desc_map = XnnUtils::GetTensorDescMap(m_ctx, m_xnn_handle, DT_TRUE);
        tensor_desc_map = m_model->MapTensorDescNames(tensor_desc_map, DT_TRUE);
        return {tensor_desc_map};
    }

    std::vector<TensorDescMap> GetOutputs() override
    {
        TensorDescMap tensor_desc_map = XnnUtils::GetTensorDescMap(m_ctx, m_xnn_handle, DT_FALSE);
        tensor_desc_map = m_model->MapTensorDescNames(tensor_desc_map, DT_FALSE);
        return {tensor_desc_map};
    }

    std::string GetVersion() override
    {
        return (m_model->GetVersion() + " device(xnn." + m_version + ")");
    }

    Status Update(const std::string &name, AnyParams &params) override
    {
        if ("register_mem" == name)
        {
            return RegisterMem(params);
        }
        else if ("deregister_mem" == name)
        {
            return DeRegisterMem(params);
        }
        else
        {
            std::string info = "the specified function '" + name + "' does not exist";
            AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
            return  Status::ERROR;
        }

        return  Status::ERROR;
    }

private:
    Status CreatePredictor()
    {
        Status ret = Status::ERROR;
        DT_S32 err_code = XNN_PREDICTOR_NO_ERROR;
        XNN_MobileConfig xnn_config;

        std::vector<std::string> model_name = NNSplit(m_model->GetModelName(), '/');

        Buffer buffer = m_model->GetModelBuffer();
        if (!buffer.IsValid())
        {
            AURA_ADD_ERROR_STRING(m_ctx, "GetModelBuffer failed");
            goto EXIT;
        }

        xnn_config.mPowerMode = GetXnnPerfOption(m_config.perf_level).power_mode;
        xnn_config.mPriority  = GetXnnPerfOption(m_config.perf_level).priority;
        xnn_config.mLogLevel  = GetXnnLogLevel(m_config.log_level);
        xnn_config.mModelBuf  = reinterpret_cast<DT_CHAR*>(buffer.m_data);
        xnn_config.mModelSize = buffer.m_size;

        if (model_name.size() > 0)
        {
            xnn_config.mModelDmaBufName = (DT_CHAR *)model_name[model_name.size() - 1].c_str();
        }

        if (m_model->GetMinnVersion() == 0x010000)
        {
            xnn_config.mTarget = XNN_NPU;
        }
        else if (m_model->GetMinnVersion() == 0x010001)
        {
            xnn_config.mTarget = XNN_F1NPU;
        }
        else
        {
            std::string info = "XNN not support minn version : " + std::to_string(m_model->GetMinnVersion());
            AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
            goto EXIT;
        }

        err_code = XNNPredictor_create(m_ctx, &m_xnn_handle, &xnn_config);
        if (err_code != XNN_PREDICTOR_NO_ERROR)
        {
            std::string info = "XNNPredictor_create failed, err : " + std::to_string(err_code);
            AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
            goto EXIT;
        }

        ret = Status::OK;
    EXIT:
        m_model->ReleaseModelBuffer();
        return ret;
    }

    RegisterBufferMap* GetRegisterBufferMap(const MatMap &input, const MatMap &output)
    {
        RegisterBufferMap *register_buffer_map = NULL;

        auto check_mat_register_func = [](const MatMap &matmap, const std::vector<Mat> &regster_mat) -> DT_BOOL
        {
            DT_BOOL is_register = DT_TRUE;

            for (const auto &iter : matmap)
            {
                auto match_mat = std::find_if(regster_mat.begin(), regster_mat.end(), [&iter](const Mat &mat)
                                                                    {
                                                                        return (mat.GetData() == iter.second->GetData());
                                                                    });
                if (match_mat == regster_mat.end())
                {
                    is_register = DT_FALSE;
                    break;
                }
            }

            return is_register;
        };

        for (auto &iter : m_register_buffer_map)
        {
            std::vector<Mat> *input_register_mat  = iter->input_map->GetRegisterMat();
            std::vector<Mat> *output_register_mat = iter->output_map->GetRegisterMat();

            if ((check_mat_register_func(input,  *input_register_mat)  == DT_TRUE) &&
                (check_mat_register_func(output, *output_register_mat) == DT_TRUE))
            {
                register_buffer_map = iter;
                break;
            }
        }

        return register_buffer_map;
    }

    Status RegisterImpl(const MatMap &input, const MatMap &output)
    {
        Status ret = Status::ERROR;

        MatMap input_mapped  = m_model->MapMatNames(input,  DT_TRUE);
        MatMap output_mapped = m_model->MapMatNames(output, DT_FALSE);

        // check wether registed
        if (GetRegisterBufferMap(input_mapped, output_mapped) != NULL)
        {
            return Status::OK;
        }

        RegisterBufferMap *register_buffer_map = (RegisterBufferMap *)AURA_ALLOC_PARAM(m_ctx, AURA_MEM_HEAP, sizeof(RegisterBufferMap), 0);
        if (NULL == register_buffer_map)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "AURA_ALLOC_PARAM failed");
            return ret;
        }

        register_buffer_map->input_map  = NULL;
        register_buffer_map->output_map = NULL;
        m_register_buffer_map.push_back(register_buffer_map);

        // create XnnIOBufferMap
        XnnIOBufferMap *input_map = NULL;
        XnnIOBufferMap *output_map = NULL;
        input_map = Create<XnnIOBufferMap>(m_ctx, m_xnn_handle, DT_TRUE);
        if (NULL == input_map)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "create XnnIOBufferMap failed");
            goto EXIT;
        }

        register_buffer_map->input_map  = input_map;

        output_map = Create<XnnIOBufferMap>(m_ctx, m_xnn_handle, DT_FALSE);
        if (NULL == output_map)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "create XnnIOBufferMap failed");
            goto EXIT;
        }

        register_buffer_map->output_map = output_map;

        // recode mat
        {
            std::vector<Mat> *input_register_mat  = input_map->GetRegisterMat();
            for (auto &iter : input_mapped)
            {
                input_register_mat->push_back(*iter.second);
            }

            std::vector<Mat> *output_register_mat = output_map->GetRegisterMat();
            for (auto &iter : output_mapped)
            {
                output_register_mat->push_back(*iter.second);
            }
        }

        ret = input_map->Initialize(&input_mapped);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "input_map Initialize failed");
            goto EXIT;
        }

        ret = input_map->RegisterMem();
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "input_map RegisterMem failed");
            goto EXIT;
        }

        ret = output_map->Initialize(&output_mapped);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "output_map Initialize failed");
            goto EXIT;
        }

        ret = output_map->RegisterMem();
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "output_map RegisterMem failed");
            goto EXIT;
        }

        ret = Status::OK;
EXIT:
        if (ret != Status::OK)
        {
            if (DeInitialize() != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "DeInitialize failed");
                ret = Status::OK;
            }
        }

        return ret;
    }

    Status DeRegisterImpl(const MatMap &input, const MatMap &output)
    {
        Status ret = Status::ERROR;

        MatMap input_mapped  = m_model->MapMatNames(input,  DT_TRUE);
        MatMap output_mapped = m_model->MapMatNames(output, DT_FALSE);

        // check wether registed
        RegisterBufferMap *register_buffer_map = GetRegisterBufferMap(input_mapped, output_mapped);
        if (NULL == register_buffer_map)
        {
            return Status::OK;
        }

        ret = register_buffer_map->input_map->DeRegisterMem();
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "input DeRegisterMem failed");
            ret = Status::ERROR;
        }

        ret = register_buffer_map->output_map->DeRegisterMem();
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "output DeRegisterMem failed");
            ret = Status::ERROR;
        }

        // clear register mat
        std::vector<Mat> *input_register_mat = register_buffer_map->input_map->GetRegisterMat();
        std::vector<Mat>().swap(*input_register_mat);

        std::vector<Mat> *output_register_mat = register_buffer_map->output_map->GetRegisterMat();
        std::vector<Mat>().swap(*output_register_mat);

        Delete<XnnIOBufferMap>(m_ctx, &register_buffer_map->input_map);
        Delete<XnnIOBufferMap>(m_ctx, &register_buffer_map->output_map);

        AURA_FREE(m_ctx, register_buffer_map);

        auto it = std::find(m_register_buffer_map.begin(), m_register_buffer_map.end(), register_buffer_map);
        m_register_buffer_map.erase(it);

        return ret;
    }

    Status RegisterMem(AnyParams &params)
    {
        Status ret = Status::ERROR;

        MatMap input;
        ret = params.Get("input_matmap", input);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "parse input_matmap failed");
            return ret;
        }

        MatMap output;
        ret = params.Get("output_matmap", output);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "parse output_matmap failed");
            return ret;
        }

        ret = RegisterImpl(input, output);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "Register mem failed");
            return ret;
        }

        return Status::OK;
    }

    Status DeRegisterMem(AnyParams &params)
    {
        Status ret = Status::ERROR;

        MatMap input;
        ret = params.Get("input_matmap", input);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "parse input_matmap failed");
            return ret;
        }

        MatMap output;
        ret = params.Get("output_matmap", output);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "parse output_matmap failed");
            return ret;
        }

        ret = DeRegisterImpl(input, output);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "DeRegister mem failed");
            return ret;
        }

        return Status::OK;
    }

    Status DeInitialize()
    {
        Status ret = Status::OK;

        m_input_map.reset();
        m_output_map.reset();

        for (auto &iter : m_register_buffer_map)
        {
            if (iter == NULL)
            {
                continue;
            }

            if (iter->input_map != NULL)
            {
                std::vector<Mat> *input_register_mat = iter->input_map->GetRegisterMat();
                std::vector<Mat>().swap(*input_register_mat);
                Delete<XnnIOBufferMap>(m_ctx, &iter->input_map);
                iter->input_map = NULL;
            }

            if (iter->output_map != NULL)
            {
                std::vector<Mat> *output_register_mat = iter->output_map->GetRegisterMat();
                std::vector<Mat>().swap(*output_register_mat);
                Delete<XnnIOBufferMap>(m_ctx, &iter->output_map);
                iter->output_map = NULL;
            }

            if (iter != NULL)
            {
                AURA_FREE(m_ctx, iter);
                iter = NULL;
            }
        }

        return ret;
    }

private:
    std::string m_version;
    DT_VOID *m_xnn_handle;
    std::shared_ptr<XnnIOBufferMap> m_input_map;
    std::shared_ptr<XnnIOBufferMap> m_output_map;
    std::vector<RegisterBufferMap*> m_register_buffer_map;
};

} // namespace aura