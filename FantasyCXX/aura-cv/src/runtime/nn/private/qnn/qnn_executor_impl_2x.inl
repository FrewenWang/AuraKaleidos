
namespace aura
{

// from QnnInterfaceUtils.hpp
enum ModelError_t
{
    MODEL_NO_ERROR = 0,
    MODEL_TENSOR_ERROR,
    MODEL_PARAMS_ERROR,
    MODEL_NODES_ERROR,
    MODEL_GRAPH_ERROR,
    MODEL_CONTEXT_ERROR,
    MODEL_GENERATION_ERROR,
    MODEL_SETUP_ERROR,
    MODEL_INVALID_ARGUMENT_ERROR,
    MODEL_FILE_ERROR,
    MODEL_MEMORY_ALLOCATE_ERROR,
    MODEL_UNKNOWN_ERROR = 0x7FFFFFFF
};

// from QnnInterfaceUtils.hpp
struct GraphConfigInfo_t
{
    DT_CHAR *graph_name;
    const QnnGraph_Config_t **graph_configs;
};

// from QnnInterfaceUtils.hpp
class QnnTensorMap;

struct GraphInfo_t
{
    Qnn_GraphHandle_t graph;
    DT_CHAR           *graph_name;
    Qnn_Tensor_t      *input_tensors;
    DT_U32            num_input_tensors;
    Qnn_Tensor_t      *output_tensors;
    DT_U32            num_output_tensors;
    QnnTensorMap      *input_tensor_map;
    QnnTensorMap      *output_tensor_map;
};

typedef GraphInfo_t *GraphInfoPtr_t;

enum class ProfileStage
{
    INIT_STATUS        = 0,
    EXECUTE_STATUS,
    DEINIT_STATUS,
};

AURA_INLINE QnnLog_Level_t GetQnnLogLevel(NNLogLevel log_level)
{
    QnnLog_Level_t qnn_log_level;

    if (NNLogLevel::LOG_DEBUG == log_level)
    {
        qnn_log_level = QNN_LOG_LEVEL_DEBUG;
    }
    else if (NNLogLevel::LOG_INFO == log_level)
    {
        qnn_log_level = QNN_LOG_LEVEL_INFO;
    }
    else
    {
        qnn_log_level = QNN_LOG_LEVEL_ERROR;
    }

    return qnn_log_level;
}

AURA_INLINE std::ostream& operator << (std::ostream &os, ProfileStage profile_stage)
{
    switch (profile_stage)
    {
        case ProfileStage::INIT_STATUS:
        {
            os << "Init Status";
            break;
        }

        case ProfileStage::EXECUTE_STATUS:
        {
            os << "Total Inference Time";
            break;
        }

        case ProfileStage::DEINIT_STATUS:
        {
            os << "De-Init Status";
            break;
        }

        default:
        {
            os << "undefined profile stage";
            break;
        }
    }

    return os;
}

AURA_INLINE std::string ProfileStageToString(ProfileStage stage)
{
    std::ostringstream ss;
    ss << stage;
    return ss.str();
}

AURA_INLINE ElemType GetElemType(Qnn_DataType_t data_type)
{
    switch (data_type)
    {
        case QNN_DATATYPE_UFIXED_POINT_8:
        {
            return ElemType::U8;
        }

        case QNN_DATATYPE_UFIXED_POINT_16:
        {
            return ElemType::U16;
        }

        case QNN_DATATYPE_UINT_8:
        {
            return ElemType::U8;
        }

        case QNN_DATATYPE_INT_8:
        {
            return ElemType::S8;
        }

        case QNN_DATATYPE_UINT_16:
        {
            return ElemType::U16;
        }

        case QNN_DATATYPE_INT_16:
        {
            return ElemType::S16;
        }

        case QNN_DATATYPE_UINT_32:
        {
            return ElemType::U32;
        }

        case QNN_DATATYPE_INT_32:
        {
            return ElemType::S32;
        }

        case QNN_DATATYPE_FLOAT_16:
        {
            return ElemType::F16;
        }

        case QNN_DATATYPE_FLOAT_32:
        {
            return ElemType::F32;
        }

        default:
        {
            return ElemType::INVALID;
        }
    }
}

static const std::string g_qnn_system_lib_name = {"libQnnSystem.so"};

class QnnLibrary : public NNLibrary
{
public:
    static QnnLibrary& Get(const std::string &backend_type)
    {
        if ("npu" == backend_type)
        {
            static QnnLibrary npu_library(backend_type);
            return npu_library;
        }
        else if ("cpu" == backend_type)
        {
            static QnnLibrary cpu_library(backend_type);
            return cpu_library;
        }
        else if ("gpu" == backend_type)
        {
            static QnnLibrary gpu_library(backend_type);
            return gpu_library;
        }
        else
        {
            std::string info = "Unsupported backend_type " + backend_type;
            AURA_PRINTE(AURA_TAG, "%s\n", info.c_str());
            static QnnLibrary err_library("err_backend");
            return err_library;
        }
    }

    QNN_INTERFACE_VER_TYPE GetInterface()
    {
        return m_interface;
    }

    QNN_SYSTEM_INTERFACE_VER_TYPE GetSystemInterface()
    {
        return m_system_interface;
    }

    Qnn_DeviceHandle_t GetDeviceHandle()
    {
        static DT_BOOL call_once_flag = DT_FALSE;

        if (!call_once_flag)
        {
            CreateDevice();
            call_once_flag = DT_TRUE;
        }

        return m_device_handle;
    }

    Status Destroy() override
    {
        Status ret = Status::ERROR;

        if (DeleteDevice() != Status::OK)
        {
            AURA_PRINTE(AURA_TAG, "DeleteDevice failed\n");
        }
        if (UnLoad() != Status::OK)
        {
            AURA_PRINTE(AURA_TAG, "UnLoad failed\n");
            goto EXIT;
        }

        ret = Status::OK;
EXIT:
        return ret;
    }
private:
    QnnLibrary(const std::string &backend_type) : NNLibrary(), m_backend_type(backend_type), m_qnn_backend_handle(DT_NULL),
                                                  m_qnn_sys_lib_handle(DT_NULL), m_interface(QNN_INTERFACE_VER_TYPE_INIT),
                                                  m_system_interface(QNN_SYSTEM_INTERFACE_VER_TYPE_INIT), m_device_handle(DT_NULL)
    {
        if (Load() == Status::OK)
        {
            CheckProviders();
#if defined(AURA_BUILD_HEXAGON)
            NNLibraryManager::GetInstance().AddNNLibrary(this);
#endif
        }
        else
        {
            AURA_PRINTE(AURA_TAG, "QnnLibrary Load failed\n");
        }
    }

    ~QnnLibrary()
    {
        Destroy();
    }

    AURA_DISABLE_COPY_AND_ASSIGN(QnnLibrary);

    Status Load() override
    {
        Status ret = Status::ERROR;

        do
        {
            std::string qnn_lib_name = GetQnnLibName();
            if (qnn_lib_name.empty())
            {
                AURA_PRINTE(AURA_TAG, "GetQnnLibName failed, qnn_lib_name is empty\n");
                break;
            }

            dlerror();
            m_qnn_backend_handle = dlopen(qnn_lib_name.c_str(), RTLD_LAZY | RTLD_LOCAL);
            if (DT_NULL == m_qnn_backend_handle)
            {
                std::string info = "Unable to load backend, dlerror : " + std::string(dlerror());
                AURA_PRINTE(AURA_TAG, "%s\n", info.c_str());
                break;
            }
            AURA_DLSYM_API(m_qnn_backend_handle, QnnInterface_getProviders);

            m_qnn_sys_lib_handle = dlopen(g_qnn_system_lib_name.c_str(), RTLD_LAZY | RTLD_LOCAL);
            if (DT_NULL == m_qnn_sys_lib_handle)
            {
                std::string info = "Unable to load sys_backend, dlerror : " + std::string(dlerror());
                AURA_PRINTE(AURA_TAG, "%s\n", info.c_str());
                break;
            }
            AURA_DLSYM_API(m_qnn_sys_lib_handle, QnnSystemInterface_getProviders);

            ret = Status::OK;
        } while (0);

        if (ret != Status::OK)
        {
            UnLoad();
        }

       return ret;
    }

    Status UnLoad() override
    {
        Status ret = Status::OK;

        if (m_qnn_backend_handle != DT_NULL)
        {
            if (dlclose(m_qnn_backend_handle) != 0)
            {
                AURA_PRINTE(AURA_TAG, "dlclose m_qnn_backend_handle failed\n");
                ret = Status::ERROR;
            }
            m_qnn_backend_handle = DT_NULL;
            QnnInterface_getProviders = DT_NULL;
        }

        if (m_qnn_sys_lib_handle != DT_NULL)
        {
            if (dlclose(m_qnn_sys_lib_handle) != 0)
            {
                AURA_PRINTE(AURA_TAG, "dlclose m_qnn_sys_lib_handle failed\n");
                ret = Status::ERROR;
            }
            m_qnn_sys_lib_handle = DT_NULL;
            QnnSystemInterface_getProviders = DT_NULL;
        }

        return ret;
    }

    std::string GetQnnLibName()
    {
        std::string lib_name;
#if defined(AURA_BUILD_HEXAGON)
        std::unordered_map<int, std::string> qnn_lib_name =
        {
            //frome sm8750 start suppose, sm8750: 0x79
            {0x79, "libQnnHtpV79.so"},
            {0x81, "libQnnHtpV81.so"}
        };

        qurt_arch_version_t qurt_version;
        DT_S32 ret = qurt_sysenv_get_arch_version(&qurt_version);
        if (QURT_EOK == ret)
        {
            DT_S32 cdsp_version = qurt_version.arch_version & 0xff;
            if (qnn_lib_name.count(cdsp_version))
            {
                lib_name = qnn_lib_name[cdsp_version];
            }
        }
#else
        std::unordered_map<std::string, std::string> qnn_lib_name =
        {
            {"npu", "libQnnHtp.so"},
            {"gpu", "libQnnGpu.so"},
            {"cpu", "libQnnCpu.so"}
        };

        if (qnn_lib_name.count(m_backend_type))
        {
            lib_name = qnn_lib_name[m_backend_type];
        }
#endif // AURA_BUILD_HEXAGON
        return lib_name;
    }

    Status CheckProviders()
    {
        Status ret = Status::ERROR;

        do
        {
            Qnn_ErrorHandle_t qnn_err = QNN_SUCCESS;

            //check interface_providers
            DT_BOOL found_valid_interface = DT_FALSE;
            DT_U32 num_providers = 0;

            QnnInterface_t **interface_providers = DT_NULL;
            QnnSystemInterface_t **system_interface_providers = DT_NULL;

            qnn_err = QnnInterface_getProviders((const QnnInterface_t ***)&interface_providers, &num_providers);
            if (qnn_err != QNN_SUCCESS)
            {
                std::string info = std::string("QnnInterface_getProviders failed, err : ") + std::to_string(qnn_err);
                AURA_PRINTE(AURA_TAG, "%s\n", info.c_str());
                break;
            }

            if ((DT_NULL == interface_providers) || (0 == num_providers))
            {
                AURA_PRINTE(AURA_TAG, "QnnInterface_getProviders failed : null interface providers received or 0 interface providers\n");
                break;
            }

            for (DT_U32 idx = 0; idx < num_providers; idx++)
            {
                if (DT_NULL == interface_providers[idx])
                {
                    AURA_PRINTE(AURA_TAG, "interface_providers[idx] is null\n");
                    break;
                }

                if ((QNN_API_VERSION_MAJOR == interface_providers[idx]->apiVersion.coreApiVersion.major) &&
                    (QNN_API_VERSION_MINOR <= interface_providers[idx]->apiVersion.coreApiVersion.minor))
                {
                    found_valid_interface = DT_TRUE;
                    m_interface = interface_providers[idx]->QNN_INTERFACE_VER_NAME;
                    break;
                }
            }

            if (!found_valid_interface)
            {
                AURA_PRINTE(AURA_TAG, "Unable to find a valid interface\n");
                break;
            }

            // check system_interface_providers
            found_valid_interface = DT_FALSE;
            num_providers = 0;

            qnn_err = QnnSystemInterface_getProviders((const QnnSystemInterface_t***)&system_interface_providers, &num_providers);
            if (qnn_err != QNN_SUCCESS)
            {
                std::string info = std::string("QnnSystemInterface_getProviders failed, err : ") + std::to_string(qnn_err);
                AURA_PRINTE(AURA_TAG, "%s\n", info.c_str());
                break;
            }

            if (DT_NULL == system_interface_providers || 0 == num_providers)
            {
                AURA_PRINTE(AURA_TAG, "QnnSystemInterface_getProviders failed : null interface providers received or 0 interface providers\n");
                break;
            }

            for (DT_U32 idx = 0; idx < num_providers; idx++)
            {
                if ((QNN_SYSTEM_API_VERSION_MAJOR == system_interface_providers[idx]->systemApiVersion.major) &&
                    (QNN_SYSTEM_API_VERSION_MINOR <= system_interface_providers[idx]->systemApiVersion.minor))
                {
                    found_valid_interface = DT_TRUE;
                    m_system_interface = system_interface_providers[idx]->QNN_SYSTEM_INTERFACE_VER_NAME;
                    break;
                }
            }

            if (!found_valid_interface)
            {
                AURA_PRINTE(AURA_TAG, "Unable to find a valid interface\n");
                break;
            }

            ret = Status::OK;
        } while (0);

        return ret;
    }

    Status CreateDevice()
    {
        Qnn_ErrorHandle_t qnn_err = QNN_SUCCESS;
        DT_BOOL device_supported = DT_TRUE;

        if (m_interface.propertyHasCapability != DT_NULL)
        {
            qnn_err = m_interface.propertyHasCapability(QNN_PROPERTY_GROUP_DEVICE);
            if (qnn_err != QNN_PROPERTY_NO_ERROR)
            {
                device_supported = DT_FALSE;
            }
        }

        if (device_supported)
        {
            if (m_interface.deviceCreate != DT_NULL)
            {
                qnn_err = m_interface.deviceCreate(DT_NULL, DT_NULL, &m_device_handle);
                if (qnn_err != QNN_SUCCESS)
                {
                    std::string info = std::string("Failed to create divice, err : ") + std::to_string(qnn_err);
                    AURA_PRINTE(AURA_TAG, "%s\n", info.c_str());
                    return Status::ERROR;
                }
            }
        }

        return Status::OK;
    }

    Status DeleteDevice()
    {
        if (m_device_handle)
        {
            if (m_interface.deviceFree != DT_NULL)
            {
                Qnn_ErrorHandle_t qnn_err = m_interface.deviceFree(m_device_handle);
                if (qnn_err != QNN_SUCCESS)
                {
                    std::string info = "Failed to free device, err : " + std::to_string(qnn_err);
                    AURA_PRINTE(AURA_TAG, "%s\n", info.c_str());
                    return Status::ERROR;
                }
            }
            m_device_handle = DT_NULL;
        }

        return Status::OK;
    }

    AURA_API_DEF(QnnInterface_getProviders) = Qnn_ErrorHandle_t(*)(const QnnInterface_t ***provider_list, DT_U32 *num_providers);
    AURA_API_PTR(QnnInterface_getProviders);

    AURA_API_DEF(QnnSystemInterface_getProviders) = Qnn_ErrorHandle_t(*)(const QnnSystemInterface_t ***providerList, DT_U32 *numProviders);
    AURA_API_PTR(QnnSystemInterface_getProviders);

private:
    std::string m_backend_type;
    DT_VOID *m_qnn_backend_handle;
    DT_VOID *m_qnn_sys_lib_handle;

    // qnn interface
    QNN_INTERFACE_VER_TYPE m_interface;
    QNN_SYSTEM_INTERFACE_VER_TYPE m_system_interface;

    Qnn_DeviceHandle_t m_device_handle;
};

using CreateQnnGraphFunc = ModelError_t (*)(Qnn_BackendHandle_t, QNN_INTERFACE_VER_TYPE, Qnn_ContextHandle_t,
                                            const GraphConfigInfo_t**, const uint32_t, GraphInfoPtr_t**, uint32_t*,
                                            bool, QnnLog_Callback_t, QnnLog_Level_t);
using DeleteQnnGraphFunc = ModelError_t (*)(GraphInfoPtr_t**, uint32_t);

struct ModelInterface
{
    CreateQnnGraphFunc create_func;
    DeleteQnnGraphFunc delete_func;
};

class QnnUtils
{
public:
    static std::string GetBuildId(Context *ctx, QNN_INTERFACE_VER_TYPE &interface)
    {
        if (DT_NULL == ctx)
        {
            return "";
        }

        DT_CHAR *backend_build_id = DT_NULL;
        if (DT_NULL == interface.backendGetBuildId)
        {
            AURA_ADD_ERROR_STRING(ctx, "m_qnn_interface.backendGetBuildId is null");
        }
        else
        {
            Qnn_ErrorHandle_t qnn_err = interface.backendGetBuildId((const DT_CHAR **)&backend_build_id);
            if ( qnn_err != QNN_SUCCESS)
            {
                std::string info = "Unable to get build id from the backend, err : " + std::to_string(qnn_err);
                AURA_ADD_ERROR_STRING(ctx, info.c_str());
            }
        }

        return (DT_NULL == backend_build_id) ? "" : std::string(backend_build_id);
    }

    static Status CreateLogHandle(Context *ctx, const QNN_INTERFACE_VER_TYPE &interface, Qnn_LogHandle_t &log_handle, const NNLogLevel &log_level)
    {
        if (DT_NULL == ctx)
        {
            return Status::ERROR;
        }

        Qnn_ErrorHandle_t qnn_err = interface.logCreate(DT_NULL, GetQnnLogLevel(log_level), &log_handle);
        if (qnn_err != QNN_SUCCESS)
        {
            std::string info = "Unable to initialize logging in the backend, err : " + std::to_string(qnn_err);
            AURA_ADD_ERROR_STRING(ctx, info.c_str());
            return Status::ERROR;
        }

        return Status::OK;
    }

    static Status DeleteLogHandle(Context *ctx, const QNN_INTERFACE_VER_TYPE &interface, Qnn_LogHandle_t &log_handle)
    {
        if (DT_NULL == ctx)
        {
            return Status::ERROR;
        }

        if (log_handle)
        {
            Qnn_ErrorHandle_t qnn_err = interface.logFree(log_handle);
            if (qnn_err != QNN_SUCCESS)
            {
                std::string info = "Unable to terminate logging in the backend, err : " + std::to_string(qnn_err);
                AURA_ADD_ERROR_STRING(ctx, info.c_str());
                return Status::ERROR;
            }
            log_handle = DT_NULL;
        }

        return Status::OK;
    }

    static Status CreateBackend(Context *ctx, const QNN_INTERFACE_VER_TYPE &interface, const Qnn_LogHandle_t &log_handle, Qnn_BackendHandle_t &backend_handle)
    {
        if (DT_NULL == ctx)
        {
            return Status::ERROR;
        }

        QnnBackend_Config_t **backend_config = DT_NULL;

        // create a QnnBackend.
        Qnn_ErrorHandle_t qnn_err = interface.backendCreate(log_handle, (const QnnBackend_Config_t **)backend_config, &backend_handle);
        if (qnn_err != QNN_BACKEND_NO_ERROR)
        {
            std::string info = std::string("Backend Create failed, err : ") + std::to_string(qnn_err);
            AURA_ADD_ERROR_STRING(ctx, info.c_str());
            return Status::ERROR;
        }

        return Status::OK;
    }

    static Status DeleteBackend(Context *ctx, const QNN_INTERFACE_VER_TYPE &interface, Qnn_BackendHandle_t &backend_handle)
    {
        if (DT_NULL == ctx)
        {
            return Status::ERROR;
        }

        if (backend_handle)
        {
            Qnn_ErrorHandle_t qnn_err = interface.backendFree(backend_handle);
            if (qnn_err != QNN_BACKEND_NO_ERROR)
            {
                std::string info = "Unable to terminate backend, err : " + std::to_string(qnn_err);
                AURA_ADD_ERROR_STRING(ctx, info.c_str());
                return Status::ERROR;
            }
            backend_handle = DT_NULL;
        }

        return Status::OK;
    }

    static Status CreateProfiler(Context *ctx, const QNN_INTERFACE_VER_TYPE &interface, const Qnn_BackendHandle_t &backend_handle, const NNProfilingLevel &profiling_level, Qnn_ProfileHandle_t &profile_handle)
    {
        if (DT_NULL == ctx)
        {
            return Status::ERROR;
        }

        if (profiling_level != NNProfilingLevel::PROFILING_OFF)
        {
            QnnProfile_Level_t qnn_profile_level = (NNProfilingLevel::PROFILING_DETAILED == profiling_level) ? QNN_PROFILE_LEVEL_DETAILED : QNN_PROFILE_LEVEL_BASIC;

            Qnn_ErrorHandle_t qnn_err = interface.profileCreate(backend_handle, qnn_profile_level, &profile_handle);
            if (qnn_err != QNN_PROFILE_NO_ERROR)
            {
                std::string info = "Unable to create profile handle in the backend, err : " + std::to_string(qnn_err);
                AURA_ADD_ERROR_STRING(ctx, info.c_str());
                return Status::ERROR;
            }
        }

        return Status::OK;
    }

    static Status DeleteProfiler(Context *ctx, const QNN_INTERFACE_VER_TYPE &interface, Qnn_ProfileHandle_t &profile_handle)
    {
        if (DT_NULL == ctx)
        {
            return Status::ERROR;
        }

        if (profile_handle)
        {
            Qnn_ErrorHandle_t qnn_err = interface.profileFree(profile_handle);
            if (qnn_err != QNN_CONTEXT_NO_ERROR)
            {
                std::string info = "Unable to terminate profile, err : " + std::to_string(qnn_err);
                AURA_ADD_ERROR_STRING(ctx, info.c_str());
                return Status::ERROR;
            }
            profile_handle = DT_NULL;
        }

        return Status::OK;
    }

    static Status CreatePowerConfigId(Context *ctx, const QNN_INTERFACE_VER_TYPE &interface, const std::string &backend_type,
                                      QnnHtpDevice_PerfInfrastructure_t &perf_infra, DT_U32 &power_config_id)
    {
        if (DT_NULL == ctx)
        {
            return Status::ERROR;
        }

        if ("npu" == backend_type)
        {
            QnnDevice_Infrastructure_t device_infra = DT_NULL;

            Qnn_ErrorHandle_t qnn_err = interface.deviceGetInfrastructure(&device_infra);
            if (qnn_err != QNN_SUCCESS)
            {
                std::string info = "deviceGetInfrastrcture failed, err : " + std::to_string(qnn_err);
                AURA_ADD_ERROR_STRING(ctx, info.c_str());
                return Status::ERROR;
            }

            QnnHtpDevice_Infrastructure_t *htp_infra = static_cast<QnnHtpDevice_Infrastructure_t *>(device_infra);
            perf_infra = htp_infra->perfInfra;

            DT_U32 device_id = 0;
            DT_U32 core_id = 0;

            qnn_err = perf_infra.createPowerConfigId(device_id, core_id, &power_config_id);
            if (qnn_err != QNN_SUCCESS)
            {
                std::string info = "createPowerConfigId failed, err : " + std::to_string(qnn_err);
                AURA_ADD_ERROR_STRING(ctx, info.c_str());
                return Status::ERROR;
            }
        }

        return Status::OK;
    }

    static Status SetMemStepSize(Context *ctx, const QnnHtpDevice_PerfInfrastructure_t &perf_infra, DT_S32 mem_step_size)
    {
        if (DT_NULL == ctx)
        {
            return Status::ERROR;
        }

        if (mem_step_size <= 0)
        {
            return Status::OK;
        }

        QnnHtpPerfInfrastructure_MemoryConfig_t memGrowSize;
        memGrowSize.option = QNN_HTP_PERF_INFRASTRUCTURE_MEMORY_CONFIGOPTION_GROW_SIZE;
        memGrowSize.memGrowSizeConfig = mem_step_size * 1024 * 1024;

        const QnnHtpPerfInfrastructure_MemoryConfig_t *memConfigs[] = {&memGrowSize, NULL, NULL};

        Qnn_ErrorHandle_t qnn_err = perf_infra.setMemoryConfig(0, 0, memConfigs);
        if (qnn_err != QNN_HTP_PERF_INFRASTRUCTURE_NO_ERROR)
        {
            std::string info = std::string("setMemoryConfig failed, err : ") + std::to_string(qnn_err);
            AURA_ADD_ERROR_STRING(ctx, info.c_str());
            return Status::ERROR;
        }

        return Status::OK;
    }

    static Status DeletePowerConfigId(Context *ctx, DT_U32 &power_config_id, const QnnHtpDevice_PerfInfrastructure_t &perf_infra)
    {
        if (DT_NULL == ctx)
        {
            return Status::ERROR;
        }

        if (power_config_id != 0)
        {
            Qnn_ErrorHandle_t qnn_err = perf_infra.destroyPowerConfigId(power_config_id);
            if (qnn_err != QNN_SUCCESS)
            {
                std::string info = "destroyPowerConfigId failed, err : " + std::to_string(qnn_err);
                AURA_ADD_ERROR_STRING(ctx, info.c_str());
                return Status::ERROR;
            }
            power_config_id = 0;
        }

        return Status::OK;
    }

    static Status RegisterOpPackages(Context *ctx, const QNN_INTERFACE_VER_TYPE &interface, const Qnn_BackendHandle_t &backend_handle, const std::string &udo_lib)
    {
        if (DT_NULL == ctx)
        {
            return Status::ERROR;
        }

        if (udo_lib.empty())
        {
            return Status::OK;
        }

        if (DT_NULL == interface.backendRegisterOpPackage)
        {
            AURA_ADD_ERROR_STRING(ctx, "backendRegisterOpPackage is null");
            return Status::ERROR;
        }

        std::vector<std::string> udo_strs;
        udo_strs = NNSplit(udo_lib, ',');

        for (auto udo_str : udo_strs)
        {
            std::vector<std::string> udo_info;
            udo_info = NNSplit(udo_str, ':');

            const DT_CHAR *target = DT_NULL;

            if (udo_info.size() == 3)
            {
                target = udo_info[2].c_str();
            }
            else if(udo_info.size() != 2)
            {
                std::string info = "udo_lib path " + udo_lib + " is wrong";
                AURA_ADD_ERROR_STRING(ctx, info.c_str());
                return Status::ERROR;
            }

            Qnn_ErrorHandle_t qnn_err = interface.backendRegisterOpPackage(backend_handle, (DT_CHAR *)udo_info[0].c_str(), (DT_CHAR *)udo_info[1].c_str(), target);
            if (qnn_err != QNN_BACKEND_NO_ERROR)
            {
                std::string info = "register op package failed, err : " + std::to_string(qnn_err) + " Package lib: " + udo_info[0] + " interface Provider: " + udo_info[1];
                AURA_ADD_ERROR_STRING(ctx, info.c_str());
                return Status::ERROR;
            }
        }

        return Status::OK;
    }

    static Status DeleteContext(Context *ctx, const QNN_INTERFACE_VER_TYPE &interface, const Qnn_ProfileHandle_t &profile_handle, Qnn_ContextHandle_t &context_handle)
    {
        if (DT_NULL == ctx)
        {
            return Status::ERROR;
        }

        if (context_handle)
        {
            Qnn_ErrorHandle_t qnn_err = interface.contextFree(context_handle, profile_handle);
            if (qnn_err != QNN_CONTEXT_NO_ERROR)
            {
                std::string info = "Unable to free context, err : " + std::to_string(qnn_err);
                AURA_ADD_ERROR_STRING(ctx, info.c_str());
                return Status::ERROR;
            }
            context_handle = DT_NULL;
        }

        return Status::OK;
    }

    static Status CreateGraphsFromBinary(Context *ctx, const QNN_INTERFACE_VER_TYPE &interface, const QNN_SYSTEM_INTERFACE_VER_TYPE &system_interface,
                                         const Buffer &binary_buffer, const Qnn_BackendHandle_t &backend_handle, const Qnn_DeviceHandle_t &device_handle,
                                         const Qnn_ProfileHandle_t &profile_handle, Qnn_ContextHandle_t &context_handle, GraphInfoPtr_t **graphs_info_ptr,
                                         DT_U32 &graphs_count, std::vector<DT_S8> &graph_ids, DT_S32 budget)
    {
        AURA_UNUSED(graph_ids);
        AURA_UNUSED(budget);

        if (DT_NULL == ctx)
        {
            return Status::ERROR;
        }

        // generate context and graph
        if (DT_NULL == system_interface.systemContextCreate ||
            DT_NULL == system_interface.systemContextGetBinaryInfo ||
            DT_NULL == system_interface.systemContextFree)
        {
            AURA_ADD_ERROR_STRING(ctx, "Qnn System function pointers are not populated.");
            return Status::ERROR;
        }

        Status ret = Status::ERROR;
        QnnSystemContext_Handle_t sys_ctx_handle = DT_NULL;

        Qnn_ErrorHandle_t qnn_err = system_interface.systemContextCreate(&sys_ctx_handle);
        if (qnn_err != QNN_SUCCESS)
        {
            std::string info = std::string("Cound not create system handle, err : ") + std::to_string(qnn_err);
            AURA_ADD_ERROR_STRING(ctx, info.c_str());
            return ret;
        }

        const QnnSystemContext_BinaryInfo_t *binary_info = DT_NULL;
        Qnn_ContextBinarySize_t binary_info_size = 0;

        std::vector<QnnContext_Config_t *> cfg;
// qnn 2.17.0 -> qnn API version 2.11.0
#if ((QNN_API_VERSION_MAJOR >= 2) && (QNN_API_VERSION_MINOR >= 11))
        QnnContext_Config_t enable_graphs_config;
        QnnContext_Config_t weight_sharing_config;
        QnnHtpContext_CustomConfig_t weight_sharing_enabled;
        std::vector<const DT_CHAR*> graphs_nams;
#endif

// qnn 2.19.0 -> qnn API version 2.13.0
#if ((QNN_API_VERSION_MAJOR >= 2) && (QNN_API_VERSION_MINOR >= 13))
        QnnContext_Config_t file_read_memory_budget_config;
        QnnHtpContext_CustomConfig_t file_read_memory_budget;
#endif

        std::unordered_set<DT_S8> ids_set;

        qnn_err = system_interface.systemContextGetBinaryInfo(sys_ctx_handle, binary_buffer.m_data, binary_buffer.m_size,
                                                              &binary_info, &binary_info_size);
        if (qnn_err != QNN_SUCCESS)
        {
            std::string info = "Failed to get context binary info, err : " + std::to_string(qnn_err);
            AURA_ADD_ERROR_STRING(ctx, info.c_str());
            goto EXIT;
        }

        if (QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1 == binary_info->version)
        {
            if (!binary_info->contextBinaryInfoV1.graphs)
            {
                AURA_ADD_ERROR_STRING(ctx, "graphs error");
                goto EXIT;
            }

            // need to copy graphs -> graphs_info
            graphs_count = binary_info->contextBinaryInfoV1.numGraphs;
            if (QnnUtils::InitGraphsInfoV1((Context *)ctx, binary_info->contextBinaryInfoV1.graphs, graphs_info_ptr, binary_info->contextBinaryInfoV1.numGraphs) != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "InitGraphsInfoV1 failed");
                goto EXIT;
            }
        }

// qnn 2.28.0 -> qnn API version 2.21.0
#if ((QNN_API_VERSION_MAJOR >= 2) && (QNN_API_VERSION_MINOR >= 21))
        else if (QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3 == binary_info->version)
        {
            if (!binary_info->contextBinaryInfoV3.graphs)
            {
                AURA_ADD_ERROR_STRING(ctx, "graphs error");
                goto EXIT;
            }

            // need to copy graphs -> graphs_info
            graphs_count = binary_info->contextBinaryInfoV3.numGraphs;
            if (QnnUtils::InitGraphsInfoV3((Context *)ctx, binary_info->contextBinaryInfoV3.graphs, graphs_info_ptr, binary_info->contextBinaryInfoV3.numGraphs) != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "InitGraphsInfoV3 failed");
                goto EXIT;
            }
        }
#endif

        else
        {
            std::string info = "version error : binary info version = " + std::to_string(binary_info->version);
            AURA_ADD_ERROR_STRING(ctx, info.c_str());
            goto EXIT;
        }

        system_interface.systemContextFree(sys_ctx_handle);
        sys_ctx_handle = DT_NULL;

        if (DT_NULL == interface.contextCreateFromBinary)
        {
            AURA_ADD_ERROR_STRING(ctx, "contextCreateFromBinary is nullptr");
            goto EXIT;
        }

// qnn 2.17.0 -> qnn API version 2.11.0
#if ((QNN_API_VERSION_MAJOR >= 2) && (QNN_API_VERSION_MINOR >= 11))
        if (graph_ids[0] < 0)
        {
            if (graphs_count > 1)
            {
                weight_sharing_enabled.option  = QNN_HTP_CONTEXT_CONFIG_OPTION_WEIGHT_SHARING_ENABLED;
                weight_sharing_enabled.weightSharingEnabled = DT_TRUE;

                weight_sharing_config.option = QNN_CONTEXT_CONFIG_OPTION_CUSTOM;
                weight_sharing_config.customConfig = &weight_sharing_enabled;
                cfg.push_back(&weight_sharing_config);
            }
        }
        else
        {
            enable_graphs_config.option = QNN_CONTEXT_CONFIG_ENABLE_GRAPHS;

            DT_S8 max_id = -1;
            for (auto &id : graph_ids)
            {
                if (id >= 0)
                {
                    ids_set.insert(id);
                    max_id = Max(max_id, id);
                }
            }

            if (static_cast<DT_U32>(max_id) >= graphs_count)
            {
                AURA_ADD_ERROR_STRING(ctx, "invalid graph_ids");
                goto EXIT;
            }

            for (DT_U32 i = 0; i < graphs_count; i++)
            {
                if (ids_set.find(i) != ids_set.end())
                {
                    graphs_nams.push_back((**graphs_info_ptr)[i].graph_name);
                }
            }
            graphs_nams.push_back(DT_NULL);

            enable_graphs_config.enableGraphs = graphs_nams.data();
            cfg.push_back(&enable_graphs_config);

            if (graphs_nams.size() > 2)
            {
                weight_sharing_enabled.option  = QNN_HTP_CONTEXT_CONFIG_OPTION_WEIGHT_SHARING_ENABLED;
                weight_sharing_enabled.weightSharingEnabled = DT_TRUE;

                weight_sharing_config.option = QNN_CONTEXT_CONFIG_OPTION_CUSTOM;
                weight_sharing_config.customConfig = &weight_sharing_enabled;
                cfg.push_back(&weight_sharing_config);
            }
        }
#endif

// qnn 2.19.0 -> qnn API version 2.13.0
#if ((QNN_API_VERSION_MAJOR >= 2) && (QNN_API_VERSION_MINOR >= 13))
        if ((budget > 0) && (budget < (binary_buffer.m_size >> 20)) && (binary_buffer.m_property != 0))
        {
            file_read_memory_budget.option = QNN_HTP_CONTEXT_CONFIG_OPTION_FILE_READ_MEMORY_BUDGET;
            file_read_memory_budget.fileReadMemoryBudgetInMb = budget;

            file_read_memory_budget_config.option = QNN_CONTEXT_CONFIG_OPTION_CUSTOM;
            file_read_memory_budget_config.customConfig = reinterpret_cast<QnnContext_CustomConfig_t>(&file_read_memory_budget);

            cfg.push_back(&file_read_memory_budget_config);
        }
#endif

        cfg.push_back(DT_NULL);
        qnn_err = interface.contextCreateFromBinary(backend_handle, device_handle, const_cast<const QnnContext_Config_t**>(cfg.data()),
                                                    binary_buffer.m_data, binary_buffer.m_size, &context_handle, profile_handle);
        if (qnn_err != QNN_SUCCESS)
        {
            std::string info = std::string("contextCreateFromBinary failed, err : ") + std::to_string(qnn_err);
            AURA_ADD_ERROR_STRING(ctx, info.c_str());
            goto EXIT;
        }

        if (DT_NULL == interface.graphRetrieve)
        {
            AURA_ADD_ERROR_STRING(ctx, "graphRetrieve is null");
            goto EXIT;
        }

        for (DT_U32 graph_idx = 0; graph_idx < graphs_count; graph_idx++)
        {
            if (!ids_set.empty() && ids_set.find(graph_idx) == ids_set.end())
            {
                continue;
            }

            qnn_err = interface.graphRetrieve(context_handle,
                                              (*(*(graphs_info_ptr)))[graph_idx].graph_name,
                                              &((*(*(graphs_info_ptr)))[graph_idx].graph));
            if (qnn_err != QNN_SUCCESS)
            {
                std::string info = std::string("graphRetrieve failed for graph idx : ") +
                                   std::to_string(graph_idx) + ", err : " + std::to_string(qnn_err);
                AURA_ADD_ERROR_STRING(ctx, info.c_str());
                goto EXIT;
            }
        }

        ret = Status::OK;
EXIT:
        if (ret != Status::OK)
        {
            if (sys_ctx_handle != DT_NULL)
            {
                system_interface.systemContextFree(sys_ctx_handle);
                sys_ctx_handle = DT_NULL;
            }

            DeinitGraphsInfo((Context *)ctx, graphs_info_ptr, graphs_count);
        }

        AURA_RETURN(ctx, ret);
    }

    static Status CreateGraphsFromModel(Context *ctx, const QNN_INTERFACE_VER_TYPE &interface, const Qnn_BackendHandle_t &backend_handle, const Qnn_DeviceHandle_t &device_handle,
                                        const Qnn_ProfileHandle_t &profile_handle, const ModelInterface &model_interface, const NNLogLevel &log_level,
                                        Qnn_ContextHandle_t &context_handle, GraphInfoPtr_t **graphs_info_ptr, DT_U32 &graphs_count)
    {
        if (DT_NULL == ctx)
        {
            return Status::ERROR;
        }

        Status ret = Status::ERROR;

        // create a context in a backend
        Qnn_ErrorHandle_t qnn_err = interface.contextCreate(backend_handle, device_handle, DT_NULL, &context_handle);
        if (qnn_err != QNN_CONTEXT_NO_ERROR)
        {
            std::string info = "contextCreate failed, err : " + std::to_string(qnn_err);
            AURA_ADD_ERROR_STRING(ctx, info.c_str());
            return ret;
        }

        QnnLog_Level_t qnn_log_level = GetQnnLogLevel(log_level);

        // compose graphs
        const GraphConfigInfo_t **graph_configs_info = DT_NULL;
        DT_U32 graph_configs_count = 0;
        DT_BOOL debug = DT_FALSE;

        ModelError_t model_status = model_interface.create_func(backend_handle, interface, context_handle, graph_configs_info, graph_configs_count,
                                                                graphs_info_ptr, &graphs_count, debug, DT_NULL, qnn_log_level);
        if (model_status != ModelError_t::MODEL_NO_ERROR)
        {
            std::string info = "QnnModel_composeGraphs failed, err : " + std::to_string(model_status);
            AURA_ADD_ERROR_STRING(ctx, info.c_str());
            goto EXIT;
        }

        // finalize graphs
        for (size_t graph_idx = 0; graph_idx < graphs_count; graph_idx++)
        {
            Qnn_ErrorHandle_t qnn_err = interface.graphFinalize((*(*graphs_info_ptr))[graph_idx].graph, profile_handle, DT_NULL);
            if (qnn_err != QNN_GRAPH_NO_ERROR)
            {
                std::string info = "graphFinalize failed, err : " + std::to_string(qnn_err);
                AURA_ADD_ERROR_STRING(ctx, info.c_str());
                goto EXIT;
            }

            (*(*graphs_info_ptr))[graph_idx].input_tensor_map  = NULL;
            (*(*graphs_info_ptr))[graph_idx].output_tensor_map = NULL;
        }

        ret = Status::OK;
EXIT:
        if (ret != Status::OK)
        {
            model_interface.delete_func(graphs_info_ptr, graphs_count);
            if (QnnUtils::DeleteContext(ctx, interface, profile_handle, context_handle) != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "DeleteContext failed");
            }
        }

        AURA_RETURN(ctx, ret);
    }

    static Status ExtractBackendProfilingInfo(Context *ctx, const QNN_INTERFACE_VER_TYPE &interface, const Qnn_ProfileHandle_t &profile_handle,
                                              const NNProfilingLevel &profiling_level, std::string &profile_data, ProfileStage profile_stage)
    {
        if (DT_NULL == ctx)
        {
            return Status::ERROR;
        }

        if (NNProfilingLevel::PROFILING_OFF == profiling_level)
        {
            return Status::OK;
        }

        if (!profile_handle)
        {
            AURA_ADD_ERROR_STRING(ctx, "profile_handle is null");
            return Status::ERROR;
        }

        const QnnProfile_EventId_t *profile_events = DT_NULL;
        DT_U32 num_events = 0;

        Qnn_ErrorHandle_t qnn_err = interface.profileGetEvents(profile_handle, &profile_events, &num_events);
        if (qnn_err != QNN_PROFILE_NO_ERROR)
        {
            std::string info = "failure in profile get events, err : " + std::to_string(qnn_err);
            AURA_ADD_ERROR_STRING(ctx, info.c_str());
            return Status::ERROR;
        }

        for (size_t event = 0; event < num_events; event++)
        {
            Status ret = ExtractProfilingEvent(ctx, profile_events[event], interface, profile_data, profile_stage);
            ret |= ExtractProfilingSubEvents(ctx, profile_events[event], interface, profile_data, profile_stage);
            if (ret != Status::OK)
            {
                std::string info = "ExtractBackendProfilingInfo: ExtractProfilingEvent or ExtractProfilingSubEvents failed, err : " + std::to_string(qnn_err);
                AURA_ADD_ERROR_STRING(ctx, info.c_str());
                return Status::ERROR;
            }
        }

        return Status::OK;
    }

    static Status SetPerformance(Context *ctx, const QnnHtpDevice_PerfInfrastructure_t &perf_infra, NNPerfLevel htp_perf_level,
                                 NNPerfLevel hmx_perf_level, const std::string &backend_type, DT_U32 power_config_id,
                                 QnnHtpDevice_Arch_t arch)
    {
        if (DT_NULL == ctx)
        {
            return Status::ERROR;
        }

        if (0 == power_config_id)
        {
            return Status::OK;
        }

        if (backend_type != "npu")
        {
            AURA_ADD_ERROR_STRING(ctx, "not supported backend");
            return Status::OK;
        }

        if (NNPerfLevel::PERF_DEFAULT == htp_perf_level)
        {
            hmx_perf_level = NNPerfLevel::PERF_DEFAULT;
        }

        Status ret = Status::ERROR;

        QnnHtpPerfInfrastructure_PowerConfig_t power_config;
        memset(&power_config, 0, sizeof(power_config));
        Qnn_ErrorHandle_t qnn_err;

        power_config.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
        power_config.dcvsV3Config.contextId = power_config_id;

        QnnHtpPerfInfrastructure_SleepLatency_t latency_value = 0;
        QnnHtpPerfInfrastructure_VoltageCorner_t min_vc = DCVS_VOLTAGE_CORNER_DISABLE;
        QnnHtpPerfInfrastructure_VoltageCorner_t max_vc = DCVS_VOLTAGE_CORNER_DISABLE;
        QnnHtpPerfInfrastructure_VoltageCorner_t tar_vc = DCVS_VOLTAGE_CORNER_DISABLE;
        QnnHtpPerfInfrastructure_PowerMode_t power_mode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_POWER_SAVER_MODE;

        switch (htp_perf_level)
        {
            case NNPerfLevel::PERF_LOW:
            {
                power_mode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_POWER_SAVER_MODE;
                min_vc = DCVS_VOLTAGE_VCORNER_SVS2;
                max_vc = DCVS_VOLTAGE_VCORNER_SVS;
                tar_vc = DCVS_VOLTAGE_VCORNER_SVS;
                latency_value = 1000;
                break;
            }

            case NNPerfLevel::PERF_NORMAL:
            {
                power_mode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
                min_vc = DCVS_VOLTAGE_VCORNER_NOM_PLUS;
                max_vc = DCVS_VOLTAGE_VCORNER_NOM_PLUS;
                tar_vc = DCVS_VOLTAGE_VCORNER_NOM_PLUS;
                latency_value = 400;
                break;
            }

            case NNPerfLevel::QNN_PERF_CUSTOM_0:
            {
                power_mode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_DUTY_CYCLE_MODE;
                min_vc = DCVS_VOLTAGE_CORNER_DISABLE;
                max_vc = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
                tar_vc = DCVS_VOLTAGE_VCORNER_NOM;
                latency_value = 1000;
                break;
            }

            case NNPerfLevel::PERF_DEFAULT:
            {
                power_config.dcvsV3Config.setDcvsEnable = 1;
                power_config.dcvsV3Config.dcvsEnable = 0;
                break;
            }

            default:
            {
                power_mode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
                min_vc = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
                max_vc = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
                tar_vc = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
                latency_value = 10;
                break;
            }
        }

        if (htp_perf_level != NNPerfLevel::PERF_DEFAULT)
        {
            power_config.dcvsV3Config.setDcvsEnable = 1;
            power_config.dcvsV3Config.dcvsEnable = 0;

            // set sleep latency parameter
            power_config.dcvsV3Config.setSleepLatency = 1; // True to consider Latency parameter otherwise False
            power_config.dcvsV3Config.sleepLatency = latency_value;

            // set bus params
            power_config.dcvsV3Config.setBusParams = 1; // True to consider Bus parameter otherwise False
            power_config.dcvsV3Config.busVoltageCornerMin = min_vc;
            power_config.dcvsV3Config.busVoltageCornerTarget = tar_vc;
            power_config.dcvsV3Config.busVoltageCornerMax = max_vc;

            // set core params
            power_config.dcvsV3Config.setCoreParams = 1; // True to consider Core parameter otherwise False
            power_config.dcvsV3Config.coreVoltageCornerMin = min_vc;
            power_config.dcvsV3Config.coreVoltageCornerTarget = tar_vc;
            power_config.dcvsV3Config.coreVoltageCornerMax = max_vc;

            power_config.dcvsV3Config.powerMode = power_mode;

            power_config.dcvsV3Config.sleepDisable    = 0;  // True to consider sleep/LPM modes, False to enable
            power_config.dcvsV3Config.setSleepDisable = 0;  // True to consider sleep disable/enable parameter otherwise False
        }

        // qnn sdk larger than 2.22.1
        // qnn can not set hmx perf on linux
        const QnnHtpPerfInfrastructure_PowerConfig_t *power_configs[] = {&power_config, DT_NULL, DT_NULL};
#if (((AURA_QNN_V2_MAGIC & 0xFFFF) >= 0x08AD) && !defined(AURA_BUILD_LINUX))
        QnnHtpPerfInfrastructure_PowerConfig_t power_config_hmx;
        memset(&power_config_hmx, 0, sizeof(power_config_hmx));

        if ((arch != QnnHtpDevice_Arch_t::QNN_HTP_DEVICE_ARCH_UNKNOWN) &&
            (static_cast<DT_S32>(arch) >= static_cast<DT_S32>(QnnHtpDevice_Arch_t::QNN_HTP_DEVICE_ARCH_V75)))
        {
            QnnHtpPerfInfrastructure_ClkPerfMode_t clk_perf_model = QNN_HTP_PERF_INFRASTRUCTURE_CLK_PERF_HIGH;
            QnnHtpPerfInfrastructure_ExpVoltageCorner_t exp_volt_corner = DCVS_EXP_VCORNER_DISABLE;
            QnnHtpPerfInfrastructure_HmxDefault_Vote_t hmx_pick_default = 1;

            switch (hmx_perf_level)
            {
                case NNPerfLevel::PERF_LOW:
                {
                    clk_perf_model   = QNN_HTP_PERF_INFRASTRUCTURE_CLK_PERF_LOW;
                    exp_volt_corner  = DCVS_EXP_VCORNER_MIN;
                    hmx_pick_default = 0;
                    break;
                }

                case NNPerfLevel::PERF_NORMAL:
                {
                    clk_perf_model   = QNN_HTP_PERF_INFRASTRUCTURE_CLK_PERF_LOW;
                    exp_volt_corner  = DCVS_EXP_VCORNER_SVS;
                    hmx_pick_default = 0;
                    break;
                }

                case NNPerfLevel::PERF_DEFAULT:
                {
                    hmx_pick_default = 1;
                    break;
                }

                default:
                {
                    clk_perf_model = QNN_HTP_PERF_INFRASTRUCTURE_CLK_PERF_HIGH;
                    exp_volt_corner = DCVS_EXP_VCORNER_MAX;
                    hmx_pick_default = 0;
                    break;
                }
            }

            power_config_hmx.option                     = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_HMX_V2;
            power_config_hmx.hmxV2Config.hmxPickDefault = hmx_pick_default;
            power_config_hmx.hmxV2Config.hmxPerfMode    = clk_perf_model;

            power_config_hmx.hmxV2Config.hmxVoltageCornerMin    = exp_volt_corner;
            power_config_hmx.hmxV2Config.hmxVoltageCornerTarget = exp_volt_corner;
            power_config_hmx.hmxV2Config.hmxVoltageCornerMax    = exp_volt_corner;

            power_configs[1] = &power_config_hmx;
        }
#else
        AURA_UNUSED(arch);
#endif

        qnn_err = perf_infra.setPowerConfig(power_config_id, power_configs);
        if (qnn_err != QNN_SUCCESS)
        {
            std::string info = std::string("setPowerConfig failed, htp perf level = ") + std::to_string((DT_U32)htp_perf_level) +
                                           " hmx perf level = " + std::to_string((DT_U32)hmx_perf_level) + ", err : " + std::to_string(qnn_err);
            AURA_ADD_ERROR_STRING(ctx, info.c_str());
            return Status::ERROR;
        }

        ret = Status::OK;
        AURA_RETURN(ctx, ret);
    }

    static Status GetGraphsBinary(Context *ctx, const QNN_INTERFACE_VER_TYPE &interface, const Qnn_ContextHandle_t &context_handle, Buffer &binary_buffer)
    {
        if (DT_NULL == ctx)
        {
            return Status::ERROR;
        }

        Status ret = Status::ERROR;

        DT_U64 required_buffer_size = 0;
        DT_CHAR *buffer_data = DT_NULL;
        DT_U64 written_buffer_size = 0;

        Qnn_ErrorHandle_t qnn_err = interface.contextGetBinarySize(context_handle, &required_buffer_size);
        if (qnn_err != QNN_CONTEXT_NO_ERROR)
        {
            std::string info = "Could not get the required binary size, err : " + std::to_string(qnn_err);
            AURA_ADD_ERROR_STRING(ctx, info.c_str());
            return ret;
        }

        buffer_data = (DT_CHAR *)(AURA_ALLOC(ctx, required_buffer_size));
        if (DT_NULL == buffer_data)
        {
            std::string info = "AURA_ALLOC size " + std::to_string(required_buffer_size) + " failed";
            AURA_ADD_ERROR_STRING(ctx, info.c_str());
            goto EXIT;
        }

        if (DT_NULL == interface.contextGetBinarySize ||
            DT_NULL == interface.contextGetBinary)
        {
            AURA_ADD_ERROR_STRING(ctx, "contextGetBinarySize or contextGetBinary is nullptr.");
            goto EXIT;
        }

        qnn_err = interface.contextGetBinary(context_handle, buffer_data, required_buffer_size, &written_buffer_size);
        if (qnn_err != QNN_CONTEXT_NO_ERROR)
        {
            std::string info = "Could not get binary, err : " + std::to_string(qnn_err);
            AURA_ADD_ERROR_STRING(ctx, info.c_str());
            goto EXIT;
        }

        if (required_buffer_size < written_buffer_size)
        {
            std::string info = std::string("Illegal written buffer size [") +
                               std::to_string(written_buffer_size) +
                               std::string("] bytes. Cannot exceed allocated memory of [") +
                               std::to_string(required_buffer_size) + "]";
            AURA_ADD_ERROR_STRING(ctx, info.c_str());
            goto EXIT;
        }

        binary_buffer = Buffer(AURA_MEM_HEAP, required_buffer_size, written_buffer_size, buffer_data, buffer_data, 0);
        ret = Status::OK;
EXIT:
        if (ret != Status::OK)
        {
            if (buffer_data)
            {
                AURA_FREE(ctx, buffer_data);
                buffer_data = DT_NULL;
                binary_buffer.Clear();
            }
        }

        AURA_RETURN(ctx, ret);
    }

    static Status DeinitGraphsInfo(Context *ctx, GraphInfoPtr_t **graphs_info_ptr, DT_U32 graphs_count)
    {
        if (DT_NULL == ctx)
        {
            return Status::ERROR;
        }

        if (DT_NULL == *graphs_info_ptr || DT_NULL == *(*graphs_info_ptr))
        {
            return Status::OK;
        }

        GraphInfo_t *graphs_info = *(*graphs_info_ptr);

        for (DT_U32 idx = 0; idx < graphs_count; idx++)
        {
            if (graphs_info[idx].graph_name)
            {
                AURA_FREE(ctx, graphs_info[idx].graph_name);
                graphs_info[idx].graph_name = DT_NULL;
            }

            DeinitTensorsInfoV1(ctx, &graphs_info[idx].input_tensors, graphs_info[idx].num_input_tensors);
            graphs_info[idx].num_input_tensors = 0;

            DeinitTensorsInfoV1(ctx, &graphs_info[idx].output_tensors, graphs_info[idx].num_output_tensors);
            graphs_info[idx].num_output_tensors = 0;

            if (graphs_info->input_tensor_map != NULL)
            {
                Delete<QnnTensorMap>(ctx, &graphs_info->input_tensor_map);
                graphs_info->input_tensor_map = NULL;
            }

            if (graphs_info->output_tensor_map != NULL)
            {
                Delete<QnnTensorMap>(ctx, &graphs_info->output_tensor_map);
                graphs_info->output_tensor_map = NULL;
            }
        }

        if (graphs_info)
        {
            AURA_FREE(ctx, graphs_info);
            graphs_info = DT_NULL;
            *(*graphs_info_ptr) = DT_NULL;
        }

        if (*graphs_info_ptr)
        {
            AURA_FREE(ctx, *graphs_info_ptr);
            *graphs_info_ptr = DT_NULL;
        }

        return Status::OK;
    }

    static Status CreateGraphsInfo(Context *ctx, GraphInfoPtr_t **src_grpahs_info_ptr, GraphInfoPtr_t **dst_grpahs_info_ptr, DT_U32 graph_count)
    {
        if (DT_NULL == ctx)
        {
            return Status::ERROR;
        }

        Status ret = Status::ERROR;

        GraphInfo_t *graphs_info = DT_NULL;
        GraphInfo_t *src_graphs_info = *(*src_grpahs_info_ptr);

        *dst_grpahs_info_ptr = (GraphInfoPtr_t *)AURA_ALLOC_PARAM(ctx, AURA_MEM_HEAP, sizeof(GraphInfoPtr_t), 0);
        if (DT_NULL == *dst_grpahs_info_ptr)
        {
            AURA_ADD_ERROR_STRING(ctx, "AURA_ALLOC_PARAM failed");
            goto EXIT;
        }

        graphs_info = (GraphInfo_t *)AURA_ALLOC_PARAM(ctx, AURA_MEM_HEAP, graph_count * sizeof(GraphInfo_t), 0);
        if (DT_NULL == graphs_info)
        {
            AURA_ADD_ERROR_STRING(ctx, "AURA_ALLOC_PARAM failed");
            goto EXIT;
        }

        *(*dst_grpahs_info_ptr) = graphs_info;

        for (DT_U32 idx = 0; idx < graph_count; idx++)
        {
            graphs_info[idx].graph = DT_NULL;
            graphs_info[idx].graph_name = (DT_CHAR *)AURA_ALLOC_PARAM(ctx, AURA_MEM_HEAP, strlen(src_graphs_info[idx].graph_name) + 1, 0);
            if (DT_NULL == graphs_info[idx].graph_name)
            {
                AURA_ADD_ERROR_STRING(ctx, "AURA_ALLOC_PARAM failed");
                goto EXIT;
            }

            strcpy(graphs_info[idx].graph_name, src_graphs_info[idx].graph_name);

            graphs_info[idx].input_tensors = DT_NULL;
            graphs_info[idx].num_input_tensors = 0;

            if (src_graphs_info[idx].input_tensors)
            {
                if (QnnUtils::InitTensorsInfoV1(ctx, src_graphs_info[idx].input_tensors, &graphs_info[idx].input_tensors,
                                                src_graphs_info[idx].num_input_tensors) != Status::OK)
                {
                    AURA_ADD_ERROR_STRING(ctx, "InitTensorsInfoV1 failed");
                    goto EXIT;
                }
                graphs_info[idx].num_input_tensors = src_graphs_info[idx].num_input_tensors;
            }

            graphs_info[idx].output_tensors = DT_NULL;
            graphs_info[idx].num_output_tensors = 0;

            if (src_graphs_info[idx].output_tensors)
            {
                if (QnnUtils::InitTensorsInfoV1(ctx, src_graphs_info[idx].output_tensors, &graphs_info[idx].output_tensors,
                                                src_graphs_info[idx].num_output_tensors) != Status::OK)
                {
                    AURA_ADD_ERROR_STRING(ctx, "InitTensorsInfoV1 failed");
                    goto EXIT;
                }
                graphs_info[idx].num_output_tensors = src_graphs_info[idx].num_output_tensors;
            }

            graphs_info[idx].graph = src_graphs_info[idx].graph;

            graphs_info->input_tensor_map  = NULL;
            graphs_info->output_tensor_map = NULL;
        }

        ret = Status::OK;
EXIT:
        if (ret != Status::OK)
        {
            if (DeinitGraphsInfo(ctx, dst_grpahs_info_ptr, graph_count) != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "DeinitGraphsInfo failed");
            }
        }

        AURA_RETURN(ctx, ret);
    }

    static Status DeleteGraphsInfo(Context *ctx, GraphInfoPtr_t **graphs_info_ptr, DT_U32 graph_count)
    {
        if (DT_NULL == ctx)
        {
            return Status::ERROR;
        }

        if ((DT_NULL == *graphs_info_ptr) || (DT_NULL == *(*graphs_info_ptr)))
        {
            return Status::OK;
        }

        if (DeinitGraphsInfo(ctx, graphs_info_ptr, graph_count) != Status::OK)
        {
            AURA_ADD_ERROR_STRING(ctx, "DeinitGraphsInfo failed");
            return Status::ERROR;
        }

        return Status::OK;
    }

    static Status InitGraphsInfoV1(Context *ctx, const QnnSystemContext_GraphInfo_t *graphs_input, GraphInfoPtr_t **graphs_info_ptr, DT_U32 graphs_count)
    {
        if (DT_NULL == ctx)
        {
            return Status::ERROR;
        }

        if (DT_NULL == graphs_input)
        {
            AURA_ADD_ERROR_STRING(ctx, "graphs_input is null");
            return Status::ERROR;
        }

        Status ret = Status::ERROR;

        GraphInfo_t *graphs_info = DT_NULL;

        *graphs_info_ptr = (GraphInfoPtr_t *)AURA_ALLOC_PARAM(ctx, AURA_MEM_HEAP, sizeof(GraphInfoPtr_t), 0);
        graphs_info = (GraphInfo_t *)AURA_ALLOC_PARAM(ctx, AURA_MEM_HEAP, graphs_count * sizeof(GraphInfo_t), 0);
        if ((DT_NULL == graphs_info) || (DT_NULL == *graphs_info_ptr))
        {
            AURA_ADD_ERROR_STRING(ctx, "AURA_ALLOC_PARAM failed");
            goto EXIT;
        }

        *(*graphs_info_ptr) = graphs_info;

        for (DT_U32 idx = 0; idx < graphs_count; idx++)
        {
            graphs_info[idx].graph = DT_NULL;
            graphs_info[idx].graph_name = (DT_CHAR *)AURA_ALLOC_PARAM(ctx, AURA_MEM_HEAP, strlen(graphs_input[idx].graphInfoV1.graphName) + 1, 0);
            if (DT_NULL == graphs_info[idx].graph_name)
            {
                AURA_ADD_ERROR_STRING(ctx, "AURA_ALLOC_PARAM failed");
                goto EXIT;
            }
            strcpy(graphs_info[idx].graph_name, graphs_input[idx].graphInfoV1.graphName);

            graphs_info[idx].input_tensors = DT_NULL;
            graphs_info[idx].num_input_tensors = 0;

            if (graphs_input[idx].graphInfoV1.graphInputs)
            {
                if (QnnUtils::InitTensorsInfoV1(ctx, graphs_input[idx].graphInfoV1.graphInputs, &graphs_info[idx].input_tensors,
                                                graphs_input[idx].graphInfoV1.numGraphInputs) != Status::OK)
                {
                    AURA_ADD_ERROR_STRING(ctx, "InitTensorsInfoV1 failed");
                    goto EXIT;
                }
                graphs_info[idx].num_input_tensors = graphs_input[idx].graphInfoV1.numGraphInputs;
            }

            graphs_info[idx].output_tensors = DT_NULL;
            graphs_info[idx].num_output_tensors = 0;

            if (graphs_input[idx].graphInfoV1.graphOutputs)
            {
                if (QnnUtils::InitTensorsInfoV1(ctx, graphs_input[idx].graphInfoV1.graphOutputs, &graphs_info[idx].output_tensors,
                                                graphs_input[idx].graphInfoV1.numGraphOutputs) != Status::OK)
                {
                    AURA_ADD_ERROR_STRING(ctx, "InitTensorsInfoV1 failed");
                    goto EXIT;
                }
                graphs_info[idx].num_output_tensors = graphs_input[idx].graphInfoV1.numGraphOutputs;
            }

            graphs_info->input_tensor_map  = NULL;
            graphs_info->output_tensor_map = NULL;
        }

        ret = Status::OK;
EXIT:
        if (ret != Status::OK)
        {
            if (DeinitGraphsInfo(ctx, graphs_info_ptr, graphs_count) != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "DeinitGraphsInfo failed");
            }
        }

        AURA_RETURN(ctx, ret);
    }

// qnn 2.28.0 -> qnn API version 2.21.0
#if ((QNN_API_VERSION_MAJOR >= 2) && (QNN_API_VERSION_MINOR >= 21))
    static Status InitGraphsInfoV3(Context *ctx, const QnnSystemContext_GraphInfo_t *graphs_input, GraphInfoPtr_t **graphs_info_ptr, DT_U32 graphs_count)
    {
        if (DT_NULL == ctx)
        {
            return Status::ERROR;
        }

        if (DT_NULL == graphs_input)
        {
            AURA_ADD_ERROR_STRING(ctx, "graphs_input is null");
            return Status::ERROR;
        }

        Status ret = Status::ERROR;

        GraphInfo_t *graphs_info = DT_NULL;

        *graphs_info_ptr = (GraphInfoPtr_t *)AURA_ALLOC_PARAM(ctx, AURA_MEM_HEAP, sizeof(GraphInfoPtr_t), 0);
        graphs_info = (GraphInfo_t *)AURA_ALLOC_PARAM(ctx, AURA_MEM_HEAP, graphs_count * sizeof(GraphInfo_t), 0);
        if ((DT_NULL == graphs_info) || (DT_NULL == *graphs_info_ptr))
        {
            AURA_ADD_ERROR_STRING(ctx, "AURA_ALLOC_PARAM failed");
            goto EXIT;
        }

        *(*graphs_info_ptr) = graphs_info;

        for (DT_U32 idx = 0; idx < graphs_count; idx++)
        {
            graphs_info[idx].graph = DT_NULL;
            graphs_info[idx].graph_name = (DT_CHAR *)AURA_ALLOC_PARAM(ctx, AURA_MEM_HEAP, strlen(graphs_input[idx].graphInfoV3.graphName) + 1, 0);
            if (DT_NULL == graphs_info[idx].graph_name)
            {
                AURA_ADD_ERROR_STRING(ctx, "AURA_ALLOC_PARAM failed");
                goto EXIT;
            }
            strcpy(graphs_info[idx].graph_name, graphs_input[idx].graphInfoV3.graphName);

            graphs_info[idx].input_tensors = DT_NULL;
            graphs_info[idx].num_input_tensors = 0;

            if (graphs_input[idx].graphInfoV3.graphInputs)
            {
                if (QnnUtils::InitTensorsInfoV1(ctx, graphs_input[idx].graphInfoV3.graphInputs, &graphs_info[idx].input_tensors,
                                                graphs_input[idx].graphInfoV3.numGraphInputs) != Status::OK)
                {
                    AURA_ADD_ERROR_STRING(ctx, "InitTensorsInfoV1 failed");
                    goto EXIT;
                }
                graphs_info[idx].num_input_tensors = graphs_input[idx].graphInfoV3.numGraphInputs;
            }

            graphs_info[idx].output_tensors = DT_NULL;
            graphs_info[idx].num_output_tensors = 0;

            if (graphs_input[idx].graphInfoV3.graphOutputs)
            {
                if (QnnUtils::InitTensorsInfoV1(ctx, graphs_input[idx].graphInfoV3.graphOutputs, &graphs_info[idx].output_tensors,
                                                graphs_input[idx].graphInfoV3.numGraphOutputs) != Status::OK)
                {
                    AURA_ADD_ERROR_STRING(ctx, "InitTensorsInfoV1 failed");
                    goto EXIT;
                }
                graphs_info[idx].num_output_tensors = graphs_input[idx].graphInfoV3.numGraphOutputs;
            }
        }

        ret = Status::OK;
EXIT:
        if (ret != Status::OK)
        {
            if (DeinitGraphsInfo(ctx, graphs_info_ptr, graphs_count) != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "DeinitGraphsInfo failed");
            }
        }

        AURA_RETURN(ctx, ret);
    }
#endif

    static Status GetHtpDeviceArch(Context *ctx, QNN_INTERFACE_VER_TYPE &interface, Qnn_LogHandle_t &log_handle, QnnHtpDevice_Arch_t &arch)
    {
        arch = QnnHtpDevice_Arch_t::QNN_HTP_DEVICE_ARCH_UNKNOWN;

        if (DT_NULL == ctx)
        {
            return Status::ERROR;
        }

        const QnnDevice_PlatformInfo_t *platform_info = DT_NULL;
        Qnn_ErrorHandle_t qnn_err = interface.deviceGetPlatformInfo(log_handle, &platform_info);
        if ((qnn_err != QNN_SUCCESS) || (DT_NULL == platform_info))
        {
            AURA_ADD_ERROR_STRING(ctx, ("failed to get platform info from the device, err : " + std::to_string(qnn_err)).c_str());
            return Status::ERROR;
        }

        if (platform_info->v1.numHwDevices > 0)
        {
            arch = platform_info->v1.hwDevices[0].v1.deviceInfoExtension->onChipDevice.arch;
        }

        qnn_err = interface.deviceFreePlatformInfo(log_handle, platform_info);
        if (qnn_err != QNN_SUCCESS)
        {
            AURA_ADD_ERROR_STRING(ctx, ("failed to free platform info from the device, err : " + std::to_string(qnn_err)).c_str());
            return Status::ERROR;
        }

        return Status::OK;
    }

private:
    static Status InitTensorsInfoV1(Context *ctx, const Qnn_Tensor_t *tensor_src, Qnn_Tensor_t **tensors_wrapper, const DT_U32 count)
    {
        if (DT_NULL == ctx)
        {
            return Status::ERROR;
        }

        if (DT_NULL == tensor_src)
        {
            AURA_ADD_ERROR_STRING(ctx, "tensor_src is null");
            return Status::ERROR;
        }

        Status ret = Status::ERROR;

        *tensors_wrapper = (Qnn_Tensor_t *)AURA_ALLOC_PARAM(ctx, AURA_MEM_HEAP, count * sizeof(Qnn_Tensor_t), 0);
        if (DT_NULL == *tensors_wrapper)
        {
            AURA_ADD_ERROR_STRING(ctx, "AURA_ALLOC_PARAM failed");
            return ret;
        }

        for (DT_U32 idx = 0; idx < count; idx++)
        {
            (*tensors_wrapper)[idx] = QNN_TENSOR_INIT;

            // copy tensor name
            (*tensors_wrapper)[idx].v1.name = (DT_CHAR *)AURA_ALLOC_PARAM(ctx, AURA_MEM_HEAP, strlen(tensor_src[idx].v1.name) + 1, 0);
            if (DT_NULL == (*tensors_wrapper)[idx].v1.name)
            {
                AURA_ADD_ERROR_STRING(ctx, "AURA_ALLOC_PARAM failed");
                goto EXIT;
            }

            strcpy((DT_CHAR *)(*tensors_wrapper)[idx].v1.name, tensor_src[idx].v1.name);

            (*tensors_wrapper)[idx].version = tensor_src[idx].version;
            (*tensors_wrapper)[idx].v1.id = tensor_src[idx].v1.id;
            (*tensors_wrapper)[idx].v1.type = tensor_src[idx].v1.type;
            (*tensors_wrapper)[idx].v1.dataFormat = tensor_src[idx].v1.dataFormat;
            (*tensors_wrapper)[idx].v1.dataType = tensor_src[idx].v1.dataType;

            Qnn_QuantizeParams_t src_q_params = tensor_src[idx].v1.quantizeParams;
            Qnn_QuantizeParams_t q_params = QNN_QUANTIZE_PARAMS_INIT;
            q_params.encodingDefinition = src_q_params.encodingDefinition;
            q_params.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;

            if (QNN_QUANTIZATION_ENCODING_SCALE_OFFSET == src_q_params.quantizationEncoding)
            {
                q_params.quantizationEncoding = src_q_params.quantizationEncoding;
                q_params.scaleOffsetEncoding  = src_q_params.scaleOffsetEncoding;
            }
            else if (QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET == src_q_params.quantizationEncoding)
            {
                q_params.quantizationEncoding = src_q_params.quantizationEncoding;
                q_params.axisScaleOffsetEncoding.axis = src_q_params.axisScaleOffsetEncoding.axis;
                q_params.axisScaleOffsetEncoding.numScaleOffsets = src_q_params.axisScaleOffsetEncoding.numScaleOffsets;

                if (src_q_params.axisScaleOffsetEncoding.numScaleOffsets > 0)
                {
                    q_params.axisScaleOffsetEncoding.scaleOffset = (Qnn_ScaleOffset_t *)AURA_ALLOC_PARAM(
                                                                    ctx, AURA_MEM_HEAP,
                                                                    src_q_params.axisScaleOffsetEncoding.numScaleOffsets * sizeof(Qnn_ScaleOffset_t),
                                                                    0);
                    if (DT_NULL == q_params.axisScaleOffsetEncoding.scaleOffset)
                    {
                        AURA_ADD_ERROR_STRING(ctx, "AURA_ALLOC_PARAM failed");
                        goto EXIT;
                    }

                    for (DT_U32 idx = 0; idx < q_params.axisScaleOffsetEncoding.numScaleOffsets; idx++)
                    {
                        q_params.axisScaleOffsetEncoding.scaleOffset[idx].scale = src_q_params.axisScaleOffsetEncoding.scaleOffset[idx].scale;
                        q_params.axisScaleOffsetEncoding.scaleOffset[idx].offset = src_q_params.axisScaleOffsetEncoding.scaleOffset[idx].offset;
                    }
                }
            }

            (*tensors_wrapper)[idx].v1.quantizeParams = q_params;
            (*tensors_wrapper)[idx].v1.rank = tensor_src[idx].v1.rank;
            (*tensors_wrapper)[idx].v1.dimensions = NULL;

            if (tensor_src[idx].v1.rank > 0)
            {
                DT_S32 len = tensor_src[idx].v1.rank * sizeof(DT_U32);
                (*tensors_wrapper)[idx].v1.dimensions = (DT_U32 *)AURA_ALLOC_PARAM(ctx, AURA_MEM_HEAP, len, 0);
                if ((*tensors_wrapper)[idx].v1.dimensions)
                {
                    memcpy((*tensors_wrapper)[idx].v1.dimensions, tensor_src[idx].v1.dimensions, len);
                }
                else
                {
                    AURA_ADD_ERROR_STRING(ctx, "AURA_ALLOC_PARAM failed");
                    goto EXIT;
                }
            }

            (*tensors_wrapper)[idx].v1.memType = QNN_TENSORMEMTYPE_UNDEFINED;
            (*tensors_wrapper)[idx].v1.clientBuf = QNN_CLIENT_BUFFER_INIT;
            (*tensors_wrapper)[idx].v1.memHandle = DT_NULL;
        }

        ret = Status::OK;
EXIT:
        if (ret != Status::OK)
        {
            DeinitTensorsInfoV1(ctx, tensors_wrapper, count);
        }

        AURA_RETURN(ctx, ret);
    }

    static Status DeinitTensorsInfoV1(Context *ctx, Qnn_Tensor_t **tensors_wrapper, DT_U32 count)
    {
        if (DT_NULL == ctx)
        {
            return Status::ERROR;
        }

        if (DT_NULL == tensors_wrapper || DT_NULL == *tensors_wrapper)
        {
            AURA_ADD_ERROR_STRING(ctx, "null ptr");
            return Status::OK;
        }

        for (DT_U32 idx = 0; idx < count; idx++)
        {
            if ((*tensors_wrapper)[idx].v1.name != DT_NULL)
            {
                AURA_FREE(ctx, (DT_VOID*)(*tensors_wrapper)[idx].v1.name);
                (*tensors_wrapper)[idx].v1.name = DT_NULL;
            }

            if ((*tensors_wrapper)[idx].v1.dimensions != DT_NULL)
            {
                AURA_FREE(ctx, (*tensors_wrapper)[idx].v1.dimensions);
                (*tensors_wrapper)[idx].v1.dimensions = DT_NULL;
            }

            if ((*tensors_wrapper)[idx].v1.quantizeParams.axisScaleOffsetEncoding.scaleOffset != DT_NULL)
            {
                AURA_FREE(ctx, (*tensors_wrapper)[idx].v1.quantizeParams.axisScaleOffsetEncoding.scaleOffset);
                (*tensors_wrapper)[idx].v1.quantizeParams.axisScaleOffsetEncoding.scaleOffset = DT_NULL;
            }

            (*tensors_wrapper)[idx] = QNN_TENSOR_INIT;
        }

        AURA_FREE(ctx, *tensors_wrapper);
        *tensors_wrapper = DT_NULL;

        return Status::OK;
    }

    static Status ExtractProfilingEvent(Context *ctx, const QnnProfile_EventId_t profile_event_id, const QNN_INTERFACE_VER_TYPE &interface, std::string &profile_data, ProfileStage profile_stage)
    {
        if (DT_NULL == ctx)
        {
            return Status::ERROR;
        }

        QnnProfile_EventData_t event_data;
        Qnn_ErrorHandle_t qnn_err = interface.profileGetEventData(profile_event_id, &event_data);
        if (qnn_err != QNN_PROFILE_NO_ERROR)
        {
            std::string info = "Failure in profile get event type, err : " + std::to_string(qnn_err);
            AURA_ADD_ERROR_STRING(ctx, info.c_str());
            return Status::ERROR;
        }

        std::string event_info = ",," + ProfileStageToString(profile_stage) + " [" + std::string(event_data.identifier) + "]," + std::to_string(event_data.value) + "\n";
        profile_data += event_info;

        return Status::OK;
    }

    static Status ExtractProfilingSubEvents(Context *ctx, const QnnProfile_EventId_t profile_event_id, const QNN_INTERFACE_VER_TYPE &interface, std::string &profile_data, ProfileStage profile_stage)
    {
        if (DT_NULL == ctx)
        {
            return Status::ERROR;
        }

        const QnnProfile_EventId_t *profile_sub_events = DT_NULL;
        DT_U32 num_sub_events = 0;

        Qnn_ErrorHandle_t qnn_err = interface.profileGetSubEvents(profile_event_id, &profile_sub_events, &num_sub_events);
        if (qnn_err != QNN_PROFILE_NO_ERROR)
        {
            std::string info = "failure in get sub events, err : " + std::to_string(qnn_err);
            AURA_ADD_ERROR_STRING(ctx, info.c_str());
            return Status::ERROR;
        }

        for (size_t sub_event = 0; sub_event < num_sub_events; sub_event++)
        {
            Status ret = ExtractProfilingEvent(ctx, profile_sub_events[sub_event], interface, profile_data, profile_stage);
            ret |= ExtractProfilingSubEvents(ctx, profile_sub_events[sub_event], interface, profile_data, profile_stage);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(ctx, "ExtractProfilingSubEvents : ExtractProfilingEvent or ExtractProfilingSubEvents failed");
                return Status::ERROR;
            }
        }

        return Status::OK;
    }
};

class QnnTensorMap
{
public:
    QnnTensorMap(Context *ctx, Qnn_Tensor_t *tensors, DT_U32 tensors_num, DT_BOOL is_input, Qnn_ContextHandle_t context_handle, QNN_INTERFACE_VER_TYPE *interface, std::string backend_type)
                 : m_ctx(ctx), m_is_input(is_input), m_tensors(tensors),
                   m_tensors_num(tensors_num), m_backend_type(backend_type), m_context_handle(context_handle), m_interface(interface)
    {
        do
        {
            if ((DT_NULL == m_ctx) || (DT_NULL == m_tensors) || (DT_NULL == m_context_handle) || (DT_NULL == m_interface))
            {
                AURA_ADD_ERROR_STRING(m_ctx, "null ptr");
                break;
            }

            m_is_valid = DT_TRUE;
        } while (0);
    }

    ~QnnTensorMap()
    {
        if (DeInitialize() != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "~QnnTensorMap failed");
        }
    }

    Status Initialize(const MatMap *mat_map)
    {
        if (!m_is_valid)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "m_is_valid is false");
            return Status::ERROR;
        }

        if ((DT_NULL == mat_map) || (mat_map->size() != m_tensors_num))
        {
            AURA_ADD_ERROR_STRING(m_ctx, "size match error");
            return Status::ERROR;
        }

        m_mat_map = *mat_map;

        for (DT_U32 i = 0; i < m_tensors_num; i++)
        {
            std::string tensor_name = std::string(m_tensors[i].v1.name);
            if (m_mat_map.find(tensor_name) == m_mat_map.end())
            {
                std::string info = "mat names " + tensor_name + " not provided";
                AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
                return Status::ERROR;
            }

            if (InitTensor(&m_tensors[i], m_mat_map[tensor_name], m_backend_type, tensor_name) != Status::OK)
            {
                std::string info = "InitTensor " + tensor_name + " failed";
                AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
                return Status::ERROR;
            }
        }

        return Status::OK;
    }

    Status DeInitialize()
    {
        for (auto iter = m_quant_mat.begin(); iter != m_quant_mat.end(); ++iter)
        {
            Mat *mat = iter->second;
            Delete<Mat>(m_ctx, &mat);
        }
        m_quant_mat.clear();
        m_register_mat_map.clear();

        return Status::OK;
    }

    Status Quant()
    {
        if (DT_FALSE == m_is_input)
        {
            return Status::OK;
        }

        if (m_quant_mat.size() > m_tensors_num)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "quant_mat size is wrong");
            return Status::ERROR;
        }

        for (DT_U32 i = 0; i < m_tensors_num; i++)
        {
            std::string tensor_name = std::string(m_tensors[i].v1.name);

            if (m_quant_mat.count(tensor_name))
            {
                if (NNQuantize(m_ctx, *(m_mat_map[tensor_name]), *(m_quant_mat[tensor_name]),
                               -m_tensors[i].v1.quantizeParams.scaleOffsetEncoding.offset,
                               m_tensors[i].v1.quantizeParams.scaleOffsetEncoding.scale) != Status::OK)
                {
                    AURA_ADD_ERROR_STRING(m_ctx, "NN Quantize failed");
                    return Status::ERROR;
                }
            }
        }

        return Status::OK;
    }

    Status DeQuant()
    {
        if (DT_TRUE == m_is_input)
        {
            return Status::OK;
        }

        if (m_quant_mat.size() > m_tensors_num)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "quant_mat size is wrong");
            return Status::ERROR;
        }

        for (DT_U32 i = 0; i < m_tensors_num; i++)
        {
            std::string tensor_name = std::string(m_tensors[i].v1.name);

            if (m_quant_mat.count(tensor_name))
            {
                if (NNDeQuantize(m_ctx, *(m_quant_mat[tensor_name]), *(m_mat_map[tensor_name]),
                                 -m_tensors[i].v1.quantizeParams.scaleOffsetEncoding.offset,
                                 m_tensors[i].v1.quantizeParams.scaleOffsetEncoding.scale) != Status::OK)
                {
                    AURA_ADD_ERROR_STRING(m_ctx, "NNDeQuantize failed");
                    return Status::ERROR;
                }
            }
        }

        return Status::OK;
    }

    Status RegisterMem()
    {
        for (DT_U32 idx = 0; idx < m_tensors_num; idx++)
        {
            std::string tensor_name = std::string(m_tensors[idx].v1.name);

            if (!m_register_mat_map.count(tensor_name))
            {
                continue;
            }

            Qnn_MemDescriptor_t mem_descriptor = QNN_MEM_DESCRIPTOR_INIT;
            mem_descriptor.memShape = {m_tensors[idx].v1.rank, m_tensors[idx].v1.dimensions, DT_NULL};
            mem_descriptor.dataType = m_tensors[idx].v1.dataType;
            mem_descriptor.memType = QNN_MEM_TYPE_ION;
            mem_descriptor.ionInfo.fd = m_register_mat_map[tensor_name]->GetBuffer().m_property;

            Qnn_ErrorHandle_t qnn_err = m_interface->memRegister(m_context_handle, &mem_descriptor, 1u, &(m_tensors[idx].v1.memHandle));
            if (qnn_err != QNN_SUCCESS)
            {
                std::string info = std::string("memRegister failed, err = ") + std::to_string(qnn_err);
                AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
                return Status::ERROR;
            }
        }

        return Status::OK;
    }

    Status DeRegisterMem()
    {
        for (DT_U32 idx = 0; idx < m_tensors_num; idx++)
        {
            if (QNN_TENSORMEMTYPE_RAW == m_tensors[idx].v1.memType)
            {
                m_tensors[idx].v1.clientBuf = QNN_CLIENT_BUFFER_INIT;
            }
            else if (QNN_TENSORMEMTYPE_MEMHANDLE == m_tensors[idx].v1.memType)
            {
                if (m_tensors[idx].v1.memHandle != DT_NULL)
                {
                    Qnn_ErrorHandle_t qnn_err = m_interface->memDeRegister(&(m_tensors[idx].v1.memHandle), 1u);
                    if (qnn_err != QNN_SUCCESS)
                    {
                        std::string info = std::string("memDeRegister failed, err : ") + std::to_string(qnn_err);
                        AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
                        return Status::ERROR;
                    }

                    m_tensors[idx].v1.memHandle = DT_NULL;
                }

            }
            m_tensors[idx].v1.memType = QNN_TENSORMEMTYPE_UNDEFINED;
        }

        m_register_mat_map.clear();

        return Status::OK;
    }

    std::vector<Mat>* GetRegisterMat()
    {
        return &m_register_mat;
    }

private:
    Status InitTensor(Qnn_Tensor_t *tensor, Mat *mat, const std::string &backend_type, const std::string &tensor_name)
    {
        if (tensor->v1.rank > 4 || tensor->v1.rank <= 0)
        {
            std::string info = "not supported rank, rank is " + std::to_string(tensor->v1.rank);
            AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
            return Status::ERROR;
        }

        DT_BOOL is_quant = DT_FALSE;
        ElemType src_elem_type = mat->GetElemType();
        ElemType dst_elem_type = GetElemType(tensor->v1.dataType);
        if ((src_elem_type != ElemType::INVALID) && (src_elem_type == dst_elem_type))
        {
            if (!mat->IsContinuous())
            {
                AURA_ADD_ERROR_STRING(m_ctx, "external mat data memory must be continuous");
                return Status::ERROR;
            }
        }
        else if ((ElemType::F32 == src_elem_type) && ((QNN_DATATYPE_UFIXED_POINT_8 == tensor->v1.dataType) ||
                                                      (QNN_DATATYPE_UFIXED_POINT_16 == tensor->v1.dataType)))
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
            if (!m_quant_mat.count(tensor_name))
            {
                m_quant_mat[tensor_name] = Create<Mat>(m_ctx, dst_elem_type, mat->GetSizes());
                if (!m_quant_mat[tensor_name]->IsValid())
                {
                    AURA_ADD_ERROR_STRING(m_ctx, "create tensor mat failed");
                    return Status::ERROR;
                }
            }

            mat = m_quant_mat[tensor_name];
        }

        // config tensor
#if defined(AURA_BUILD_HEXAGON)
        AURA_UNUSED(backend_type);
        if (AURA_MEM_DMA_BUF_HEAP == mat->GetMemType() || AURA_MEM_HEAP == mat->GetMemType())
        {
            tensor->v1.memType = QNN_TENSORMEMTYPE_RAW;

            Qnn_ClientBuffer_t client_buffer = QNN_CLIENT_BUFFER_INIT;
            client_buffer.dataSize = mat->GetSizes().Total() * ElemTypeSize(mat->GetElemType());
            client_buffer.data = mat->GetBuffer().m_data;
            tensor->v1.clientBuf = client_buffer;
        }
#else
        if (AURA_MEM_HEAP == mat->GetMemType() || "cpu" == backend_type || "gpu" == backend_type)
        {
            tensor->v1.memType = QNN_TENSORMEMTYPE_RAW;

            Qnn_ClientBuffer_t client_buffer = QNN_CLIENT_BUFFER_INIT;
            client_buffer.dataSize = mat->GetSizes().Total() * ElemTypeSize(mat->GetElemType());
            client_buffer.data = mat->GetBuffer().m_data;
            tensor->v1.clientBuf = client_buffer;
        }
        else if (AURA_MEM_DMA_BUF_HEAP == mat->GetMemType())
        {
            tensor->v1.memType = QNN_TENSORMEMTYPE_MEMHANDLE;
            tensor->v1.memHandle = DT_NULL;

            m_register_mat_map[tensor_name] = mat;
        }
#endif // AURA_BUILD_HEXAGON
        else
        {
            AURA_ADD_ERROR_STRING(m_ctx, "unknown mem type");
            return Status::ERROR;
        }
        return Status::OK;
    }

    Context *m_ctx;

    DT_BOOL m_is_valid;
    DT_BOOL m_is_input;

    Qnn_Tensor_t *m_tensors;
    DT_U32 m_tensors_num;
    std::string m_backend_type;

    MatMap m_mat_map;
    MatMap m_quant_mat;
    MatMap m_register_mat_map;

    std::vector<Mat> m_register_mat;

    Qnn_ContextHandle_t m_context_handle;
    QNN_INTERFACE_VER_TYPE *m_interface;
};

class QnnExecutorImplV2 : public QnnExecutorImpl
{
public:
    QnnExecutorImplV2(Context *ctx, const std::shared_ptr<QnnModel> &model, const NNConfig &config)
                      : QnnExecutorImpl(ctx, model, config),
                        m_interface(QNN_INTERFACE_VER_TYPE_INIT),
                        m_system_interface(QNN_SYSTEM_INTERFACE_VER_TYPE_INIT),
                        m_log_handle(DT_NULL),
                        m_arch(QnnHtpDevice_Arch_t::QNN_HTP_DEVICE_ARCH_UNKNOWN),
                        m_backend_handle(DT_NULL),
                        m_profile_handle(DT_NULL),
                        m_power_config_id(0),
                        m_perf_infra(QNN_HTP_DEVICE_PERF_INFRASTRUCTURE_INIT),
                        m_context(DT_NULL),
                        m_graphs_info(DT_NULL),
                        m_graphs_count(0)
    {
        do
        {
            if (DT_NULL == m_model)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "m_model is null");
                break;
            }

            m_interface = QnnLibrary::Get(m_model->GetBackendType()).GetInterface();
            if (DT_NULL == m_interface.graphExecute)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "m_interface is null");
                break;
            }

            m_system_interface = QnnLibrary::Get(m_model->GetBackendType()).GetSystemInterface();

            // check version : v2.7.xxxxx
            std::vector<std::string> build_id;
            m_build_id = QnnUtils::GetBuildId(m_ctx, m_interface);
            build_id = NNSplit(m_build_id, '.');

            // framework version
            std::vector<std::string> framework_version;
            framework_version = NNSplit(m_model->GetFrameWorkVersion(), '.');

            if ((4 == framework_version.size()) && (build_id.size() >= 4))
            {
                if ((framework_version[1] != build_id[0]) || (framework_version[2] > build_id[1]))
                {
                    std::string info = "frame version not match qnn version, frame verison: " + m_model->GetFrameWorkVersion() + " build_id: " + m_build_id;
                    AURA_ADD_ERROR_STRING(ctx, info.c_str());
                    break;
                }

                if (framework_version[2] != build_id[1])
                {
                    std::string info = "WARNING: frame version not match qnn version, model init may slower, frame verison: " +
                                        m_model->GetFrameWorkVersion() + " build_id: " + m_build_id + "\n";
                    AURA_LOGD(m_ctx, AURA_TAG, info.c_str());
                }

            }
            else
            {
                AURA_ADD_ERROR_STRING(ctx, "frame version size must be 4 and build_id size can not be less than 4");
                break;
            }

            auto init_func = [=]() -> Status
            {
                if (CreateQnnHandles() != Status::OK)
                {
                    AURA_LOGE(m_ctx, AURA_TAG, "CreateQnnHandles failed\n");
                    return Status::ERROR;
                }

                if (CreateRuntime() != Status::OK)
                {
                    AURA_LOGE(m_ctx, AURA_TAG, "CreateRuntime failed\n");
                    return Status::ERROR;
                }

                m_is_valid = DT_TRUE;
                return Status::OK;
            };
#if defined(AURA_BUILD_HEXAGON)
            if (init_func() != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "m_interface is null");
                break;
            }
            m_config.async_call = DT_FALSE;
#else
            if (m_wp)
            {
                m_token = m_wp->AsyncRun(init_func);
            }
            else
            {
                AURA_ADD_ERROR_STRING(ctx, "m_wp null ptr");
            }
#endif // AURA_BUILD_HEXAGON
        } while (0);
    }

    ~QnnExecutorImplV2()
    {
        if (m_token.valid())
        {
            m_token.wait();
        }

        Status ret = Status::OK;

        ret |= DeInitialize();
        ret |= DeleteQnnHandles();
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "~QnnExecutorImplV2 failed");
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
            AURA_ADD_ERROR_STRING(m_ctx, "m_is_valid is invalid ");
            return Status::ERROR;
        }

        Status ret = Status::ERROR;

#if !defined(AURA_BUILD_LINUX)
        if (QnnUtils::GetHtpDeviceArch(m_ctx, m_interface, m_log_handle, m_arch) != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "failed to get htp device arch");
            goto EXIT;
        }
#endif // AURA_BUILD_LINUX

        // profiling
        if (ExtractBackendProfilingInfo(ProfileStage::INIT_STATUS) != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "ExtractBackendProfilingInfo failed");
            goto EXIT;
        }

        for (DT_U32 idx = 0; idx < m_graphs_count; idx++)
        {
            QnnTensorMap *in_tensor_map = Create<QnnTensorMap>(m_ctx, (*m_graphs_info)[idx].input_tensors, (*m_graphs_info)[idx].num_input_tensors, DT_TRUE, m_context, &m_interface, m_model->GetBackendType());
            QnnTensorMap *out_tensor_map = Create<QnnTensorMap>(m_ctx, (*m_graphs_info)[idx].output_tensors, (*m_graphs_info)[idx].num_output_tensors, DT_FALSE, m_context, &m_interface, m_model->GetBackendType());

            if (DT_NULL == in_tensor_map || DT_NULL == out_tensor_map)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "Create QnnTensorMap failed");
                goto EXIT;
            }

            m_input_tensor_map.push_back(in_tensor_map);
            m_output_tensor_map.push_back(out_tensor_map);
        }

#if defined(AURA_BUILD_HEXAGON)
        ret = QnnUtils::SetPerformance(m_ctx, m_perf_infra, m_config.htp_perf_level, m_config.hmx_perf_level, m_model->GetBackendType(), m_power_config_id, m_arch);
#else
        if (m_config.async_call && m_wp)
        {
            ret = m_wp->AsyncRun(QnnUtils::SetPerformance, m_ctx, m_perf_infra, m_config.htp_perf_level, m_config.hmx_perf_level, m_model->GetBackendType(), m_power_config_id, m_arch).get();
        }
        else
        {
            ret = QnnUtils::SetPerformance(m_ctx, m_perf_infra, m_config.htp_perf_level, m_config.hmx_perf_level, m_model->GetBackendType(), m_power_config_id, m_arch);
        }
#endif // AURA_BUILD_HEXAGON
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "SetPerf perf_level failed");
        }
EXIT:
        if (ret != Status::OK)
        {
            DeInitialize();
        }

        AURA_RETURN(m_ctx, ret);
    }

    Status Update(const std::string &name, AnyParams &params) override
    {
        if ("register_mem" == name)
        {
            return Register(params);
        }
        else if ("deregister_mem" == name)
        {
            return DeRegister(params);
        }
        else if ("update_perf" == name)
        {
            auto update_perf = [=]() -> Status
            {
                return UpdatePerf(params);
            };
#if defined(AURA_BUILD_HEXAGON)
            return update_perf();
#else
            if (m_config.async_call && m_wp)
            {
                return m_wp->AsyncRun(update_perf).get();
            }
            else
            {
                return update_perf();
            }
#endif // AURA_BUILD_HEXAGON
        }
        else
        {
            std::string info = "the specified function '" + name + "' does not exist";
            AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
            return  Status::ERROR;
        }

        return  Status::ERROR;
    }

    Status Forward(const MatMap &input, MatMap &output, DT_S32 graph_id) override
    {
        auto process_func = [=]() -> Status
        {
            Status ret = Status::OK;
            Qnn_ErrorHandle_t qnn_err;

            DT_BOOL valid_graph_id = DT_TRUE;

            MatMap input_mapped = m_model->MapMatNames(input, DT_TRUE);
            MatMap output_mapped = m_model->MapMatNames(output, DT_FALSE);

            if (input_mapped.empty() || output_mapped.empty())
            {
                AURA_ADD_ERROR_STRING(m_ctx, "map names failed");
                return Status::ERROR;
            }

            if (graph_id >= static_cast<DT_S32>(m_graphs_count) || graph_id < 0)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "invalid graph id");
                valid_graph_id = DT_FALSE;
                ret = Status::ERROR;
                return ret;
            }

            if (m_config.graph_ids[0] != -1)
            {
                valid_graph_id = DT_FALSE;
                for (auto &id : m_config.graph_ids)
                {
                    if (graph_id == id)
                    {
                        valid_graph_id = DT_TRUE;
                        break;
                    }
                }

                if (!valid_graph_id)
                {
                    AURA_ADD_ERROR_STRING(m_ctx, "invalid graph id");
                    ret = Status::ERROR;
                    return ret;
                }
            }

            QnnTensorMap *input_tensor_map  = NULL;
            QnnTensorMap *output_tensor_map = NULL;

            GraphInfo_t **graphs_info = GetGraphsInfo(input_mapped, output_mapped, graph_id);
            if (graphs_info != NULL)
            {
                input_tensor_map  = graphs_info[graph_id]->input_tensor_map;
                output_tensor_map = graphs_info[graph_id]->output_tensor_map;
            }
            else
            {
                graphs_info = m_graphs_info;
                input_tensor_map  = m_input_tensor_map[graph_id];
                output_tensor_map = m_output_tensor_map[graph_id];
                if (input_tensor_map->Initialize(&input_mapped) != Status::OK)
                {
                    AURA_ADD_ERROR_STRING(m_ctx, "input map initilize failed");
                    ret = Status::ERROR;
                    goto EXIT;
                }

                if (output_tensor_map->Initialize(&output_mapped) != Status::OK)
                {
                    AURA_ADD_ERROR_STRING(m_ctx, "output map initilize failed");
                    ret = Status::ERROR;
                    goto EXIT;
                }

                if (input_tensor_map->RegisterMem() != Status::OK)
                {
                    AURA_ADD_ERROR_STRING(m_ctx, "input map Register failed");
                    ret = Status::ERROR;
                    goto EXIT;
                }

                if (output_tensor_map->RegisterMem() != Status::OK)
                {
                    AURA_ADD_ERROR_STRING(m_ctx, "output map Register failed");
                    ret = Status::ERROR;
                    goto EXIT;
                }
            }

            if (input_tensor_map->Quant() != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "input map Quant failed");
                ret = Status::ERROR;
                goto EXIT;
            }

            qnn_err = m_interface.graphExecute((*graphs_info)[graph_id].graph,
                                               (*graphs_info)[graph_id].input_tensors,
                                               (*graphs_info)[graph_id].num_input_tensors,
                                               (*graphs_info)[graph_id].output_tensors,
                                               (*graphs_info)[graph_id].num_output_tensors,
                                                m_profile_handle,
                                                DT_NULL);
            if (qnn_err != QNN_GRAPH_NO_ERROR)
            {
                AURA_ADD_ERROR_STRING(m_ctx, ("graphExecute failed, err : " + std::to_string(qnn_err)).c_str());
                ret = Status::ERROR;
                goto EXIT;
            }

            if (output_tensor_map->DeQuant() != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "output_tensor_map DeQuant failed");
                ret = Status::ERROR;
                goto EXIT;
            }

            if (ExtractBackendProfilingInfo(ProfileStage::EXECUTE_STATUS) != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "ExtractBackendProfilingInfo failed");
                ret = Status::ERROR;
                goto EXIT;
            }
    EXIT:
            if (!valid_graph_id)
            {
                return ret;
            }

            if (graphs_info == m_graphs_info)
            {
                if (output_tensor_map->DeRegisterMem() != Status::OK)
                {
                    AURA_ADD_ERROR_STRING(m_ctx, "output map deregister failed");
                    ret = Status::ERROR;
                }

                if (input_tensor_map->DeRegisterMem() != Status::OK)
                {
                    AURA_ADD_ERROR_STRING(m_ctx, "input map deregister failed");
                    ret = Status::ERROR;
                }
            }

            AURA_RETURN(m_ctx, ret);
        };
#if defined(AURA_BUILD_HEXAGON)
        return process_func();
#else
        if (m_config.async_call && m_wp)
        {
            return m_wp->AsyncRun(process_func).get();
        }
        else
        {
            return process_func();
        }
#endif // AURA_BUILD_HEXAGON
    }

    std::vector<TensorDescMap> GetInputs() override
    {
        std::vector<TensorDescMap> result;

        for (DT_U32 i = 0; i < m_graphs_count; i++)
        {
            GraphInfo_t graph_info = (*m_graphs_info)[i];
            TensorDescMap graph_inputs;

            for (DT_U32 j = 0; j < graph_info.num_input_tensors; j++)
            {
                TensorDesc desc;
                std::string name = graph_info.input_tensors[j].v1.name;

                desc.elem_type = GetElemType(graph_info.input_tensors[j].v1.dataType);
                for (DT_U32 k = 0; k < graph_info.input_tensors[j].v1.rank; k++)
                {
                    desc.sizes.push_back(graph_info.input_tensors[j].v1.dimensions[k]);
                }

                desc.scale = graph_info.input_tensors[j].v1.quantizeParams.scaleOffsetEncoding.scale;
                desc.zero_point = -graph_info.input_tensors[j].v1.quantizeParams.scaleOffsetEncoding.offset;
                desc.graph_id = i;

                graph_inputs[name] = desc;
            }

            graph_inputs = m_model->MapTensorDescNames(graph_inputs, DT_TRUE);
            result.push_back(graph_inputs);
        }

        return result;
    }

    std::vector<TensorDescMap> GetOutputs() override
    {
        std::vector<TensorDescMap> result;

        for (DT_U32 i = 0; i < m_graphs_count; i++)
        {
            GraphInfo_t graph_info = (*m_graphs_info)[i];
            TensorDescMap graph_outputs;

            for (DT_U32 j = 0; j < graph_info.num_output_tensors; j++)
            {
                TensorDesc desc;
                std::string name = graph_info.output_tensors[j].v1.name;

                desc.elem_type = GetElemType(graph_info.output_tensors[j].v1.dataType);
                for (DT_U32 k = 0; k < graph_info.output_tensors[j].v1.rank; k++)
                {
                    desc.sizes.push_back(graph_info.output_tensors[j].v1.dimensions[k]);
                }

                desc.scale = graph_info.output_tensors[j].v1.quantizeParams.scaleOffsetEncoding.scale;
                desc.zero_point = -graph_info.output_tensors[j].v1.quantizeParams.scaleOffsetEncoding.offset;
                desc.graph_id = i;

                graph_outputs[name] = desc;
            }

            graph_outputs = m_model->MapTensorDescNames(graph_outputs, DT_FALSE);
            result.push_back(graph_outputs);
        }

        return result;
    }

    std::string GetVersion() override
    {
        return (m_model->GetVersion() + " device(qnn." + m_build_id + ")");
    }

private:
    Status CreateQnnHandles()
    {
        Status ret = Status::ERROR;

        if (QnnUtils::CreateLogHandle(m_ctx, m_interface, m_log_handle, m_config.log_level) != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "CreateLogHandle failed");
            goto EXIT;
        }

        if (QnnUtils::CreateBackend(m_ctx, m_interface, m_log_handle, m_backend_handle) != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "CreateBackend failed");
            goto EXIT;
        }

        if (QnnUtils::CreateProfiler(m_ctx, m_interface, m_backend_handle, m_config.profiling_level, m_profile_handle) != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "CreateProfiler failed");
            goto EXIT;
        }

        if (QnnUtils::CreatePowerConfigId(m_ctx, m_interface, m_model->GetBackendType(), m_perf_infra, m_power_config_id) != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "CreatePowerConfigId failed");
            goto EXIT;
        }

        if (QnnUtils::SetMemStepSize(m_ctx, m_perf_infra, m_config.mem_step_size) != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "CreatePowerConfigId failed");
            goto EXIT;
        }

        if (QnnUtils::RegisterOpPackages(m_ctx, m_interface, m_backend_handle, m_config.udo_path) != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "RegisterOpPackages failed");
            goto EXIT;
        }

        ret = Status::OK;
EXIT:
        if (ret != Status::OK)
        {
            DeleteQnnHandles();
        }

        AURA_RETURN(m_ctx, ret);
    }

    Status DeleteQnnHandles()
    {
        Status ret = Status::OK;

        ret |= QnnUtils::DeletePowerConfigId(m_ctx, m_power_config_id, m_perf_infra);

        ret |= ExtractBackendProfilingInfo(ProfileStage::DEINIT_STATUS, DT_TRUE);

        ret |= QnnUtils::DeleteProfiler(m_ctx, m_interface, m_profile_handle);

        ret |= QnnUtils::DeleteBackend(m_ctx, m_interface, m_backend_handle);

        ret |= QnnUtils::DeleteLogHandle(m_ctx, m_interface, m_log_handle);

        AURA_RETURN(m_ctx, ret);
    }

    Status CreateRuntime()
    {
        Status ret = Status::ERROR;

        Buffer buffer = m_model->GetModelBuffer();
        if (!buffer.IsValid())
        {
            AURA_ADD_ERROR_STRING(m_ctx, "GetModelBuffer failed");
            goto EXIT;
        }

        if (QnnUtils::CreateGraphsFromBinary(m_ctx, m_interface, m_system_interface, buffer, m_backend_handle,
                                             QnnLibrary::Get(m_model->GetBackendType()).GetDeviceHandle(),
                                             m_profile_handle, m_context, &m_graphs_info, m_graphs_count,
                                             m_config.graph_ids, m_config.budget) != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "CreateGraphsFromBinary failed");
            goto EXIT;
        }

        ret = Status::OK;
EXIT:
        m_model->ReleaseModelBuffer();
        AURA_RETURN(m_ctx, ret);
    }

    Status DeInitialize()
    {
        Status ret = Status::OK;
#if defined(AURA_BUILD_HEXAGON)
        ret = QnnUtils::SetPerformance(m_ctx, m_perf_infra, NNPerfLevel::PERF_DEFAULT, NNPerfLevel::PERF_DEFAULT, m_model->GetBackendType(), m_power_config_id, m_arch);
#else
        if (m_config.async_call && m_wp)
        {
            ret = m_wp->AsyncRun(QnnUtils::SetPerformance, m_ctx, m_perf_infra, NNPerfLevel::PERF_DEFAULT, NNPerfLevel::PERF_DEFAULT, m_model->GetBackendType(), m_power_config_id, m_arch).get();
        }
        else
        {
            ret = QnnUtils::SetPerformance(m_ctx, m_perf_infra, NNPerfLevel::PERF_DEFAULT, NNPerfLevel::PERF_DEFAULT, m_model->GetBackendType(), m_power_config_id, m_arch);
        }
#endif // AURA_BUILD_HEXAGON
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "SetPerf default failed");
        }

        for (DT_U32 i = 0; i < m_input_tensor_map.size(); i++)
        {
            Delete<QnnTensorMap>(m_ctx, &m_input_tensor_map[i]);
            Delete<QnnTensorMap>(m_ctx, &m_output_tensor_map[i]);
        }

        m_input_tensor_map.clear();
        m_output_tensor_map.clear();

        if (QnnUtils::DeinitGraphsInfo(m_ctx, &m_graphs_info, m_graphs_count) != Status::OK)
        {
            ret = Status::ERROR;
            AURA_ADD_ERROR_STRING(m_ctx, "DeinitGraphsInfo failed");
        }

        if (QnnUtils::DeleteContext(m_ctx, m_interface, m_profile_handle, m_context) != Status::OK)
        {
            ret = Status::ERROR;
            AURA_ADD_ERROR_STRING(m_ctx, "DeleteContext failed");
        }

        AURA_RETURN(m_ctx, ret);
    }

    std::string GenerateProfileHeadInfo()
    {
        std::string head_info;

        head_info += "Qnn SDK Version:," + m_build_id + "\n";
        head_info += "Model Name:," + m_model->GetFrameWorkVersion() + "\n";
        head_info += "Backends:," + m_model->GetBackendType() + "\n";
        head_info += "PerfLevel htp:," + NNPerfLevelToString(m_config.htp_perf_level) + "\n";
        head_info += "PerfLevel hmx:," + NNPerfLevelToString(m_config.hmx_perf_level) + "\n";
        head_info += "ProfilingLevel:," + NNProfilingLevelToString(m_config.profiling_level) + "\n";
        head_info += "LogLevel:," + NNLogLevelToString(m_config.log_level) + "\n";

        head_info += "\n";
        head_info += ",,," + m_model->GetBackendType() + " timing(us)\n";

        return head_info;
    }

    Status ExtractBackendProfilingInfo(ProfileStage profile_stage, DT_BOOL save_profiling = DT_FALSE)
    {
        if (QnnUtils::ExtractBackendProfilingInfo(m_ctx, m_interface, m_profile_handle, m_config.profiling_level, m_profile_data, profile_stage) != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "ExtractBackendProfilingInfo failed");
            return Status::ERROR;
        }

        if (save_profiling && (m_config.profiling_level != NNProfilingLevel::PROFILING_OFF))
        {
            std::string profile_head_info = GenerateProfileHeadInfo();

            TimeStamp now = TimeStamp::Now();

            std::string fname = std::string(m_config.profiling_path) + "/" + now.ToString() + ".csv";
            std::ofstream ofs(fname, std::ios::out | std::ios::binary | std::ios::trunc);

            if (!ofs.is_open())
            {
                AURA_ADD_ERROR_STRING(m_ctx, ("open file failed :" + fname).c_str());
                return Status::ERROR;
            }

            ofs << profile_head_info;
            ofs << m_profile_data;
            ofs.close();

            m_profile_data.clear();
        }

        return Status::OK;
    }

    Status Register(AnyParams &params)
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

        DT_S32 graph_id = 0;
        if (params.HasKeys("graph_id") == DT_TRUE)
        {
            ret = params.Get("graph_id", graph_id);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "parse graph id failed");
                return ret;
            }
        }

        ret = RegisterImpl(input, output);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "Register mem failed");
            return ret;
        }

        return Status::OK;
    }

    Status DeRegister(AnyParams &params)
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

        DT_S32 graph_id = 0;
        if (params.HasKeys("graph_id") == DT_TRUE)
        {
            ret = params.Get("graph_id", graph_id);
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "parse graph id failed");
                return ret;
            }
        }

        ret = DeRegisterImpl(input, output);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "DeRegister mem failed");
            return ret;
        }

        return Status::OK;
    }

    Status RegisterImpl(const MatMap &input, MatMap &output, DT_S32 graph_id = 0)
    {
        Status ret = Status::ERROR;

        MatMap input_mapped  = m_model->MapMatNames(input, DT_TRUE);
        MatMap output_mapped = m_model->MapMatNames(output, DT_FALSE);

        // check wether registed
        GraphInfo_t **graphs_info = GetGraphsInfo(input_mapped, output_mapped);
        if (graphs_info != NULL)
        {
            return Status::OK;
        }

        // create graph info
        ret = QnnUtils::CreateGraphsInfo(m_ctx, &m_graphs_info, &graphs_info, m_graphs_count);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "create graphs info failed");
            return Status::ERROR;
        }

        // recode tensor map
        for (DT_U32 idx = 0; idx < m_graphs_count; idx++)
        {
            QnnTensorMap *input_tensor_map  = Create<QnnTensorMap>(m_ctx, (*graphs_info)[idx].input_tensors,  (*graphs_info)[idx].num_input_tensors,  DT_TRUE,  m_context, &m_interface, m_model->GetBackendType());
            QnnTensorMap *output_tensor_map = Create<QnnTensorMap>(m_ctx, (*graphs_info)[idx].output_tensors, (*graphs_info)[idx].num_output_tensors, DT_FALSE, m_context, &m_interface, m_model->GetBackendType());
            if ((DT_NULL == input_tensor_map) || (DT_NULL == output_tensor_map))
            {
                AURA_ADD_ERROR_STRING(m_ctx, "Create QnnTensorMap failed");
                ret = Status::ERROR;
                goto EXIT;
            }

            (*graphs_info)[idx].input_tensor_map  = input_tensor_map;
            (*graphs_info)[idx].output_tensor_map = output_tensor_map;
        }

        // recode mat
        {
            std::vector<Mat> *input_register_mat = (*graphs_info)[graph_id].input_tensor_map->GetRegisterMat();
            for (auto &iter : input_mapped)
            {
                input_register_mat->push_back(*iter.second);
            }

            std::vector<Mat> *output_register_mat = (*graphs_info)[graph_id].output_tensor_map->GetRegisterMat();
            for (auto &iter : output_mapped)
            {
                output_register_mat->push_back(*iter.second);
            }
        }

        // Initialize and register
        if ((*graphs_info)[graph_id].input_tensor_map->Initialize(&input_mapped) != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "input map initilize failed");
            ret = Status::ERROR;
            goto EXIT;
        }

        if ((*graphs_info)[graph_id].input_tensor_map->RegisterMem() != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "input map Register failed");
            ret = Status::ERROR;
            goto EXIT;
        }

        if ((*graphs_info)[graph_id].output_tensor_map->Initialize(&output_mapped) != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "output map initilize failed");
            ret = Status::ERROR;
            goto EXIT;
        }

        if ((*graphs_info)[graph_id].output_tensor_map->RegisterMem() != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "output map Register failed");
            ret = Status::ERROR;
            goto EXIT;
        }

        m_register_graphs_info.push_back(graphs_info);

        ret = Status::OK;
EXIT:
        if (ret != Status::OK)
        {
            DeInitialize();
        }

        return ret;
    }

    Status DeRegisterImpl(const MatMap &input, MatMap &output, DT_S32 graph_id = 0)
    {
        Status ret = Status::ERROR;

        MatMap input_mapped  = m_model->MapMatNames(input, DT_TRUE);
        MatMap output_mapped = m_model->MapMatNames(output, DT_FALSE);

        // check Whether registed
        GraphInfo_t **graphs_info = GetGraphsInfo(input_mapped, output_mapped);
        if (NULL == graphs_info)
        {
            return Status::OK;
        }

        QnnTensorMap *input_tensor_map = (*graphs_info)[graph_id].input_tensor_map;
        if (input_tensor_map->DeRegisterMem() != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "input mem DeRegist failed");
            ret = Status::ERROR;
        }

        QnnTensorMap *output_tensor_map = (*graphs_info)[graph_id].output_tensor_map;
        if (output_tensor_map->DeRegisterMem() != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "output mem DeRegist failed");
            ret = Status::ERROR;
        }

        // clear register mat
        std::vector<Mat> *input_register_mat = (*graphs_info)[graph_id].input_tensor_map->GetRegisterMat();
        std::vector<Mat>().swap(*input_register_mat);

        std::vector<Mat> *output_register_mat = (*graphs_info)[graph_id].output_tensor_map->GetRegisterMat();
        std::vector<Mat>().swap(*output_register_mat);

        // when on other graph in using, delete graphs info
        DT_BOOL is_graph_using = DT_FALSE;

        for (DT_U32 i = 0; i < m_graphs_count; i++)
        {
            if (((*graphs_info)[graph_id].input_tensor_map->GetRegisterMat()->size() != 0) || ((*graphs_info)[graph_id].output_tensor_map->GetRegisterMat()->size() != 0))
            {
                is_graph_using = DT_TRUE;
            }
        }

        if (DT_FALSE == is_graph_using)
        {
            auto it = std::find(m_register_graphs_info.begin(), m_register_graphs_info.end(), graphs_info);

            m_register_graphs_info.erase(it);

            QnnUtils::DeleteGraphsInfo(m_ctx, &graphs_info, m_graphs_count);
            graphs_info = NULL;
        }

        ret = Status::OK;

        return ret;
    }

    GraphInfo_t** GetGraphsInfo(const MatMap &input, const MatMap &output, DT_S32 graph_id = 0)
    {
        GraphInfo_t **graph_info = NULL;

        // check whether mat registed
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

        // check in all registed graph info
        for (auto &iter : m_register_graphs_info)
        {
            std::vector<Mat> *input_register_mat  = (*iter)[graph_id].input_tensor_map->GetRegisterMat();
            std::vector<Mat> *output_register_mat = (*iter)[graph_id].output_tensor_map->GetRegisterMat();

            if ((check_mat_register_func(input,  *input_register_mat)  == DT_TRUE) &&
                (check_mat_register_func(output, *output_register_mat) == DT_TRUE))
            {
                graph_info = iter;
                break;
            }
        }

        return graph_info;
    }

    Status UpdatePerf(const AnyParams &params)
    {
        Status ret = Status::ERROR;

        NNPerfLevel htp_perf_level = NNPerfLevel::PERF_HIGH;
        NNPerfLevel hmx_perf_level = NNPerfLevel::PERF_HIGH;

        if (GetNNPerfLevel(m_ctx, params, htp_perf_level) != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "GetNNPerfLevel failed");
            return Status::ERROR;
        }

        if (GetQnnHmxPerfLevel(m_ctx, params, hmx_perf_level) != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "GetQnnHmxPerfLevel failed");
            return Status::ERROR;
        }

        if ((htp_perf_level != m_config.htp_perf_level) || (hmx_perf_level != m_config.hmx_perf_level))
        {
            m_config.htp_perf_level = htp_perf_level;
            m_config.hmx_perf_level = hmx_perf_level;

            ret = QnnUtils::SetPerformance(m_ctx, m_perf_infra, m_config.htp_perf_level, m_config.hmx_perf_level, m_model->GetBackendType(), m_power_config_id, m_arch);

            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "SetPerformance failed");
                return Status::ERROR;
            }
        }

        return Status::OK;
    }

    // qnn interface
    QNN_INTERFACE_VER_TYPE m_interface;
    QNN_SYSTEM_INTERFACE_VER_TYPE m_system_interface;

    std::string m_build_id;

    Qnn_LogHandle_t m_log_handle;

    QnnHtpDevice_Arch_t m_arch;

    Qnn_BackendHandle_t m_backend_handle;

    Qnn_ProfileHandle_t m_profile_handle;
    std::string m_profile_data;

    DT_U32 m_power_config_id;
    QnnHtpDevice_PerfInfrastructure_t m_perf_infra;

    Qnn_ContextHandle_t m_context;

    GraphInfo_t **m_graphs_info;
    DT_U32 m_graphs_count;

    std::vector<QnnTensorMap*> m_input_tensor_map;
    std::vector<QnnTensorMap*> m_output_tensor_map;

    std::vector<GraphInfo_t**> m_register_graphs_info;
};

} //namespace aura