namespace aura
{

AURA_INLINE Snpe_Runtime_t GetSnpeRuntimeType(const NNBackend &backend_type)
{
    Snpe_Runtime_t snpe_runtime_type;

    if (NNBackend::CPU == backend_type)
    {
        snpe_runtime_type = SNPE_RUNTIME_CPU;
    }
    else if (NNBackend::GPU == backend_type)
    {
        snpe_runtime_type = SNPE_RUNTIME_GPU;
    }
    else if (NNBackend::NPU == backend_type)
    {
        snpe_runtime_type = SNPE_RUNTIME_DSP;
    }
    else
    {
        snpe_runtime_type = SNPE_RUNTIME_UNSET;
    }

    return snpe_runtime_type;
}

AURA_INLINE Snpe_PerformanceProfile_t GetSnpePerfLevel(NNPerfLevel perf_level)
{
    Snpe_PerformanceProfile_t snpe_perf_level;

    if (NNPerfLevel::PERF_LOW == perf_level)
    {
        snpe_perf_level = SNPE_PERFORMANCE_PROFILE_POWER_SAVER;
    }
    else if (NNPerfLevel::PERF_NORMAL == perf_level)
    {
        snpe_perf_level = SNPE_PERFORMANCE_PROFILE_BALANCED;
    }
    else
    {
        snpe_perf_level = SNPE_PERFORMANCE_PROFILE_BURST;
    }

    return snpe_perf_level;
}

AURA_INLINE Snpe_ProfilingLevel_t GetSnpeProfilingLevel(NNProfilingLevel profiling_level)
{
    Snpe_ProfilingLevel_t snpe_profiling_level;

    if (NNProfilingLevel::PROFILING_OFF == profiling_level)
    {
        snpe_profiling_level = SNPE_PROFILING_LEVEL_OFF;
    }
    else if (NNProfilingLevel::PROFILING_BASIC == profiling_level)
    {
        snpe_profiling_level = SNPE_PROFILING_LEVEL_BASIC;
    }
    else if (NNProfilingLevel::PROFILING_DETAILED == profiling_level)
    {
        snpe_profiling_level = SNPE_PROFILING_LEVEL_DETAILED;
    }
    else
    {
        snpe_profiling_level = SNPE_PROFILING_LEVEL_OFF;
    }

    return snpe_profiling_level;
}

AURA_INLINE Snpe_LogLevel_t GetSnpeLogLevel(NNLogLevel log_level)
{
    Snpe_LogLevel_t snpe_log_level;

    if (NNLogLevel::LOG_DEBUG == log_level)
    {
        snpe_log_level = SNPE_LOG_LEVEL_VERBOSE;
    }
    else if (NNLogLevel::LOG_INFO == log_level)
    {
        snpe_log_level = SNPE_LOG_LEVEL_INFO;
    }
    else
    {
        snpe_log_level = SNPE_LOG_LEVEL_ERROR;
    }

    return snpe_log_level;
}

AURA_INLINE ElemType GetElemType(Snpe_UserBufferEncoding_ElementType_t elem_type)
{
    switch (elem_type)
    {
        case SNPE_USERBUFFERENCODING_ELEMENTTYPE_TF8:
        case SNPE_USERBUFFERENCODING_ELEMENTTYPE_UINT8:
        {
            return ElemType::U8;
        }

        case SNPE_USERBUFFERENCODING_ELEMENTTYPE_TF16:
        case SNPE_USERBUFFERENCODING_ELEMENTTYPE_UINT16:
        {
            return ElemType::U16;
        }

        case SNPE_USERBUFFERENCODING_ELEMENTTYPE_INT8:
        {
            return ElemType::S8;
        }

        case SNPE_USERBUFFERENCODING_ELEMENTTYPE_INT16:
        {
            return ElemType::S16;
        }

        case SNPE_USERBUFFERENCODING_ELEMENTTYPE_INT32:
        {
            return ElemType::S32;
        }

        case SNPE_USERBUFFERENCODING_ELEMENTTYPE_UINT32:
        {
            return ElemType::U32;
        }

        case SNPE_USERBUFFERENCODING_ELEMENTTYPE_FLOAT16:
        {
            return ElemType::F16;
        }

        case SNPE_USERBUFFERENCODING_ELEMENTTYPE_FLOAT:
        {
            return ElemType::F32;
        }

        default:
        {
            return ElemType::INVALID;
        }
    }
}

static const std::string g_snpe_lib_name = "libSNPE.so";

class SnpeLibrary : public NNLibrary
{
public:
    static SnpeLibrary& Get()
    {
        static SnpeLibrary snpe_library;
        return snpe_library;
    }

    Status Destroy() override
    {
        return UnLoad();
    }
public:
    AURA_API_DEF(Snpe_Util_GetLibraryVersion) = Snpe_DlVersion_Handle_t (*)();
    AURA_API_PTR(Snpe_Util_GetLibraryVersion);

    AURA_API_DEF(Snpe_DlVersion_Delete) = Snpe_ErrorCode_t (*)(Snpe_DlVersion_Handle_t);
    AURA_API_PTR(Snpe_DlVersion_Delete);

    AURA_API_DEF(Snpe_DlVersion_ToString) = const char* (*)(Snpe_DlVersion_Handle_t);
    AURA_API_PTR(Snpe_DlVersion_ToString);

    AURA_API_DEF(Snpe_DlContainer_OpenBuffer) = Snpe_DlContainer_Handle_t (*)(const uint8_t*, const size_t);
    AURA_API_PTR(Snpe_DlContainer_OpenBuffer);

    AURA_API_DEF(Snpe_DlContainer_Delete) = Snpe_ErrorCode_t (*)(Snpe_DlContainer_Handle_t);
    AURA_API_PTR(Snpe_DlContainer_Delete);

    AURA_API_DEF(Snpe_Util_IsRuntimeAvailableCheckOption) = int (*)(Snpe_Runtime_t, Snpe_RuntimeCheckOption_t);
    AURA_API_PTR(Snpe_Util_IsRuntimeAvailableCheckOption);

    AURA_API_DEF(Snpe_RuntimeList_Create) = Snpe_RuntimeList_Handle_t (*)();
    AURA_API_PTR(Snpe_RuntimeList_Create);

    AURA_API_DEF(Snpe_RuntimeList_Empty) = int (*)(Snpe_RuntimeList_Handle_t);
    AURA_API_PTR(Snpe_RuntimeList_Empty);

    AURA_API_DEF(Snpe_RuntimeList_Add) = Snpe_ErrorCode_t (*)(Snpe_RuntimeList_Handle_t, Snpe_Runtime_t);
    AURA_API_PTR(Snpe_RuntimeList_Add);

    AURA_API_DEF(Snpe_RuntimeList_Delete) = Snpe_ErrorCode_t (*)(Snpe_RuntimeList_Handle_t);
    AURA_API_PTR(Snpe_RuntimeList_Delete);

    AURA_API_DEF(Snpe_SNPEBuilder_Create) = Snpe_SNPEBuilder_Handle_t (*)(Snpe_DlContainer_Handle_t);
    AURA_API_PTR(Snpe_SNPEBuilder_Create);

    AURA_API_DEF(Snpe_SNPEBuilder_SetOutputLayers) = Snpe_ErrorCode_t (*)(Snpe_SNPEBuilder_Handle_t, Snpe_StringList_Handle_t);
    AURA_API_PTR(Snpe_SNPEBuilder_SetOutputLayers);

    AURA_API_DEF(Snpe_SNPEBuilder_SetRuntimeProcessorOrder) = Snpe_ErrorCode_t (*)(Snpe_SNPEBuilder_Handle_t, Snpe_RuntimeList_Handle_t);
    AURA_API_PTR(Snpe_SNPEBuilder_SetRuntimeProcessorOrder);

    AURA_API_DEF(Snpe_SNPEBuilder_SetUseUserSuppliedBuffers) = Snpe_ErrorCode_t (*)(Snpe_SNPEBuilder_Handle_t, int);
    AURA_API_PTR(Snpe_SNPEBuilder_SetUseUserSuppliedBuffers);

    AURA_API_DEF(Snpe_SNPEBuilder_SetInitCacheMode) = Snpe_ErrorCode_t (*)(Snpe_SNPEBuilder_Handle_t, int);
    AURA_API_PTR(Snpe_SNPEBuilder_SetInitCacheMode);

    AURA_API_DEF(Snpe_SNPEBuilder_Build) = Snpe_SNPE_Handle_t (*)(Snpe_SNPEBuilder_Handle_t);
    AURA_API_PTR(Snpe_SNPEBuilder_Build);

    AURA_API_DEF(Snpe_SNPEBuilder_SetPlatformConfig) = Snpe_ErrorCode_t (*)(Snpe_SNPEBuilder_Handle_t, Snpe_PlatformConfig_Handle_t);
    AURA_API_PTR(Snpe_SNPEBuilder_SetPlatformConfig);

#if !defined(SNPE_EXECUTOR_IMPL_V2133)
    AURA_API_DEF(Snpe_SNPEBuilder_SetModelName) = Snpe_ErrorCode_t (*)(Snpe_SNPEBuilder_Handle_t, const char *);
    AURA_API_PTR(Snpe_SNPEBuilder_SetModelName);
#endif

    AURA_API_DEF(Snpe_SNPEBuilder_Delete) = Snpe_ErrorCode_t (*)(Snpe_SNPEBuilder_Handle_t);
    AURA_API_PTR(Snpe_SNPEBuilder_Delete);

    AURA_API_DEF(Snpe_SNPEBuilder_SetPerformanceProfile) = Snpe_ErrorCode_t (*)(Snpe_SNPEBuilder_Handle_t, Snpe_PerformanceProfile_t);
    AURA_API_PTR(Snpe_SNPEBuilder_SetPerformanceProfile);

    AURA_API_DEF(Snpe_SNPEBuilder_SetProfilingLevel) = Snpe_ErrorCode_t (*)(Snpe_SNPEBuilder_Handle_t, Snpe_ProfilingLevel_t);
    AURA_API_PTR(Snpe_SNPEBuilder_SetProfilingLevel);

    AURA_API_DEF(Snpe_SNPE_Delete) = Snpe_ErrorCode_t (*)(Snpe_SNPE_Handle_t);
    AURA_API_PTR(Snpe_SNPE_Delete);

    AURA_API_DEF(Snpe_SNPE_ExecuteUserBuffers) = Snpe_ErrorCode_t (*)(Snpe_SNPE_Handle_t, Snpe_UserBufferMap_Handle_t, Snpe_UserBufferMap_Handle_t);
    AURA_API_PTR(Snpe_SNPE_ExecuteUserBuffers);

    AURA_API_DEF(Snpe_SNPE_GetDiagLogInterface_Ref) = Snpe_IDiagLog_Handle_t (*)(Snpe_SNPE_Handle_t);
    AURA_API_PTR(Snpe_SNPE_GetDiagLogInterface_Ref);

    AURA_API_DEF(Snpe_SNPE_GetInputTensorNames) = Snpe_StringList_Handle_t (*)(Snpe_SNPE_Handle_t);
    AURA_API_PTR(Snpe_SNPE_GetInputTensorNames);

    AURA_API_DEF(Snpe_SNPE_GetOutputTensorNames) = Snpe_StringList_Handle_t (*)(Snpe_SNPE_Handle_t);
    AURA_API_PTR(Snpe_SNPE_GetOutputTensorNames);

    AURA_API_DEF(Snpe_SNPE_GetInputOutputBufferAttributes) = Snpe_IBufferAttributes_Handle_t (*)(Snpe_SNPE_Handle_t, const char *);
    AURA_API_PTR(Snpe_SNPE_GetInputOutputBufferAttributes);

    AURA_API_DEF(Snpe_PlatformConfig_Create) = Snpe_PlatformConfig_Handle_t (*)();
    AURA_API_PTR(Snpe_PlatformConfig_Create);

    AURA_API_DEF(Snpe_PlatformConfig_SetPlatformOptions) = int (*)(Snpe_PlatformConfig_Handle_t, const char* options);
    AURA_API_PTR(Snpe_PlatformConfig_SetPlatformOptions);

    AURA_API_DEF(Snpe_PlatformConfig_Delete) = Snpe_ErrorCode_t (*)(Snpe_PlatformConfig_Handle_t);
    AURA_API_PTR(Snpe_PlatformConfig_Delete);

    AURA_API_DEF(Snpe_IDiagLog_GetOptions) = Snpe_Options_Handle_t (*)(Snpe_IDiagLog_Handle_t);
    AURA_API_PTR(Snpe_IDiagLog_GetOptions);

    AURA_API_DEF(Snpe_IDiagLog_SetOptions) = Snpe_ErrorCode_t (*)(Snpe_IDiagLog_Handle_t, Snpe_Options_Handle_t);
    AURA_API_PTR(Snpe_IDiagLog_SetOptions);

    AURA_API_DEF(Snpe_IDiagLog_Start) = Snpe_ErrorCode_t (*)(Snpe_IDiagLog_Handle_t);
    AURA_API_PTR(Snpe_IDiagLog_Start);

    AURA_API_DEF(Snpe_IDiagLog_Stop) = Snpe_ErrorCode_t (*)(Snpe_IDiagLog_Handle_t);
    AURA_API_PTR(Snpe_IDiagLog_Stop);

    AURA_API_DEF(Snpe_Options_SetLogFileDirectory) = void (*)(Snpe_Options_Handle_t, const char*);
    AURA_API_PTR(Snpe_Options_SetLogFileDirectory);

    AURA_API_DEF(Snpe_Options_Delete) = Snpe_ErrorCode_t (*)(Snpe_Options_Handle_t);
    AURA_API_PTR(Snpe_Options_Delete);

    AURA_API_DEF(Snpe_UserBufferMap_Create) = Snpe_UserBufferMap_Handle_t (*)();
    AURA_API_PTR(Snpe_UserBufferMap_Create);

    AURA_API_DEF(Snpe_UserBufferMap_Delete) = Snpe_ErrorCode_t (*)(Snpe_UserBufferMap_Handle_t);
    AURA_API_PTR(Snpe_UserBufferMap_Delete);

    AURA_API_DEF(Snpe_StringList_Size) = size_t (*)(Snpe_StringList_Handle_t);
    AURA_API_PTR(Snpe_StringList_Size);

    AURA_API_DEF(Snpe_StringList_At) = const char* (*)(Snpe_StringList_Handle_t, size_t);
    AURA_API_PTR(Snpe_StringList_At);

    AURA_API_DEF(Snpe_StringList_Delete) = Snpe_ErrorCode_t (*)(Snpe_StringList_Handle_t);
    AURA_API_PTR(Snpe_StringList_Delete);

    AURA_API_DEF(Snpe_IBufferAttributes_GetDims) = Snpe_TensorShape_Handle_t (*)(Snpe_IBufferAttributes_Handle_t);
    AURA_API_PTR(Snpe_IBufferAttributes_GetDims);

    AURA_API_DEF(Snpe_TensorShape_Delete) = Snpe_ErrorCode_t (*)(Snpe_TensorShape_Handle_t);
    AURA_API_PTR(Snpe_TensorShape_Delete);

    AURA_API_DEF(Snpe_TensorShape_Rank) = size_t (*)(Snpe_TensorShape_Handle_t);
    AURA_API_PTR(Snpe_TensorShape_Rank);

    AURA_API_DEF(Snpe_UserBufferEncodingFloatN_Create) = Snpe_UserBufferEncoding_Handle_t (*)(uint8_t);
    AURA_API_PTR(Snpe_UserBufferEncodingFloatN_Create);

    AURA_API_DEF(Snpe_UserBufferEncodingFloatN_Delete) = Snpe_ErrorCode_t (*)(Snpe_UserBufferEncoding_Handle_t);
    AURA_API_PTR(Snpe_UserBufferEncodingFloatN_Delete);

    AURA_API_DEF(Snpe_UserBufferEncodingTfN_Create) = Snpe_UserBufferEncoding_Handle_t (*)(uint64_t, float, uint8_t);
    AURA_API_PTR(Snpe_UserBufferEncodingTfN_Create);

    AURA_API_DEF(Snpe_UserBufferEncodingTfN_Delete) = Snpe_ErrorCode_t (*)(Snpe_UserBufferEncoding_Handle_t);
    AURA_API_PTR(Snpe_UserBufferEncodingTfN_Delete);

    AURA_API_DEF(Snpe_TensorShape_At) = size_t (*)(Snpe_TensorShape_Handle_t, size_t);
    AURA_API_PTR(Snpe_TensorShape_At);

    AURA_API_DEF(Snpe_TensorShape_CreateDimsSize) = Snpe_TensorShape_Handle_t (*)(const size_t *, size_t);
    AURA_API_PTR(Snpe_TensorShape_CreateDimsSize);

    AURA_API_DEF(Snpe_UserBufferEncodingTfN_GetStepExactly0) = uint64_t (*)(Snpe_UserBufferEncoding_Handle_t);
    AURA_API_PTR(Snpe_UserBufferEncodingTfN_GetStepExactly0);

    AURA_API_DEF(Snpe_UserBufferEncodingTfN_GetQuantizedStepSize) = float (*)(Snpe_UserBufferEncoding_Handle_t);
    AURA_API_PTR(Snpe_UserBufferEncodingTfN_GetQuantizedStepSize);

    AURA_API_DEF(Snpe_IBufferAttributes_GetEncoding_Ref) = Snpe_UserBufferEncoding_Handle_t (*)(Snpe_IBufferAttributes_Handle_t);
    AURA_API_PTR(Snpe_IBufferAttributes_GetEncoding_Ref);

    AURA_API_DEF(Snpe_IBufferAttributes_Delete) = Snpe_ErrorCode_t (*)(Snpe_IBufferAttributes_Handle_t);
    AURA_API_PTR(Snpe_IBufferAttributes_Delete);

    AURA_API_DEF(Snpe_Util_InitializeLogging) = int (*)(Snpe_LogLevel_t);
    AURA_API_PTR(Snpe_Util_InitializeLogging);

    AURA_API_DEF(Snpe_Util_InitializeLoggingPath) = int (*)(Snpe_LogLevel_t, const char*);
    AURA_API_PTR(Snpe_Util_InitializeLoggingPath);

    AURA_API_DEF(Snpe_Util_SetLogLevel) = int (*)(Snpe_LogLevel_t);
    AURA_API_PTR(Snpe_Util_SetLogLevel);

    AURA_API_DEF(Snpe_Util_TerminateLogging) = int (*)();
    AURA_API_PTR(Snpe_Util_TerminateLogging);

    AURA_API_DEF(Snpe_Util_GetLastError) = const char* (*)();
    AURA_API_PTR(Snpe_Util_GetLastError);

    AURA_API_DEF(Snpe_Util_CreateUserBuffer) = Snpe_IUserBuffer_Handle_t (*)(void*, size_t, Snpe_TensorShape_Handle_t, Snpe_IUserBuffer_Handle_t);
    AURA_API_PTR(Snpe_Util_CreateUserBuffer);

    AURA_API_DEF(Snpe_IUserBuffer_Delete) = Snpe_ErrorCode_t (*)(Snpe_IUserBuffer_Handle_t);
    AURA_API_PTR(Snpe_IUserBuffer_Delete);

    AURA_API_DEF(Snpe_UserBufferMap_Add) = void (*)(Snpe_UserBufferMap_Handle_t, const char*, Snpe_IUserBuffer_Handle_t);
    AURA_API_PTR(Snpe_UserBufferMap_Add);

    AURA_API_DEF(Snpe_IBufferAttributes_GetEncodingType) = Snpe_UserBufferEncoding_ElementType_t (*)(Snpe_IBufferAttributes_Handle_t);
    AURA_API_PTR(Snpe_IBufferAttributes_GetEncodingType);

    AURA_API_DEF(Snpe_StringList_Create) = Snpe_StringList_Handle_t (*)();
    AURA_API_PTR(Snpe_StringList_Create);

    AURA_API_DEF(Snpe_StringList_Append) = Snpe_ErrorCode_t (*)(Snpe_StringList_Handle_t, const char*);
    AURA_API_PTR(Snpe_StringList_Append);

private:
    SnpeLibrary() : NNLibrary(), m_handle(MI_NULL)
    {
        Load();
    }

    ~SnpeLibrary()
    {
        Destroy();
    }

    AURA_DISABLE_COPY_AND_ASSIGN(SnpeLibrary);

    Status Load() override
    {
        Status ret = Status::ERROR;

        dlerror();
        m_handle = dlopen(g_snpe_lib_name.c_str(), RTLD_LAZY | RTLD_LOCAL);
        if (MI_NULL == m_handle)
        {
            std::string info = "dlopen libSNPE.so failed, err : " + std::string(dlerror());
            AURA_PRINTE(AURA_TAG, "%s\n", info.c_str());
            return ret;
        }

        do
        {
            AURA_DLSYM_API(m_handle, Snpe_Util_GetLibraryVersion);
            AURA_DLSYM_API(m_handle, Snpe_DlVersion_Delete);
            AURA_DLSYM_API(m_handle, Snpe_DlVersion_ToString);
            AURA_DLSYM_API(m_handle, Snpe_DlContainer_OpenBuffer);
            AURA_DLSYM_API(m_handle, Snpe_DlContainer_Delete);
            AURA_DLSYM_API(m_handle, Snpe_Util_IsRuntimeAvailableCheckOption);
            AURA_DLSYM_API(m_handle, Snpe_RuntimeList_Create);
            AURA_DLSYM_API(m_handle, Snpe_RuntimeList_Empty);
            AURA_DLSYM_API(m_handle, Snpe_RuntimeList_Add);
            AURA_DLSYM_API(m_handle, Snpe_RuntimeList_Delete);
            AURA_DLSYM_API(m_handle, Snpe_SNPEBuilder_Create);
            AURA_DLSYM_API(m_handle, Snpe_SNPEBuilder_SetOutputLayers);
            AURA_DLSYM_API(m_handle, Snpe_SNPEBuilder_SetRuntimeProcessorOrder);
            AURA_DLSYM_API(m_handle, Snpe_SNPEBuilder_SetUseUserSuppliedBuffers);
            AURA_DLSYM_API(m_handle, Snpe_SNPEBuilder_SetInitCacheMode);
            AURA_DLSYM_API(m_handle, Snpe_SNPEBuilder_Build);
            AURA_DLSYM_API(m_handle, Snpe_SNPEBuilder_SetPlatformConfig);
#if !defined(SNPE_EXECUTOR_IMPL_V2133)
            AURA_DLSYM_API(m_handle, Snpe_SNPEBuilder_SetModelName);
#endif
            AURA_DLSYM_API(m_handle, Snpe_SNPEBuilder_Delete);
            AURA_DLSYM_API(m_handle, Snpe_SNPEBuilder_SetPerformanceProfile);
            AURA_DLSYM_API(m_handle, Snpe_SNPEBuilder_SetProfilingLevel);
            AURA_DLSYM_API(m_handle, Snpe_SNPE_Delete);
            AURA_DLSYM_API(m_handle, Snpe_SNPE_ExecuteUserBuffers);
            AURA_DLSYM_API(m_handle, Snpe_SNPE_GetDiagLogInterface_Ref);
            AURA_DLSYM_API(m_handle, Snpe_SNPE_GetInputTensorNames);
            AURA_DLSYM_API(m_handle, Snpe_SNPE_GetOutputTensorNames);
            AURA_DLSYM_API(m_handle, Snpe_SNPE_GetInputOutputBufferAttributes);
            AURA_DLSYM_API(m_handle, Snpe_PlatformConfig_Create);
            AURA_DLSYM_API(m_handle, Snpe_PlatformConfig_SetPlatformOptions);
            AURA_DLSYM_API(m_handle, Snpe_PlatformConfig_Delete);
            AURA_DLSYM_API(m_handle, Snpe_IDiagLog_GetOptions);
            AURA_DLSYM_API(m_handle, Snpe_IDiagLog_SetOptions);
            AURA_DLSYM_API(m_handle, Snpe_IDiagLog_Start);
            AURA_DLSYM_API(m_handle, Snpe_IDiagLog_Stop);
            AURA_DLSYM_API(m_handle, Snpe_Options_SetLogFileDirectory);
            AURA_DLSYM_API(m_handle, Snpe_Options_Delete);
            AURA_DLSYM_API(m_handle, Snpe_UserBufferMap_Create);
            AURA_DLSYM_API(m_handle, Snpe_UserBufferMap_Delete);
            AURA_DLSYM_API(m_handle, Snpe_StringList_Size);
            AURA_DLSYM_API(m_handle, Snpe_StringList_At);
            AURA_DLSYM_API(m_handle, Snpe_StringList_Delete);
            AURA_DLSYM_API(m_handle, Snpe_IBufferAttributes_GetDims);
            AURA_DLSYM_API(m_handle, Snpe_TensorShape_Delete);
            AURA_DLSYM_API(m_handle, Snpe_TensorShape_Rank);
            AURA_DLSYM_API(m_handle, Snpe_UserBufferEncodingFloatN_Create);
            AURA_DLSYM_API(m_handle, Snpe_UserBufferEncodingFloatN_Delete);
            AURA_DLSYM_API(m_handle, Snpe_UserBufferEncodingTfN_Create);
            AURA_DLSYM_API(m_handle, Snpe_UserBufferEncodingTfN_Delete);
            AURA_DLSYM_API(m_handle, Snpe_TensorShape_At);
            AURA_DLSYM_API(m_handle, Snpe_TensorShape_CreateDimsSize);
            AURA_DLSYM_API(m_handle, Snpe_UserBufferEncodingTfN_GetStepExactly0);
            AURA_DLSYM_API(m_handle, Snpe_UserBufferEncodingTfN_GetQuantizedStepSize);
            AURA_DLSYM_API(m_handle, Snpe_IBufferAttributes_GetEncoding_Ref);
            AURA_DLSYM_API(m_handle, Snpe_IBufferAttributes_Delete);
            AURA_DLSYM_API(m_handle, Snpe_Util_InitializeLogging);
            AURA_DLSYM_API(m_handle, Snpe_Util_InitializeLoggingPath);
            AURA_DLSYM_API(m_handle, Snpe_Util_SetLogLevel);
            AURA_DLSYM_API(m_handle, Snpe_Util_TerminateLogging);
            AURA_DLSYM_API(m_handle, Snpe_Util_GetLastError);
            AURA_DLSYM_API(m_handle, Snpe_Util_CreateUserBuffer);
            AURA_DLSYM_API(m_handle, Snpe_IUserBuffer_Delete);
            AURA_DLSYM_API(m_handle, Snpe_UserBufferMap_Add);
            AURA_DLSYM_API(m_handle, Snpe_IBufferAttributes_GetEncodingType);
            AURA_DLSYM_API(m_handle, Snpe_StringList_Create);
            AURA_DLSYM_API(m_handle, Snpe_StringList_Append);

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

static Snpe_DlVersion_Handle_t Snpe_Util_GetLibraryVersion(Context *ctx)
{
    auto func = SnpeLibrary::Get().Snpe_Util_GetLibraryVersion;
    if (func)
    {
        return func();
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "Snpe_Util_GetLibraryVersion is null ptr");
        return MI_NULL;
    }
}

static Snpe_ErrorCode_t Snpe_DlVersion_Delete(Context *ctx, Snpe_DlVersion_Handle_t handle)
{
    auto func = SnpeLibrary::Get().Snpe_DlVersion_Delete;
    if (func)
    {
        return func(handle);
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "Snpe_DlVersion_Delete is null ptr");
        return SNPE_ERRORCODE_CAPI_CREATE_FAILURE;
    }
}

static const char *Snpe_DlVersion_ToString(Context *ctx, Snpe_DlVersion_Handle_t handle)
{
    auto func = SnpeLibrary::Get().Snpe_DlVersion_ToString;
    if (func)
    {
        return func(handle);
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "Snpe_DlVersion_ToString is null ptr");
        return MI_NULL;
    }
}

static Snpe_DlContainer_Handle_t Snpe_DlContainer_OpenBuffer(Context *ctx, const MI_UCHAR *buffer, const size_t size)
{
    auto func = SnpeLibrary::Get().Snpe_DlContainer_OpenBuffer;
    if (func)
    {
        return func(buffer, size);
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "Snpe_DlContainer_OpenBuffer is null ptr");
        return MI_NULL;
    }
}

static Snpe_ErrorCode_t Snpe_DlContainer_Delete(Context *ctx, Snpe_DlContainer_Handle_t handle)
{
    auto func = SnpeLibrary::Get().Snpe_DlContainer_Delete;
    if (func)
    {
        return func(handle);
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "Snpe_DlContainer_Delete is null ptr");
        return SNPE_ERRORCODE_CAPI_CREATE_FAILURE;
    }
}

static int Snpe_Util_IsRuntimeAvailableCheckOption(Context *ctx, Snpe_Runtime_t runtime, Snpe_RuntimeCheckOption_t runtime_check_option)
{
    auto func = SnpeLibrary::Get().Snpe_Util_IsRuntimeAvailableCheckOption;
    if (func)
    {
        return func(runtime, runtime_check_option);
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "Snpe_Util_IsRuntimeAvailableCheckOption is null ptr");
        return 0;
    }
}

static Snpe_RuntimeList_Handle_t Snpe_RuntimeList_Create(Context *ctx)
{
    auto func = SnpeLibrary::Get().Snpe_RuntimeList_Create;
    if (func)
    {
        return func();
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "Snpe_RuntimeList_Create is null ptr");
        return MI_NULL;
    }
}

static Snpe_ErrorCode_t Snpe_RuntimeList_Add(Context *ctx, Snpe_RuntimeList_Handle_t handle, Snpe_Runtime_t runtime)
{
    auto func = SnpeLibrary::Get().Snpe_RuntimeList_Add;
    if (func)
    {
        return func(handle, runtime);
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "Snpe_RuntimeList_Add is null ptr");
        return SNPE_ERRORCODE_CAPI_CREATE_FAILURE;
    }
}

static Snpe_ErrorCode_t Snpe_RuntimeList_Delete(Context *ctx, Snpe_RuntimeList_Handle_t handle)
{
    auto func = SnpeLibrary::Get().Snpe_RuntimeList_Delete;
    if (func)
    {
        return func(handle);
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "Snpe_RuntimeList_Delete is null ptr");
        return SNPE_ERRORCODE_CAPI_CREATE_FAILURE;
    }
}

static Snpe_SNPEBuilder_Handle_t Snpe_SNPEBuilder_Create(Context *ctx, Snpe_DlContainer_Handle_t handle)
{
    auto func = SnpeLibrary::Get().Snpe_SNPEBuilder_Create;
    if (func)
    {
        return func(handle);
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "Snpe_SNPEBuilder_Create is null ptr");
        return MI_NULL;
    }
}

static Snpe_ErrorCode_t Snpe_SNPEBuilder_SetOutputLayers(Context *ctx, Snpe_SNPEBuilder_Handle_t handle, Snpe_StringList_Handle_t output_layer_names)
{
    auto func = SnpeLibrary::Get().Snpe_SNPEBuilder_SetOutputLayers;
    if (func)
    {
        return func(handle, output_layer_names);
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "Snpe_SNPEBuilder_SetOutputLayers is null ptr");
        return SNPE_ERRORCODE_CAPI_CREATE_FAILURE;
    }
}

static Snpe_ErrorCode_t Snpe_SNPEBuilder_SetRuntimeProcessorOrder(Context *ctx, Snpe_SNPEBuilder_Handle_t handle, Snpe_RuntimeList_Handle_t runtime_list_handle)
{
    auto func = SnpeLibrary::Get().Snpe_SNPEBuilder_SetRuntimeProcessorOrder;
    if (func)
    {
        return func(handle, runtime_list_handle);
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "Snpe_SNPEBuilder_SetRuntimeProcessorOrder is null ptr");
        return SNPE_ERRORCODE_CAPI_CREATE_FAILURE;
    }
}

static Snpe_ErrorCode_t Snpe_SNPEBuilder_SetUseUserSuppliedBuffers(Context *ctx, Snpe_SNPEBuilder_Handle_t handle, int buffer_mode)
{
    auto func = SnpeLibrary::Get().Snpe_SNPEBuilder_SetUseUserSuppliedBuffers;
    if (func)
    {
        return func(handle, buffer_mode);
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "Snpe_SNPEBuilder_SetUseUserSuppliedBuffers is null ptr");
        return SNPE_ERRORCODE_CAPI_CREATE_FAILURE;
    }
}

static Snpe_ErrorCode_t Snpe_SNPEBuilder_SetInitCacheMode(Context *ctx, Snpe_SNPEBuilder_Handle_t handle, int cache_mode)
{
    auto func = SnpeLibrary::Get().Snpe_SNPEBuilder_SetInitCacheMode;
    if (func)
    {
        return func(handle, cache_mode);
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "Snpe_SNPEBuilder_SetInitCacheMode is null ptr");
        return SNPE_ERRORCODE_CAPI_CREATE_FAILURE;
    }
}

static Snpe_SNPE_Handle_t Snpe_SNPEBuilder_Build(Context *ctx, Snpe_SNPEBuilder_Handle_t handle)
{
    auto func = SnpeLibrary::Get().Snpe_SNPEBuilder_Build;
    if (func)
    {
        return func(handle);
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "Snpe_SNPEBuilder_Build is null ptr");
        return MI_NULL;
    }
}

static Snpe_ErrorCode_t Snpe_SNPEBuilder_SetPlatformConfig(Context *ctx, Snpe_SNPEBuilder_Handle_t handle, Snpe_PlatformConfig_Handle_t config)
{
    auto func = SnpeLibrary::Get().Snpe_SNPEBuilder_SetPlatformConfig;
    if (func)
    {
        return func(handle, config);
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "Snpe_SNPEBuilder_SetPlatformConfig is null ptr");
        return SNPE_ERRORCODE_CAPI_CREATE_FAILURE;
    }
}

#if !defined(SNPE_EXECUTOR_IMPL_V2133)
static Snpe_ErrorCode_t Snpe_SNPEBuilder_SetModelName(Context *ctx, Snpe_SNPEBuilder_Handle_t handle, const char *model_name)
{
    auto func = SnpeLibrary::Get().Snpe_SNPEBuilder_SetModelName;
    if (func)
    {
        return func(handle, model_name);
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "Snpe_SNPEBuilder_SetModelName is null ptr");
        return SNPE_ERRORCODE_CAPI_CREATE_FAILURE;
    }
}
#endif

static Snpe_ErrorCode_t Snpe_SNPEBuilder_Delete(Context *ctx, Snpe_SNPEBuilder_Handle_t handle)
{
    auto func = SnpeLibrary::Get().Snpe_SNPEBuilder_Delete;
    if (func)
    {
        return func(handle);
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "Snpe_SNPEBuilder_Delete is null ptr");
        return SNPE_ERRORCODE_CAPI_CREATE_FAILURE;
    }
}

static Snpe_ErrorCode_t Snpe_SNPEBuilder_SetPerformanceProfile(Context *ctx, Snpe_SNPEBuilder_Handle_t handle, Snpe_PerformanceProfile_t performance_profile)
{
    auto func = SnpeLibrary::Get().Snpe_SNPEBuilder_SetPerformanceProfile;
    if (func)
    {
        return func(handle, performance_profile);
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "Snpe_SNPEBuilder_SetPerformanceProfile is null ptr");
        return SNPE_ERRORCODE_CAPI_CREATE_FAILURE;
    }
}

static Snpe_ErrorCode_t Snpe_SNPEBuilder_SetProfilingLevel(Context *ctx, Snpe_SNPEBuilder_Handle_t handle, Snpe_ProfilingLevel_t profiling_level)
{
    auto func = SnpeLibrary::Get().Snpe_SNPEBuilder_SetProfilingLevel;
    if (func)
    {
        return func(handle, profiling_level);
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "Snpe_SNPEBuilder_SetProfilingLevel is null ptr");
        return SNPE_ERRORCODE_CAPI_CREATE_FAILURE;
    }
}

static Snpe_ErrorCode_t Snpe_SNPE_Delete(Context *ctx, Snpe_SNPE_Handle_t handle)
{
    auto func = SnpeLibrary::Get().Snpe_SNPE_Delete;
    if (func)
    {
        return func(handle);
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "Snpe_SNPE_Delete is null ptr");
        return SNPE_ERRORCODE_CAPI_CREATE_FAILURE;
    }
}

static Snpe_ErrorCode_t Snpe_SNPE_ExecuteUserBuffers(Context *ctx, Snpe_SNPE_Handle_t handle, Snpe_UserBufferMap_Handle_t input_handle, Snpe_UserBufferMap_Handle_t output_handle)
{
    auto func = SnpeLibrary::Get().Snpe_SNPE_ExecuteUserBuffers;
    if (func)
    {
        return func(handle, input_handle, output_handle);
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "Snpe_SNPE_ExecuteUserBuffers is null ptr");
        return SNPE_ERRORCODE_CAPI_CREATE_FAILURE;
    }
}

static Snpe_IDiagLog_Handle_t Snpe_SNPE_GetDiagLogInterface_Ref(Context *ctx, Snpe_SNPE_Handle_t handle)
{
    auto func = SnpeLibrary::Get().Snpe_SNPE_GetDiagLogInterface_Ref;
    if (func)
    {
        return func(handle);
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "Snpe_SNPE_GetDiagLogInterface_Ref is null ptr");
        return MI_NULL;
    }
}

static Snpe_StringList_Handle_t Snpe_SNPE_GetInputTensorNames(Context *ctx, Snpe_SNPE_Handle_t handle)
{
    auto func = SnpeLibrary::Get().Snpe_SNPE_GetInputTensorNames;
    if (func)
    {
        return func(handle);
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "Snpe_SNPE_GetInputTensorNames is null ptr");
        return MI_NULL;
    }
}

static Snpe_StringList_Handle_t Snpe_SNPE_GetOutputTensorNames(Context *ctx, Snpe_SNPE_Handle_t handle)
{
    auto func = SnpeLibrary::Get().Snpe_SNPE_GetOutputTensorNames;
    if (func)
    {
        return func(handle);
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "Snpe_SNPE_GetOutputTensorNames is null ptr");
        return MI_NULL;
    }
}

static Snpe_PlatformConfig_Handle_t Snpe_PlatformConfig_Create(Context *ctx)
{
    auto func = SnpeLibrary::Get().Snpe_PlatformConfig_Create;
    if (func)
    {
        return func();
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "Snpe_PlatformConfig_Create is null ptr");
        return MI_NULL;
    }
}

static int Snpe_PlatformConfig_SetPlatformOptions(Context *ctx, Snpe_PlatformConfig_Handle_t handle, const char *options)
{
    auto func = SnpeLibrary::Get().Snpe_PlatformConfig_SetPlatformOptions;
    if (func)
    {
        return func(handle, options);
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "Snpe_PlatformConfig_SetPlatformOptions is null ptr");
        return SNPE_ERRORCODE_CAPI_CREATE_FAILURE;
    }
}

static Snpe_ErrorCode_t Snpe_PlatformConfig_Delete(Context *ctx, Snpe_PlatformConfig_Handle_t handle)
{
    auto func = SnpeLibrary::Get().Snpe_PlatformConfig_Delete;
    if (func)
    {
        return func(handle);
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "Snpe_PlatformConfig_Delete is null ptr");
        return SNPE_ERRORCODE_CAPI_CREATE_FAILURE;
    }
}

static Snpe_IBufferAttributes_Handle_t Snpe_SNPE_GetInputOutputBufferAttributes(Context *ctx, Snpe_SNPE_Handle_t handle, const char *name)
{
    auto func = SnpeLibrary::Get().Snpe_SNPE_GetInputOutputBufferAttributes;
    if (func)
    {
        return func(handle, name);
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "Snpe_SNPE_GetInputOutputBufferAttributes is null ptr");
        return MI_NULL;
    }
}

static Snpe_Options_Handle_t Snpe_IDiagLog_GetOptions(Context *ctx, Snpe_IDiagLog_Handle_t handle)
{
    auto func = SnpeLibrary::Get().Snpe_IDiagLog_GetOptions;
    if (func)
    {
        return func(handle);
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "Snpe_IDiagLog_GetOptions is null ptr");
        return MI_NULL;
    }
}

static Snpe_ErrorCode_t Snpe_IDiagLog_SetOptions(Context *ctx, Snpe_IDiagLog_Handle_t handle, Snpe_Options_Handle_t logging_options_handle)
{
    auto func = SnpeLibrary::Get().Snpe_IDiagLog_SetOptions;
    if (func)
    {
        return func(handle, logging_options_handle);
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "Snpe_IDiagLog_SetOptions is null ptr");
        return SNPE_ERRORCODE_CAPI_CREATE_FAILURE;
    }
}

static Snpe_ErrorCode_t Snpe_IDiagLog_Start(Context *ctx, Snpe_IDiagLog_Handle_t handle)
{
    auto func = SnpeLibrary::Get().Snpe_IDiagLog_Start;
    if (func)
    {
        return func(handle);
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "Snpe_IDiagLog_Start is null ptr");
        return SNPE_ERRORCODE_CAPI_CREATE_FAILURE;
    }
}

static Snpe_ErrorCode_t Snpe_IDiagLog_Stop(Context *ctx, Snpe_IDiagLog_Handle_t handle)
{
    auto func = SnpeLibrary::Get().Snpe_IDiagLog_Stop;
    if (func)
    {
        return func(handle);
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "Snpe_IDiagLog_Stop is null ptr");
        return SNPE_ERRORCODE_CAPI_CREATE_FAILURE;
    }
}

static void Snpe_Options_SetLogFileDirectory(Context *ctx, Snpe_Options_Handle_t handle, const char *log_file_directory)
{
    auto func = SnpeLibrary::Get().Snpe_Options_SetLogFileDirectory;
    if (func)
    {
        func(handle, log_file_directory);
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "Snpe_Options_SetLogFileDirectory is null ptr");
    }
}

static Snpe_ErrorCode_t Snpe_Options_Delete(Context *ctx, Snpe_Options_Handle_t handle)
{
    auto func = SnpeLibrary::Get().Snpe_Options_Delete;
    if (func)
    {
        return func(handle);
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "Snpe_Options_Delete is null ptr");
        return SNPE_ERRORCODE_CAPI_CREATE_FAILURE;
    }
}

static Snpe_UserBufferMap_Handle_t Snpe_UserBufferMap_Create(Context *ctx)
{
    auto func = SnpeLibrary::Get().Snpe_UserBufferMap_Create;
    if (func)
    {
        return func();
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "Snpe_UserBufferMap_Create is null ptr");
        return MI_NULL;
    }
}

static Snpe_ErrorCode_t Snpe_UserBufferMap_Delete(Context *ctx, Snpe_UserBufferMap_Handle_t handle)
{
    auto func = SnpeLibrary::Get().Snpe_UserBufferMap_Delete;
    if (func)
    {
        return func(handle);
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "Snpe_UserBufferMap_Delete is null ptr");
        return SNPE_ERRORCODE_CAPI_CREATE_FAILURE;
    }
}

static size_t Snpe_StringList_Size(Context *ctx, Snpe_StringList_Handle_t handle)
{
    auto func = SnpeLibrary::Get().Snpe_StringList_Size;
    if (func)
    {
        return func(handle);
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "Snpe_StringList_Size is null ptr");
        return 0;
    }
}

static const char* Snpe_StringList_At(Context *ctx, Snpe_StringList_Handle_t handle, size_t idx)
{
    auto func = SnpeLibrary::Get().Snpe_StringList_At;
    if (func)
    {
        return func(handle, idx);
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "Snpe_StringList_At is null ptr");
        return MI_NULL;
    }
}

static Snpe_ErrorCode_t Snpe_StringList_Delete(Context *ctx, Snpe_StringList_Handle_t handle)
{
    auto func = SnpeLibrary::Get().Snpe_StringList_Delete;
    if (func)
    {
        return func(handle);
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "Snpe_StringList_Delete is null ptr");
        return SNPE_ERRORCODE_CAPI_CREATE_FAILURE;
    }
}

static Snpe_TensorShape_Handle_t Snpe_IBufferAttributes_GetDims(Context *ctx, Snpe_IBufferAttributes_Handle_t handle)
{
    auto func = SnpeLibrary::Get().Snpe_IBufferAttributes_GetDims;
    if (func)
    {
        return func(handle);
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "Snpe_IBufferAttributes_GetDims is null ptr");
        return MI_NULL;
    }
}

static Snpe_ErrorCode_t Snpe_TensorShape_Delete(Context *ctx, Snpe_TensorShape_Handle_t handle)
{
    auto func = SnpeLibrary::Get().Snpe_TensorShape_Delete;
    if (func)
    {
        return func(handle);
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "Snpe_TensorShape_Delete is null ptr");
        return SNPE_ERRORCODE_CAPI_CREATE_FAILURE;
    }
}

static size_t Snpe_TensorShape_Rank(Context *ctx, Snpe_TensorShape_Handle_t tensor_shape)
{
    auto func = SnpeLibrary::Get().Snpe_TensorShape_Rank;
    if (func)
    {
        return func(tensor_shape);
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "Snpe_TensorShape_Rank is null ptr");
        return 0;
    }
}

static Snpe_UserBufferEncoding_Handle_t Snpe_UserBufferEncodingFloatN_Create(Context *ctx, MI_U8 bit_width)
{
    auto func = SnpeLibrary::Get().Snpe_UserBufferEncodingFloatN_Create;
    if (func)
    {
        return func(bit_width);
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "Snpe_UserBufferEncodingFloatN_Create is null ptr");
        return MI_NULL;
    }
}

static Snpe_ErrorCode_t Snpe_UserBufferEncodingFloatN_Delete(Context *ctx, Snpe_UserBufferEncoding_Handle_t handle)
{
    auto func = SnpeLibrary::Get().Snpe_UserBufferEncodingFloatN_Delete;
    if (func)
    {
        return func(handle);
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "Snpe_UserBufferEncodingFloatN_Delete is null ptr");
        return SNPE_ERRORCODE_CAPI_CREATE_FAILURE;
    }
}

static Snpe_UserBufferEncoding_Handle_t Snpe_UserBufferEncodingTfN_Create(Context *ctx, uint64_t step_for0, float step_size, uint8_t b_width)
{
    auto func = SnpeLibrary::Get().Snpe_UserBufferEncodingTfN_Create;
    if (func)
    {
        return func(step_for0, step_size, b_width);
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "Snpe_UserBufferEncodingTfN_Create is null ptr");
        return MI_NULL;
    }
}

static Snpe_ErrorCode_t Snpe_UserBufferEncodingTfN_Delete(Context *ctx, Snpe_UserBufferEncoding_Handle_t handle)
{
    auto func = SnpeLibrary::Get().Snpe_UserBufferEncodingTfN_Delete;
    if (func)
    {
        return func(handle);
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "Snpe_UserBufferEncodingTfN_Delete is null ptr");
        return SNPE_ERRORCODE_CAPI_CREATE_FAILURE;
    }
}

static size_t Snpe_TensorShape_At(Context *ctx, Snpe_TensorShape_Handle_t handle, size_t index)
{
    auto func = SnpeLibrary::Get().Snpe_TensorShape_At;
    if (func)
    {
        return func(handle, index);
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "Snpe_TensorShape_At is null ptr");
        return SNPE_ERRORCODE_CAPI_CREATE_FAILURE;
    }
}

static Snpe_TensorShape_Handle_t Snpe_TensorShape_CreateDimsSize(Context *ctx, const size_t *dims, size_t size)
{
    auto func = SnpeLibrary::Get().Snpe_TensorShape_CreateDimsSize;
    if (func)
    {
        return func(dims, size);
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "Snpe_TensorShape_At is null ptr");
        return MI_NULL;
    }
}

static uint64_t Snpe_UserBufferEncodingTfN_GetStepExactly0(Context *ctx, Snpe_UserBufferEncoding_Handle_t handle)
{
    auto func = SnpeLibrary::Get().Snpe_UserBufferEncodingTfN_GetStepExactly0;
    if (func)
    {
        return func(handle);
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "Snpe_UserBufferEncodingTfN_GetStepExactly0 is null ptr");
        return 0;
    }
}

static float Snpe_UserBufferEncodingTfN_GetQuantizedStepSize(Context *ctx, Snpe_UserBufferEncoding_Handle_t handle)
{
    auto func = SnpeLibrary::Get().Snpe_UserBufferEncodingTfN_GetQuantizedStepSize;
    if (func)
    {
        return func(handle);
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "Snpe_UserBufferEncodingTfN_GetQuantizedStepSize is null ptr");
        return 0.f;
    }
}

static Snpe_UserBufferEncoding_Handle_t Snpe_IBufferAttributes_GetEncoding_Ref(Context *ctx, Snpe_IBufferAttributes_Handle_t handle)
{
    auto func = SnpeLibrary::Get().Snpe_IBufferAttributes_GetEncoding_Ref;
    if (func)
    {
        return func(handle);
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "Snpe_IBufferAttributes_GetEncoding_Ref is null ptr");
        return MI_NULL;
    }
}

static Snpe_ErrorCode_t Snpe_IBufferAttributes_Delete(Context *ctx, Snpe_IBufferAttributes_Handle_t handle)
{
    auto func = SnpeLibrary::Get().Snpe_IBufferAttributes_Delete;
    if (func)
    {
        return func(handle);
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "Snpe_IBufferAttributes_Delete is null ptr");
        return SNPE_ERRORCODE_CAPI_CREATE_FAILURE;
    }
}

static int Snpe_Util_InitializeLogging(Context *ctx, Snpe_LogLevel_t level)
{
    auto func = SnpeLibrary::Get().Snpe_Util_InitializeLogging;
    if (func)
    {
        return func(level);
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "Snpe_Util_InitializeLogging is null ptr");
        return 0;
    }
}

static Snpe_IUserBuffer_Handle_t Snpe_Util_CreateUserBuffer(Context *ctx, void *buffer, size_t buffer_size, Snpe_TensorShape_Handle_t strides_handle,
                                                            Snpe_IUserBuffer_Handle_t user_buffer_encoding_handle)
{
    auto func = SnpeLibrary::Get().Snpe_Util_CreateUserBuffer;
    if (func)
    {
        return func(buffer, buffer_size, strides_handle, user_buffer_encoding_handle);
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "Snpe_Util_CreateUserBuffer is null ptr");
        return MI_NULL;
    }
}

static Snpe_ErrorCode_t Snpe_IUserBuffer_Delete(Context *ctx, Snpe_IUserBuffer_Handle_t handle)
{
    auto func = SnpeLibrary::Get().Snpe_IUserBuffer_Delete;
    if (func)
    {
        return func(handle);
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "Snpe_IUserBuffer_Delete is null ptr");
        return SNPE_ERRORCODE_CAPI_CREATE_FAILURE;
    }
}

static void Snpe_UserBufferMap_Add(Context *ctx, Snpe_UserBufferMap_Handle_t handle, const char *name, Snpe_IUserBuffer_Handle_t buffer_handle)
{
    auto func = SnpeLibrary::Get().Snpe_UserBufferMap_Add;
    if (func)
    {
        func(handle, name, buffer_handle);
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "Snpe_UserBufferMap_Add is null ptr");
    }
}

static Snpe_UserBufferEncoding_ElementType_t Snpe_IBufferAttributes_GetEncodingType(Context *ctx, Snpe_IBufferAttributes_Handle_t handle)
{
    auto func = SnpeLibrary::Get().Snpe_IBufferAttributes_GetEncodingType;
    if (func)
    {
        return func(handle);
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "Snpe_IBufferAttributes_GetEncodingType is null ptr");
        return SNPE_USERBUFFERENCODING_ELEMENTTYPE_UNKNOWN;
    }
}

static Snpe_StringList_Handle_t Snpe_StringList_Create(Context *ctx)
{
    auto func = SnpeLibrary::Get().Snpe_StringList_Create;
    if (func)
    {
        return func();
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "Snpe_StringList_Create is null ptr");
        return MI_NULL;
    }
}

static Snpe_ErrorCode_t Snpe_StringList_Append(Context *ctx, Snpe_StringList_Handle_t handle, const char *string)
{
    auto func = SnpeLibrary::Get().Snpe_StringList_Append;
    if (func)
    {
        return func(handle, string);
    }
    else
    {
        AURA_ADD_ERROR_STRING(ctx, "Snpe_StringList_Append is null ptr");
        return SNPE_ERRORCODE_CAPI_CREATE_FAILURE;
    }
}

class SnpeUtils
{
public:
    static std::string GetLibraryVersion(Context *ctx)
    {
        if (MI_NULL == ctx)
        {
            return std::string();
        }

        Snpe_DlVersion_Handle_t handle = Snpe_Util_GetLibraryVersion(ctx);
        if (handle)
        {
            std::string version = Snpe_DlVersion_ToString(ctx, handle);
            Snpe_DlVersion_Delete(ctx, handle);
            return "v" + version;
        }
        else
        {
            AURA_ADD_ERROR_STRING(ctx, "Snpe_Util_GetLibraryVersion failed");
            return std::string();
        }
    }

    static Snpe_ErrorCode_t InitDiagLog(Context *ctx, const std::string &profilgin_path, Snpe_SNPE_Handle_t snpe_handle)
    {
        Snpe_ErrorCode_t ret = SNPE_ERRORCODE_CAPI_BAD_ARGUMENT;

        if (MI_NULL == ctx)
        {
            return ret;
        }

        Snpe_IDiagLog_Handle_t log_handle = Snpe_SNPE_GetDiagLogInterface_Ref(ctx, snpe_handle);
        if (MI_NULL == log_handle)
        {
            AURA_ADD_ERROR_STRING(ctx, "Snpe_SNPE_GetDiagLogInterface_Ref failed");
            return ret;
        }

        Snpe_Options_Handle_t options_handle = Snpe_IDiagLog_GetOptions(ctx, log_handle);
        if (MI_NULL == options_handle)
        {
            AURA_ADD_ERROR_STRING(ctx, "Snpe_IDiagLog_GetOptions failed");
            return ret;
        }

        if (!profilgin_path.empty())
        {
            Snpe_Options_SetLogFileDirectory(ctx, options_handle, profilgin_path.c_str());
        }
        else
        {
            Snpe_Options_SetLogFileDirectory(ctx, options_handle, m_log_path.c_str());
        }

        ret = Snpe_IDiagLog_SetOptions(ctx, log_handle, options_handle);
        if (ret != SNPE_SUCCESS)
        {
            std::string info = "Snpe_IDiagLog_SetOptions failed, err : " + std::to_string(ret);
            AURA_ADD_ERROR_STRING(ctx, info.c_str());
            goto EXIT;
        }

        Snpe_IDiagLog_Start(ctx, log_handle);
EXIT:
        Snpe_Options_Delete(ctx, options_handle);
        return ret;
    }

    static Status GetTensorAttributes(Context *ctx, Snpe_SNPE_Handle_t handle, const std::string &tensor_name, MI_U32 &dims, MI_U32 &batch,
                                      std::vector<size_t> &shape, Snpe_UserBufferEncoding_ElementType_t &elem_type, MI_F32 &scale, MI_U16 &zero_point)
    {
        Status ret = Status::ERROR;

        if (MI_NULL == ctx)
        {
            return ret;
        }

        if (MI_NULL == handle)
        {
            AURA_ADD_ERROR_STRING(ctx, "snpe handle is null");
            return ret;
        }

        Snpe_IBufferAttributes_Handle_t attributes_handle = MI_NULL;
        Snpe_TensorShape_Handle_t shape_handle = MI_NULL;
        Snpe_UserBufferEncoding_Handle_t encode_handle = MI_NULL;
        size_t stride = 0;

        attributes_handle = Snpe_SNPE_GetInputOutputBufferAttributes(ctx, handle, tensor_name.c_str());
        if (MI_NULL == attributes_handle)
        {
            AURA_ADD_ERROR_STRING(ctx, "Snpe_SNPE_GetInputOutputBufferAttributes failed");
            goto EXIT;
        }

        shape_handle = Snpe_IBufferAttributes_GetDims(ctx, attributes_handle);
        if (MI_NULL == shape_handle)
        {
            AURA_ADD_ERROR_STRING(ctx, "Snpe_IBufferAttributes_GetDims failed");
            goto EXIT;
        }

        encode_handle = Snpe_IBufferAttributes_GetEncoding_Ref(ctx, attributes_handle);
        if (MI_NULL == encode_handle)
        {
            AURA_ADD_ERROR_STRING(ctx, "Snpe_IBufferAttributes_GetEncoding_Ref failed");
            goto EXIT;
        }

        dims = Snpe_TensorShape_Rank(ctx, shape_handle);
        if (dims > 4)
        {
            AURA_ADD_ERROR_STRING(ctx, "Snpe_TensorShape_Rank failed, only suppose dim is 1/2/3/4");
            goto EXIT;
        }

        shape.resize(dims);

        elem_type = Snpe_IBufferAttributes_GetEncodingType(ctx, attributes_handle);
        if ((SNPE_USERBUFFERENCODING_ELEMENTTYPE_TF8 == elem_type) || (SNPE_USERBUFFERENCODING_ELEMENTTYPE_TF16 == elem_type))
        {
            shape[dims - 1] = ElemTypeSize(GetElemType(elem_type));
            scale = Snpe_UserBufferEncodingTfN_GetQuantizedStepSize(ctx, encode_handle);
            zero_point = Snpe_UserBufferEncodingTfN_GetStepExactly0(ctx, encode_handle);
        }
        else if ((SNPE_USERBUFFERENCODING_ELEMENTTYPE_FLOAT == elem_type) || (SNPE_USERBUFFERENCODING_ELEMENTTYPE_FLOAT16 == elem_type))
        {
            shape[dims - 1] = ElemTypeSize(GetElemType(elem_type));
            scale = 0.f;
            zero_point = 0;
        }
        else
        {
            AURA_ADD_ERROR_STRING(ctx, "Snpe_IBufferAttributes_GetEncodingType failed, elem_type only suppose tf8/tf16/float");
            goto EXIT;
        }

        stride = shape[dims - 1];
        for (MI_S32 i = dims - 1; i > 0 ; i--)
        {
            stride *= Snpe_TensorShape_At(ctx, shape_handle, i);
            shape[i - 1] = stride;
        }
        batch = Snpe_TensorShape_At(ctx, shape_handle, 0);

        ret = Status::OK;
EXIT:
        if (attributes_handle)
        {
            Snpe_ErrorCode_t snpe_ret = Snpe_IBufferAttributes_Delete(ctx, attributes_handle);
            if (snpe_ret != SNPE_SUCCESS)
            {
                std::string info = "Snpe_IBufferAttributes_Delete failed, error : " + std::to_string(snpe_ret);
                AURA_ADD_ERROR_STRING(ctx, info.c_str());
                ret = Status::ERROR;
            }
        }

        if (shape_handle)
        {
            Snpe_ErrorCode_t snpe_ret = Snpe_TensorShape_Delete(ctx, shape_handle);
            if (snpe_ret != SNPE_SUCCESS)
            {
                std::string info = "Snpe_TensorShape_Delete failed, error : " + std::to_string(snpe_ret);
                AURA_ADD_ERROR_STRING(ctx, info.c_str());
                ret = Status::ERROR;
            }
        }

        return ret;
    }

    static Status GetTensorAttributes(Context *ctx, Snpe_SNPE_Handle_t handle, const std::string &tensor_name, MI_F32 &scale, MI_U16 &zero_point)
    {
        Status ret = Status::ERROR;

        if (MI_NULL == ctx)
        {
            return ret;
        }

        if (MI_NULL == handle)
        {
            AURA_ADD_ERROR_STRING(ctx, "snpe handle is null");
            return ret;
        }

        Snpe_IBufferAttributes_Handle_t attributes_handle = MI_NULL;
        Snpe_UserBufferEncoding_Handle_t encode_handle = MI_NULL;
        Snpe_UserBufferEncoding_ElementType_t elem_type;

        attributes_handle = Snpe_SNPE_GetInputOutputBufferAttributes(ctx, handle, tensor_name.c_str());
        if (MI_NULL == attributes_handle)
        {
            AURA_ADD_ERROR_STRING(ctx, "Snpe_SNPE_GetInputOutputBufferAttributes failed");
            goto EXIT;
        }

        encode_handle = Snpe_IBufferAttributes_GetEncoding_Ref(ctx, attributes_handle);
        if (MI_NULL == encode_handle)
        {
            AURA_ADD_ERROR_STRING(ctx, "Snpe_IBufferAttributes_GetEncoding_Ref failed");
            goto EXIT;
        }

        elem_type = Snpe_IBufferAttributes_GetEncodingType(ctx, attributes_handle);
        if ((SNPE_USERBUFFERENCODING_ELEMENTTYPE_TF8 == elem_type) || (SNPE_USERBUFFERENCODING_ELEMENTTYPE_TF16 == elem_type))
        {
            scale = Snpe_UserBufferEncodingTfN_GetQuantizedStepSize(ctx, encode_handle);
            zero_point = Snpe_UserBufferEncodingTfN_GetStepExactly0(ctx, encode_handle);
        }
        else if ((SNPE_USERBUFFERENCODING_ELEMENTTYPE_FLOAT == elem_type) || (SNPE_USERBUFFERENCODING_ELEMENTTYPE_FLOAT16 == elem_type))
        {
            scale = 0.f;
            zero_point = 0;
        }
        else
        {
            AURA_ADD_ERROR_STRING(ctx, "Snpe_IBufferAttributes_GetEncodingType failed, elem_type only suppose tf8/tf16/float");
            goto EXIT;
        }

        ret = Status::OK;
EXIT:
        if (attributes_handle)
        {
            Snpe_ErrorCode_t snpe_ret = Snpe_IBufferAttributes_Delete(ctx, attributes_handle);
            if (snpe_ret != SNPE_SUCCESS)
            {
                std::string info = "Snpe_IBufferAttributes_Delete failed, error : " + std::to_string(snpe_ret);
                AURA_ADD_ERROR_STRING(ctx, info.c_str());
                ret = Status::ERROR;
            }
        }

        return ret;
    }

    static TensorDescMap GetTensorDesc(Context *ctx, Snpe_SNPE_Handle_t snpe_handle,
                                       Snpe_StringList_Handle_t list_handle)
    {
        Status status = Status::ERROR;

        TensorDescMap result;
        size_t list_size = Snpe_StringList_Size(ctx, list_handle);
        for (size_t i = 0; i < list_size; i++)
        {
            TensorDesc desc;
            MI_S32 rank;
            std::string tensor_name = std::string(Snpe_StringList_At(ctx, list_handle, i));
            Snpe_IBufferAttributes_Handle_t attributes_handle = MI_NULL;
            Snpe_TensorShape_Handle_t shape_handle = MI_NULL;
            Snpe_UserBufferEncoding_Handle_t encode_handle = MI_NULL;
            Snpe_UserBufferEncoding_ElementType_t elem_type;

            attributes_handle = Snpe_SNPE_GetInputOutputBufferAttributes(ctx, snpe_handle, tensor_name.c_str());
            if (MI_NULL == attributes_handle)
            {
                AURA_ADD_ERROR_STRING(ctx, "Snpe_SNPE_GetInputOutputBufferAttributes failed");
                goto EXIT;
            }

            shape_handle = Snpe_IBufferAttributes_GetDims(ctx, attributes_handle);
            if (MI_NULL == shape_handle)
            {
                AURA_ADD_ERROR_STRING(ctx, "Snpe_IBufferAttributes_GetDims failed");
                goto EXIT;
            }

            encode_handle = Snpe_IBufferAttributes_GetEncoding_Ref(ctx, attributes_handle);
            if (MI_NULL == encode_handle)
            {
                AURA_ADD_ERROR_STRING(ctx, "Snpe_IBufferAttributes_GetEncoding_Ref failed");
                goto EXIT;
            }

            rank = Snpe_TensorShape_Rank(ctx, shape_handle);
            for (MI_S32 j = 0; j < rank; j++)
            {
                MI_S32 dim = Snpe_TensorShape_At(ctx, shape_handle, j);
                desc.sizes.push_back(dim);
            }

            elem_type = Snpe_IBufferAttributes_GetEncodingType(ctx, attributes_handle);
            desc.elem_type = GetElemType(elem_type);

            if ((SNPE_USERBUFFERENCODING_ELEMENTTYPE_TF8 == elem_type) || (SNPE_USERBUFFERENCODING_ELEMENTTYPE_TF16 == elem_type))
            {
                desc.scale = Snpe_UserBufferEncodingTfN_GetQuantizedStepSize(ctx, encode_handle);
                desc.zero_point = Snpe_UserBufferEncodingTfN_GetStepExactly0(ctx, encode_handle);
            }
            else if ((SNPE_USERBUFFERENCODING_ELEMENTTYPE_FLOAT == elem_type) || (SNPE_USERBUFFERENCODING_ELEMENTTYPE_FLOAT16 == elem_type))
            {
                desc.scale = 0.f;
                desc.zero_point = 0;
            }
            else
            {
                AURA_ADD_ERROR_STRING(ctx, "Snpe_IBufferAttributes_GetEncodingType failed, elem_type only suppose tf8/tf16/float");
                goto EXIT;
            }

            status = Status::OK;
EXIT:
            if (attributes_handle)
            {
                Snpe_ErrorCode_t snpe_ret = Snpe_IBufferAttributes_Delete(ctx, attributes_handle);
                if (snpe_ret != SNPE_SUCCESS)
                {
                    std::string info = "Snpe_IBufferAttributes_Delete failed, error : " + std::to_string(snpe_ret);
                    AURA_ADD_ERROR_STRING(ctx, info.c_str());
                    status = Status::ERROR;
                }
            }

            if (shape_handle)
            {
                Snpe_ErrorCode_t snpe_ret = Snpe_TensorShape_Delete(ctx, shape_handle);
                if (snpe_ret != SNPE_SUCCESS)
                {
                    std::string info = "Snpe_TensorShape_Delete failed, error : " + std::to_string(snpe_ret);
                    AURA_ADD_ERROR_STRING(ctx, info.c_str());
                    status = Status::ERROR;
                }
            }

            if (status != Status::OK)
            {
                result.clear();
                break;
            }

            result[tensor_name] = desc;
        }

        Snpe_ErrorCode_t snpe_ret = Snpe_StringList_Delete(ctx, list_handle);
        if (snpe_ret != SNPE_SUCCESS)
        {
            std::string info = "Snpe_StringList_Delete failed, error : " + std::to_string(snpe_ret);
            AURA_ADD_ERROR_STRING(ctx, info.c_str());
            return TensorDescMap();
        }

        return result;
    }

private:
    static std::string m_log_path;
};

std::string SnpeUtils::m_log_path = "/data/vendor/camera";

class SnpeUserBufferMap
{
public:
    SnpeUserBufferMap(Context *ctx, Snpe_SNPE_Handle_t handle, MI_BOOL is_input)
                      : m_ctx(ctx), m_handle(handle), m_is_valid(MI_FALSE),
                        m_is_input(is_input), m_user_buffer_map(MI_NULL)
    {
        do
        {
            if (MI_NULL == m_ctx)
            {
                break;
            }

            m_user_buffer_map = Snpe_UserBufferMap_Create(m_ctx);
            if (MI_NULL == m_user_buffer_map)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "Snpe_UserBufferMap_Create failed");
                break;
            }

            Snpe_StringList_Handle_t tensor_list = MI_NULL;
            if (m_is_input)
            {
                tensor_list = Snpe_SNPE_GetInputTensorNames(m_ctx, m_handle);
            }
            else
            {
                tensor_list = Snpe_SNPE_GetOutputTensorNames(m_ctx, m_handle);
            }

            size_t list_size = Snpe_StringList_Size(m_ctx, tensor_list);
            if (list_size < 1)
            {
                Snpe_ErrorCode_t snpe_ret = Snpe_StringList_Delete(m_ctx, tensor_list);
                std::string info = "Snpe_StringList_Size list_size less 1, Snpe_StringList_Delete ret : " + std::to_string(snpe_ret);
                AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
                break;
            }

            for (size_t i = 0; i < list_size; i++)
            {
                std::string tensor_name = std::string(Snpe_StringList_At(m_ctx, tensor_list, i));
                m_tensor_name.push_back(tensor_name);
            }

            Snpe_ErrorCode_t snpe_ret = Snpe_StringList_Delete(m_ctx, tensor_list);
            if (snpe_ret != SNPE_SUCCESS)
            {
                std::string info = "Snpe_StringList_Delete failed, error : " + std::to_string(snpe_ret);
                AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
                break;
            }

            m_is_valid = MI_TRUE;

        } while (0);
    }

    ~SnpeUserBufferMap()
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

        if ((MI_NULL == mat_map) || (mat_map->size() != m_tensor_name.size()))
        {
            AURA_ADD_ERROR_STRING(m_ctx, "size match error");
            return Status::ERROR;
        }

        m_mat_map = *mat_map;

        for (size_t i = 0; i < m_tensor_name.size(); i++)
        {
            std::string tensor_name = m_tensor_name[i];
            if (m_mat_map.find(tensor_name) == m_mat_map.end())
            {
                std::string info = "mat names " + tensor_name + " not provided";
                AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
                return Status::ERROR;
            }

            if (InitUserBuffer(tensor_name) != Status::OK)
            {
                std::string info = "InitUserBuffer " + tensor_name + " failed";
                AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
                return Status::ERROR;
            }
        }

        return Status::OK;
    }

    AURA_VOID DeInitialize()
    {
        for (auto iter = m_quant_mat.begin(); iter != m_quant_mat.end(); ++iter)
        {
            Mat *mat = iter->second;
            Delete<Mat>(m_ctx, &mat);
        }
        m_quant_mat.clear();

        for (auto iter = m_user_buffer_handle.begin(); iter != m_user_buffer_handle.end(); ++iter)
        {
            if (*iter)
            {
                Snpe_ErrorCode_t snpe_ret = Snpe_IUserBuffer_Delete(m_ctx, *iter);
                if (snpe_ret != SNPE_SUCCESS)
                {
                    AURA_LOGE(m_ctx, AURA_TAG, "Snpe_IUserBuffer_Delete failed, snpe_ret=%d\n", snpe_ret);
                }
                *iter = MI_NULL;
            }
        }

        m_user_buffer_handle.clear();

        if (m_user_buffer_map)
        {
            Snpe_ErrorCode_t snpe_ret = Snpe_UserBufferMap_Delete(m_ctx, m_user_buffer_map);
            if (snpe_ret != SNPE_SUCCESS)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "Snpe_UserBufferMap_Delete failed, snpe_ret=%d\n", snpe_ret);
            }
            m_user_buffer_map = MI_NULL;
        }
    }

    Snpe_UserBufferMap_Handle_t GetSnpeUserBufferMapHandle()
    {
        return m_user_buffer_map;
    }

    Status Quant()
    {
        if (MI_FALSE == m_is_input)
        {
            return Status::OK;
        }

        Status ret        = Status::ERROR;
        MI_F32 scale      = 0.f;
        MI_U16 zero_point = 0;

        for (size_t i = 0; i < m_tensor_name.size(); i++)
        {
            std::string tensor_name = m_tensor_name[i];

            if (m_quant_mat.count(tensor_name))
            {
                ret = SnpeUtils::GetTensorAttributes(m_ctx, m_handle, tensor_name, scale, zero_point);
                if (ret != Status::OK)
                {
                    std::string info = "GetTensorAttributes failed, i=" + std::to_string(i) + " tensor_name=" + tensor_name;
                    AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
                    goto EXIT;
                }

                ret = NNQuantize(m_ctx, *(m_mat_map[tensor_name]), *(m_quant_mat[tensor_name]), zero_point, scale);
                if (ret != Status::OK)
                {
                    std::string info = "NNQuantize failed, i=" + std::to_string(i) + " tensor_name=" + tensor_name;
                    AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
                    goto EXIT;
                }
            }
        }

        ret = Status::OK;
EXIT:
        return ret;
    }

    Status DeQuant()
    {
        if (MI_TRUE == m_is_input)
        {
            return Status::OK;
        }

        Status ret        = Status::ERROR;
        MI_F32 scale      = 0.f;
        MI_U16 zero_point = 0;

        for (size_t i = 0; i < m_tensor_name.size(); i++)
        {
            std::string tensor_name = m_tensor_name[i];

            if (m_quant_mat.count(tensor_name))
            {
                ret = SnpeUtils::GetTensorAttributes(m_ctx, m_handle, tensor_name, scale, zero_point);
                if (ret != Status::OK)
                {
                    std::string info = "GetTensorAttributes failed, i=" + std::to_string(i) + " tensor_name=" + tensor_name;
                    AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
                    goto EXIT;
                }

                ret = NNDeQuantize(m_ctx, *(m_quant_mat[tensor_name]), *(m_mat_map[tensor_name]), zero_point, scale);
                if (ret != Status::OK)
                {
                    std::string info = "NNQuantize failed, i=" + std::to_string(i) + " tensor_name=" + tensor_name;
                    AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
                    goto EXIT;
                }
            }
        }

        ret = Status::OK;
EXIT:
        return ret;
    }

private:
    Status InitUserBuffer(const std::string &tensor_name)
    {
        MI_U32 dims = 0;
        MI_U32 batch = 0;
        MI_F32 scale = 0.f;
        MI_U16 zero_point = 0;
        std::vector<size_t> shape;
        Snpe_UserBufferEncoding_ElementType_t elem_type;
        if (SnpeUtils::GetTensorAttributes(m_ctx, m_handle, tensor_name, dims, batch, shape, elem_type, scale, zero_point) != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "GetTensorAttributes failed");
            return Status::ERROR;
        }

        Status ret = Status::ERROR;
        Mat *mat = m_mat_map[tensor_name];

        MI_BOOL is_quant = MI_FALSE;
        ElemType src_elem_type = mat->GetElemType();
        ElemType dst_elem_type = GetElemType(elem_type);
        Snpe_UserBufferEncoding_Handle_t encode_handle = MI_NULL;
        if (src_elem_type == dst_elem_type)
        {
            if (!mat->IsContinuous())
            {
                AURA_ADD_ERROR_STRING(m_ctx, "external mat data memory must be continuous");
                return Status::ERROR;
            }

            if ((elem_type == SNPE_USERBUFFERENCODING_ELEMENTTYPE_TF8) || (elem_type == SNPE_USERBUFFERENCODING_ELEMENTTYPE_TF16))
            {
                encode_handle = Snpe_UserBufferEncodingTfN_Create(m_ctx, zero_point, scale, ElemTypeSize(dst_elem_type) << 3);
                if (MI_NULL == encode_handle)
                {
                    AURA_ADD_ERROR_STRING(m_ctx, "Snpe_UserBufferEncodingTfN_Create failed");
                    return Status::ERROR;
                }
            }
            else if ((elem_type == SNPE_USERBUFFERENCODING_ELEMENTTYPE_FLOAT) || (elem_type == SNPE_USERBUFFERENCODING_ELEMENTTYPE_FLOAT16))
            {
                encode_handle = Snpe_UserBufferEncodingFloatN_Create(m_ctx, ElemTypeSize(dst_elem_type) << 3);
                if (MI_NULL == encode_handle)
                {
                    AURA_ADD_ERROR_STRING(m_ctx, "Snpe_UserBufferEncodingFloatN_Create failed");
                    return Status::ERROR;
                }
            }
            else
            {
                std::string info = "unsupport tensor dataType " + std::to_string(elem_type);
                AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
                return Status::ERROR;
            }
        }
        else if ((ElemType::F32 == src_elem_type) && ((SNPE_USERBUFFERENCODING_ELEMENTTYPE_TF8 == elem_type) ||
                                                      (SNPE_USERBUFFERENCODING_ELEMENTTYPE_TF16 == elem_type)))
        {
            encode_handle = Snpe_UserBufferEncodingTfN_Create(m_ctx, zero_point, scale, ElemTypeSize(dst_elem_type) << 3);
            if (MI_NULL == encode_handle)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "Snpe_UserBufferEncodingTfN_Create failed");
                return Status::ERROR;
            }

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
            if (!m_quant_mat.count(tensor_name))
            {
                m_quant_mat[tensor_name] = Create<Mat>(m_ctx, dst_elem_type, mat->GetSizes());
                if (!m_quant_mat[tensor_name]->IsValid())
                {
                    AURA_ADD_ERROR_STRING(m_ctx, "create tensor mat failed");
                    goto EXIT;
                }
            }

            mat = m_quant_mat[tensor_name];
        }

        {
            MI_S32 elem_counts = batch * shape[0] / ElemTypeSize(dst_elem_type);
            if (elem_counts != mat->GetSizes().Total())
            {
                std::string info = "tensor " + std::string(tensor_name) + ": expected " + std::to_string(elem_counts) + " bytes, "
                                   "but got " + std::to_string(mat->GetSizes().Total()) + " bytes";
                AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
                goto EXIT;
            }

            Snpe_TensorShape_Handle_t shape_handle = MI_NULL;
            Snpe_IUserBuffer_Handle_t user_buffer_handle = MI_NULL;
            Snpe_ErrorCode_t snpe_ret = SNPE_SUCCESS;
            shape_handle = Snpe_TensorShape_CreateDimsSize(m_ctx, shape.data(), dims);
            if (MI_NULL == shape_handle)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "Snpe_TensorShape_CreateDimsSize failed");
                goto EXIT;
            }

            user_buffer_handle = Snpe_Util_CreateUserBuffer(m_ctx, mat->GetData(), mat->GetSizes().Total() * ElemTypeSize(mat->GetElemType()), shape_handle, encode_handle);
            if (MI_NULL == user_buffer_handle)
            {
                snpe_ret = Snpe_TensorShape_Delete(m_ctx, shape_handle);
                std::string info = "Snpe_Util_CreateUserBuffer failed, Snpe_TensorShape_Delete ret : " + std::to_string(snpe_ret);
                AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
                goto EXIT;
            }

            Snpe_UserBufferMap_Add(m_ctx, m_user_buffer_map, tensor_name.c_str(), user_buffer_handle);
            m_user_buffer_handle.push_back(user_buffer_handle);
            snpe_ret = Snpe_TensorShape_Delete(m_ctx, shape_handle);
            if (snpe_ret != SNPE_SUCCESS)
            {
                std::string info = "Snpe_TensorShape_Delete failed, error : " + std::to_string(snpe_ret);
                AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
                goto EXIT;
            }
        }

        ret = Status::OK;
EXIT:
        if (encode_handle)
        {
            if ((SNPE_USERBUFFERENCODING_ELEMENTTYPE_FLOAT == elem_type) || (SNPE_USERBUFFERENCODING_ELEMENTTYPE_FLOAT16 == elem_type))
            {
                Snpe_ErrorCode_t snpe_ret = Snpe_UserBufferEncodingFloatN_Delete(m_ctx, encode_handle);
                if (snpe_ret != SNPE_SUCCESS)
                {
                    std::string info = "Snpe_UserBufferEncodingFloatN_Delete failed, error : " + std::to_string(snpe_ret);
                    AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
                    ret = Status::ERROR;
                }
            }
            else
            {
                Snpe_ErrorCode_t snpe_ret = Snpe_UserBufferEncodingTfN_Delete(m_ctx, encode_handle);
                if (snpe_ret != SNPE_SUCCESS)
                {
                    std::string info = "Snpe_UserBufferEncodingTfN_Delete failed, error : " + std::to_string(snpe_ret);
                    AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
                    ret = Status::ERROR;
                }
            }
        }
        return ret;
    }

private:
    Context *m_ctx;
    Snpe_SNPE_Handle_t m_handle;
    MatMap m_mat_map;
    MatMap m_quant_mat;
    MI_BOOL m_is_valid;
    MI_BOOL m_is_input;
    Snpe_UserBufferMap_Handle_t m_user_buffer_map;
    std::vector<std::string> m_tensor_name;
    std::vector<Snpe_IUserBuffer_Handle_t> m_user_buffer_handle;
};

class SnpeExecutorImplV2 : public SnpeExecutorImpl
{
public:
    SnpeExecutorImplV2(Context *ctx, const std::shared_ptr<SnpeModel> &model, const NNConfig &config)
                       : SnpeExecutorImpl(ctx, model, config), m_container_handle(MI_NULL), m_snpe_handle(MI_NULL)
    {
        do
        {
            // get snpe version : 2.12.1.230626174329_59328
            std::vector<std::string> snpe_version;
            m_version = SnpeUtils::GetLibraryVersion(ctx);
            snpe_version = NNSplit(m_version, '.');

            // framework version
            std::vector<std::string> framework_version;
            framework_version = NNSplit(m_model->GetFrameWorkVersion(), '.');

            if ((4 == framework_version.size()) && (snpe_version.size() >= 3))
            {
                if ((framework_version[1] != snpe_version[0]) || (framework_version[2] > snpe_version[1]))
                {
                    std::string info = "frame version not match snpe version, frame verison: " + m_model->GetFrameWorkVersion() + " snpe version: " + m_version;
                    AURA_ADD_ERROR_STRING(ctx, info.c_str());
                    break;
                }

                if (framework_version[2] != snpe_version[1])
                {
                    std::string info = "WARNING: frame version not match snpe version, model init may slower, frame verison: " +
                                        m_model->GetFrameWorkVersion() + " snpe version: " + m_version + "\n";
                    AURA_LOGD(m_ctx, AURA_TAG, info.c_str());
                }
            }
            else
            {
                AURA_ADD_ERROR_STRING(ctx, "frame version size must be 4 and snpe version size can not be less than 3");
                break;
            }

            auto init_snpe_func = [=]() -> Status
            {
                Snpe_Util_InitializeLogging(ctx, GetSnpeLogLevel(m_config.log_level));

                Snpe_RuntimeCheckOption_t rt_opt = m_config.unsigned_pd ? SNPE_RUNTIME_CHECK_OPTION_DEFAULT : SNPE_RUNTIME_CHECK_OPTION_NORMAL_CHECK;

                MI_S32 ret = Snpe_Util_IsRuntimeAvailableCheckOption(ctx, GetSnpeRuntimeType(m_config.backend), rt_opt);
                if (!ret)
                {
                    std::string info = "Snpe_Util_IsRuntimeAvailableCheckOption failed, backend=" + config.at("backend") + " unsigned_pd=" + std::to_string(m_config.unsigned_pd);
                    AURA_ADD_ERROR_STRING(ctx, info.c_str());
                    return Status::ERROR;
                }

                if (CreateRuntime() != Status::OK)
                {
                    AURA_ADD_ERROR_STRING(m_ctx, "CreateRuntime failed");
                    return Status::ERROR;
                }

                m_is_valid = MI_TRUE;
                return Status::OK;
            };

            if (m_wp)
            {
                m_token = m_wp->AsyncRun(init_snpe_func);
            }
            else
            {
                AURA_ADD_ERROR_STRING(ctx, "m_wp null ptr");
            }
        } while (0);
    }

    ~SnpeExecutorImplV2()
    {
        if (m_token.valid())
        {
            m_token.wait();
        }

        m_input_map.reset();
        m_output_map.reset();

        if (NNProfilingLevel::PROFILING_DETAILED == m_config.profiling_level && m_snpe_handle)
        {
            Snpe_IDiagLog_Handle_t log_handle = Snpe_SNPE_GetDiagLogInterface_Ref(m_ctx, m_snpe_handle);
            if (log_handle)
            {
                Snpe_IDiagLog_Stop(m_ctx, log_handle);
            }
        }

        if (m_container_handle)
        {
            Snpe_ErrorCode_t snpe_ret = Snpe_DlContainer_Delete(m_ctx, m_container_handle);
            if (snpe_ret != SNPE_SUCCESS)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "Snpe_DlContainer_Delete failed, error=%d\n", snpe_ret);
            }
            m_container_handle = MI_NULL;
        }

        if (m_snpe_handle)
        {
            Snpe_ErrorCode_t snpe_ret = Snpe_SNPE_Delete(m_ctx, m_snpe_handle);
            if (snpe_ret != SNPE_SUCCESS)
            {
                AURA_LOGE(m_ctx, AURA_TAG, "Snpe_SNPE_Delete failed, error=%d\n", snpe_ret);
            }
            m_snpe_handle = MI_NULL;
        }
    }

    Status Initialize() override
    {
        Status ret = Status::ERROR;

        if (m_token.valid())
        {
            m_token.wait();
        }

        if (!m_is_valid)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "m_is_valid is false");
            return ret;
        }

        // set log
        if (NNProfilingLevel::PROFILING_DETAILED == m_config.profiling_level)
        {
            if (SnpeUtils::InitDiagLog(m_ctx, m_config.profiling_path, m_snpe_handle) != SNPE_SUCCESS)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "InitDiagLog failed");
            }
        }

        // create SnpeUserBufferMap
        m_input_map = std::make_shared<SnpeUserBufferMap>(m_ctx, m_snpe_handle, MI_TRUE);
        m_output_map = std::make_shared<SnpeUserBufferMap>(m_ctx, m_snpe_handle, MI_FALSE);

        ret = Status::OK;

        return ret;
    }

    Status Forward(const MatMap &input, MatMap &output, MI_S32 graph_id) override
    {
        auto process_func = [=]() -> Status
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

            if (m_input_map->GetSnpeUserBufferMapHandle() == MI_NULL)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "m_input_map GetSnpeUserBufferMapHandle is null");
                return ret;
            }

            if (m_output_map->GetSnpeUserBufferMapHandle() == MI_NULL)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "m_output_map GetSnpeUserBufferMapHandle is null");
                return ret;
            }

            ret = m_input_map->Quant();
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "m_input_map Quant failed");
                return ret;
            }

            Snpe_ErrorCode_t rt = Snpe_SNPE_ExecuteUserBuffers(m_ctx, m_snpe_handle, m_input_map->GetSnpeUserBufferMapHandle(), m_output_map->GetSnpeUserBufferMapHandle());
            if (rt != SNPE_SUCCESS)
            {
                MI_CHAR err_str[128];
                snprintf(err_str, sizeof(err_str), "Snpe_SNPE_ExecuteUserBuffers failed, error: %d", rt);
                return ret;
            }

            ret = m_output_map->DeQuant();
            if (ret != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "m_output_map DeQuant failed");
                return ret;
            }

            ret = Status::OK;
            return ret;
        };

        if (m_config.async_call && m_wp)
        {
            return m_wp->AsyncRun(process_func).get();
        }
        else
        {
            return process_func();
        }
    }

    std::vector<TensorDescMap> GetInputs() override
    {
        Snpe_StringList_Handle_t list_handle = Snpe_SNPE_GetInputTensorNames(m_ctx, m_snpe_handle);
        if (MI_NULL == list_handle)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "Snpe_SNPE_GetInputTensorNames failed");
            return {};
        }

        TensorDescMap result = SnpeUtils::GetTensorDesc(m_ctx, m_snpe_handle, list_handle);
        result = m_model->MapTensorDescNames(result, MI_TRUE);

        return {result};
    }

    std::vector<TensorDescMap> GetOutputs() override
    {
        Snpe_StringList_Handle_t list_handle = Snpe_SNPE_GetOutputTensorNames(m_ctx, m_snpe_handle);
        if (MI_NULL == list_handle)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "Snpe_SNPE_GetOutputTensorNames failed");
            return {};
        }

        TensorDescMap result = SnpeUtils::GetTensorDesc(m_ctx, m_snpe_handle, list_handle);
        result = m_model->MapTensorDescNames(result, MI_FALSE);

        return {result};
    }

    std::string GetVersion() override
    {
        return (m_model->GetVersion() + " device(snpe." + m_version + ")");
    }

private:
    Status SetCustomConfig(std::string &config_options)
    {
        AURA_UNUSED(config_options);

        return Status::OK;
    }

    Status CreateRuntime()
    {
        Status ret = Status::ERROR;

        Snpe_SNPEBuilder_Handle_t builder_handle = MI_NULL;
        Snpe_StringList_Handle_t string_list = MI_NULL;
        Snpe_RuntimeList_Handle_t runtime_list = MI_NULL;
        Snpe_PlatformConfig_Handle_t config_handle = MI_NULL;
        std::string config_options;
        std::string model_name = m_model->GetModelName();

        std::vector<std::string> output_layers = m_model->GetOutputLayerNames();
        Buffer buffer = m_model->GetModelBuffer();
        if (!buffer.IsValid())
        {
            AURA_ADD_ERROR_STRING(m_ctx, "GetDlcBuffer failed");
            goto EXIT;
        }

        m_container_handle = Snpe_DlContainer_OpenBuffer(m_ctx, (const MI_UCHAR *)buffer.m_data, buffer.m_size);
        if (MI_NULL == m_container_handle)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "Snpe_DlContainer_OpenBuffer failed");
            goto EXIT;
        }

        // create builder
        builder_handle = Snpe_SNPEBuilder_Create(m_ctx, m_container_handle);
        if (MI_NULL == builder_handle)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "Snpe_SNPEBuilder_Create failed");
            goto EXIT;
        }
#if !defined(SNPE_EXECUTOR_IMPL_V2133)
        // register model name
        if (!model_name.empty())
        {
            Snpe_SNPEBuilder_SetModelName(m_ctx, builder_handle, model_name.c_str());
        }
#endif
        // ceate string list
        string_list = Snpe_StringList_Create(m_ctx);
        if (MI_NULL == string_list)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "Snpe_SNPEBuilder_Create failed");
            goto EXIT;
        }

        // set output layers
        if (output_layers.size() > 1)
        {
            for (MI_U32 i = 0; i < output_layers.size(); i++)
            {
                if (Snpe_StringList_Append(m_ctx, string_list, output_layers[i].c_str()) != SNPE_SUCCESS)
                {
                    AURA_ADD_ERROR_STRING(m_ctx, "Snpe_StringList_Append failed");
                    goto EXIT;
                }
            }
        }

        if (Snpe_SNPEBuilder_SetOutputLayers(m_ctx, builder_handle, string_list) != SNPE_SUCCESS)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "Snpe_SNPEBuilder_SetOutputLayers failed");
            goto EXIT;
        }

        //set config
        config_handle = Snpe_PlatformConfig_Create(m_ctx);
        if (MI_NULL == config_handle)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "Snpe_PlatformConfig_Create failed");
            goto EXIT;
        }

        if (MI_FALSE == m_config.unsigned_pd)
        {
            config_options = "unsignedPD:OFF";
        }

        if (NNPerfLevel::PERF_HIGH == m_config.perf_level)
        {
            config_options = config_options.empty() ? "rpcPollTime:0" : (config_options + ";rpcPollTime:0");
        }

        if (m_config.mem_step_size > 0)
        {
            std::string mem_step_size_str = "dspGrowSize:" + std::to_string(m_config.mem_step_size);
            config_options = config_options.empty() ? mem_step_size_str : (config_options + ";" + mem_step_size_str);
        }

        if (!config_options.empty())
        {
            if (!Snpe_PlatformConfig_SetPlatformOptions(m_ctx, config_handle, config_options.c_str()))
            {
                std::string info = "Snpe_PlatformConfig_SetPlatformOptions failed, option is " + config_options;
                AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
                goto EXIT;
            }
        }

        // create runtime list
        runtime_list = Snpe_RuntimeList_Create(m_ctx);
        if (MI_NULL == runtime_list)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "Snpe_RuntimeList_Create failed");
            goto EXIT;
        }

        if (Snpe_RuntimeList_Add(m_ctx, runtime_list, GetSnpeRuntimeType(m_config.backend)) != SNPE_SUCCESS)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "Snpe_RuntimeList_Add failed");
            goto EXIT;
        }

        if (Snpe_SNPEBuilder_SetRuntimeProcessorOrder(m_ctx, builder_handle, runtime_list) != SNPE_SUCCESS)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "Snpe_SNPEBuilder_SetRuntimeProcessorOrder failed");
            goto EXIT;
        }

        if (Snpe_SNPEBuilder_SetUseUserSuppliedBuffers(m_ctx, builder_handle, MI_TRUE) != SNPE_SUCCESS)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "Snpe_SNPEBuilder_SetUseUserSuppliedBuffers failed");
            goto EXIT;
        }

        if (Snpe_SNPEBuilder_SetPerformanceProfile(m_ctx, builder_handle, GetSnpePerfLevel(m_config.perf_level)) != SNPE_SUCCESS)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "Snpe_SNPEBuilder_SetPerformanceProfile failed");
            goto EXIT;
        }

        if (Snpe_SNPEBuilder_SetProfilingLevel(m_ctx, builder_handle, GetSnpeProfilingLevel(m_config.profiling_level)) != SNPE_SUCCESS)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "Snpe_SNPEBuilder_SetProfilingLevel failed");
            goto EXIT;
        }

        if (Snpe_SNPEBuilder_SetInitCacheMode(m_ctx, builder_handle, MI_TRUE) != SNPE_SUCCESS)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "Snpe_SNPEBuilder_SetInitCacheMode failed");
            goto EXIT;
        }

        if (Snpe_SNPEBuilder_SetPlatformConfig(m_ctx, builder_handle, config_handle) != SNPE_SUCCESS)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "Snpe_SNPEBuilder_SetPlatformConfig failed");
            goto EXIT;
        }

        m_snpe_handle = Snpe_SNPEBuilder_Build(m_ctx, builder_handle);
        if (MI_NULL == m_snpe_handle)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "Snpe_SNPEBuilder_Build failed");
            goto EXIT;
        }

        ret = Status::OK;
EXIT:
        if (builder_handle)
        {
            Snpe_ErrorCode_t snpe_ret = Snpe_SNPEBuilder_Delete(m_ctx, builder_handle);
            if (snpe_ret != SNPE_SUCCESS)
            {
                std::string info = "Snpe_SNPEBuilder_Delete failed, error : " + std::to_string(snpe_ret);
                AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
                ret = Status::ERROR;
            }
        }

        if (string_list)
        {
            Snpe_ErrorCode_t snpe_ret = Snpe_StringList_Delete(m_ctx, string_list);
            if (snpe_ret != SNPE_SUCCESS)
            {
                std::string info = "Snpe_StringList_Delete failed, error : " + std::to_string(snpe_ret);
                AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
                ret = Status::ERROR;
            }
        }

        if (runtime_list)
        {
            Snpe_ErrorCode_t snpe_ret = Snpe_RuntimeList_Delete(m_ctx, runtime_list);
            if (snpe_ret != SNPE_SUCCESS)
            {
                std::string info = "Snpe_RuntimeList_Delete failed, error : " + std::to_string(snpe_ret);
                AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
                ret = Status::ERROR;
            }
        }

        if (config_handle)
        {
            Snpe_ErrorCode_t snpe_ret = Snpe_PlatformConfig_Delete(m_ctx, config_handle);
            if (snpe_ret != SNPE_SUCCESS)
            {
                std::string info = "Snpe_PlatformConfig_Delete failed, error : " + std::to_string(snpe_ret);
                AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
                ret = Status::ERROR;
            }
        }

        m_model->ReleaseModelBuffer();

        if (m_container_handle)
        {
            Snpe_DlContainer_Delete(m_ctx, m_container_handle);
            m_container_handle = MI_NULL;
        }

        return ret;
    }

private:
    std::string m_version;
    Snpe_DlContainer_Handle_t m_container_handle;
    Snpe_SNPE_Handle_t m_snpe_handle;
    std::shared_ptr<SnpeUserBufferMap> m_input_map;
    std::shared_ptr<SnpeUserBufferMap> m_output_map;
};

} // namespace aura