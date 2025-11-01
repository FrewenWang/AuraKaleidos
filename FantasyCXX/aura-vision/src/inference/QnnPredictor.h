#pragma once

#include <dlfcn.h>
#include <getopt.h>

#include <iostream>
#include <fstream>
#include <memory>
#include <map>
#include <sstream>
#include <string>

#include "BuildId.hpp"
#include "DataUtil.hpp"
#include "IOTensor.hpp"
#include "Logger.hpp"
#include "QnnCommon.h"
#include "QnnInterface.h"
#include "QnnWrapperUtils.hpp"
#include "System/QnnSystemInterface.h"
#include "AbsPredictor.h"
#include "vision/core/common/VMacro.h"

#define QNN_MIN_ERROR_COMMON          1000
#define QNN_MAX_ERROR_COMMON          1999
namespace aura::vision {

enum class StatusCode {
    SUCCESS,
    FAIL_INIT_BACKEND,
    FAIL_LOAD_BACKEND,
    FAIL_PROFILE_CREATE,
    FAIL_OP_PACKAGE,
    FAIL_CONTEXT_FROM_BIN,
    FAIL_CONTEXT_CREATE,
    FAIL_COMPOSE_GRAPH,
    FAIL_FINALIZE_GRAPH,
    FAIL_LOAD_MODEL,
    FAIL_SYM_FUNCTION,
    FAIL_GET_INTERFACE_PROVIDERS,
    FAIL_CONTEXT_FERR,
    FAILURE,
    FAILURE_SYSTEM_ERROR,
    QNN_FEATURE_UNSUPPORTED,
    FAILURE_SYSTEM_COMMUNICATION_ERROR,
};
/**
 * @brief Set device ID which HTP to use
 * HTP0 -> 0, HTP1 -> 1
 */
enum HtpCode : short {
    DEVICE_HTP0 = 0,
    DEVICE_HTP1 = 1
};

enum QNN_HTP_PERF_INFRASTRUCTURE_POWERLEVEL {
    QNN_HTP_PERF_INFRASTRUCTURE_POWERLEVEL_BURST = 0,
    QNN_HTP_PERF_INFRASTRUCTURE_POWERLEVEL_SUSTAINED_HIGH_PERFORMANCE,
    QNN_HTP_PERF_INFRASTRUCTURE_POWERLEVEL_HIGH_PERFORMANCE,
    QNN_HTP_PERF_INFRASTRUCTURE_POWERLEVEL_POWER_SAVER,
    QNN_HTP_PERF_INFRASTRUCTURE_POWERLEVEL_LOW_POWER_SAVER,
    QNN_HTP_PERF_INFRASTRUCTURE_POWERLEVEL_HIGH_POWER_SAVER,
    QNN_HTP_PERF_INFRASTRUCTURE_POWERLEVEL_LOW_BALANCED,
    QNN_HTP_PERF_INFRASTRUCTURE_POWERLEVEL_BALANCED,
    QNN_HTP_PERF_INFRASTRUCTURE_POWERLEVEL_DEFAULT,
    QNN_HTP_PERF_INFRASTRUCTURE_POWERLEVEL_SYSTEM_SETTINGS
};

enum class ProfilingLevel { OFF, BASIC, DETAILED, INVALID };

// Graph Related Function Handle Types
typedef qnn_wrapper_api::ModelError_t (*ComposeGraphsFnHandleType_t)(
        Qnn_BackendHandle_t,
        QNN_INTERFACE_VER_TYPE,
        Qnn_ContextHandle_t,
        const qnn_wrapper_api::GraphConfigInfo_t **,
        const uint32_t,
        qnn_wrapper_api::GraphInfo_t ***,
        uint32_t *,
        bool,
        QnnLog_Callback_t,
        QnnLog_Level_t);
typedef qnn_wrapper_api::ModelError_t (*FreeGraphInfoFnHandleType_t)(
        qnn_wrapper_api::GraphInfo_t ***, uint32_t);


class QnnPredictor : public AbsPredictor {
public:
    ~QnnPredictor() override = default;
    /**
     * @brief QNN 模型推理器的初始化方法
     * @param model  模型数据
     * @return int  初始化状态
     */
    int init(ModelInfo &model) override;

    int doPredict(TensorArray& inputs, TensorArray& outputs, PerfUtil* perf) override;

    // 暂时未使用
    // int doPredict(std::vector<cv::Mat> &inputs, std::vector<cv::Mat> &outputs, PerfUtil* perf) override;
    int deinit() override;
    bool valid() override;
    DLayout getSupportedLayout() override;
    std::vector<ModelInput> get_input_desc() const override;

    static void logStdoutCallback(const char* fmt, QnnLog_Level_t level, uint64_t timestamp, va_list argp);

private:
    /** 全局静态的QNN后端初始化的标志变量 */
    static bool gInited;

    ModelInfo modelInfo;
    /**  QnnPredictor 是否初始化标志变量，避免重复初始化和反初始化  */
    bool mInited = false;

    // QnnPredictor的默认加载CachedBinary
    bool mLoadFromCachedBinary = true;

    QnnLog_Callback_t logCallback;

    // qnn interface QNN接口对象
    const QnnInterface_t **m_interfaceProviders;
    uint32_t m_numProviders;
    QNN_INTERFACE_VER_TYPE m_qnnInterface;
    bool m_foundValidInterface;

    // qnn system interface pointer
    const QnnSystemInterface_t **m_systemInterfaceProviders;
    uint32_t m_numSystemProviders;
    QNN_SYSTEM_INTERFACE_VER_TYPE m_qnnSystemInterface;
    bool m_foundValidSystemInterface;

    std::string m_backend_build_version;
    char* m_backendBuildId = nullptr;

    // backend
    QnnBackend_Config_t **m_backendConfig;
    bool m_isBackendInitialized;

    // profiling level
    ProfilingLevel m_profilingLevel;
    Qnn_ProfileHandle_t m_profileBackendHandle;

    // register op package
    std::vector<std::string> m_opPackagePaths;

    // create context from cached bin
    QnnContext_Config_t **m_contextConfig;
    qnn_wrapper_api::GraphInfo_t **m_graphsInfo;
    uint32_t m_graphsCount;
    Qnn_ContextHandle_t m_context;
    bool m_isContextCreated;

    // compose graph
    qnn_wrapper_api::GraphConfigInfo_t **m_graphConfigsInfo;
    uint32_t m_graphConfigsInfoCount;
    bool m_debug;

    ComposeGraphsFnHandleType_t m_composeGraphsFnHandle;
    FreeGraphInfoFnHandleType_t m_freeGraphInfoFnHandle;

    // execute graph
    qnn::tools::iotensor::IOTensor m_ioTensor;
    qnn::tools::iotensor::OutputDataType m_outputDataType;
    qnn::tools::iotensor::InputDataType m_inputDataType;

    Qnn_Tensor_t* t_qnn_inputs;
    Qnn_Tensor_t* t_qnn_outputs;
    int t_numInputTensors;
    int t_numOutputTensors;

    Qnn_LogHandle_t m_logHandle         = nullptr;
    Qnn_BackendHandle_t m_backendHandle = nullptr;
    Qnn_DeviceHandle_t m_deviceHandle   = nullptr;
    //设置使用的HTP
    short deviceID = 0;
    // HTP中使用核的类型
    short CORE_TYPE_0 = 0;
    // HTP中核的ID
    short CORE_ID_0 = 0;
    //refer to type of the device, other than hta, gpu or dsp
    short DEVICE_HTP = 0;
    //HTP中对应的HMX数量
    short HTP_WITH_ONE_HMX = 1;
    //HTP中对应的core数量
    short HTP_WITH_ONE_CORE = 1;
    //0 copy with share buffer or not, qnx(true), other (false)
    const bool shareBuffer = true;
    //debug log for QNN SDK, default false
    const bool enableQnnLog = true;

private:
    std::string get_qnn_build_version();
    void logging_initialize();

    /** 判断指定设备的支持 */
    StatusCode isDevicePropertySupported();

    /** 创建指定设备的配置信息  */
    StatusCode createDevice();
    /** 释放指定设备的配置信息  */
    StatusCode freeDevice();
    /** 释放HTP后端  */
    StatusCode freeBackend();
    StatusCode backend_initialize();
    StatusCode profiling_initialize();
    StatusCode register_op_package();
    /**
     * @param is_load_from_cached_binary
     * @param mem
     * @param mem_len
     * @return
     */
    StatusCode create_context(bool is_load_from_cached_binary, char *mem, int mem_len);
    /**
     * 设置初始化能耗级别的ConfigId
     * @return
     */
    StatusCode initConfigId();
    /**
     * 设置QNN的能耗级别
     * @param powerConfigId
     * @param powerLevel
     * @return
     */
    StatusCode setPowerLevel(uint32_t powerConfigId, uint32_t powerLevel);

    /// 设置的power的Config id
    uint32_t powerConfigId = 0;

    bool onInferenceCmd(int cmd) override;

    bool isPredicted = false;
    /**
     * 判断此QnnPredictor是否是从模型Buffer加载
     * TODO @wangzhijiang 考虑和原来的m_loadFromCachedBinary合并
     * 后续重构QnnPredictor
     */
    bool isLoadFromBinaryBuffer = false;
};

template <>
inline std::shared_ptr<AbsPredictor> make_predictor<QNN>(ModelInfo& model) {
    auto predictor = std::make_shared<QnnPredictor>();
    if(predictor->init(model) != 0) {
        return nullptr;
    }
    return std::dynamic_pointer_cast<AbsPredictor>(predictor);
}

} // namespace vision
