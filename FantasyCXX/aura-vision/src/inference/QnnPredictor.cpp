#include <thread>
#include "QnnPredictor.h"
#include "util/InferenceConverter.hpp"
#include "System/QnnSystemContext.h"
#include "QnnSampleAppUtils.hpp"
#include "QnnTypeMacros.hpp"
#include "util/DebugUtil.h"
#include "HTP/QnnHtpPerfInfrastructure.h"
#include "HTP/QnnHtpDevice.h"

// 定义QNN2.6版本的启用编译开关，后续待2.6版本稳定之后，放开此开关
// #define USE_QNN26

static const char *TAG = "QnnPredictor(4000)";
using namespace std;
using namespace qnn::tools::sample_app;
/// 默认QNN后端全局初始化的标志变量为false
bool vision::QnnPredictor::gInited = false;
std::mutex gLogStdoutCallbackMutex;
char gBuffer[1024];

namespace aura::vision {

int QnnPredictor::init(ModelInfo &model) {
    mPerfTag = "[Predictor]: [Qnn] " + std::to_string(model.id);
    if (mInited) {
        VLOGW(TAG, "%s init failed !!! cause of mInited:%d", std::to_string(model.id).c_str(), mInited);
        V_RET(Error::INFER_ERR);
    }
    modelInfo = model;
    if (modelInfo.blobs.size() != 1) {
        VLOGE(TAG, "Predictor model mem error, one blob needed! error model info:%s", modelInfo.to_string().c_str());
        V_RET(Error::MODEL_INIT_ERR);
    }
    // load model
    auto *mem = static_cast<char *>(model.blobs[0].data);
    auto mem_len = model.blobs[0].len;
    // QNN 推理引擎加载Binary的模型数据
    m_inputDataType = qnn::tools::iotensor::InputDataType::FLOAT;
    m_outputDataType = qnn::tools::iotensor::OutputDataType::FLOAT_ONLY;
    m_profilingLevel = ProfilingLevel::OFF;

    m_interfaceProviders = nullptr;
    m_numProviders = 0;
    delete []m_backendBuildId;
    m_backendBuildId = nullptr;

    m_systemInterfaceProviders = nullptr;
    m_numSystemProviders = 0;
    m_isBackendInitialized = false;

    t_qnn_inputs = nullptr;
    t_qnn_outputs = nullptr;
    StatusCode returnStatus;
    // qnn interface init  获取 QnnInterfaceProviders
    QnnInterface_getProviders((const QnnInterface_t ***)&m_interfaceProviders, &m_numProviders);
    if (nullptr == m_interfaceProviders || 0 == m_numProviders) {
        VLOGE(TAG, "QnnInterface_getProviders  error interfaceProviders or numProviders NULL !!!");
        V_RET(Error::MODEL_INIT_ERR);
    }
    // get qnn interface
    for (size_t pIdx = 0; pIdx < m_numProviders; pIdx++) {
        if (QNN_API_VERSION_MAJOR == m_interfaceProviders[pIdx]->apiVersion.coreApiVersion.major
            && QNN_API_VERSION_MINOR <= m_interfaceProviders[pIdx]->apiVersion.coreApiVersion.minor) {
            // 标记QNN interface 初始化成功.从QnnProviders获取对应qnnInterface
            m_foundValidInterface = true;
            m_qnnInterface = m_interfaceProviders[pIdx]->QNN_INTERFACE_VER_NAME;
            break;
        }
    }
#if defined(USE_QNN26) and defined(BUILD_QNX)
    // 如果使用QNN2.6版本则传入共享Buffer进行初始化m_ioTensor
    m_ioTensor = qnn::tools::iotensor::IOTensor(shareBuffer, &m_qnnInterface);
#endif
    if (!m_foundValidInterface) {
        VLOGE(TAG, "Unable to find a valid interface.");
        V_RET(Error::MODEL_INIT_ERR);
    }

    // 开始初始化 qnnSystemInterface init
    if (mLoadFromCachedBinary) {
        // 获取QNN的接口Providers
        QnnSystemInterface_getProviders((const QnnSystemInterface_t ***)&m_systemInterfaceProviders,
                                        &m_numSystemProviders);
        if (nullptr == m_systemInterfaceProviders || 0 == m_numSystemProviders) {
            QNN_ERROR("Failed to get system interface providers: null interface providers received.");
            V_RET(Error::MODEL_INIT_ERR);
        }
        bool foundValidSystemInterface{false};
        for (size_t pIdx = 0; pIdx < m_numSystemProviders; pIdx++) {
            //  QNN_SYSTEM_API_VERSION_MINOR 恒为0 。
            //  && QNN_SYSTEM_API_VERSION_MINOR <= m_systemInterfaceProviders[pIdx]->systemApiVersion.minor 恒成立
            //  故屏蔽此条件，避免编译警告
            if (QNN_SYSTEM_API_VERSION_MAJOR == m_systemInterfaceProviders[pIdx]->systemApiVersion.major) {
                // 找到对应版本的systemInterfaceProviders。同时获取对应的SystemInterface
                foundValidSystemInterface = true;
                m_qnnSystemInterface = m_systemInterfaceProviders[pIdx]->QNN_SYSTEM_INTERFACE_VER_NAME;
                break;
            }
        }
        if (!foundValidSystemInterface) {
            VLOGE(TAG, "Unable to find a valid interface.");
            V_RET(Error::MODEL_INIT_ERR);
        }
    }

    // get qnn build version
    m_qnnInterface.backendGetBuildId((const char **)&m_backendBuildId);
    std::string qnn_version = get_qnn_build_version();
    if (!gInited) {
        VLOGD(TAG, "Qnn Backend build version: %s", qnn_version.c_str());
        gInited = true;
    }

    // initialize logging in the backend
    if (enableQnnLog) {
        QnnPredictor::logging_initialize();
    }
    // initialize a qnnBackend
    if (!m_isBackendInitialized) {
        returnStatus = backend_initialize();
        if (returnStatus != StatusCode::SUCCESS) {
            VLOGE(TAG, "Backend initialize error: %d", (int)returnStatus);
            V_RET(returnStatus);
        }
    }

    if (StatusCode::FAILURE != isDevicePropertySupported()) {
        returnStatus = createDevice();
        if (StatusCode::SUCCESS != returnStatus) {
            VLOGE(TAG, "create QNN Device error:  ret=%d", returnStatus);
            V_RET(returnStatus);
        }
    }
    // initialize profiling
    returnStatus = profiling_initialize();
    if (returnStatus != StatusCode::SUCCESS) {
        VLOGE(TAG, "Profiling Initialize Error ret=%d", returnStatus);
        V_RET(returnStatus);
    }

    // qnn UDO OpPackages
    returnStatus = register_op_package();
    if (returnStatus != StatusCode::SUCCESS) {
        VLOGE(TAG, "Register op package Error ret=%d", returnStatus);
        V_RET(returnStatus);
    }

    // 进行创建QNN的上下文对象 同时将模型Buffer数据传入QNN后端
    if (!model.hasLoaded) {
        returnStatus = QnnPredictor::create_context(mLoadFromCachedBinary, mem, mem_len);
        isLoadFromBinaryBuffer = true;
        model.hasLoaded = true;
        model.graphsInfo = m_graphsInfo;
        model.graphsCount = m_graphsCount;
    } else {
        isLoadFromBinaryBuffer = false;
        m_graphsInfo = model.graphsInfo;
        m_graphsCount = model.graphsCount;
        model.hasLoaded = true;
    }
    if (returnStatus != StatusCode::SUCCESS) {
        VLOGE(TAG, "QnnPredictor init create_context Error ret=%d", returnStatus);
        V_RET(returnStatus);
    }
    // 暂时注释掉设置PowerLevel的接口
    // init power level config and setPowerLevel
    // returnStatus = initConfigId();
    // if (returnStatus != StatusCode::SUCCESS) {
    //     VLOGE(TAG, "QnnPredictor init initConfigId Error ret=%d", returnStatus);
    //     V_RET(returnStatus);
    // }
    // returnStatus = setPowerLevel(powerConfigId, QNN_HTP_PERF_INFRASTRUCTURE_POWERLEVEL_BURST);
    // if (returnStatus != StatusCode::SUCCESS) {
    //     VLOGE(TAG, "QnnPredictor init setPowerLevel Error ret=%d", returnStatus);
    //     V_RET(returnStatus);
    // }
    VLOGI(TAG, "init model: %s from %s", modelInfo.port_desc.ability().c_str(),
          isLoadFromBinaryBuffer ? "BinaryBuffer" : "GraphsInfo");
    mInited = true;
    V_RET(Error::OK);
}

int QnnPredictor::deinit() {
    if (!mInited) {
        VLOGW(TAG, "%s deinit failed !!! cause of mInited:%d", modelInfo.port_desc.ability().c_str(), mInited);
        V_RET(Error::INFER_ERR);
    }
    mInited = false;
    // 暂时注释掉设置PowerLevel的逻辑
    // QnnPredictor 在进行deinit的时候，先进行设置为节能模式
    // setPowerLevel(powerConfigId, QNN_HTP_PERF_INFRASTRUCTURE_POWERLEVEL_POWER_SAVER);

    if (t_qnn_inputs != nullptr || t_qnn_outputs != nullptr) {
        m_ioTensor.tearDownInputAndOutputTensors(t_qnn_inputs, t_qnn_outputs, t_numInputTensors, t_numOutputTensors);
        t_qnn_inputs = nullptr;
        t_qnn_outputs = nullptr;
    }

    // 如果某个模型是初次加载的时候，需要初始化上下文。则需要free context
    // 如果是使用ModelInfo进行二次初始化时候不需要初始化上下文。不需要free context
    if (isLoadFromBinaryBuffer) {
        auto freeRet = m_qnnInterface.contextFree(m_context, nullptr);
        if (freeRet != QNN_CONTEXT_NO_ERROR) {
            VLOGE(TAG, "free context error!!!! Could not free context，ret = %ld", freeRet);
            return static_cast<int>(StatusCode::FAIL_CONTEXT_FERR);
        }
        if (m_graphsInfo) {
            qnn_wrapper_api::freeGraphsInfo(&m_graphsInfo, m_graphsCount);
            m_graphsInfo = nullptr;
            m_graphsCount = 0;
        }
    } else {
        VLOGI(TAG, "is not LoadFromBinaryBuffer no need free context!!!");
    }

    if (StatusCode::FAILURE != isDevicePropertySupported()) {
        if (StatusCode::SUCCESS != freeDevice()) {
            VLOGE(TAG, "Could not free device");
            return static_cast<int>(StatusCode::FAILURE);
        }
    }
    if (StatusCode::SUCCESS != freeBackend()) {
        VLOGE(TAG, "Could not free backend");
        return static_cast<int>(StatusCode::FAILURE);
    }

    if (enableQnnLog) {
        m_qnnInterface.logFree(&m_logHandle);
    }

    V_RET(Error::OK);
}

int QnnPredictor::doPredict(TensorArray &inputs, TensorArray &outputs, PerfUtil *perf) {
#ifdef DEBUG_SUPPORT
    VLOGD(TAG, "start predict with model: %s", modelInfo.port_desc.ability().c_str());
#endif
    auto returnStatus = Error::OK;
    outputs.clear();
    V_CHECK_COND(!mInited, Error::MODEL_UN_INITED, "Error: predictor not initialized!");
    V_CHECK_COND(inputs.empty(), Error::MODEL_UN_INITED, "Error: predictor inputs are empty!");

    std::vector<uint8_t *> qnn_inputs;

    // 目前所有模型推理，输出的input的size都为1.故直接取inputs[0]
    VTensor src_tensor = inputs[0];
    if (src_tensor.dLayout != NHWC) {
        src_tensor = src_tensor.changeLayout(NHWC);
    }
    DBG_PRINT_ARRAY((float *) src_tensor.data, 100, "QnnPredictor_doPredict_prepare");

    for (size_t i = 0; i < inputs.size(); i++) {
        qnn_inputs.push_back(reinterpret_cast<uint8_t *>(src_tensor.data));
    }

    // VLOGD(TAG, "Model id = %d", m_model_info.id);
    for (size_t graphIdx = 0; graphIdx < m_graphsCount; graphIdx++) {
        // VLOGD(TAG, "Starting execution for graphIdx: %d", graphIdx);
        if (graphIdx >= 1) {
            VLOGE(TAG, "No Inputs available for graphIdx: %d", graphIdx);
            returnStatus = Error::ERROR_PREDICT;
            break;
        }

        if (t_qnn_inputs == nullptr && t_qnn_outputs == nullptr) {
            if (qnn::tools::iotensor::StatusCode::SUCCESS
                != m_ioTensor.setupInputAndOutputTensors(&t_qnn_inputs, &t_qnn_outputs, (*m_graphsInfo)[graphIdx])) {
                VLOGE(TAG, "Error in setting up Input and output Tensors for graphIdx: %d", graphIdx);
                returnStatus = Error::ERROR_PREDICT;
                break;
            }
        }

        // auto inputFileList = m_inputFileLists[graphIdx];
        auto graphInfo = (*m_graphsInfo)[graphIdx];

        if (qnn::tools::iotensor::StatusCode::SUCCESS
            != m_ioTensor.populateInputTensors(graphIdx, qnn_inputs, t_qnn_inputs, graphInfo, m_inputDataType)) {
            VLOGE(TAG, "populateInputTensors Error!!!");
            returnStatus = Error::ERROR_PREDICT;
        }
        Qnn_ErrorHandle_t executeStatus = QNN_GRAPH_NO_ERROR;
        if (Error::OK == returnStatus) {
            t_numInputTensors = graphInfo.numInputTensors;
            t_numOutputTensors = graphInfo.numOutputTensors;

            PERF_TICK(perf, mPerfTag)
            int mid = modelInfo.id;
            if (mid == PerfUtil::qnnLoopModel) {
                while (true) {
                    executeStatus = m_qnnInterface.graphExecute(
                            graphInfo.graph, t_qnn_inputs, graphInfo.numInputTensors, t_qnn_outputs,
                            graphInfo.numOutputTensors, m_profileBackendHandle, nullptr);
                }
            } else {
                executeStatus = m_qnnInterface.graphExecute(graphInfo.graph, t_qnn_inputs, graphInfo.numInputTensors,
                                                            t_qnn_outputs, graphInfo.numOutputTensors,
                                                            m_profileBackendHandle, nullptr);
            }
            PERF_TOCK(perf, mPerfTag)
            if (QNN_GRAPH_NO_ERROR != executeStatus) {
                returnStatus = Error::ERROR_PREDICT;
            }
        }

        if (Error::OK != returnStatus) {
            VLOGE(TAG, "Execution of Graph: %d failed! ret=%d graphExecute=%d", graphIdx, returnStatus, executeStatus);
            break;
        }

        // get output layer blob name
        // 在qnn中部分网络层会被重命名.导致相同的模型 QNN 的网络层名称不一致. 所以增加qnn_output
        auto output_layer_names = modelInfo.port_desc.qnn_output_size() > 0 ? modelInfo.port_desc.qnn_output() :
                                                                              modelInfo.port_desc.output();

        if (graphInfo.numOutputTensors != output_layer_names.size()) {
            VLOGD(TAG, "output number can not match.");
            V_RET(Error::INVALID_PARAM);
        }
        // TODO: support only batch size 1
        // TODO: support only output datatype float
        for (size_t outputIdx = 0; outputIdx < graphInfo.numOutputTensors; outputIdx++) {
            std::string layer_name = graphInfo.outputTensors[outputIdx].v1.name;
            if (output_layer_names[outputIdx] != layer_name) {
                VLOGE(TAG, "Can not find output name: %s, require output name : %s for outputIdx: %d",
                      layer_name.c_str(), output_layer_names[outputIdx].c_str(), outputIdx);
                continue;
            }
            // VLOGD(TAG, "Writing output for outputIdx: %d", outputIdx);
            VTensor out_tensor;
            int w = 1, h = 1, c = 1;
            int length = 0;
            uint8_t *bufferToWrite = nullptr;
            std::vector<size_t> dims{};
            qnn::tools::datautil::StatusCode err{qnn::tools::datautil::StatusCode::SUCCESS};
            if (t_qnn_outputs != nullptr) {
                if (2 == t_qnn_outputs[outputIdx].v1.rank) {
                    h = t_qnn_outputs[outputIdx].v1.dimensions[0];
                    w = t_qnn_outputs[outputIdx].v1.dimensions[1];
                    dims.push_back(h);
                    dims.push_back(w);
                } else if (3 == t_qnn_outputs[outputIdx].v1.rank) {
                    h = t_qnn_outputs[outputIdx].v1.dimensions[0];
                    w = t_qnn_outputs[outputIdx].v1.dimensions[1];
                    c = t_qnn_outputs[outputIdx].v1.dimensions[2];
                    dims.push_back(h);
                    dims.push_back(w);
                    dims.push_back(c);
                } else if (4 == t_qnn_outputs[outputIdx].v1.rank) {
                    int n = t_qnn_outputs[outputIdx].v1.dimensions[0];
                    h = t_qnn_outputs[outputIdx].v1.dimensions[1];
                    w = t_qnn_outputs[outputIdx].v1.dimensions[2];
                    c = t_qnn_outputs[outputIdx].v1.dimensions[3];
                    dims.push_back(n);
                    dims.push_back(h);
                    dims.push_back(w);
                    dims.push_back(c);
                }
            }
            //TODO 模型推理的后处理逻辑需要优化.减少数据转化和内存拷贝的逻辑
            //uint8_t * pointer = static_cast<uint8_t *>(m_ioTensor.getTensorBuffer(t_qnn_outputs + outputIdx));
            // create vision tensor
            out_tensor.create(w, h, c, FP32, NHWC);
            if (t_qnn_outputs != nullptr) {
                std::tie(err, length) =
                        qnn::tools::datautil::calculateLength(dims, t_qnn_outputs[outputIdx].v1.dataType);
                if (t_qnn_outputs[outputIdx].v1.dataType == QNN_DATATYPE_FLOAT_32) {
                    // VLOGD(TAG, "Writing in output->dataType == QNN_DATATYPE_FLOAT_32");
                    //length *= sizeof(float);
#if defined(USE_QNN26) and defined(BUILD_QNX)
                    // 如果使用QNN2.6则直接获取共享内存里面的t_qnn_outputs数据
                    bufferToWrite = reinterpret_cast<uint8_t *>(m_ioTensor.getTensorBuffer(t_qnn_outputs + outputIdx));
#else
                    bufferToWrite = reinterpret_cast<uint8_t *>(t_qnn_outputs[outputIdx].v1.clientBuf.data);
#endif
                    memcpy(reinterpret_cast<char *>(out_tensor.data), reinterpret_cast<char *>(bufferToWrite), length);
                } else if (m_outputDataType == qnn::tools::iotensor::OutputDataType::FLOAT_ONLY) {
                    // VLOGD(TAG, "Writing in output->dataType == OutputDataType::FLOAT_ONLY");
                    // TODO 模型推理的后处理逻辑需要优化.减少数据转化和内存拷贝的逻辑
                    length *= sizeof(float);
                    float *floatBuffer = nullptr;
                    m_ioTensor.convertToFloat(&floatBuffer, static_cast<Qnn_Tensor_t *>(&t_qnn_outputs[outputIdx]));

                    uint8_t *bufferToWrite = reinterpret_cast<uint8_t *>(floatBuffer);
                    memcpy(reinterpret_cast<char *>(out_tensor.data), reinterpret_cast<char *>(bufferToWrite), length);
                    if (nullptr != floatBuffer) {
                        // VLOGD(TAG, "freeing floatBuffer");
                        free(floatBuffer);
                        floatBuffer = nullptr;
                    }
                }
            }
            DBG_PRINT_ARRAY((char *) out_tensor.data, 100, "QnnPredictor_doPredict_post_" + std::to_string(outputIdx));
            outputs.push_back(out_tensor);
        }
    }
    V_RET(returnStatus);
}

bool QnnPredictor::valid() {
    return mInited;
}

void QnnPredictor::logStdoutCallback(const char *fmt, QnnLog_Level_t level, uint64_t timestamp, va_list argp) {
    std::lock_guard<std::mutex> lock(gLogStdoutCallbackMutex);
    va_list args;
    va_copy(args, argp);
    vsprintf(gBuffer, fmt, args);
    va_end(args);
    switch (level) {
        case QNN_LOG_LEVEL_ERROR:
            VLOGE(TAG, "%s", gBuffer);
            break;
        case QNN_LOG_LEVEL_WARN:
            VLOGW(TAG, "%s", gBuffer);
            break;
        case QNN_LOG_LEVEL_INFO:
            VLOGI(TAG, "%s", gBuffer);
            break;
        case QNN_LOG_LEVEL_DEBUG:
            VLOGD(TAG, "%s", gBuffer);
            break;
        case QNN_LOG_LEVEL_VERBOSE:
            VLOGV(TAG, "%s", gBuffer);
            break;
        case QNN_LOG_LEVEL_MAX:
            VLOGD(TAG, "%s", gBuffer);
            break;
    }
}

void QnnPredictor::logging_initialize() {
    if (!qnn::log::initializeLogging()) {
        VLOGD(TAG, "QNN log init failed");
    }
    if (qnn::log::isLogInitialized()) {
        logCallback = logStdoutCallback;
        auto level = Logger::getLogLevel();
        QnnLog_Level_t qnnLogLevel;
        switch (level) {
            case LogLevel::FATAL:
            case LogLevel::ERROR:
                qnnLogLevel = QnnLog_Level_t::QNN_LOG_LEVEL_ERROR;
                break;
            case LogLevel::WARN:
                qnnLogLevel = QnnLog_Level_t::QNN_LOG_LEVEL_WARN;
                break;
            case LogLevel::INFO:
                qnnLogLevel = QnnLog_Level_t::QNN_LOG_LEVEL_INFO;
                break;
            case LogLevel::DEBUGGER:
                qnnLogLevel = QnnLog_Level_t::QNN_LOG_LEVEL_DEBUG;
                break;
            default:
                qnnLogLevel = QnnLog_Level_t::QNN_LOG_LEVEL_VERBOSE;
                break;
        }
        if (!qnn::log::setLogLevel(qnnLogLevel)) {
            VLOGD(TAG, "QNN log set error");
        }
        auto logLevel = qnn::log::getLogLevel();
        VLOGI(TAG, "Initializing logging in the backend. Callback: [%p], Log Level: [%d]", logCallback, logLevel);
        if (QNN_SUCCESS != m_qnnInterface.logCreate(logCallback, logLevel, &m_logHandle)) {
            VLOGD(TAG, "Unable to initialize logging in the backend.");
        }
    } else {
        VLOGD(TAG, "Logging not available in the backend.");
    }
}

StatusCode QnnPredictor::backend_initialize() {
    if (QNN_BACKEND_NO_ERROR
        != m_qnnInterface.backendCreate(m_logHandle, (const QnnBackend_Config_t **)m_backendConfig, &m_backendHandle)) {
        VLOGE(TAG, "Could not initialize backend");
        return StatusCode::FAIL_INIT_BACKEND;
    }
    m_isBackendInitialized = true;
    return StatusCode::SUCCESS;
}

std::string QnnPredictor::get_qnn_build_version() {
    return m_backendBuildId == nullptr ? std::string("") : std::string(m_backendBuildId);
}

StatusCode QnnPredictor::profiling_initialize() {
    if (ProfilingLevel::OFF != m_profilingLevel) {
        VLOGI(TAG, "Profiling turned on; level = %d", m_profilingLevel);
        if (ProfilingLevel::BASIC == m_profilingLevel) {
            VLOGI(TAG, "Basic profiling requested. Creating Qnn Profile object.");
            if (QNN_PROFILE_NO_ERROR
                != m_qnnInterface.profileCreate(m_backendHandle, QNN_PROFILE_LEVEL_BASIC, &m_profileBackendHandle)) {
                VLOGD(TAG, "Unable to create profile handle in the backend.");
                return StatusCode::FAIL_PROFILE_CREATE;
            }
        } else if (ProfilingLevel::DETAILED == m_profilingLevel) {
            VLOGI(TAG, "Detailed profiling requested. Creating Qnn Profile object.");
            if (QNN_PROFILE_NO_ERROR
                != m_qnnInterface.profileCreate(m_backendHandle, QNN_PROFILE_LEVEL_DETAILED, &m_profileBackendHandle)) {
                VLOGE(TAG, "Unable to create profile handle in the backend.");
                return StatusCode::FAIL_PROFILE_CREATE;
            }
        }
    }
    return StatusCode::SUCCESS;
}

// Register op packages and interface providers supplied during object creation.
// If there are multiple op packages, register them sequentially in the order provided.
StatusCode QnnPredictor::register_op_package() {
    const size_t pathIdx = 0;
    const size_t interfaceProviderIdx = 1;
    for (auto &opPackagePath : m_opPackagePaths) {
        std::vector<std::string> opPackage;
        opPackage.clear();

        std::istringstream tokenizedStringStream(opPackagePath);
        while (!tokenizedStringStream.eof()) {
            std::string value;
            getline(tokenizedStringStream, value, ':');
            if (!value.empty()) {
                opPackage.push_back(value);
            }
        }

        VLOGD(TAG, "opPackagePath: %s", opPackagePath.c_str());
        if (opPackage.size() != 2) {
            VLOGE(TAG, "Malformed opPackageString provided: %s", opPackagePath.c_str());
            return StatusCode::FAIL_OP_PACKAGE;
        }
        if (nullptr == m_qnnInterface.backendRegisterOpPackage) {
            VLOGE(TAG, "backendRegisterOpPackageFnHandle is nullptr.");
            return StatusCode::FAIL_OP_PACKAGE;
        }
        // 注册自定义算法子
        if (QNN_BACKEND_NO_ERROR
            != m_qnnInterface.backendRegisterOpPackage(m_backendHandle, (char *)opPackage[pathIdx].c_str(),
                                                       (char *)opPackage[interfaceProviderIdx].c_str(), nullptr)) {
            VLOGE(TAG, "Could not register Op Package: %s and interface provider: %s", opPackage[pathIdx].c_str(),
                  opPackage[interfaceProviderIdx].c_str());
            return StatusCode::FAIL_OP_PACKAGE;
        }
        VLOGI(TAG, "Registered Op Package: %s and interface provider: %s", opPackage[pathIdx].c_str(),
              opPackage[interfaceProviderIdx].c_str());
    }

    return StatusCode::SUCCESS;
}

StatusCode QnnPredictor::create_context(bool is_load_from_cached_binary, char *mem, int mem_len) {
    // create context from binary
    auto returnStatus = StatusCode::SUCCESS;
    if (mem == nullptr || mem_len == 0) {
        VLOGE(TAG, "No provided model binary");
        return StatusCode::FAILURE;
    }
    // 创建QNN推理的上下文对象
    if (is_load_from_cached_binary) {
        std::shared_ptr<uint8_t> buffer{nullptr};
        // 创建QNN系统接口的上下文Handler对象
        QnnSystemContext_Handle_t sysCtxHandle{nullptr};
        if (QNN_SUCCESS != m_qnnSystemInterface.systemContextCreate(&sysCtxHandle)) {
            VLOGE(TAG, "Could not create system handle.");
            returnStatus = StatusCode::FAILURE;
        }
        const QnnSystemContext_BinaryInfo_t *binaryInfo{nullptr};
        Qnn_ContextBinarySize_t binaryInfoSize{0};
        if (StatusCode::SUCCESS == returnStatus
            && QNN_SUCCESS
                       != m_qnnSystemInterface.systemContextGetBinaryInfo(sysCtxHandle, mem, mem_len, &binaryInfo,
                                                                          &binaryInfoSize)) {
            VLOGE(TAG, "Failed to get context binary info");
            returnStatus = StatusCode::FAILURE;
        }
        if (StatusCode::SUCCESS == returnStatus
            && !qnn::tools::sample_app::copyMetadataToGraphsInfo(binaryInfo, m_graphsInfo, m_graphsCount)) {
            VLOGE(TAG, "Failed to copy metadata.");
            returnStatus = StatusCode::FAILURE;
        }
        m_qnnSystemInterface.systemContextFree(sysCtxHandle);
        sysCtxHandle = nullptr;

        if (StatusCode::SUCCESS == returnStatus && nullptr == m_qnnInterface.contextCreateFromBinary) {
            VLOGE(TAG, "contextCreateFromBinaryFnHandle is nullptr.");
            returnStatus = StatusCode::FAIL_CONTEXT_FROM_BIN;
        }
        if (StatusCode::SUCCESS == returnStatus) {
            auto res = m_qnnInterface.contextCreateFromBinary(m_backendHandle, m_deviceHandle,
                                                              (const QnnContext_Config_t **)&m_contextConfig, mem,
                                                              mem_len, &m_context, m_profileBackendHandle);
            if (res) {
                VLOGE(TAG, "Could not create context from binary. contextCreateFromBinary: %d", res);
                returnStatus = StatusCode::FAIL_CONTEXT_FROM_BIN;
            }
        }
#if defined(USE_QNN26) and defined(BUILD_QNX)
        // @since Qnn2.6  QNN2.6新增的获取m_context的接口
        m_ioTensor.getContextInfo(&m_context);
#endif
        // 进行QNN模型的图展开
        if (StatusCode::SUCCESS == returnStatus) {
            for (size_t graphIdx = 0; graphIdx < m_graphsCount; graphIdx++) {
                if (nullptr == m_qnnInterface.graphRetrieve) {
                    VLOGE(TAG, "graphRetrieveFnHandle is nullptr.");
                    returnStatus = StatusCode::FAIL_CONTEXT_FROM_BIN;
                    break;
                }
                if (QNN_SUCCESS
                    != m_qnnInterface.graphRetrieve(m_context, (*m_graphsInfo)[graphIdx].graphName,
                                                    &((*m_graphsInfo)[graphIdx].graph))) {
                    VLOGE(TAG, "Unable to retrieve graph handle for graph Idx: %d", graphIdx);
                    returnStatus = StatusCode::FAIL_CONTEXT_FROM_BIN;
                }
            }
        }

        if (StatusCode::SUCCESS != returnStatus) {
            VLOGD(TAG, "Cleaning up graph Info structures.");
            qnn_wrapper_api::freeGraphsInfo(&m_graphsInfo, m_graphsCount);
            m_graphsInfo = nullptr;
            m_graphsCount = 0;
            return returnStatus;
        }
    } else {
        // create Context config
        if (QNN_CONTEXT_NO_ERROR
            != m_qnnInterface.contextCreate(m_backendHandle, m_deviceHandle,
                                            (const QnnContext_Config_t **)&m_contextConfig, &m_context)) {
            VLOGE(TAG, "Could not create context");
            return StatusCode::FAIL_CONTEXT_CREATE;
        }
        m_isContextCreated = true;

        // compose graph
        m_debug = false;
        if (qnn_wrapper_api::ModelError_t::MODEL_NO_ERROR
            != m_composeGraphsFnHandle(m_backendHandle, m_qnnInterface, m_context,
                                       (const qnn_wrapper_api::GraphConfigInfo_t **)m_graphConfigsInfo,
                                       m_graphConfigsInfoCount, &m_graphsInfo, &m_graphsCount, m_debug,
                                       qnn::log::getLogCallback(), qnn::log::getLogLevel())) {
            VLOGE(TAG, "Failed in composeGraphs()");
            return StatusCode::FAIL_COMPOSE_GRAPH;
        }
        // finalize graphs
        for (size_t graphIdx = 0; graphIdx < m_graphsCount; graphIdx++) {
            if (QNN_GRAPH_NO_ERROR
                != m_qnnInterface.graphFinalize((*m_graphsInfo)[graphIdx].graph, m_profileBackendHandle, nullptr)) {
                return StatusCode::FAIL_FINALIZE_GRAPH;
            }
        }
    }
    return StatusCode::SUCCESS;
}

std::vector<ModelInput> QnnPredictor::get_input_desc() const {
    std::vector<ModelInput> inputs;
    for (const auto &input : modelInfo.port_desc.input()) {
        inputs.emplace_back(input);
    }
    return inputs;
}

StatusCode QnnPredictor::isDevicePropertySupported() {
    if (nullptr != m_qnnInterface.propertyHasCapability) {
        auto qnnStatus = m_qnnInterface.propertyHasCapability(QNN_PROPERTY_GROUP_DEVICE);
        if (QNN_PROPERTY_NOT_SUPPORTED == qnnStatus || QNN_PROPERTY_ERROR_UNKNOWN_KEY == qnnStatus) {
            VLOGE(TAG, "Device property not supported or not known to backend");
            return StatusCode::FAILURE;
        }
    }
    return StatusCode::SUCCESS;
}

StatusCode QnnPredictor::initConfigId() {
    powerConfigId = 0;
    QnnDevice_Infrastructure_t deviceInfra = nullptr;
    Qnn_ErrorHandle_t devErr = m_qnnInterface.deviceGetInfrastructure(&deviceInfra);
    if (devErr != QNN_DEVICE_NO_ERROR) {
        VLOGE(TAG, "initConfigId deviceInfra is nullptr");
        return StatusCode::FAILURE;
    }
    QnnHtpDevice_Infrastructure_t *htpInfra = static_cast<QnnHtpDevice_Infrastructure_t *>(deviceInfra);
    if (htpInfra == nullptr) {
        VLOGE(TAG, "initConfigId QnnDevice_getInfrastructure returned nullptr\n");
        return StatusCode::FAILURE;
    }
    Qnn_ErrorHandle_t perfInfraErr = QNN_HTP_PERF_INFRASTRUCTURE_NO_ERROR;
    QnnHtpDevice_PerfInfrastructure_t perfInfra = htpInfra->perfInfra;
    perfInfraErr = perfInfra.createPowerConfigId(0, 0, &powerConfigId);
    if (perfInfraErr != QNN_HTP_PERF_INFRASTRUCTURE_NO_ERROR) {
        VLOGE(TAG, "initConfigId Perf Infra createPowerConfigId return failed: [%d]", perfInfraErr);
        return StatusCode::FAILURE;
    }
    return StatusCode::SUCCESS;
}

StatusCode QnnPredictor::createDevice() {
    // set HTP0 or HTP1
    if (RtConfig::scheduleHtp == HTP1) {
        deviceID = HTP1;
    } else {
        deviceID = HTP0;
    }
    // HTP configuration
    QnnDevice_CoreInfo_t coreSpec = QNN_DEVICE_CORE_INFO_INIT;
    coreSpec.v1.coreId = CORE_ID_0;
    coreSpec.v1.coreType = CORE_TYPE_0;
    QnnDevice_HardwareDeviceInfo_t deviceSpec = QNN_DEVICE_HARDWARE_DEVICE_INFO_INIT;
    deviceSpec.v1.deviceId = deviceID;
    deviceSpec.v1.deviceType = DEVICE_HTP;
    deviceSpec.v1.numCores = HTP_WITH_ONE_CORE;
    deviceSpec.v1.cores = &coreSpec;
    QnnDevice_PlatformInfo_t hardwareSpec = QNN_DEVICE_PLATFORM_INFO_INIT;
    hardwareSpec.v1.numHwDevices = HTP_WITH_ONE_HMX;
    hardwareSpec.v1.hwDevices = &deviceSpec;
    QnnDevice_Config_t hardwareConfig = QNN_DEVICE_CONFIG_INIT;
    hardwareConfig.option = QNN_DEVICE_CONFIG_OPTION_PLATFORM_INFO;
    hardwareConfig.hardwareInfo = &hardwareSpec;
    const QnnDevice_Config_t *configs[] = {&hardwareConfig, NULL};

    if (nullptr != m_qnnInterface.deviceCreate) {
        auto qnnStatus = m_qnnInterface.deviceCreate(m_logHandle, configs, &m_deviceHandle);
        if (QNN_SUCCESS != qnnStatus) {
            VLOGE(TAG, "Failed to create device");
            return StatusCode::FAIL_INIT_BACKEND;
        }
    }
    return StatusCode::SUCCESS;
}

StatusCode QnnPredictor::freeDevice() {
    if (nullptr != m_qnnInterface.deviceFree) {
        auto qnnStatus = m_qnnInterface.deviceFree(m_deviceHandle);
        if (QNN_SUCCESS != qnnStatus) {
            VLOGE(TAG, "Failed to free device");
            return StatusCode::FAIL_CONTEXT_FERR;
        }
    }
    return StatusCode::SUCCESS;
}

// Terminate backend
StatusCode QnnPredictor::freeBackend() {
    if (nullptr != m_qnnInterface.backendFree) {
        if (QNN_SUCCESS != m_qnnInterface.backendFree(m_backendHandle)) {
            VLOGE(TAG, "Failed to free backend");
            return StatusCode::FAIL_SYM_FUNCTION;
        }
    }
    m_isBackendInitialized = false;
    return StatusCode::SUCCESS;
}

DLayout QnnPredictor::getSupportedLayout() {
    return NHWC;
}

StatusCode QnnPredictor::setPowerLevel(uint32_t powerConfigId, uint32_t powerLevel) {
    VLOGI(TAG, "setPowerLevel powerConfigId[%d], powerLevel[%d]", powerConfigId, powerLevel);
    QnnHtpPerfInfrastructure_PowerConfig_t powerConfig;
    memset(&powerConfig, 0, sizeof(powerConfig));
    powerConfig.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
    powerConfig.dcvsV3Config.dcvsEnable = 0;
    powerConfig.dcvsV3Config.setDcvsEnable = 1;
    powerConfig.dcvsV3Config.contextId = powerConfigId;
    if (powerLevel == QNN_HTP_PERF_INFRASTRUCTURE_POWERLEVEL_POWER_SAVER
        || powerLevel == QNN_HTP_PERF_INFRASTRUCTURE_POWERLEVEL_LOW_POWER_SAVER
        || powerLevel == QNN_HTP_PERF_INFRASTRUCTURE_POWERLEVEL_HIGH_POWER_SAVER) {
        powerConfig.dcvsV3Config.powerMode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_POWER_SAVER_MODE;
    } else {
        powerConfig.dcvsV3Config.powerMode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
        powerConfig.dcvsV3Config.setSleepLatency = 1;
        powerConfig.dcvsV3Config.setBusParams = 1;
        powerConfig.dcvsV3Config.setCoreParams = 1;
        powerConfig.dcvsV3Config.sleepDisable = 0;
        powerConfig.dcvsV3Config.setSleepDisable = 0;
    }

    switch (powerLevel) {
        case QNN_HTP_PERF_INFRASTRUCTURE_POWERLEVEL_BURST:
            powerConfig.dcvsV3Config.sleepLatency = 65535;
            powerConfig.dcvsV3Config.busVoltageCornerMin = DCVS_VOLTAGE_VCORNER_TURBO_PLUS;
            powerConfig.dcvsV3Config.busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_TURBO_PLUS;
            powerConfig.dcvsV3Config.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_TURBO_PLUS;
            powerConfig.dcvsV3Config.coreVoltageCornerMin = DCVS_VOLTAGE_VCORNER_TURBO_PLUS;
            powerConfig.dcvsV3Config.coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_TURBO_PLUS;
            powerConfig.dcvsV3Config.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_TURBO_PLUS;
            break;
        case QNN_HTP_PERF_INFRASTRUCTURE_POWERLEVEL_SUSTAINED_HIGH_PERFORMANCE:
        case QNN_HTP_PERF_INFRASTRUCTURE_POWERLEVEL_HIGH_PERFORMANCE:
            powerConfig.dcvsV3Config.sleepLatency = 65535;
            powerConfig.dcvsV3Config.busVoltageCornerMin = DCVS_VOLTAGE_VCORNER_TURBO;
            powerConfig.dcvsV3Config.busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_TURBO;
            powerConfig.dcvsV3Config.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_TURBO;
            powerConfig.dcvsV3Config.coreVoltageCornerMin = DCVS_VOLTAGE_VCORNER_TURBO;
            powerConfig.dcvsV3Config.coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_TURBO;
            powerConfig.dcvsV3Config.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_TURBO;
            break;
        case QNN_HTP_PERF_INFRASTRUCTURE_POWERLEVEL_POWER_SAVER:
            powerConfig.dcvsV3Config.sleepLatency = 65535;
            powerConfig.dcvsV3Config.busVoltageCornerMin = DCVS_VOLTAGE_VCORNER_SVS;
            powerConfig.dcvsV3Config.busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_SVS;
            powerConfig.dcvsV3Config.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_SVS;
            powerConfig.dcvsV3Config.coreVoltageCornerMin = DCVS_VOLTAGE_VCORNER_SVS;
            powerConfig.dcvsV3Config.coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_SVS;
            powerConfig.dcvsV3Config.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_SVS;
            break;
        case QNN_HTP_PERF_INFRASTRUCTURE_POWERLEVEL_LOW_POWER_SAVER:
            powerConfig.dcvsV3Config.sleepLatency = 65535;
            powerConfig.dcvsV3Config.busVoltageCornerMin = DCVS_VOLTAGE_VCORNER_SVS2;
            powerConfig.dcvsV3Config.busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_SVS2;
            powerConfig.dcvsV3Config.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_SVS2;
            powerConfig.dcvsV3Config.coreVoltageCornerMin = DCVS_VOLTAGE_VCORNER_SVS2;
            powerConfig.dcvsV3Config.coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_SVS2;
            powerConfig.dcvsV3Config.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_SVS2;
            break;
        case QNN_HTP_PERF_INFRASTRUCTURE_POWERLEVEL_HIGH_POWER_SAVER:
            powerConfig.dcvsV3Config.sleepLatency = 65535;
            powerConfig.dcvsV3Config.busVoltageCornerMin = DCVS_VOLTAGE_VCORNER_SVS_PLUS;
            powerConfig.dcvsV3Config.busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_SVS_PLUS;
            powerConfig.dcvsV3Config.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_SVS_PLUS;
            powerConfig.dcvsV3Config.coreVoltageCornerMin = DCVS_VOLTAGE_VCORNER_SVS_PLUS;
            powerConfig.dcvsV3Config.coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_SVS_PLUS;
            powerConfig.dcvsV3Config.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_SVS_PLUS;
            break;
        case QNN_HTP_PERF_INFRASTRUCTURE_POWERLEVEL_LOW_BALANCED:
            powerConfig.dcvsV3Config.sleepLatency = 65535;
            powerConfig.dcvsV3Config.busVoltageCornerMin = DCVS_VOLTAGE_VCORNER_NOM;
            powerConfig.dcvsV3Config.busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_NOM;
            powerConfig.dcvsV3Config.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_NOM;
            powerConfig.dcvsV3Config.coreVoltageCornerMin = DCVS_VOLTAGE_VCORNER_NOM;
            powerConfig.dcvsV3Config.coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_NOM;
            powerConfig.dcvsV3Config.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_NOM;
            break;
        case QNN_HTP_PERF_INFRASTRUCTURE_POWERLEVEL_BALANCED:
            powerConfig.dcvsV3Config.sleepLatency = 65535;
            powerConfig.dcvsV3Config.busVoltageCornerMin = DCVS_VOLTAGE_VCORNER_NOM_PLUS;
            powerConfig.dcvsV3Config.busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_NOM_PLUS;
            powerConfig.dcvsV3Config.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_NOM_PLUS;
            powerConfig.dcvsV3Config.coreVoltageCornerMin = DCVS_VOLTAGE_VCORNER_NOM_PLUS;
            powerConfig.dcvsV3Config.coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_NOM_PLUS;
            powerConfig.dcvsV3Config.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_NOM_PLUS;
            break;
        default:
            VLOGE(TAG, "setPowerLevel Invalid powerlevel %d.", powerLevel);
            return StatusCode::FAILURE;
    }
    const QnnHtpPerfInfrastructure_PowerConfig_t *powerConfigs[] = {&powerConfig, NULL};
    QnnDevice_Infrastructure_t deviceInfra = nullptr;
    Qnn_ErrorHandle_t devErr = m_qnnInterface.deviceGetInfrastructure(&deviceInfra);
    if (devErr != QNN_DEVICE_NO_ERROR) {
        VLOGE(TAG, "setPowerLevel deviceInfra is nullptr");
        return StatusCode::FAILURE;
    }

    QnnHtpDevice_Infrastructure_t *htpInfra = static_cast<QnnHtpDevice_Infrastructure_t *>(deviceInfra);

    if (htpInfra == nullptr) {
        VLOGE(TAG, "setPowerLevel QnnDevice_getInfrastructure returned nullptr\n");
        return StatusCode::FAILURE;
    }

    Qnn_ErrorHandle_t perfInfraErr = QNN_HTP_PERF_INFRASTRUCTURE_NO_ERROR;
    QnnHtpDevice_PerfInfrastructure_t perfInfra = htpInfra->perfInfra;

    perfInfraErr = perfInfra.setPowerConfig(powerConfigId, powerConfigs);
    if (perfInfraErr != QNN_HTP_PERF_INFRASTRUCTURE_NO_ERROR) {
        VLOGE(TAG, "setPowerLevel Perf Infra set power config return: [%d]", perfInfraErr);
        return StatusCode::FAILURE;
    }
    VLOGI(TAG, "setPowerLevel Perf Infra set power config return: success [%d]", perfInfraErr);
    return StatusCode::SUCCESS;
}

bool QnnPredictor::onInferenceCmd(int cmd) {
    switch (cmd) {
        case CMD_INFERENCE_POWER_DEFAULT: {
            setPowerLevel(powerConfigId, QNN_HTP_PERF_INFRASTRUCTURE_POWERLEVEL_DEFAULT);
            return true;
        };
        case CMD_INFERENCE_POWER_BUST: {
            setPowerLevel(powerConfigId, QNN_HTP_PERF_INFRASTRUCTURE_POWERLEVEL_BURST);
            return true;
        };
        case CMD_INFERENCE_POWER_POWER_SAVE: {
            setPowerLevel(powerConfigId, QNN_HTP_PERF_INFRASTRUCTURE_POWERLEVEL_POWER_SAVER);
            return true;
        }
        default:
            VLOGE(TAG, "onInferenceCmd error cmd:[%d]", cmd);
    }
    return false;
}

} // namespace vision
