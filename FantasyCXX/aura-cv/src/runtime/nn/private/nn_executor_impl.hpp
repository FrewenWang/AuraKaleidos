#ifndef AURA_RUNTIME_NN_NN_EXECUTOR_IMPL_HPP__
#define AURA_RUNTIME_NN_NN_EXECUTOR_IMPL_HPP__

#include "aura/runtime/nn/nn_engine.hpp"
#include "aura/runtime/worker_pool.h"
#include "nn_deserialize.hpp"
#include "nn_model.hpp"
#include "nb_model.hpp"

namespace aura
{

enum class NNBackend
{
    CPU = 0,              /*!< CPU backend */
    GPU,                  /*!< GPU backend */
    NPU                   /*!< NPU backend */
};

enum class NNPerfLevel
{
    PERF_DEFAULT = 0,        /*!< Default performance level */
    PERF_LOW,                /*!< Low performance level     */
    PERF_NORMAL,             /*!< Normal performance level  */
    PERF_HIGH,               /*!< High performance level    */

    QNN_PERF_CUSTOM_0 = 100
};

enum class NNProfilingLevel
{
    PROFILING_OFF = 0,    /*!< Profiling is turned off */
    PROFILING_BASIC,      /*!< Basic profiling         */
    PROFILING_DETAILED    /*!< Detailed profiling      */
};

enum class NNLogLevel
{
    LOG_ERROR = 0,        /*!< ERROR log level*/
    LOG_INFO,             /*!< INFO log level */
    LOG_DEBUG             /*!< DEBUG log level*/
};

AURA_INLINE DT_VOID NNStringToLower(std::string &str)
{
    std::transform(str.begin(), str.end(), str.begin(), ::tolower);
}

AURA_INLINE NNPerfLevel GetNNPerfLevel(const NNConfig &config)
{
    NNPerfLevel nn_perf_level = NNPerfLevel::PERF_HIGH;

    if (config.count("perf_level"))
    {
        std::string value = config.at("perf_level");

        NNStringToLower(value);

        if (std::string("perf_high") == value)
        {
            nn_perf_level = NNPerfLevel::PERF_HIGH;
        }
        else if (std::string("perf_normal") == value)
        {
            nn_perf_level = NNPerfLevel::PERF_NORMAL;
        }
        else if (std::string("perf_low") == value)
        {
            nn_perf_level = NNPerfLevel::PERF_LOW;
        }
        else if (std::string("qnn_perf_custom_0") == value)
        {
            nn_perf_level = NNPerfLevel::QNN_PERF_CUSTOM_0;
        }
    }

    return nn_perf_level;
}

AURA_INLINE Status GetNNPerfLevel(const Context *ctx, const AnyParams &params, NNPerfLevel &nn_perf_level)
{
    nn_perf_level = NNPerfLevel::PERF_HIGH;

    std::string value;
    Status status = params.Get("perf_level", value);
    if (status != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "AnyParams get failed, must set perf_level");
        return status;
    }

    NNStringToLower(value);

    if (std::string("perf_high") == value)
    {
        nn_perf_level = NNPerfLevel::PERF_HIGH;
    }
    else if (std::string("perf_normal") == value)
    {
        nn_perf_level = NNPerfLevel::PERF_NORMAL;
    }
    else if (std::string("perf_low") == value)
    {
        nn_perf_level = NNPerfLevel::PERF_LOW;
    }
    else if (std::string("qnn_perf_custom_0") == value)
    {
        nn_perf_level = NNPerfLevel::QNN_PERF_CUSTOM_0;
    }
    else
    {
        std::string info = "nn_perf_level set wrong value " + value;
        AURA_ADD_ERROR_STRING(ctx, info.c_str());
        return Status::ERROR;
    }

    return Status::OK;
}

AURA_INLINE NNProfilingLevel GetNNProfilingLevel(const NNConfig &config)
{
    NNProfilingLevel nn_profiling_level = NNProfilingLevel::PROFILING_OFF;

    if (config.count("profiling_level"))
    {
        std::string value = config.at("profiling_level");

        NNStringToLower(value);

        if (value == std::string("profiling_off"))
        {
            nn_profiling_level = NNProfilingLevel::PROFILING_OFF;
        }
        else if (value == std::string("profiling_basic"))
        {
            nn_profiling_level = NNProfilingLevel::PROFILING_BASIC;
        }
        else if (value == std::string("profiling_detailed"))
        {
            nn_profiling_level = NNProfilingLevel::PROFILING_DETAILED;
        }
    }

    return nn_profiling_level;
}

AURA_INLINE NNLogLevel GetNNLogLevel(const NNConfig &config)
{
    NNLogLevel nn_log_level = NNLogLevel::LOG_ERROR;

    if (config.count("log_level"))
    {
        std::string value = config.at("log_level");

        NNStringToLower(value);

        if (value == std::string("log_error"))
        {
            nn_log_level = NNLogLevel::LOG_ERROR;
        }
        else if (value == std::string("log_info"))
        {
            nn_log_level = NNLogLevel::LOG_INFO;
        }
        else if (value == std::string("log_debug"))
        {
            nn_log_level = NNLogLevel::LOG_DEBUG;
        }
    }

    return nn_log_level;
}

AURA_INLINE std::string GetNNProfilingPath(const NNConfig &config)
{
    std::string nn_profiling_path = std::string();

    std::string key = "profiling_path";

    if (config.count("profiling_path"))
    {
        nn_profiling_path = config.at("profiling_path");
    }

    return nn_profiling_path;
}

AURA_INLINE NNBackend GetNNBackend(const NNConfig &config)
{
    NNBackend backend = NNBackend::NPU;

    if (config.count("backend"))
    {
        std::string value = config.at("backend");

        NNStringToLower(value);

        if (value == std::string("cpu"))
        {
            backend = NNBackend::CPU;
        }
        else if (value == std::string("gpu"))
        {
            backend = NNBackend::GPU;
        }
    }

    return backend;
}

AURA_INLINE DT_S32 GetHtpMemStepSize(const NNConfig &config)
{
    DT_S32 mem_step_size = 0;

    if (config.count("htp_mem_step_size"))
    {
        std::string value = config.at("htp_mem_step_size");

        mem_step_size = std::stoi(value);
    }

    return mem_step_size;
}

AURA_INLINE DT_BOOL GetHtpAsyncCall(const NNConfig &config)
{
    DT_BOOL htp_async_call_flag = DT_TRUE;

    if (config.count("htp_async_call"))
    {
        std::string value = config.at("htp_async_call");

        NNStringToLower(value);

        if (value == std::string("false"))
        {
            htp_async_call_flag = DT_FALSE;
        }
    }

    return htp_async_call_flag;
}

AURA_INLINE std::ostream& operator<<(std::ostream &os, NNBackend backend)
{
    switch (backend)
    {
        case NNBackend::GPU:
        {
            os << "gpu";
            break;
        }

        case NNBackend::NPU:
        {
            os << "npu";
            break;
        }

        default:
        {
            os << "cpu";
            break;
        }
    }

    return os;
}

AURA_INLINE std::string NNBackendToString(NNBackend backend)
{
    std::ostringstream ss;
    ss << backend;
    return ss.str();
}

AURA_INLINE std::ostream& operator<<(std::ostream &os, NNPerfLevel perf_level)
{
    switch (perf_level)
    {
        case NNPerfLevel::PERF_LOW:
        {
            os << "PERF_LOW";
            break;
        }

        case NNPerfLevel::PERF_NORMAL:
        {
            os << "PERF_NORMAL";
            break;
        }

        case NNPerfLevel::PERF_HIGH:
        {
            os << "PERF_HIGH";
            break;
        }

        default:
        {
            os << "PERF_DEFAULT";
            break;
        }
    }

    return os;
}

AURA_INLINE std::string NNPerfLevelToString(NNPerfLevel perf_level)
{
    std::ostringstream ss;
    ss << perf_level;
    return ss.str();
}

AURA_INLINE std::ostream& operator<<(std::ostream &os, NNProfilingLevel profiling_level)
{
    switch (profiling_level)
    {
        case NNProfilingLevel::PROFILING_BASIC:
        {
            os << "PROFILING_BASIC";
            break;
        }

        case NNProfilingLevel::PROFILING_DETAILED:
        {
            os << "PROFILING_DETAILED";
            break;
        }

        default:
        {
            os << "PROFILING_OFF";
            break;
        }
    }

    return os;
}

AURA_INLINE std::ostream& operator<<(std::ostream &os, NNLogLevel log_level)
{
    switch (log_level)
    {
        case NNLogLevel::LOG_DEBUG:
        {
            os << "LOG_DEBUG";
            break;
        }

        case NNLogLevel::LOG_INFO:
        {
            os << "LOG_INFO";
            break;
        }

        default:
        {
            os << "LOG_ERROR";
            break;
        }
    }

    return os;
}

AURA_INLINE std::string NNProfilingLevelToString(NNProfilingLevel profiling_level)
{
    std::ostringstream ss;
    ss << profiling_level;
    return ss.str();
}

AURA_INLINE std::string NNLogLevelToString(NNLogLevel log_level)
{
    std::ostringstream ss;
    ss << log_level;
    return ss.str();
}

class NNLibrary
{
public:
    NNLibrary()
    {}

    virtual ~NNLibrary()
    {}

    virtual Status Load() = 0;

    virtual Status UnLoad() = 0;

    virtual Status Destroy() = 0;
};

class NNLibraryManager
{
public:
    static NNLibraryManager& GetInstance()
    {
        static NNLibraryManager instance;
        return instance;
    }

    DT_VOID AddRefCount()
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_ref_count++;
    }

    DT_VOID AddNNLibrary(NNLibrary *library)
    {
        if (library)
        {
            m_librarys.push_back(library);
        }
    }

    DT_VOID Destroy()
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_ref_count--;

        if (m_ref_count <= 0)
        {
            for (auto &library : m_librarys)
            {
                library->Destroy();
            }
            m_librarys.clear();
        }
    }
private:
    NNLibraryManager() : m_librarys(), m_ref_count(0), m_mutex()
    {}

    ~NNLibraryManager()
    {
        Destroy();
    }

    AURA_DISABLE_COPY_AND_ASSIGN(NNLibraryManager);
private:
    std::vector<NNLibrary*> m_librarys;
    DT_S32 m_ref_count;
    std::mutex m_mutex;
};

class NNExecutorImpl : public NNExecutor
{
public:
    NNExecutorImpl(Context *ctx) : NNExecutor(ctx), m_is_valid(DT_FALSE)
    {
#if defined(AURA_BUILD_HOST)
        m_wp.reset(new WorkerPool(ctx, AURA_TAG, CpuAffinity::ALL, CpuAffinity::ALL, 0, 1));
#endif
    }

    Status Update(const std::string &name, AnyParams &params) override
    {
        AURA_UNUSED(name);
        AURA_UNUSED(params);
        return Status::OK;
    }

protected:
    DT_BOOL                     m_is_valid;
    std::future<Status>         m_token;
    std::shared_ptr<WorkerPool> m_wp;
};

class NNExecutorInterface : public NNExecutor
{
public:
    NNExecutorInterface(Context *ctx) : NNExecutor(ctx), m_ready(DT_FALSE)
    {}

    Status Initialize() override
    {
        Status ret = Status::ERROR;
        if (m_ctx && m_impl)
        {
            if (DT_FALSE == m_ready)
            {
                ret = m_impl->Initialize();
                if (Status::OK == ret)
                {
                    m_ready = DT_TRUE;
                }
            }
            else
            {
                ret = Status::OK;
            }
        }
        return ret;
    }

    Status Update(const std::string &name, AnyParams &params) override
    {
        Status ret = Status::ERROR;
        if (m_ctx && m_impl)
        {
            if (DT_TRUE == m_ready)
            {
                ret = m_impl->Update(name, params);
            }
        }
        return ret;
    }

    Status Forward(const MatMap &input, MatMap &output, DT_S32 graph_id) override
    {
        Status ret = Status::ERROR;
        if (m_ctx && m_impl)
        {
            if (DT_TRUE == m_ready)
            {
                ret = m_impl->Forward(input, output, graph_id);
            }
        }
        return ret;
    }

    std::vector<TensorDescMap> GetInputs() override
    {
        if (m_ctx && m_impl)
        {
            if (DT_TRUE == m_ready)
            {
                return m_impl->GetInputs();
            }
        }
        return {};
    }

    std::vector<TensorDescMap> GetOutputs() override
    {
        if (m_ctx && m_impl)
        {
            if (DT_TRUE == m_ready)
            {
                return m_impl->GetOutputs();
            }
        }
        return {};
    }

    std::string GetVersion() override
    {
        if (m_ctx && m_impl)
        {
            return m_impl->GetVersion();
        }
        else
        {
            return std::string();
        }
    }

protected:
    DT_BOOL                         m_ready;
    std::shared_ptr<NNExecutorImpl> m_impl;
};

} // namespace aura

#endif // AURA_RUNTIME_NN_NN_EXECUTOR_IMPL_HPP__