#ifndef AURA_RUNTIME_CONTEXT_HOST_CONFIG_HPP__
#define AURA_RUNTIME_CONTEXT_HOST_CONFIG_HPP__

#include "aura/runtime/core.h"

#include <string>

/**
 * @defgroup runtime Runtime
 * @{
 *      @defgroup context Context
 *      @{
 *          @defgroup context_host Context Host
 *      @}
 * @}
*/

namespace aura
{
/**
 * @addtogroup context_host
 * @{
*/

/**
 * @brief Enumeration representing CPU affinity options.
 */
enum class CpuAffinity
{
    ALL    = 0, /*!< The initiate threads with no CPU affinity */
    BIG   ,     /*!< The all big cores are used */
    LITTLE,     /*!< The all small cores are used */
};

/**
 * @brief Enumeration representing OpenCL performance levels.
 */
enum class CLPerfLevel
{
    PERF_DEFAULT    = 0,        /*!< Default performance level. */
    PERF_LOW,                   /*!< Low performance level. */
    PERF_NORMAL,                /*!< Normal performance level. */
    PERF_HIGH,                  /*!< High performance level. */
};

/**
 * @brief Enumeration representing OpenCL priority levels.
 */
enum class CLPriorityLevel
{
    PRIORITY_DEFAULT    = 0,    /*!< Default priority level. */
    PRIORITY_LOW,               /*!< Low priority level. */
    PRIORITY_NORMAL,            /*!< Normal priority level. */
    PRIORITY_HIGH,              /*!< High priority level. */
};

/**
 * @brief Enumeration representing types of OpenCL precompiled sources.
 */
enum class CLPrecompiledType
{
    INVALID    = 0, /*!< Invalid precompiled type. */
    PATH,           /*!< Precompiled sources provided as file path. */
    STRING,         /*!< Precompiled sources provided as a string. */
};

/**
 * @brief Enumeration representing Vdsp priority levels.
 */
enum class XtensaPriorityLevel
{
    PRIORITY_DEFAULT    = 0,    /*!< Default priority level. */
    PRIORITY_LOW,               /*!< Low priority level. */
    PRIORITY_NORMAL,            /*!< Normal priority level. */
    PRIORITY_HIGH,              /*!< High priority level. */
};

/**
 * @brief Configuration class for AURA library.
 *
 * This class provides a way to configure various aspects of the AURA library.
 */
class AURA_EXPORTS Config
{
public:
    /**
     * @brief Default constructor for Config class.
     *
     * Initializes default values for configuration options.
     */
    Config() : m_log_output(LogOutput::STDOUT),
               m_log_level(LogLevel::DEBUG),
               m_log_file(""),
               m_thread_tag(""),
               m_compute_affinity(CpuAffinity::ALL),
               m_async_affinity(CpuAffinity::ALL),
               m_compute_threads(0),
               m_async_threads(0),
               m_enable_systrace(MI_FALSE),
               m_enable_cl(MI_FALSE),
               m_enable_hexagon(MI_FALSE),
               m_enable_nn(MI_FALSE),
               m_enable_xtensa(MI_FALSE)
    {}

    ~Config() = default;

    /**
     * @brief Set log configuration options.
     *
     * @param output Log output destination.
     * @param level Log level.
     * @param file Log file path.
     * 
     * @return Reference to the modified Config object.
     */
    Config& SetLog(LogOutput output, LogLevel level, const std::string &file = std::string())
    {
        m_log_output    = output;
        m_log_level     = level;
        if (!file.empty())
        {
            m_log_file = file;
        }

        return *this;
    }

    /**
     * @brief Set worker pool configuration options.
     *
     * @param thread_tag Thread tag.
     * @param compute_affinity CPU affinity for compute threads.
     * @param async_affinity CPU affinity for asynchronous threads.
     * 
     * @return Reference to the modified Config object.
     */
    Config& SetWorkerPool(const std::string &thread_tag, CpuAffinity compute_affinity = CpuAffinity::ALL,
                          CpuAffinity async_affinity = CpuAffinity::ALL, MI_S32 compute_threads = 0, MI_S32 async_threads = 0)
    {
        // max_16 bytes: [thread_tag][A/C(thread_type)][idx]['\0']; [12] + [1 + 2 + 1](reserved)
        m_thread_tag       = thread_tag.size() > 12 ? thread_tag.substr(0, 12) : thread_tag;
        m_compute_affinity = compute_affinity;
        m_async_affinity   = async_affinity;
        m_compute_threads  = compute_threads;
        m_async_threads    = async_threads;

        return *this;
    }

    /**
     * @brief Set system trace configuration option.
     *
     * @param enable_syatrace Flag to enable system trace.
     * 
     * @return Reference to the modified Config object.
     */
    Config& SetSysTrace(MI_BOOL enable_syatrace)
    {
        m_enable_systrace = enable_syatrace;

        return *this;
    }

    /**
     * @brief Set OpenCL configuration options.
     *
     * @param enable_cl Flag to enable OpenCL.
     * @param cache_path Cache path for OpenCL.
     * @param cache_prefix Cache prefix for OpenCL.
     * @param cl_precompiled_type Type of OpenCL precompiled sources.
     * @param precompiled_sources Precompiled sources for OpenCL.
     * @param extern_version External version for OpenCL kernel version detection.
     * @param cl_perf_level Performance level for OpenCL.
     * @param cl_priority_level Priority level for OpenCL.
     * 
     * @return Reference to the modified Config object.
     */
    Config& SetCLConf(MI_BOOL enable_cl, const std::string &cache_path, const std::string &cache_prefix,
                      CLPrecompiledType cl_precompiled_type = CLPrecompiledType::INVALID,
                      const std::string &precompiled_sources = std::string(),
                      const std::string &extern_version = std::string(),
                      CLPerfLevel cl_perf_level = CLPerfLevel::PERF_HIGH,
                      CLPriorityLevel cl_priority_level = CLPriorityLevel::PRIORITY_LOW)
    {
        m_enable_cl              = enable_cl;
        m_cl_cache_path          = cache_path;
        m_cl_cache_prefix        = cache_prefix;
        m_cl_precompiled_type    = cl_precompiled_type;
        m_cl_precompiled_sources = precompiled_sources;
        m_cl_external_version    = extern_version;
        m_cl_perf_level          = cl_perf_level;
        m_cl_priority_level      = cl_priority_level;

        return *this;
    }

    /**
     * @brief Set Hexagon configuration options.
     *
     * @param enable_hexagon Flag to enable Hexagon.
     * @param unsigned_pd Unsigned PD for signature.
     * @param lib_prefix Library prefix for Hexagon.
     * @param output Log output destination for Hexagon.
     * @param level Log level for Hexagon.
     * @param file Log file path for Hexagon.
     * 
     * @return Reference to the modified Config object.
     */
    Config& SetHexagonConf(MI_BOOL enable_hexagon, MI_BOOL unsigned_pd, const std::string &lib_prefix, MI_BOOL async_call = MI_TRUE,
                           LogOutput output = LogOutput::FARF, LogLevel level = LogLevel::DEBUG, const std::string &file = std::string())
    {
        m_enable_hexagon     = enable_hexagon;
        m_unsigned_pd        = unsigned_pd;
        m_hexagon_lib_prefix = lib_prefix;
        m_async_call         = async_call;
        m_hexagon_log_output = output;
        m_hexagon_log_level  = level;
        m_hexagon_log_file   = file;

        return *this;
    }

    /**
     * @brief Set NN configuration options.
     *
     * @param enable_nn Flag to enable NN.
     * 
     * @return Reference to the modified Config object.
     */
    Config& SetNNConf(MI_BOOL enable_nn)
    {
        m_enable_nn = enable_nn;

        return *this;
    }

    /**
     * @brief Set Xtensa configuration options.
     *
     * @param enable_xtensa Flag to enable Xtensa.
     * @param pil_name Pil name for Xtensa.
     * 
     * @return Reference to the modified Config object.
     */
    Config& SetXtensaConf(MI_BOOL enable_xtensa, const std::string &pil_name = std::string(), 
                          XtensaPriorityLevel xtensa_priority_level = XtensaPriorityLevel::PRIORITY_LOW)
    {
        m_enable_xtensa         = enable_xtensa;
        m_pil_name              = pil_name;
        m_xtensa_priority_level = xtensa_priority_level;

        return *this;
    }

    LogOutput           m_log_output;             /*!< Log output destination. */
    LogLevel            m_log_level;              /*!< Log level. */
    std::string         m_log_file;               /*!< Log file path. */
    std::string         m_thread_tag;             /*!< Thread tag. */
    CpuAffinity         m_compute_affinity;       /*!< CPU affinity for compute threads. */
    CpuAffinity         m_async_affinity;         /*!< CPU affinity for asynchronous threads. */
    MI_S32              m_compute_threads;        /*!< number of compute threads. */
    MI_S32              m_async_threads;          /*!< number of async   threads. */
    MI_BOOL             m_enable_systrace;        /*!< Flag to enable system trace. */

    MI_BOOL             m_enable_cl;              /*!< Flag to enable OpenCL. */
    std::string         m_cl_cache_path;          /*!< Cache path for OpenCL. */
    std::string         m_cl_cache_prefix;        /*!< Cache prefix for OpenCL. */
    CLPrecompiledType   m_cl_precompiled_type;    /*!< Type of OpenCL precompiled sources. */
    std::string         m_cl_precompiled_sources; /*!< Precompiled sources for OpenCL. */
    std::string         m_cl_external_version;    /*!< For external reference cl kernel version detection. */
    CLPerfLevel         m_cl_perf_level;          /*!< Performance level for OpenCL. */
    CLPriorityLevel     m_cl_priority_level;      /*!< Priority level for OpenCL. */

    MI_BOOL             m_enable_hexagon;         /*!< Flag to enable Hexagon. */
    MI_BOOL             m_async_call;             /*!< Flag to enable use async thread call fastrpc. */
    LogOutput           m_hexagon_log_output;     /*!< Log output destination for Hexagon. */
    LogLevel            m_hexagon_log_level;      /*!< Log level for Hexagon. */
    std::string         m_hexagon_log_file;       /*!< Log file path for Hexagon. */
    MI_BOOL             m_unsigned_pd;            /*!< Unsigned PD for Hexagon signature. */
    std::string         m_hexagon_lib_prefix;     /*!< Library prefix for Hexagon. */

    MI_BOOL             m_enable_nn;              /*!< Flag to enable NN. */

    MI_BOOL             m_enable_xtensa;          /*!< Flag to enable Xtensa. */
    std::string         m_pil_name;               /*!< Pil name for Xtensa. */
    XtensaPriorityLevel m_xtensa_priority_level;  /*!< Task priority level for VDSP. */
};

/**
 * @}
*/
} // namespace aura

#endif // AURA_RUNTIME_CONTEXT_HOST_CONFIG_HPP__