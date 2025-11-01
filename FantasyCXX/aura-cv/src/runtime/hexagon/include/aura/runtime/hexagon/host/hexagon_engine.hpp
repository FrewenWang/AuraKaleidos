#ifndef AURA_RUNTIME_HEXAGON_HOST_HEXAGON_ENGINE_HPP__
#define AURA_RUNTIME_HEXAGON_HOST_HEXAGON_ENGINE_HPP__

#include "aura/runtime/hexagon/rpc_param.hpp"

/**
 * @defgroup runtime Runtime
 * @{
 *    @defgroup hexagon Hexagon
 *    @{
 *       @defgroup Hexagon_host Hexagon Host
 *    @}
 * @}
 */

namespace aura
{
/**
 * @addtogroup Hexagon_host
 * @{
 */

/**
 * @brief Structure representing profiling information for Hexagon.
 */
struct AURA_EXPORTS HexagonProfiling
{
    MI_U64 rpc_time;  /*!< Time taken by the Remote Procedure Call (RPC). */
    MI_U64 skel_time; /*!< Skeleton execution time. */
    MI_U64 clk_mhz;   /*!< Clock speed in megahertz (MHz). */
};

AURA_INLINE std::ostream& operator<<(std::ostream &os, HexagonProfiling profiling)
{
    os << "HexagonProfiling:" << std::endl
       << "  |- rpc_time       : " << profiling.rpc_time  << std::endl
       << "  |- skel_time      : " << profiling.skel_time << std::endl
       << "  |- clk_mhz        : " << profiling.clk_mhz   << std::endl;
    return os;
}

AURA_INLINE std::string HexagonProfilingToString(const HexagonProfiling &profiling)
{
    std::ostringstream oss;
    oss << profiling;
    return oss.str();
}

/**
 * @brief Enumeration representing Hexagon runtime query types.
 */
enum class HexagonRTQueryType
{
    CURRENT_FREQ = 0, /*!< Query current HTP frequency in MHz. */
    VTCM_INFO,        /*!< Query HTP VTCM layout information. */
    HTP_STATUS,       /*!< Query HTP current status. */
};

AURA_INLINE std::ostream& operator<<(std::ostream &os, HexagonRTQueryType type)
{
    switch (type)
    {
        case HexagonRTQueryType::CURRENT_FREQ:
        {
            os << "CurrentFreq";
            break;
        }
        case HexagonRTQueryType::VTCM_INFO:
        {
            os << "VtcmInfo";
            break;
        }
        case HexagonRTQueryType::HTP_STATUS:
        {
            os << "HtpStatus";
            break;
        }
        default:
        {
            break;
        }
    }

    return os;
}

AURA_INLINE std::string RTQueryTypeToString(const HexagonRTQueryType &type)
{
    std::ostringstream oss;
    oss << type;
    return oss.str();
}

/**
 * @brief Structure representing the layout of the VTCM (Vector Tightly Coupled Memory).
 */
struct AURA_EXPORTS VtcmLayout
{
    MI_S32 total_vtcm_size; /*!< Total VTCM size. */
    MI_S32 page_list_count; /*!< Count of VTCM page lists. */
    MI_S32 page_sizes[16];  /*!< VTCM page sizes (in KB). */
    MI_S32 page_count[16];  /*!< VTCM page counts. */
};

AURA_INLINE std::ostream& operator<<(std::ostream &os, const VtcmLayout &layout)
{
    os << "VtcmLayout: " << " ";

    os << "[ ";
    for (MI_S32 i = 0; i < layout.page_list_count; ++i)
    {
        os << layout.page_sizes[i] << "KBx" << layout.page_count[i] << " ";
    }
    os << "]";

    return os;
}

AURA_INLINE std::string VtcmLayoutToString(const VtcmLayout &layout)
{
    std::ostringstream oss;
    oss << layout;
    return oss.str();
}

/**
 * @brief Structure representing hardware information for Hexagon.
 */
struct AURA_EXPORTS HardwareInfo
{
    MI_S32 arch_version;    /*!< Hexagon architecture version. */
    MI_S32 num_hvx_units;   /*!< Number of HVX (Hexagon Vector eXtension) units. */
    VtcmLayout vtcm_layout; /*!< Layout information for VTCM. */
};

/**
 * @brief Union representing real-time information for Hexagon.
 */
union RealTimeInfo
{
    MI_F32 cur_freq;        /*!< Current frequency in MHz. */
    VtcmLayout vtcm_layout; /*!< VTCM layout information. */
    // remote_rpc_status_flags_t user_pd_status; // Current unused
};

/**
 * @brief Implementation class for HexagonEngine.
 *
 * This class encapsulates the private implementation details of HexagonEngine, providing an
 * interface for interacting with Hexagon hardware acceleration. It allows users to enable
 * Hexagon acceleration, set power levels, run specific operations, query hardware information,
 * and obtain real-time information about the Hexagon device.
 */
class AURA_EXPORTS HexagonEngine
{
public:
    /**
     * @brief Constructor for HexagonEngine.
     *
     * @param ctx The pointer to the Context object.
     * @param enable_hexagon Whether to enable Hexagon acceleration.
     * @param unsigned_pd Whether to use unsigned Hexagon platform.
     * @param lib_prefix The library prefix for Hexagon.
     * @param output The log output destination.
     * @param level The log level.
     * @param file The log file path.
     */
    HexagonEngine(Context *ctx,
                  MI_BOOL enable_hexagon,
                  MI_BOOL unsigned_pd,
                  const std::string &lib_prefix,
                  MI_BOOL async_call,
                  LogOutput ouput,
                  LogLevel level,
                  const std::string &file);

    /**
     * @brief Destructor for HexagonEngine.
     */
    ~HexagonEngine();

    AURA_DISABLE_COPY_AND_ASSIGN(HexagonEngine);

    /**
     * @brief Set the power level for Hexagon device.
     *
     * @param target_level The target power level.
     * @param enable_dcvs Whether to enable dynamic voltage and frequency scaling.
     * @param client_id power client id value, default 0 .
     *
     * @return Status::OK if successful; otherwise, an appropriate error status.
     */
    Status SetPower(HexagonPowerLevel target_level, MI_BOOL enable_dcvs, MI_U32 client_id = 0);

    /**
     * @brief Run the Hexagon Engine with the specified parameters.
     *
     * @param package The name of the package.
     * @param module The name of the operation module.
     * @param rpc_param The HexagonRpcParam object containing operation parameters.
     * @param profiling Optional pointer to HexagonProfiling object for profiling information (default is MI_NULL).
     *
     * @return Status::OK if the operation is successful; otherwise, an appropriate error status.
     */
    Status Run(const std::string &package, const std::string &module, HexagonRpcParam &rpc_param, HexagonProfiling *profiling = MI_NULL) const;

    /**
     * @brief Get the version information for Hexagon Engine.
     *
     * @return The version information as a string.
     */
    std::string GetVersion() const;

    /**
     * @brief Query hardware information for Hexagon.
     *
     * @param info The HardwareInfo structure to store the information.
     *
     * @return Status::OK if successful; otherwise, an appropriate error status.
     */
    Status QueryHWInfo(HardwareInfo &info);

    /**
     * @brief Query real-time information for Hexagon.
     *
     * @param type The query type.
     * @param info The RealTimeInfo structure to store the information.
     *
     * @return Status::OK if successful; otherwise, an appropriate error status.
     */
    Status QueryRTInfo(HexagonRTQueryType type, RealTimeInfo &info);

private:
    class Impl;                   /*!< Forward declaration of implementation class. */
    std::shared_ptr<Impl> m_impl; /*!< Pointer to the implementation class instance. */
};

/**
 * @}
 */
} // namespace aura

#endif // AURA_RUNTIME_HEXAGON_HOST_HEXAGON_ENGINE_HPP__