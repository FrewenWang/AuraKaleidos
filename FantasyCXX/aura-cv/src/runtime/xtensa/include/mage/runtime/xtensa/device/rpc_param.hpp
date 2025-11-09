#ifndef AURA_RUNTIME_XTENSA_DEVICE_RPC_PARAM_HPP__
#define AURA_RUNTIME_XTENSA_DEVICE_RPC_PARAM_HPP__

#include "aura/runtime/core.h"

/**
 * @defgroup runtime Runtime
 * @{
 *    @defgroup xtensa Xtensa
 *    @{
 *       @defgroup xtensa_device Xtensa Device
 *    @}
 * @}
 */
namespace aura
{
namespace xtensa
{
/**
 * @addtogroup xtensa_device
 * @{
 */

/**
 * @brief Class representing Xtensa RPC parameters.
 *
 * XtensaRpcParam provides a versatile interface for managing Xtensa RPC parameters,
 * facilitating the serialization and deserialization of various parameter types. This
 * class is designed to seamlessly handle both host and Xtensa build environments,
 * adapting to the specific memory allocation mechanisms of each platform.
 */
class XtensaRpcParam
{
public:
    /**
     * @brief Constructor for XtensaRpcParam on Xtensa.
     * 
     * @param rpc_param The RPC parameter buffer.
     * @param rpc_param_len The length of the RPC parameter buffer.
     */
    XtensaRpcParam(DT_U8 *rpc_param, DT_S32 rpc_param_len)
    {
        m_rpc_param = Buffer(AURA_XTENSA_MEM_HEAP, rpc_param_len, 0, rpc_param, rpc_param, 0);
    }

    AURA_DISABLE_COPY_AND_ASSIGN(XtensaRpcParam);

    /**
     * @brief Set the RPC parameter with a single value.
     *
     * @tparam Tp The type of the parameter.
     *
     * @param param The parameter value.
     *
     * @return Status::OK if successful; otherwise, an appropriate error status.
     */
    template <typename Tp>
    Status Set(const Tp &param)
    {
        Status ret = Serialize(this, param);
        return ret;
    }

    /**
     * @brief Set the RPC parameter with multiple values.
     *
     * @tparam Tp0 The type of the first parameter.
     * @tparam Tpn The types of the remaining parameters.
     *
     * @param param0 The value of the first parameter.
     * @param params The values of the remaining parameters.
     *
     * @return Status::OK if successful; otherwise, an appropriate error status.
     */
    template <typename Tp0, typename ...Tpn>
    Status Set(const Tp0 &param0, const Tpn &...params)
    {
        Status ret = Set(param0);
        if (ret != Status::OK)
        {
            AURA_XTENSA_LOG("Set failed");
            return Status::ERROR;
        }
        ret = Set(params...);
        return ret;
    }

    /**
     * @brief Get the RPC parameter with a single value.
     *
     * @tparam Tp The type of the parameter.
     *
     * @param param The variable to store the parameter value.
     *
     * @return Status::OK if successful; otherwise, an appropriate error status.
     */
    template <typename Tp>
    Status Get(Tp &param)
    {
        Status ret = Deserialize(this, param);
        return ret;
    }

    /**
     * @brief Get the RPC parameter with multiple values.
     *
     * @tparam Tp0 The type of the first parameter.
     * @tparam Tpn The types of the remaining parameters.
     *
     * @param param0 The variable to store the value of the first parameter.
     * @param params The variables to store the values of the remaining parameters.
     *
     * @return Status::OK if successful; otherwise, an appropriate error status.
     */
    template <typename Tp0, typename ...Tpn>
    Status Get(Tp0 &param0, Tpn &...params)
    {
        Status ret = Get(param0);
        if (ret != Status::OK)
        {
            AURA_XTENSA_LOG("Get failed");
            return Status::ERROR;
        }
        ret = Get(params...);
        return ret;
    }

    /**
     * @brief Reset the RPC parameter buffer to its original state.
     */
    DT_VOID ResetBuffer()
    {
        m_rpc_param.m_data = m_rpc_param.m_origin;
        m_rpc_param.m_size = 0;
    }

    Buffer m_rpc_param; /*!< The RPC parameter buffer. */
};

/**
 * @brief Class template representing Xtensa RPC parameters with specific types.
 *
 * XtensaRpcParamType is a versatile template class designed to simplify the management of Xtensa RPC parameters
 * by providing a type-safe interface for setting and retrieving parameters of specific types. It works in conjunction
 * with XtensaRpcParam, allowing streamlined interaction with Xtensa RPC functionalities.
 *
 * @tparam Tp The types of parameters that can be set and retrieved.
 */
template <typename ...Tp>
class XtensaRpcParamType
{
public:
    /**
     * @brief Constructor for XtensaRpcParamType.
     *
     * @param rpc_param The XtensaRpcParam to work with.
     */
    XtensaRpcParamType(XtensaRpcParam &rpc_param) : m_rpc_param(rpc_param)
    {}

    AURA_DISABLE_COPY_AND_ASSIGN(XtensaRpcParamType);

    /**
     * @brief Set RPC parameters with specific values.
     *
     * @param params The values of the RPC parameters.
     *
     * @param reset Whether to reset the RPC parameter buffer before setting values (default is true).
     *
     * @return Status::OK if successful; otherwise, an appropriate error status.
     */
    Status Set(const Tp &...params, DT_BOOL reset = DT_TRUE)
    {
        if (reset)
        {
            m_rpc_param.ResetBuffer();
        }
        Status ret = m_rpc_param.Set(params...);
        return ret;
    }

    /**
     * @brief Get RPC parameters with specific values.
     *
     *
     * @param params The variables to store the values of the RPC parameters.
     *
     * @param reset Whether to reset the RPC parameter buffer before getting values (default is false).
     * 
     * @return Status::OK if successful; otherwise, an appropriate error status.
     */
    Status Get(Tp &...params, DT_BOOL reset = DT_FALSE)
    {
        if (reset)
        {
            m_rpc_param.ResetBuffer();
        }
        Status ret = m_rpc_param.Get(params...);
        return ret;
    }

    XtensaRpcParam &m_rpc_param; /*!< The XtensaRpcParam to work with. */
};

/**
 * @}
 */
} // namespace xtensa
} // namespace aura

#include "aura/runtime/xtensa/device/rpc_serialize.hpp"

#endif // AURA_RUNTIME_XTENSA_DEVICE_RPC_PARAM_HPP__