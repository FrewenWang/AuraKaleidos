#ifndef AURA_RUNTIME_XTENSA_HOST_RPC_PARAM_HPP__
#define AURA_RUNTIME_XTENSA_HOST_RPC_PARAM_HPP__

#include "aura/runtime/mat.h"
#include "aura/runtime/memory.h"
#include "aura/runtime/logger.h"
#include <unordered_map>

/**
 * @defgroup runtime Runtime
 * @{
 *    @defgroup xtensa Xtensa
 *    @{
 *       @defgroup Xtensa_host Xtensa Host
 *    @}
 * @}
 */

namespace aura
{
/**
 * @addtogroup Xtensa_host
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
class AURA_EXPORTS XtensaRpcParam
{
public:
    /**
     * @brief Constructor for XtensaRpcParam on host.
     *
     * @param ctx The pointer to the Context object.
     * @param size The size of the RPC parameter buffer (default is 1024).
     */
    XtensaRpcParam(Context *ctx, DT_U32 size = 1024) : m_ctx(ctx)
    {
        m_rpc_param = m_ctx->GetMemPool()->GetBuffer(AURA_ALLOC_PARAM(m_ctx, AURA_MEM_DEFAULT, size, 0));
    }

    /**
     * @brief Destructor for XtensaRpcParam.
     */
    ~XtensaRpcParam()
    {
        AURA_FREE(m_ctx, m_rpc_param.m_origin);
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
        Status ret = Serialize(m_ctx, this, param);
        AURA_RETURN(m_ctx, ret);
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
            AURA_ADD_ERROR_STRING(m_ctx, "Set failed");
            return Status::ERROR;
        }
        ret |= Set(params...);
        AURA_RETURN(m_ctx, ret);
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
        Status ret = Deserialize(m_ctx, this, param);
        AURA_RETURN(m_ctx, ret);
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
            AURA_ADD_ERROR_STRING(m_ctx, "Get failed");
            return Status::ERROR;
        }
        ret |= Get(params...);
        AURA_RETURN(m_ctx, ret);
    }

    /**
     * @brief Reset the RPC parameter buffer to its original state.
     */
    DT_VOID ResetBuffer()
    {
        m_rpc_param.m_data = m_rpc_param.m_origin;
        m_rpc_param.m_size = 0;
    }

    Context *m_ctx;      /*!< The context for XtensaRpcParam. */
    Buffer  m_rpc_param; /*!< The RPC parameter buffer. */
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
class AURA_EXPORTS XtensaRpcParamType
{
public:
    /**
     * @brief Constructor for XtensaRpcParamType.
     *
     * @param ctx The pointer to the Context object.
     * @param rpc_param The XtensaRpcParam to work with.
     */
    XtensaRpcParamType(Context *ctx, XtensaRpcParam &rpc_param) : m_ctx(ctx), m_rpc_param(rpc_param)
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
        AURA_RETURN(m_ctx, ret);
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
        AURA_RETURN(m_ctx, ret);
    }

    Context *m_ctx;               /*!< The context for XtensaRpcParamType. */
    XtensaRpcParam &m_rpc_param; /*!< The XtensaRpcParam to work with. */
};

/**
 * @}
 */
} // namespace aura

#include "aura/runtime/xtensa/host/rpc_serialize.hpp"

#endif // AURA_RUNTIME_XTENSA_HOST_RPC_PARAM_HPP__
