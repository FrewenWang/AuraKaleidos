#ifndef AURA_RUNTIME_HEXAGON_RPC_PARAM_HPP__
#define AURA_RUNTIME_HEXAGON_RPC_PARAM_HPP__

#include "aura/runtime/hexagon/comm.hpp"
#include "aura/runtime/memory.h"
#include "aura/runtime/logger.h"

/**
 * aura里面运行时的构建，运行时主要是针对这个异构框架的底层封装
 * @defgroup runtime Runtime
 * @{
 *    @defgroup hexagon Hexagon
 *    @{
 *       @defgroup comm Hexagon Common
 *    @}
 * @}
 */

namespace aura
{
/**
 * @addtogroup comm
 * @{
 */

/**
 * 高通
 * @brief Class representing Hexagon RPC parameters.
 *
 * HexagonRpcParam provides a versatile interface for managing Hexagon RPC parameters,
 * facilitating the serialization and deserialization of various parameter types. This
 * class is designed to seamlessly handle both host and Hexagon build environments,
 * adapting to the specific memory allocation mechanisms of each platform.
 */
class AURA_EXPORTS HexagonRpcParam
{
public:
//// 如果是编译主机侧
#if defined(AURA_BUILD_HOST)
    /**
     * @brief Constructor for HexagonRpcParam on host.
     *
     * @param ctx The pointer to the Context object.
     * @param size The size of the RPC parameter buffer (default is 1024).
     */
    HexagonRpcParam(Context *ctx, MI_U32 size = 1024) : m_ctx(ctx)
    {
        ////
        m_rpc_param = m_ctx->GetMemPool()->GetBuffer(AURA_ALLOC_PARAM(m_ctx, AURA_MEM_DMA_BUF_HEAP, size, 0));
    }
#else // AURA_BUILD_HEXAGON
    /**
     * @brief Constructor for HexagonRpcParam on Hexagon.
     *
     * @param ctx The pointer to the Context object.
     * @param rpc_mem The Hexagon RPC memory.
     * @param rpc_param The RPC parameter buffer.
     * @param rpc_param_len The length of the RPC parameter buffer.
     */
    HexagonRpcParam(Context *ctx, const HexagonRpcMem *rpc_mem, MI_U8 *rpc_param, MI_S32 rpc_param_len)
                    : m_ctx(ctx), m_rpc_mem(rpc_mem)
    {
        m_rpc_param = Buffer(AURA_MEM_DMA_BUF_HEAP, rpc_param_len, 0, rpc_param, rpc_param, 0);
    }
#endif

    /**
     * @brief Destructor for HexagonRpcParam.
     */
    ~HexagonRpcParam()
    {
#if defined(AURA_BUILD_HOST)
        AURA_FREE(m_ctx, m_rpc_param.m_origin);
#endif
    }

    AURA_DISABLE_COPY_AND_ASSIGN(HexagonRpcParam);

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
    AURA_VOID ResetBuffer()
    {
        m_rpc_param.m_data = m_rpc_param.m_origin;
        m_rpc_param.m_size = 0;
    }

    Context                    *m_ctx;      /*!< The context for HexagonRpcParam. */
    Buffer                     m_rpc_param; /*!< The RPC parameter buffer. */
#if defined(AURA_BUILD_HOST)
    std::vector<HexagonRpcMem> m_rpc_mem;   /*!< Vector of Hexagon RPC memory on host build. */
#else // AURA_BUILD_HEXAGON
    const HexagonRpcMem        *m_rpc_mem;  /*!< Hexagon RPC memory on Hexagon build. */
#endif
};

/**
 * @brief Class template representing Hexagon RPC parameters with specific types.
 *
 * HexagonRpcParamType is a versatile template class designed to simplify the management of Hexagon RPC parameters
 * by providing a type-safe interface for setting and retrieving parameters of specific types. It works in conjunction
 * with HexagonRpcParam, allowing streamlined interaction with Hexagon RPC functionalities.
 *
 * @tparam Tp The types of parameters that can be set and retrieved.
 */
template <typename ...Tp>
class HexagonRpcParamType
{
public:
    /**
     * @brief Constructor for HexagonRpcParamType.
     *
     * @param ctx The pointer to the Context object.
     * @param rpc_param The HexagonRpcParam to work with.
     */
    HexagonRpcParamType(Context *ctx, HexagonRpcParam &rpc_param) : m_ctx(ctx), m_rpc_param(rpc_param)
    {}

    AURA_DISABLE_COPY_AND_ASSIGN(HexagonRpcParamType);

    /**
     * @brief Set RPC parameters with specific values.
     *
     * @param params The values of the RPC parameters.
     *
     * @param reset Whether to reset the RPC parameter buffer before setting values (default is true).
     *
     * @return Status::OK if successful; otherwise, an appropriate error status.
     */
    Status Set(const Tp &...params, MI_BOOL reset = MI_TRUE)
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
    Status Get(Tp &...params, MI_BOOL reset = MI_FALSE)
    {
        if (reset)
        {
            m_rpc_param.ResetBuffer();
        }
        Status ret = m_rpc_param.Get(params...);
        AURA_RETURN(m_ctx, ret);
    }

private:
    Context *m_ctx;               /*!< The context for HexagonRpcParamType. */
    HexagonRpcParam &m_rpc_param; /*!< The HexagonRpcParam to work with. */
};

/**
 * @}
 */
} // namespace aura

#include "aura/runtime/hexagon/rpc_serialize.hpp"

#endif // AURA_RUNTIME_HEXAGON_RPC_PARAM_HPP__