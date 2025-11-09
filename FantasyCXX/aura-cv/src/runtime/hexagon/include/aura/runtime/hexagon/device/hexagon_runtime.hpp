#ifndef AURA_RUNTIME_HEXAGON_DEVICE_HEXAGON_RUNTIME_HPP__
#define AURA_RUNTIME_HEXAGON_DEVICE_HEXAGON_RUNTIME_HPP__

#include "aura/runtime/hexagon/rpc_param.hpp"

/**
 * @defgroup runtime Runtime
 * @{
 *    @defgroup hexagon Hexagon
 *    @{
 *       @defgroup device Hexagon Device
 *    @}
 * @}
 */

/**
 * @addtogroup device
 * @{
 */

/**
 * @brief Macro for registering Hexagon RPC functions.
 *
 * This macro simplifies the process of registering Hexagon RPC functions.
 *
 * @param package The package name of the RPC function.
 * @param module The module name of the RPC function.
 * @param func The name of the RPC function to be registered.
 *
 * @return Rpc function register.
 */
#define AURA_HEXAGON_RPC_FUNC_REGISTER(package, module, func)    \
    static aura::RpcFuncRegister g_##func(package + std::string(".") + module, func)

/**
 * @}
*/

namespace aura
{
/**
 * @addtogroup device
 * @{
 */

/**
 * @brief Defines the function pointer type for Hexagon RPC functions.
 *
 * This typedef specifies the function pointer signature for Hexagon RPC functions.
 *
 * @param ctx The pointer to the Context object.
 * @param param The Hexagon RPC parameters.
 *
 * @return Status::OK if the operation is successful; otherwise, an appropriate error status.
 */
using RpcFunc = Status (*)(Context *, HexagonRpcParam &);

/**
 * @brief Class for registering Hexagon RPC functions.
 *
 * This class provides a mechanism for registering Hexagon Remote Procedure Call (RPC) functions.
 * It allows associating a unique name with a corresponding function pointer, enabling a centralized
 * registry for efficient lookup and invocation of Hexagon RPC functions.
 */
class AURA_EXPORTS RpcFuncRegister
{
public:
    /**
     * @brief Constructor for RpcFuncRegister.
     *
     * @param name The name of the Hexagon RPC function being registered.
     * @param func The function pointer to the Hexagon RPC function.
     */
    RpcFuncRegister(const std::string &name, RpcFunc func);
};

/**
 * @brief Set the power level for Hexagon device.
 *
 * This function configures the power level for the Hexagon device, allowing customization of the
 * target power level and the option to enable or disable Dynamic Clock and Voltage Scaling (DCVS).
 *
 * @param ctx The pointer to the Context object.
 * @param target_level The target power level.
 * @param enable_dcvs Enable or disable DCVS (Dynamic Clock and Voltage Scaling).
 * @param client_id power client id value, default 0 .
 *
 * @return Status::OK if successful; otherwise, an appropriate error status.
 */
Status SetPower(Context *ctx, HexagonPowerLevel target_level, DT_BOOL enable_dcvs, DT_U32 client_id = 0);

/**
 * @}
 */
} // namespace aura

#endif // AURA_RUNTIME_HEXAGON_DEVICE_HEXAGON_RUNTIME_HPP__