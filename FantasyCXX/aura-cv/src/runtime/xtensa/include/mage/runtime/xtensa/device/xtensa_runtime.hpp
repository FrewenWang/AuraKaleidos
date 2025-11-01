#ifndef AURA_RUNTIME_XTENSA_DEVICE_XTENSA_RUNTIME_HPP__
#define AURA_RUNTIME_XTENSA_DEVICE_XTENSA_RUNTIME_HPP__

#include "aura/config.h"
#if defined(AURA_BUILD_XTENSA)
#   include "aura/runtime/xtensa/device/rpc_param.hpp"
#else
#   include "aura/runtime/xtensa/host/rpc_param.hpp"
#endif

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

using XtensaRpcFunc = Status (*)(AURA_VOID*, XtensaRpcParam&);
using PilRpcFunc    = MI_S32 (*)(const MI_CHAR*, MI_U8*, MI_S32);

/**
 * @brief the map between op name and rpc function.
 */
struct XtensaRpcFuncRegister
{
    MI_CHAR       name[AURA_XTENSA_LEN_128];  /*!< the op name */
    XtensaRpcFunc func;                       /*!< the pointer of rpc function */
};

/**
 *
 * @brief Function to call DSP operation functions.
 *
 * @param name          The operation name string.
 * @param param         The param for opration.
 * @param param_len     The length of param.
 *
 * @return AURA_XTENSA_ERROR if it encounters an error, else returns AURA_XTENSA_OK
 *
 */
MI_S32 VdspRpcCall(const MI_CHAR *name, MI_U8 *param, MI_S32 param_len);

/**
 * @}
 */
} //namespace xtensa
} // namespace aura

#endif // AURA_RUNTIME_XTENSA_DEVICE_XTENSA_RUNTIME_HPP__
