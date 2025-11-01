#ifndef AURA_RUNTIME_HEXAGON_COMM_HPP__
#define AURA_RUNTIME_HEXAGON_COMM_HPP__

#include "aura/runtime/core.h"

/**
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
 * @brief Structure representing Hexagon RPC memory.
 */
struct AURA_EXPORTS HexagonRpcMem
{
    MI_U8 *mem;    /*!< Pointer to the memory. */
    MI_S32 memLen; /*!< Length of the memory in bytes. */
};

/**
 * @brief Enumeration representing Hexagon power levels.
 */
enum class HexagonPowerLevel
{
    DEFAULT = 0, /*!< Default power level. */
    STANDBY,     /*!< Standby power level. */
    LOW,         /*!< Low power level. */
    NORMAL,      /*!< Normal power level. */
    TURBO        /*!< Turbo power level. */
};

/**
 * @}
 */
} // namespace aura

#endif // AURA_RUNTIME_HEXAGON_COMM_HPP__