#ifndef AURA_RUNTIME_XTENSA_DEVICE_XTENSA_UTILS_HPP__
#define AURA_RUNTIME_XTENSA_DEVICE_XTENSA_UTILS_HPP__

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

AURA_VOID* AllocateBuffer(TileManager tm, MI_S32 size, MI_S32 align);

/**
 *
 * @brief Function to check buffer point.
 *
 * @param tm The tilemanager object.
 *
 * @return AURA_XTENSA_ERROR if it encounters an error, else returns AURA_XTENSA_OK
 *
 */
MI_S32 BufferCheckPointSave(TileManager tm);

/**
 *
 * @brief Function to release buffer from idx.
 *
 * @param tm  The tilemanager object.
 * @param idx The start index of the memory.
 *
 * @return Status::ERROR if it encounters an error, else returns Status::OK
 *
 */
Status BufferCheckPointRestore(TileManager tm, MI_S32 idx);

/**
 * @}
 */
} //namespace xtensa
} // namespace aura

#endif // AURA_RUNTIME_XTENSA_DEVICE_XTENSA_UTILS_HPP__
