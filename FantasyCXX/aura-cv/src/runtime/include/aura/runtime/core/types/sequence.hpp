#ifndef AURA_RUNTIME_CORE_TYPES_SEQUENCE_HPP__
#define AURA_RUNTIME_CORE_TYPES_SEQUENCE_HPP__

#include "aura/runtime/core/types/built-in.hpp"

/**
 * @defgroup runtime Runtime
 * @{
 *      @defgroup core Runtime Core
 *      @{
 *           @defgroup types Runtime Core Types
 *      @}
 * @}
 */

namespace aura
{

/**
 * @addtogroup types
 * @{
 */

/**
 * @brief Structure representing a sequence of elements.
 *
 * This structure holds a pointer to data and its length.
 */
template <typename Tp>
struct Sequence
{
    Tp    *data; /*!< Pointer to the data. */
    DT_S32 len;  /*!< Length of the data. */
};

/**
 * @}
 */

} // namespace aura

#endif // AURA_RUNTIME_CORE_TYPES_SEQUENCE_HPP__