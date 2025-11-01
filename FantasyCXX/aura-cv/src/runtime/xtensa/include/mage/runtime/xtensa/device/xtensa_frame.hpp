#ifndef AURA_RUNTIME_XTENSA_DEVICE_XTENSA_FRAME_HPP__
#define AURA_RUNTIME_XTENSA_DEVICE_XTENSA_FRAME_HPP__

#include "aura/runtime/core/types.h"
#include "aura/runtime/core/xtensa/types.h"

/**
 * @defgroup runtime Runtime
 * @{
 *      @defgroup xtensa Xtensa
 *      @{
 *           @defgroup xtensa_device Xtensa Device
 *      @}
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
 * @brief Structure representing Xtensa FrameWrapper.
 */
struct FrameWrapper
{
    FrameWrapper();

    FrameWrapper(MI_U64 buffer, MI_U32 buffer_size, MI_U64 data, MI_S32 width, MI_S32 height, MI_S32 pitch, MI_U8 pixel_res, MI_U8 num_channels,
                 MI_U8 left_edge_pad_width, MI_U8 top_edge_pad_height, MI_U8 right_edge_pad_width, MI_U8 bottom_edge_pad_height,
                 MI_U8 padding_type, MI_U32 padding_val);

    FrameWrapper(TileManager tm, const Mat *mat, BorderType border_type, const Scalar &border_value);

    MI_BOOL IsValid() const;

    MI_U64 buffer;                    /*!< The buffer of Frame. */
    MI_U32 buffer_size;               /*!< The size of buffer. */
    MI_U64 data;                      /*!< The data of Frame. */
    MI_S32 width;                     /*!< The width of Frame. */
    MI_S32 height;                    /*!< The height of Frame. */
    MI_S32 pitch;                     /*!< The pitch of Frame. */
    MI_U8  pixel_res;                 /*!< The pixel resolution of Frame. */
    MI_U8  num_channels;              /*!< The number of channels of Frame. */
    MI_U8  left_edge_pad_width;       /*!< The width of left edge for padding. */
    MI_U8  top_edge_pad_height;       /*!< The height of top edge for padding. */
    MI_U8  right_edge_pad_width;      /*!< The width of right edge for padding. */
    MI_U8  bottom_edge_pad_height;    /*!< The height of bottom edge for padding. */
    MI_U8  padding_type;              /*!< The type of padding, such as reflect, constant, zero. */
    MI_U32 padding_val;               /*!< The value of padding. */
};

/**
 * @}
*/
} // namespace xtensa
} // namespace aura

#endif // AURA_RUNTIME_XTENSA_DEVICE_XTENSA_FRAME_HPP__