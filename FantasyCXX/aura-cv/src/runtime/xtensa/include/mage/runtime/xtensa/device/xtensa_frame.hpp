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

    FrameWrapper(DT_U64 buffer, DT_U32 buffer_size, DT_U64 data, DT_S32 width, DT_S32 height, DT_S32 pitch, DT_U8 pixel_res, DT_U8 num_channels,
                 DT_U8 left_edge_pad_width, DT_U8 top_edge_pad_height, DT_U8 right_edge_pad_width, DT_U8 bottom_edge_pad_height,
                 DT_U8 padding_type, DT_U32 padding_val);

    FrameWrapper(TileManager tm, const Mat *mat, BorderType border_type, const Scalar &border_value);

    DT_BOOL IsValid() const;

    DT_U64 buffer;                    /*!< The buffer of Frame. */
    DT_U32 buffer_size;               /*!< The size of buffer. */
    DT_U64 data;                      /*!< The data of Frame. */
    DT_S32 width;                     /*!< The width of Frame. */
    DT_S32 height;                    /*!< The height of Frame. */
    DT_S32 pitch;                     /*!< The pitch of Frame. */
    DT_U8  pixel_res;                 /*!< The pixel resolution of Frame. */
    DT_U8  num_channels;              /*!< The number of channels of Frame. */
    DT_U8  left_edge_pad_width;       /*!< The width of left edge for padding. */
    DT_U8  top_edge_pad_height;       /*!< The height of top edge for padding. */
    DT_U8  right_edge_pad_width;      /*!< The width of right edge for padding. */
    DT_U8  bottom_edge_pad_height;    /*!< The height of bottom edge for padding. */
    DT_U8  padding_type;              /*!< The type of padding, such as reflect, constant, zero. */
    DT_U32 padding_val;               /*!< The value of padding. */
};

/**
 * @}
*/
} // namespace xtensa
} // namespace aura

#endif // AURA_RUNTIME_XTENSA_DEVICE_XTENSA_FRAME_HPP__