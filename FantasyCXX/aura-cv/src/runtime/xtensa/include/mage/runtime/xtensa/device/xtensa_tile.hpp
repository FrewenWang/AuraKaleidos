#ifndef AURA_RUNTIME_CORE_XTENSA_TYPES_TILE_HPP__
#define AURA_RUNTIME_CORE_XTENSA_TYPES_TILE_HPP__

#include "aura/runtime/xtensa/device/xtensa_frame.hpp"
#include "aura/runtime/core/xtensa/comm.hpp"
#include "aura/runtime/core/types/sizes.hpp"
#include "aura/runtime/core/types.h"

/**
 * @defgroup runtime Runtime
 * @{
 *      @defgroup core Runtime Core
 *      @{
 *           @defgroup tile Runtime Core Xtensa TileWrapper
 *      @}
 * @}
*/

namespace aura
{
namespace xtensa
{
/**
 * @addtogroup tile
 * @{
*/

/**
 * @brief TileWrapper class.
 *
 * The tile is a 2D array of elements with a specific element type and channel.
 * This class is used to represent a tile in the VDSP.
 */
class TileWrapper
{
public:
    /**
     * @brief Default constructor for creating an empty tile.
     */
    TileWrapper();

    /**
     * @brief Constructor for creating a tile with specified properties and existing buffer.
     *
     * @param data The pointer of data for the tile.
     * @param elem_type Element type of the tile.
     * @param channel Channel of the tile.
     */
    TileWrapper(AURA_VOID *data, ElemType elem_type, MI_S32 channel);

    /**
     * @brief Copy constructor for creating a tile by copying another tile.
     *
     * @param tile The TileWrapper object to be copied.
     *
     */
    TileWrapper(const TileWrapper &tile);

    /**
     * @brief Checks if the TileWrapper object is valid.
     *
     * @return True if the TileWrapper is valid; otherwise, false.
     */
    MI_BOOL IsValid() const;

    /**
     * @brief Update the tile with new properties.
     * @param x New x-coordinate position of the tile.
     * @param y New y-coordinate position of the tile.
     * @param width New width of the tile.
     * @param height New height of the tile.
     * @param edge_size New border edge size for padding.
     * @return MI_SUCCESS if the tile is updated successfully, MI_FAILURE otherwise.
     */
    Status Update(MI_S32 x, MI_S32 y, MI_S32 width, MI_S32 height, aura::Sizes &edge_size);

    /**
     * @brief Get the type of the tile.
     * @return Type of the tile.
     */
    MI_S32 GetXvTileType();

    /**
     * @brief Allocates single tile and associates it with a frame.
     * @param tm         The tileManager object for vdsp.
     * @param buffer     The buffer to be assigned to the frame.
     * @param frame      The frame to be associated with the tile.
     * @param in_or_out  The tile type, where 0 indicates input tile and 1 indicates output tile.
     * @param flag       The flag to indicate the tile type, where 0 indicates a normal tile and 1 indicates a border tile.
     * @return MI_SUCCESS if the tile is register successfully, MI_FAILURE otherwise.
     */
    Status Register(TileManager tm, AURA_VOID *buffer, FrameWrapper &frame, MI_U32 in_or_out, MI_S32 flag);

    /**
     * @brief Assignment operator for copying the contents of another tile.
     *
     * @param tile The TileWrapper object to be copied.
     *
     * @return A reference to the current TileWrapper object.
     */
    TileWrapper& operator=(const TileWrapper &tile);

    /**
     * @brief Get a pointer to the raw data of the tile.
     *
     * @return Pointer to the raw data of the tile.
     */
    AURA_VOID* GetData();

    /**
     * @brief Get a const pointer to the raw data of the matrix.
     *
     * @return Const pointer to the raw data of the matrix.
     */
    const AURA_VOID* GetData() const;

    /**
     * @brief Gets the element type of the tile.
     *
     * @return Element type of the tile.
     */
    ElemType GetElemType() const;

    /**
     * @brief Gets the channel of the tile.
     *
     * @return Channel of the tile.
     */
    MI_S32 GetChannel() const;

    /**
     * @brief Pads the tile according to specified parameters.
     *
     * @return Status code indicating success or failure.
     */
    Status Pad(BorderType border_type, Scalar &border_value) const;

    /**
     * @brief Resets the tile according to specific extra size and edge size.
     *
     * @return Status code indicating success or failure.
     */
    Status Reset(aura::Sizes &extra_size, aura::Sizes &edge_size) const;

    /**
     * @brief Extracts information of a specific type from the tile.
     *
     * @param type Type of information to extract.
     * @return Status code indicating success or failure.
     */
    Status Extract();

    /**
     * @brief Restores the tile to its original state.
     *
     * @return Status code indicating success or failure.
     */
    Status Restore();

private:
    /**
     * @brief Struct describing properties of a tile in a 2D array.
     *
     * This struct holds information such as data pointer, dimensions, position,
     * type, border dimensions, and padding properties for a tile.
     */
    struct TileDesc
    {
        TileDesc() : data(MI_NULL), width(0), height(0), x(0), y(0), type(0),
                    border_width(0), border_height(0), border_type(0), border_value(0)
        {}

        TileDesc(AURA_VOID *ptr, MI_S32 base_width, MI_S32 base_height, MI_S32 base_x, MI_S32 base_y,
                MI_S32 tiletype, MI_S32 edge_width, MI_S32 edge_height, MI_S32 padding_type, MI_S32 padding_value)
                : data(ptr), width(base_width), height(base_height), x(base_x), y(base_y), type(tiletype),
                border_width(edge_width), border_height(edge_height), border_type(padding_type), border_value(padding_value)
        {}

        AURA_VOID *data;        /*!< Pointer to the data of the tile. */
        MI_S32 width;         /*!< Width of the tile. */
        MI_S32 height;        /*!< Height of the tile. */
        MI_S32 x;             /*!< X-coordinate position of the tile. */
        MI_S32 y;             /*!< Y-coordinate position of the tile. */
        MI_S32 type;          /*!< Type of the tile. */
        MI_S32 border_width;  /*!< Width of the tile's border. */
        MI_S32 border_height; /*!< Height of the tile's border. */
        MI_S32 border_type;   /*!< Type of padding used for the tile. */
        MI_S32 border_value;  /*!< Value used for padding the tile. */
    };

    MI_BOOL  m_flag;      /*!< Flag indicating whether the tile is extracted. */
    AURA_VOID  *m_data;     /*!< Data of the tile. */
    ElemType m_elem_type; /*!< Element type of the tile. */
    MI_S32   m_channel;   /*!< Channel of the tile. */
    TileDesc m_desc;      /*!< Description of the tile. */
};

/**
 * @}
*/
} // namespace xtensa
} // namespace aura

#endif // AURA_RUNTIME_CORE_XTENSA_TYPES_TILE_HPP__
