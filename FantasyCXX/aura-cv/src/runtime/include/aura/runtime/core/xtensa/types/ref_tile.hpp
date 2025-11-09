#ifndef AURA_RUNTIME_CORE_XTENSA_TYPES_REF_TILE_HPP__
#define AURA_RUNTIME_CORE_XTENSA_TYPES_REF_TILE_HPP__

#include "aura/runtime/core/types/point.hpp"
#include "aura/runtime/core/types/sizes.hpp"
#include "aura/runtime/core/xtensa/comm.hpp"
#include "aura/runtime/core/xtensa/types/mat.hpp"

/**
 * @defgroup runtime Runtime
 * @{
 *      @defgroup core Runtime Core
 *      @{
 *           @defgroup ref_tile Runtime Core Xtensa RefTile
 *      @}
 * @}
*/

namespace aura
{
namespace xtensa
{
/**
 * @addtogroup ref_tile
 * @{
*/

/**
 * @brief Structure representing Xtensa RefTileWrapper.
 */
struct RefTileWrapper
{
    /**
     *
     * @brief Constructor for RefTileWrapper.
     *
     */
    RefTileWrapper() : iaura_width(0), iaura_height(0), tile_width(0), tile_height(0), x(0), y(0)
    {}

    /**
     *
     * @brief Constructor for RefTileWrapper.
     *
     * @param tile_num        The number of tiles.
     * @param pt              The start point of the tile.
     * @param size            The size of mat.
     * @param elem_type       The element type of mat.
     * @param reserved_size   The reserved size for tile impl.
     *
     */
    RefTileWrapper(DT_U32 tile_num, const Point2i &pt, const Sizes3 size, const ElemType elem_type, DT_U32 reserved_size)
    {
        DT_U32 tile_bytes  = (AURA_VDSP_LOCAL_MEM_SIZE - reserved_size) / (tile_num << 1);
        tile_height = (tile_bytes << 10) / (size.m_width * size.m_channel * ElemTypeSize(elem_type));

        x            = pt.m_x;
        y            = pt.m_y;
        tile_width   = size.m_width;
        tile_height  = tile_height < size.m_height ? tile_height : size.m_height;
        iaura_width  = size.m_width;
        iaura_height = size.m_height;
    }

    /**
     *
     * @brief Constructor for RefTileWrapper.
     *
     * @param tile_num        The number of tiles.
     * @param elem_count      The number of element in the width of a tile.
     * @param pt              The start point of the tile.
     * @param size            The size of mat.
     * @param elem_type       The element type of mat.
     * @param reserved_size   The reserved size of local buffer.
     *
     */
    RefTileWrapper(DT_U32 tile_num, DT_S32 elem_count, const Point2i &pt, const Sizes3 size, const ElemType elem_type, DT_U32 reserved_size)
    {
        DT_U32 tile_bytes  = (AURA_VDSP_LOCAL_MEM_SIZE - reserved_size) / (tile_num << 1);
        elem_count = elem_count < size.m_width ? elem_count : size.m_width;
        tile_height = (tile_bytes << 10) / (elem_count * size.m_channel * ElemTypeSize(elem_type));

        x            = pt.m_x;
        y            = pt.m_y;
        tile_width   = elem_count;
        tile_height  = tile_height < size.m_height ? tile_height : size.m_height;
        iaura_width  = size.m_width * size.m_channel;
        iaura_height = size.m_height;
    }

        /**
     *
     * @brief Constructor for RefTileWrapper.
     *
     * @param tile_num        The number of tiles.
     * @param elem_count      The number of element in the width of a tile.
     * @param pt              The start point of the tile.
     * @param src_size        The size of src mat.
     * @param dst_size        The size of dst mat.
     * @param src_elem_type   The element type of src mat.
     * @param dst_elem_type   The element type of dst mat.
     * @param reserved_size   The reserved size of local buffer.
     *
     */
    RefTileWrapper(DT_U32 src_tile_num, DT_U32 dst_tile_num, const Point2i &pt, const Sizes3 src_size, const Sizes3 dst_size,
                   const ElemType src_elem_type, const ElemType dst_elem_type, DT_U32 reserved_size)
    {
        DT_U32 tile_bytes  = (AURA_VDSP_LOCAL_MEM_SIZE - reserved_size);
        DT_U32 src_tile_bytes = src_size.m_width * src_size.m_channel * ElemTypeSize(src_elem_type) * (src_tile_num << 1);
        DT_U32 dst_tile_bytes = dst_size.m_width * dst_size.m_channel * ElemTypeSize(dst_elem_type) * (dst_tile_num << 1);
        tile_height = (tile_bytes << 10) / (src_tile_bytes + dst_tile_bytes);

        x            = pt.m_x;
        y            = pt.m_y;
        tile_width   = dst_size.m_width;
        tile_height  = tile_height < dst_size.m_height ? tile_height : dst_size.m_height;
        iaura_width  = dst_size.m_width;
        iaura_height = dst_size.m_height;
    }

    /**
     *
     * @brief Constructor for RefTileWrapper.
     *
     * @param pt            The start point of tile.
     * @param tile_size     The size of tile.
     * @param size          The size of iaura.
     *
     */
    RefTileWrapper(const Point2i &pt, const Sizes &tile_size, const Sizes3 &size)
    {
        x            = pt.m_x;
        y            = pt.m_y;
        tile_width   = tile_size.m_width;
        tile_height  = tile_size.m_height < size.m_height ? tile_size.m_height : size.m_height;
        iaura_width  = size.m_width;
        iaura_height = size.m_height;
    }

    /**
     * @brief Checks if the RefTile object is valid.
     *
     * @return True if the RefTile is valid; otherwise, false.
     */
    DT_BOOL IsValid() const
    {
        return (x >= 0 && y >= 0 && tile_width > 0 && tile_height > 0 && iaura_width > 0 && iaura_height > 0);
    }

    DT_S32 iaura_width;  /*!< The width of iaura. */
    DT_S32 iaura_height; /*!< The height of iaura. */
    DT_S32 tile_width;   /*!< The width of tile. */
    DT_U16 tile_height;  /*!< The height of tile. */
    DT_S32 x;            /*!< The x of upper left point. */
    DT_S32 y;            /*!< The y of upper left point. */
};

/**
 * @}
*/
} // namespace xtensa
} // namespace aura

#endif // AURA_RUNTIME_CORE_XTENSA_TYPES_REF_TILE_HPP__