#include "aura/runtime/xtensa/device/xtensa_tile.hpp"

#include "tileManager.h"
#include "tileManager_api.h"
#include "xm_cascade_utils.h"
#include "tileManager_FIK_api.h"

namespace aura
{
namespace xtensa
{

static xi_tile* GetXiTile(AURA_VOID *src)
{
    if (NULL == src)
    {
        AURA_XTENSA_LOG("src is null\n");
        return MI_NULL;
    }

    if (sizeof(xvTile) != sizeof(xi_tile))
    {
        AURA_XTENSA_LOG("sizeof(xvTile) != sizeof(xi_tile)\n");
        return MI_NULL;
    }

    return static_cast<xi_tile*>(src);
}

static xi_frame* GetXiFrame(xvFrame *src)
{
    if (MI_NULL == src)
    {
        AURA_XTENSA_LOG("src is null\n");
        return MI_NULL;
    }

    if (sizeof(xvFrame) != sizeof(xi_frame))
    {
        AURA_XTENSA_LOG("sizeof(xvFrame) != sizeof(xi_frame)\n");
        return MI_NULL;
    }

    return reinterpret_cast<xi_frame*>(src);
}

static xvFrame* GetxvFrame(FrameWrapper *src)
{
    if (MI_NULL == src)
    {
        AURA_XTENSA_LOG("src is null\n");
        return MI_NULL;
    }

    if (!(src->IsValid()))
    {
        AURA_XTENSA_LOG("frame is invalid\n");
        return MI_NULL;
    }

    if (sizeof(xvFrame) != sizeof(FrameWrapper))
    {
        AURA_XTENSA_LOG("sizeof(xvFrame) != sizeof(FrameWrapper)\n");
        return MI_NULL;
    }

    return reinterpret_cast<xvFrame*>(src);
}

//============================ TileWrapper ============================
TileWrapper::TileWrapper() : m_flag(MI_FALSE), m_data(MI_NULL), m_elem_type(ElemType::INVALID), m_channel(0)
{}

TileWrapper::TileWrapper(AURA_VOID *data, ElemType elem_type, MI_S32 channel)
                         : m_flag(MI_FALSE), m_data(data), m_elem_type(elem_type), m_channel(channel)
{}

TileWrapper::TileWrapper(const TileWrapper &tile)
{
    m_data      = tile.m_data;
    m_elem_type = tile.m_elem_type;
    m_channel   = tile.m_channel;

    Memcpy(&m_desc, &tile.m_desc, sizeof(TileDesc));
}

MI_BOOL TileWrapper::IsValid() const
{
    return ((m_data != MI_NULL) && (m_elem_type != ElemType::INVALID) && (m_channel > 0));
}

Status TileWrapper::Update(MI_S32 x, MI_S32 y, MI_S32 width, MI_S32 height, aura::Sizes &edge_size)
{
    xvTile *xv_tile = static_cast<xvTile*>(m_data);
    if (NULL == xv_tile)
    {
        AURA_XTENSA_LOG("xv_tile is null\n");
        return Status::ERROR;
    }

    xv_tile->x              = x;
    xv_tile->y              = y;
    xv_tile->width          = width;
    xv_tile->height         = height;
    xv_tile->tileEdgeLeft   = edge_size.m_width;
    xv_tile->tileEdgeTop    = edge_size.m_height;
    xv_tile->tileEdgeRight  = edge_size.m_width;
    xv_tile->tileEdgeBottom = edge_size.m_height;

    return Status::OK;
}

MI_S32 TileWrapper::GetXvTileType()
{
    struct Info
    {
        MI_S32   ch;
        ElemType elem_type;
        MI_S32   type;
    };

    MI_S32 type = -1;
    const Info type_map[] =
    {
        {1,  ElemType::U8,            XV_U8},
        {1,  ElemType::S8,            XV_S8},
        {1, ElemType::U16,           XV_U16},
        {1, ElemType::S16,           XV_S16},
        {1, ElemType::U32,           XV_U32},
        {1, ElemType::S32,           XV_S32},
        {2,  ElemType::U8,    XV_TILE_RG_U8},
        {2, ElemType::U16,   XV_TILE_RG_U16},
        {2, ElemType::U32,   XV_TILE_RG_U32},
        {3,  ElemType::U8,   XV_TILE_RGB_U8},
        {3, ElemType::U16,  XV_TILE_RGB_U16},
        {3, ElemType::U32,  XV_TILE_RGB_U32},
        {4,  ElemType::U8,  XV_TILE_RGBA_U8},
        {4, ElemType::U16, XV_TILE_RGBA_U16},
        {4, ElemType::U32, XV_TILE_RGBA_U32},
    };

    MI_S32 size = sizeof(type_map) / sizeof(type_map[0]);
    for (MI_S32 i = 0; i < size; i++)
    {
        if (type_map[i].ch == m_channel && type_map[i].elem_type == m_elem_type)
        {
            type = (MI_S32)(type_map[i].type);
        }
    }

    return type;
}

Status TileWrapper::Register(TileManager tm, AURA_VOID *buffer, FrameWrapper &frame, MI_U32 in_or_out, MI_S32 flag)
{
    xvTile *xv_tile = static_cast<xvTile*>(m_data);
    if (NULL == xv_tile)
    {
        AURA_XTENSA_LOG("xv_tile is null\n");
        return Status::ERROR;
    }

    if (!(frame.IsValid()))
    {
        AURA_XTENSA_LOG("frame is invalid\n");
        return Status::ERROR;
    }

    if (0 == flag)
    {
        MI_S32 tile_type = GetXvTileType();
        if (-1 == tile_type)
        {
            AURA_XTENSA_LOG("tile type is not exist\n");
            return Status::ERROR;
        }

        xv_tile->type = tile_type;
        MI_S32 ret = xvRegisterTile(static_cast<xvTileManager*>(tm), xv_tile, buffer, GetxvFrame(&frame), in_or_out);
        if (ret != AURA_XTENSA_OK)
        {
            AURA_XTENSA_LOG("xvRegisterTile failed\n");
            return Status::ERROR;
        }
    }

    return Status::OK;
}

TileWrapper& TileWrapper::operator=(const TileWrapper &tile)
{
    if (this != &tile)
    {
        m_data      = tile.m_data;
        m_elem_type = tile.m_elem_type;
        m_channel   = tile.m_channel;

        Memcpy(&m_desc, &tile.m_desc, sizeof(TileDesc));
    }

    return *this;
}

AURA_VOID* TileWrapper::GetData()
{
    return m_data;
}

const AURA_VOID* TileWrapper::GetData() const
{
    return m_data;
}

ElemType TileWrapper::GetElemType() const
{
    return m_elem_type;
}

MI_S32 TileWrapper::GetChannel() const
{
    return m_channel;
}

Status TileWrapper::Pad(BorderType border_type, Scalar &border_value) const
{
    if (!IsValid())
    {
        AURA_XTENSA_LOG("TileWrapper is invalid\n");
        return Status::ERROR;
    }

    xi_tile *tile = GetXiTile(m_data);
    if (MI_NULL == tile)
    {
        AURA_XTENSA_LOG("GetXiTile failed\n");
        return Status::ERROR;
    }

    MI_S32 ret = TilePadding(tile, static_cast<MI_S32>(border_type), border_value.m_val[0]);
    if (ret != XVF_SUCCESS)
    {
        AURA_XTENSA_LOG("TilePadding failed\n");
        return Status::ERROR;
    }

    return Status::OK;
}

Status TileWrapper::Reset(aura::Sizes &extra_size, aura::Sizes &edge_size) const
{
    if (!IsValid())
    {
        AURA_XTENSA_LOG("TileWrapper is invalid\n");
        return Status::ERROR;
    }

    xvTile *xv_tile = static_cast<xvTile*>(m_data);
    if (MI_NULL == xv_tile)
    {
        AURA_XTENSA_LOG("xv_tile is null\n");
        return Status::ERROR;
    }

    if (sizeof(TileDesc) != sizeof(TileInfo))
    {
        AURA_XTENSA_LOG("sizeof(TileDesc) != sizeof(TileInfo)\n");
        return Status::ERROR;
    }

    TileInfo *tile_info = reinterpret_cast<TileInfo*>(const_cast<TileDesc*>(&m_desc));
    xi_tile *tile       = GetXiTile(m_data);
    xi_frame *frame     = GetXiFrame(xv_tile->pFrame);

    if ((MI_NULL == tile) || (MI_NULL == tile_info) || (MI_NULL == frame))
    {
        AURA_XTENSA_LOG("tile/tile_info/frame is null\n");
        return Status::ERROR;
    }

    MI_S32 flags = GetEdgeFlags(tile, frame);
    MI_S32 ret   = TileResetting(tile, tile_info, extra_size.m_width, extra_size.m_height,
                                 edge_size.m_width, edge_size.m_height, flags);
    if (ret != XVF_SUCCESS)
    {
        AURA_XTENSA_LOG("TileResetting failed\n");
        return Status::ERROR;
    }

    return Status::OK;
}

Status TileWrapper::Extract()
{
    if (!IsValid())
    {
        AURA_XTENSA_LOG("TileWrapper is invalid\n");
        return Status::ERROR;
    }

    if (!m_flag)
    {
        if (sizeof(TileDesc) != sizeof(TileInfo))
        {
            AURA_XTENSA_LOG("sizeof(TileDesc) != sizeof(TileInfo)\n");
            return Status::ERROR;
        }

        TileInfo *tile_info = reinterpret_cast<TileInfo*>(const_cast<TileDesc*>(&m_desc));
        xi_tile *tile       = GetXiTile(m_data);

        if ((MI_NULL == tile) || (MI_NULL == tile_info))
        {
            AURA_XTENSA_LOG("tile/tile_info is null\n");
            return Status::ERROR;
        }

        ExtractTileInfo(tile_info, tile, static_cast<MI_S32>(m_elem_type));

        m_flag = MI_TRUE;
    }

    return Status::OK;
}

Status TileWrapper::Restore()
{
    if (!IsValid())
    {
        AURA_XTENSA_LOG("TileWrapper is invalid\n");
        return Status::ERROR;
    }

    if (sizeof(TileDesc) != sizeof(TileInfo))
    {
        AURA_XTENSA_LOG("sizeof(TileDesc) != sizeof(TileInfo)\n");
        return Status::ERROR;
    }

    TileInfo *tile_info = reinterpret_cast<TileInfo*>(const_cast<TileDesc*>(&m_desc));
    xi_tile *tile       = GetXiTile(m_data);

    if ((MI_NULL == tile) || (MI_NULL == tile_info))
    {
        AURA_XTENSA_LOG("tile/tile_info is null\n");
        return Status::ERROR;
    }

    TileResetToOrigin(tile, tile_info);

    m_flag = MI_FALSE;

    return Status::OK;
}

} // namespace xtensa
} // namespace aura

