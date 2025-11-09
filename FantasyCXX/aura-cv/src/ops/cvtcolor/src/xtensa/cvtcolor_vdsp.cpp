#include "xtensa/cvtcolor_vdsp_impl.hpp"

namespace aura
{
namespace xtensa
{

template <typename Tp>
static Status CheckNum(const vector<const Tp*> &src, const vector<Tp*> &dst, CvtColorType type)
{
    Status ret = Status::ERROR;

    const DT_S32 src_len = src.size();
    const DT_S32 dst_len = dst.size();

    switch (type)
    {
        case CvtColorType::BGR2BGRA:
        case CvtColorType::BGRA2BGR:
        case CvtColorType::BGR2RGB:
        case CvtColorType::BGR2GRAY:
        case CvtColorType::BGRA2GRAY:
        case CvtColorType::RGB2GRAY:
        case CvtColorType::RGBA2GRAY:
        case CvtColorType::GRAY2BGR:
        case CvtColorType::GRAY2BGRA:
        case CvtColorType::YUV2RGB_Y422:
        case CvtColorType::YUV2RGB_YUYV:
        case CvtColorType::YUV2RGB_YVYU:
        case CvtColorType::YUV2RGB_Y422_601:
        case CvtColorType::YUV2RGB_YUYV_601:
        case CvtColorType::YUV2RGB_YVYU_601:
        case CvtColorType::BAYERBG2BGR:
        case CvtColorType::BAYERGB2BGR:
        case CvtColorType::BAYERRG2BGR:
        case CvtColorType::BAYERGR2BGR:
        {
            ret = (1 == src_len && 1 == dst_len) ? Status::OK : Status::ERROR;
            break;
        }

        case CvtColorType::YUV2RGB_NV12:
        case CvtColorType::YUV2RGB_NV21:
        case CvtColorType::YUV2RGB_NV12_601:
        case CvtColorType::YUV2RGB_NV21_601:
        {
            ret = (2 == src_len && 1 == dst_len) ? Status::OK : Status::ERROR;
            break;
        }

        case CvtColorType::YUV2RGB_YU12:
        case CvtColorType::YUV2RGB_YV12:
        case CvtColorType::YUV2RGB_Y444:
        case CvtColorType::YUV2RGB_YU12_601:
        case CvtColorType::YUV2RGB_YV12_601:
        case CvtColorType::YUV2RGB_Y444_601:
        {
            ret = (3 == src_len && 1 == dst_len) ? Status::OK : Status::ERROR;
            break;
        }

        case CvtColorType::RGB2YUV_NV12:
        case CvtColorType::RGB2YUV_NV21:
        case CvtColorType::RGB2YUV_NV12_601:
        case CvtColorType::RGB2YUV_NV21_601:
        case CvtColorType::RGB2YUV_NV12_P010:
        case CvtColorType::RGB2YUV_NV21_P010:
        {
            ret = (1 == src_len && 2 == dst_len) ? Status::OK : Status::ERROR;
            break;
        }

        case CvtColorType::RGB2YUV_YU12:
        case CvtColorType::RGB2YUV_YV12:
        case CvtColorType::RGB2YUV_Y444:
        case CvtColorType::RGB2YUV_YU12_601:
        case CvtColorType::RGB2YUV_YV12_601:
        case CvtColorType::RGB2YUV_Y444_601:
        {
            ret = (1 == src_len && 3 == dst_len) ? Status::OK : Status::ERROR;
            break;
        }

        default:
        {
            AURA_XTENSA_LOG("cvtcolor type error");
            ret = Status::ERROR;
        }
    }

    if (ret != Status::OK)
    {
        AURA_XTENSA_LOG("the number of src or dst is not match to type");
    }

    AURA_STATUS_RETURN(ret);
}

template <typename Tp>
static Status CheckElemType(const vector<const Tp*> &src, const vector<Tp*> &dst, CvtColorType type)
{
    Status ret = Status::OK;

    const DT_S32 src_len = src.size();
    const DT_S32 dst_len = dst.size();

    for (DT_S32 i = 0; i < src_len; i++)
    {
        if (!(src[i]->IsValid()))
        {
            AURA_XTENSA_LOG("invalid src");
            return Status::ERROR;
        }
    }

    for (DT_S32 i = 0; i < dst_len; i++)
    {
        if (!(dst[i]->IsValid()))
        {
            AURA_XTENSA_LOG("invalid dst");
            return Status::ERROR;
        }
    }

    switch (type)
    {
        case CvtColorType::BGR2BGRA:
        case CvtColorType::BGRA2BGR:
        case CvtColorType::BGR2RGB:
        case CvtColorType::BGR2GRAY:
        case CvtColorType::RGB2GRAY:
        case CvtColorType::GRAY2BGR:
        case CvtColorType::GRAY2BGRA:
        case CvtColorType::BGRA2GRAY:
        case CvtColorType::RGBA2GRAY:
        case CvtColorType::RGB2YUV_NV12:
        case CvtColorType::RGB2YUV_NV21:
        case CvtColorType::RGB2YUV_YU12:
        case CvtColorType::RGB2YUV_YV12:
        case CvtColorType::RGB2YUV_Y444:
        case CvtColorType::RGB2YUV_NV12_601:
        case CvtColorType::RGB2YUV_NV21_601:
        case CvtColorType::RGB2YUV_YU12_601:
        case CvtColorType::RGB2YUV_YV12_601:
        case CvtColorType::RGB2YUV_Y444_601:
        case CvtColorType::YUV2RGB_NV12:
        case CvtColorType::YUV2RGB_NV21:
        case CvtColorType::YUV2RGB_YU12:
        case CvtColorType::YUV2RGB_YV12:
        case CvtColorType::YUV2RGB_Y422:
        case CvtColorType::YUV2RGB_YUYV:
        case CvtColorType::YUV2RGB_YVYU:
        case CvtColorType::YUV2RGB_Y444:
        case CvtColorType::YUV2RGB_NV12_601:
        case CvtColorType::YUV2RGB_NV21_601:
        case CvtColorType::YUV2RGB_YU12_601:
        case CvtColorType::YUV2RGB_YV12_601:
        case CvtColorType::YUV2RGB_Y422_601:
        case CvtColorType::YUV2RGB_YUYV_601:
        case CvtColorType::YUV2RGB_YVYU_601:
        case CvtColorType::YUV2RGB_Y444_601:
        {
            for (DT_S32 i = 0; i < src_len; i++)
            {
                if (src[i]->GetElemType() != ElemType::U8)
                {
                    AURA_XTENSA_LOG("src elem type should be u8");
                    return Status::ERROR;
                }
            }
            for (DT_S32 i = 0; i < dst_len; i++)
            {
                if (dst[i]->GetElemType() != ElemType::U8)
                {
                    AURA_XTENSA_LOG("dst elem type should be u8");
                    return Status::ERROR;
                }
            }
            break;
        }

        case CvtColorType::RGB2YUV_NV12_P010:
        case CvtColorType::RGB2YUV_NV21_P010:
        {
            for (DT_S32 i = 0; i < src_len; i++)
            {
                if (src[i]->GetElemType() != ElemType::U16)
                {
                    AURA_XTENSA_LOG("src elem type should be u16");
                    return Status::ERROR;
                }
            }
            for (DT_S32 i = 0; i < dst_len; i++)
            {
                if (dst[i]->GetElemType() != ElemType::U16)
                {
                    AURA_XTENSA_LOG("dst elem type should be u16");
                    return Status::ERROR;
                }
            }
            break;
        }

        case CvtColorType::BAYERBG2BGR:
        case CvtColorType::BAYERGB2BGR:
        case CvtColorType::BAYERRG2BGR:
        case CvtColorType::BAYERGR2BGR:
        {
            for (DT_S32 i = 0; i < src_len; i++)
            {
                if (src[i]->GetElemType() != ElemType::U8 && src[i]->GetElemType() != ElemType::U16)
                {
                    AURA_XTENSA_LOG("src elem type should be u8|u16");
                    ret = Status::ERROR;
                }
            }
            for (DT_S32 i = 0; i < dst_len; i++)
            {
                if (dst[i]->GetElemType() != ElemType::U8 && dst[i]->GetElemType() != ElemType::U16)
                {
                    AURA_XTENSA_LOG("dst elem type should be u8|u16");
                    ret = Status::ERROR;
                }
            }
            break;
        }

        default:
        {
            AURA_XTENSA_LOG("cvtcolor type error");
            ret = Status::ERROR;
        }
    }

    AURA_STATUS_RETURN(ret);
}

//=============================== CvtColorVdsp ===============================
CvtColorVdsp::CvtColorVdsp(TileManager tm, ExecuteMode mode) : VdspOp(tm, mode)
{
    do
    {
        xvTileManager *xv_tm = static_cast<xvTileManager*>(m_tm);
        if (NULL == xv_tm)
        {
            AURA_XTENSA_LOG("xv_tm is null ptr");
            break;
        }

        if (ExecuteMode::TILE == m_mode)
        {
            DT_VOID *buffer = xvAllocateBuffer(xv_tm, sizeof(CvtColorTile), XV_MEM_BANK_COLOR_ANY, 128);
            if (DT_NULL == buffer)
            {
                AURA_XTENSA_LOG("xvAllocateBuffer error");
                break;
            }

            m_impl = new(buffer) CvtColorTile(tm);
            if (DT_NULL == m_impl)
            {
                AURA_XTENSA_LOG("m_impl is null ptr");
                break;
            }
        }
        else if (ExecuteMode::FRAME == m_mode)
        {
            DT_VOID *buffer = xvAllocateBuffer(xv_tm, sizeof(CvtColorFrame), XV_MEM_BANK_COLOR_ANY, 128);
            if (DT_NULL == buffer)
            {
                AURA_XTENSA_LOG("xvAllocateBuffer error");
                break;
            }

            m_impl = new(buffer) CvtColorFrame(tm);
            if (DT_NULL == m_impl)
            {
                AURA_XTENSA_LOG("m_impl is null ptr");
                break;
            }
        }
        else
        {
            AURA_XTENSA_LOG("unsupport execute mode");
            break;
        }
    } while(0);
}

Status CvtColorVdsp::SetArgs(const vector<const Mat*> &src, const vector<Mat*> &dst, CvtColorType type)
{
    if (DT_NULL == m_impl)
    {
        AURA_XTENSA_LOG("m_impl is null ptr");
        return Status::ERROR;
    }

    if (src.empty() || dst.empty())
    {
        AURA_XTENSA_LOG("src/dst is empty");
        return Status::ERROR;
    }

    if (ExecuteMode::FRAME == m_mode)
    {
        CvtColorFrame *impl = static_cast<CvtColorFrame*>(m_impl);
        if (DT_NULL == impl)
        {
            AURA_XTENSA_LOG("impl is null ptr");
            return Status::ERROR;
        }

        return impl->SetArgs(src, dst, type);
    }
    else
    {
        AURA_XTENSA_LOG("execute mode is not frame");
        return Status::ERROR;
    }
}

Status CvtColorVdsp::SetArgs(const vector<TileWrapper> &src, vector<TileWrapper> &dst, CvtColorType type)
{
    if (DT_NULL == m_impl)
    {
        AURA_XTENSA_LOG("m_impl is null ptr");
        return Status::ERROR;
    }

    if (src.empty() || dst.empty())
    {
        AURA_XTENSA_LOG("src/dst is empty");
        return Status::ERROR;
    }

    if (ExecuteMode::TILE == m_mode)
    {
        CvtColorTile *impl = static_cast<CvtColorTile*>(m_impl);
        if (DT_NULL == impl)
        {
            AURA_XTENSA_LOG("impl is null ptr");
            return Status::ERROR;
        }

        return impl->SetArgs(src, dst, type);
    }
    else
    {
        AURA_XTENSA_LOG("execute mode is not tile");
        return Status::ERROR;
    }
}

AURA_VDSP_OP_CPP(CvtColor)

//============================ CvtColorTile ============================
static DT_S32 CvtColorVdspTileImpl(const xvTile *src, xvTile *dst, CvtColorType type)
{
    DT_S32 ret = AURA_XTENSA_ERROR;

    switch (type)
    {
        case CvtColorType::BGR2GRAY:
        case CvtColorType::RGB2GRAY:
        {
            ret = CvtBgr2GrayVdsp(src, dst, SwapBlue(type));
            break;
        }

        default:
        {
            AURA_XTENSA_LOG("cvtcolor type error");
            ret = AURA_XTENSA_ERROR;
        }
    }

    return ret;
}

CvtColorTile::CvtColorTile(TileManager tm) : VdspOpTile(tm), m_type(CvtColorType::INVALID), m_src_sizes(0), m_dst_sizes(0)
{}

Status CvtColorTile::SetArgs(const vector<TileWrapper> &src, vector<TileWrapper> &dst, CvtColorType type)
{
    if (DT_FALSE == m_flag)
    {
        m_type = type;
        m_src_sizes = src.size();
        m_dst_sizes = dst.size();

        for (DT_S32 i = 0; i < m_src_sizes; i++)
        {
            m_xv_src_tiles.push_back(&src[i]);
            m_src_elem_types.push_back(src[i].GetElemType());
            m_src_channels.push_back(src[i].GetChannel());
        }

        for (DT_S32 i = 0; i < m_dst_sizes; i++)
        {
            m_xv_dst_tiles.push_back(&dst[i]);
            m_dst_elem_types.push_back(dst[i].GetElemType());
            m_dst_channels.push_back(dst[i].GetChannel());
        }

        if (CheckNum<TileWrapper>(m_xv_src_tiles, m_xv_dst_tiles, type) != Status::OK)
        {
            AURA_XTENSA_LOG("tile num is not match");
            return Status::ERROR;
        }

        if (CheckElemType<TileWrapper>(m_xv_src_tiles, m_xv_dst_tiles, type) != Status::OK)
        {
            AURA_XTENSA_LOG("tile elem type is not match");
            return Status::ERROR;
        }

        m_flag = DT_TRUE;
    }
    else
    {
        for (DT_S32 i = 0; i < m_xv_src_tiles.size(); i++)
        {
            m_xv_src_tiles[i] = &src[i];
        }

        for (DT_S32 i = 0; i < m_xv_dst_tiles.size(); i++)
        {
            m_xv_dst_tiles[i] = &dst[i];
        }
    }

    return Status::OK;
}

Status CvtColorTile::DeInitialize()
{
    m_type = CvtColorType::INVALID;
    m_flag = DT_FALSE;
    m_src_sizes = 0;
    m_dst_sizes = 0;

    for (DT_S32 i = 0; i < m_src_elem_types.size(); i++)
    {
        m_src_elem_types[i] = ElemType::INVALID;
    }
    m_src_elem_types.clear();

    for (DT_S32 i = 0; i < m_dst_elem_types.size(); i++)
    {
        m_dst_elem_types[i] = ElemType::INVALID;
    }
    m_dst_elem_types.clear();

    for (DT_S32 i = 0; i < m_src_channels.size(); i++)
    {
        m_src_channels[i] = 0;
    }
    m_src_channels.clear();

    for (DT_S32 i = 0; i < m_dst_channels.size(); i++)
    {
        m_dst_channels[i] = 0;
    }
    m_dst_channels.clear();

    for (DT_S32 i = 0; i < m_xv_src_tiles.size(); i++)
    {
        m_xv_src_tiles[i] = DT_NULL;
    }
    m_xv_src_tiles.clear();

    for (DT_S32 i = 0; i < m_xv_dst_tiles.size(); i++)
    {
        m_xv_dst_tiles[i] = DT_NULL;
    }
    m_xv_dst_tiles.clear();

    return Status::OK;
}

Status CvtColorTile::Run()
{
    if (m_xv_src_tiles.empty() || m_xv_dst_tiles.empty())
    {
        AURA_XTENSA_LOG("xv_src_tiles/xv_dst_tiles is empty");
        return Status::ERROR;
    }

    vector<xvTile> src_tiles;
    for (DT_S32 i = 0; i < m_xv_src_tiles.size(); i++)
    {
        xvTile *p_src = static_cast<xvTile*>(const_cast<DT_VOID*>(m_xv_src_tiles[i]->GetData()));
        src_tiles.push_back(*p_src);
    }

    vector<xvTile> dst_tiles;
    for (DT_S32 i = 0; i < m_xv_dst_tiles.size(); i++)
    {
        xvTile *p_dst = static_cast<xvTile*>(m_xv_dst_tiles[i]->GetData());
        dst_tiles.push_back(*p_dst);
    }

    if (CvtColorVdspTileImpl(src_tiles.data(), dst_tiles.data(), m_type) != AURA_XTENSA_OK)
    {
        AURA_XTENSA_LOG("CvtColorVdspImpl failed");
        return Status::ERROR;
    }

    return Status::OK;
}

//============================ CvtColorFrame ============================
CvtColorFrame::CvtColorFrame(TileManager tm) : VdspOpFrame(tm, 2), m_type(CvtColorType::INVALID), m_src_sizes(0), m_dst_sizes(0), m_cvtcolor_tile(DT_NULL)
{
    do
    {
        xvTileManager *xv_tm = static_cast<xvTileManager*>(m_tm);
        if (DT_NULL == xv_tm)
        {
            AURA_XTENSA_LOG("xv_tm is null ptr");
            break;
        }

        DT_VOID *buffer = xvAllocateBuffer(xv_tm, sizeof(CvtColorTile), XV_MEM_BANK_COLOR_ANY, 128);
        if (DT_NULL == buffer)
        {
            AURA_XTENSA_LOG("xvAllocateBuffer failed");
            break;
        }

        m_cvtcolor_tile = new (buffer) CvtColorTile(m_tm);
        if (DT_NULL == m_cvtcolor_tile)
        {
            AURA_XTENSA_LOG("m_cvtcolor_tile is null ptr");
            break;
        }
    } while(0);
}

Status CvtColorFrame::SetArgs(const vector<const Mat*> &src, const vector<Mat*> &dst, CvtColorType type)
{
    xvTileManager *xv_tm = static_cast<xvTileManager*>(m_tm);
    if (DT_NULL == xv_tm)
    {
        AURA_XTENSA_LOG("xv_tm is null ptr");
        return Status::ERROR;
    }

    if (CheckNum<Mat>(src, dst, type) != Status::OK)
    {
        AURA_XTENSA_LOG("mat num is not match");
        return Status::ERROR;
    }

    if (CheckElemType<Mat>(src, dst, type) != Status::OK)
    {
        AURA_XTENSA_LOG("mat elem type is not match");
        return Status::ERROR;
    }

    m_type      = type;
    m_src_sizes = src.size();
    m_dst_sizes = dst.size();
    m_tile_num  = src.size() + dst.size();

    m_ref_tile = RefTileWrapper(m_src_sizes, m_dst_sizes, Point2i(0, 0), src[0]->GetSizes(), dst[0]->GetSizes(),
                                src[0]->GetElemType(), dst[0]->GetElemType(), 1);
    if (!m_ref_tile.IsValid())
    {
        AURA_XTENSA_LOG("m_ref_tile is invalid");
        return Status::ERROR;
    }

    for (DT_S32 i = 0; i < src.size(); i++)
    {
        // set BorderType::CONSTANT for no padding tile
        FrameWrapper frame(xv_tm, src[i], BorderType::CONSTANT, 0);
        if (!frame.IsValid())
        {
            AURA_XTENSA_LOG("frame is invalid");
            return Status::ERROR;
        }

        m_frames.push_back(frame);
        m_elem_types.push_back(src[i]->GetElemType());
        m_channels.push_back(src[i]->GetSizes().m_channel);
    }

    for (DT_S32 i = 0; i < dst.size(); i++)
    {
        FrameWrapper frame(xv_tm, dst[i], BorderType::CONSTANT, 0);
        if (!frame.IsValid())
        {
            AURA_XTENSA_LOG("frame is invalid");
            return Status::ERROR;
        }

        m_frames.push_back(frame);
        m_elem_types.push_back(dst[i]->GetElemType());
        m_channels.push_back(dst[i]->GetSizes().m_channel);
    }

    return Status::OK;
}

Status CvtColorFrame::DeInitialize()
{
    m_type      = CvtColorType::INVALID;
    m_src_sizes = 0;
    m_dst_sizes = 0;

    if (m_cvtcolor_tile != DT_NULL)
    {
        m_cvtcolor_tile->DeInitialize();
        m_cvtcolor_tile = DT_NULL;
    }

    VdspOpFrame::DeInitialize();

    return Status::OK;
}

DT_VOID CvtColorFrame::Prepare(xvTileManager *xv_tm, RefTile *xv_ref_tile, DT_VOID *obj, DT_VOID *tiles, DT_S32 flag)
{
    if ((DT_NULL == xv_tm) || (DT_NULL == obj) || (DT_NULL == tiles) || (DT_NULL == xv_ref_tile))
    {
        AURA_XTENSA_LOG("params are null ptr!");
        return;
    }

    if (sizeof(RefTile) != sizeof(RefTileWrapper))
    {
        AURA_XTENSA_LOG("sizeof(RefTile) != sizeof(RefTileWrapper)\n");
        return;
    }

    RefTileWrapper *ref_tile      = reinterpret_cast<RefTileWrapper*>(xv_ref_tile);
    CvtColorFrame *cvtcolor_frame = reinterpret_cast<CvtColorFrame*>(obj);

    if ((DT_NULL == ref_tile) || (DT_NULL == cvtcolor_frame))
    {
        AURA_XTENSA_LOG("ref_tile/cvtcolor_frame is null ptr");
        return;
    }

    DT_S32 tile_num  = cvtcolor_frame->m_tile_num;
    DT_S32 src_sizes = cvtcolor_frame->m_src_sizes;

    if ((cvtcolor_frame->m_frames.size() != tile_num) || (cvtcolor_frame->m_elem_types.size() != tile_num) || (cvtcolor_frame->m_channels.size() != tile_num))
    {
        AURA_XTENSA_LOG("frames/elem_types/channels size is not equal to m_tile_num\n");
        return;
    }

    aura::Sizes sizes(0, 0);
    Status ret       = Status::ERROR;
    xvTile *xv_tiles = static_cast<xvTile*>(tiles);

    for (DT_S32 i = 0; i < src_sizes; i++)
    {
        TileWrapper tile_src(static_cast<DT_VOID*>(xv_tiles + i), (cvtcolor_frame->m_elem_types)[i], (cvtcolor_frame->m_channels)[i]);
        ret = tile_src.Update(ref_tile->x, ref_tile->y, ref_tile->tile_width, ref_tile->tile_height, sizes);
        if (ret != Status::OK)
        {
            AURA_XTENSA_LOG("Update failed!\n");
            return;
        }

        ret = tile_src.Register(xv_tm, DT_NULL, cvtcolor_frame->m_frames[i], XV_INPUT_TILE, flag);
        if (ret != Status::OK)
        {
            AURA_XTENSA_LOG("Register failed!\n");
            return;
        }
    }

    for (DT_S32 i = src_sizes; i < static_cast<DT_S32>(tile_num); i++)
    {
        TileWrapper tile_dst(static_cast<DT_VOID*>(xv_tiles + i), cvtcolor_frame->m_elem_types[i], cvtcolor_frame->m_channels[i]);
        ret = tile_dst.Update(ref_tile->x, ref_tile->y, ref_tile->tile_width, ref_tile->tile_height, sizes);
        if (ret != Status::OK)
        {
            AURA_XTENSA_LOG("Update failed!\n");
            return;
        }

        ret = tile_dst.Register(xv_tm, DT_NULL, cvtcolor_frame->m_frames[i], XV_OUTPUT_TILE, flag);
        if (ret != Status::OK)
        {
            AURA_XTENSA_LOG("Register failed!\n");
            return;
        }
    }

    return;
}

DT_S32 CvtColorFrame::Execute(DT_VOID *obj, DT_VOID *tiles)
{
    if ((DT_NULL == obj) || (DT_NULL == tiles))
    {
        AURA_XTENSA_LOG("obj/tiles is null ptr");
        return AURA_XTENSA_ERROR;
    }

    CvtColorFrame *cvtcolor_frame = static_cast<CvtColorFrame*>(obj);
    if (DT_NULL == cvtcolor_frame)
    {
        AURA_XTENSA_LOG("cvtcolor_frame is null ptr");
        return AURA_XTENSA_ERROR;
    }

    DT_S32 tile_num  = cvtcolor_frame->m_tile_num;
    DT_S32 src_sizes = cvtcolor_frame->m_src_sizes;

    if ((cvtcolor_frame->m_elem_types.size() != tile_num) || (cvtcolor_frame->m_channels.size() != tile_num))
    {
        AURA_XTENSA_LOG("tiles/elem_types/channels size is not equal to m_tile_num\n");
        return AURA_XTENSA_ERROR;
    }

    /*!< Set Tiles for Op run. */
    vector<TileWrapper> vec_src;
    vector<TileWrapper> vec_dst;
    xvTile *xv_tile = static_cast<xvTile*>(tiles);

    for (DT_S32 i = 0; i < src_sizes; i++)
    {
        TileWrapper tile_src(static_cast<DT_VOID*>(xv_tile + i), cvtcolor_frame->m_elem_types[i], cvtcolor_frame->m_channels[i]);
        vec_src.push_back(tile_src);
    }

    for (DT_S32 i = src_sizes; i < tile_num; i++)
    {
        TileWrapper tile_dst(static_cast<DT_VOID*>(xv_tile + i), cvtcolor_frame->m_elem_types[i], cvtcolor_frame->m_channels[i]);
        vec_dst.push_back(tile_dst);
    }

    Status ret = cvtcolor_frame->m_cvtcolor_tile->SetArgs(vec_src, vec_dst, cvtcolor_frame->m_type);
    if (ret != Status::OK)
    {
        AURA_XTENSA_LOG("cvtcolor tile SetArgs failed");
        return AURA_XTENSA_ERROR;
    }

    ret = cvtcolor_frame->m_cvtcolor_tile->Initialize();
    if (ret != Status::OK)
    {
        AURA_XTENSA_LOG("cvtcolor tile Initialize failed");
        return AURA_XTENSA_ERROR;
    }

    ret = cvtcolor_frame->m_cvtcolor_tile->Run();
    if (ret != Status::OK)
    {
        AURA_XTENSA_LOG("cvtcolor tile Run failed");
        return AURA_XTENSA_ERROR;
    }

    return AURA_XTENSA_OK;
}

AURA_VDSP_OP_FRAME_CPP(CvtColor)

//============================ CvtColorRpc ============================
Status CvtColorRpc(TileManager xv_tm, XtensaRpcParam &rpc_param)
{
    if (DT_NULL == xv_tm)
    {
        AURA_XTENSA_LOG("xv_tm is null ptr");
        return Status::ERROR;
    }

    vector<Mat> src;
    vector<Mat> dst;
    CvtColorType type = CvtColorType::INVALID;

    CvtColorInParamVdsp in_param(rpc_param);
    Status ret = in_param.Get(src, dst, type);
    if (ret != Status::OK)
    {
        AURA_XTENSA_LOG("Get failed");
        return Status::ERROR;
    }

    vector<const Mat*> vec_src;
    vector<Mat*> vec_dst;

    for (DT_S32 i = 0; i < src.size(); i++)
    {
        const Mat *p_src = &src[i];
        vec_src.push_back(p_src);
    }

    for (DT_S32 i = 0; i < dst.size(); i++)
    {
        Mat *p_dst = &dst[i];
        vec_dst.push_back(p_dst);
    }

    CvtColorVdsp cvtcolor(xv_tm);
    return OpCall(cvtcolor, vec_src, vec_dst, type);
}

AURA_XTENSA_RPC_FUNC_REGISTER("aura.ops.cvtcolor.CvtColor", CvtColorRpc)

} // namespace xtensa
} // namespace aura