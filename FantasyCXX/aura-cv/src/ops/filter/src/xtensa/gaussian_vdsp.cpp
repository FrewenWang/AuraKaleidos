#include "xtensa/gaussian_vdsp_impl.hpp"

namespace aura
{
namespace xtensa
{

AURA_INLINE Status GetGaussianKmat(xvTileManager *xv_tm, DT_S32 ksize, const vector<DT_F32> &in_kernel, DT_VOID *&out_kernel)
{
    out_kernel = xvAllocateBuffer(xv_tm, ksize * sizeof(DT_F32), XV_MEM_BANK_COLOR_ANY, 128);
    if (DT_NULL == out_kernel)
    {
        AURA_XTENSA_LOG("xvAllocateBuffer failed");
        return Status::ERROR;
    }

    Memcpy(out_kernel, in_kernel.data(), ksize * sizeof(DT_F32));

    return Status::OK;
}

template <typename Tp, DT_U32 Q>
AURA_INLINE Status GetGaussianKmat(xvTileManager *xv_tm, DT_S32 ksize, const vector<DT_F32> &in_kernel, DT_VOID *&out_kernel)
{
    out_kernel = xvAllocateBuffer(xv_tm, ksize * sizeof(Tp), XV_MEM_BANK_COLOR_ANY, 128);
    if (DT_NULL == out_kernel)
    {
        AURA_XTENSA_LOG("xvAllocateBuffer failed");
        return Status::ERROR;
    }

    Tp *ker_row = static_cast<Tp*>(out_kernel);

    DT_S32 sum = 0;
    DT_F32 err = 0.f;

    for (DT_S32 i = 0; i < ksize / 2; i++)
    {
        DT_F32 quan_kernel     = in_kernel[i] * (1 << Q) + err;
        Tp result              = static_cast<Tp>(quan_kernel + 0.5);
        err                    = quan_kernel - (DT_F32)result;
        ker_row[i]             = result;
        ker_row[ksize - 1 - i] = result;
        sum += result;
    }

    ker_row[ksize / 2] = (1 << Q) - sum * 2;

    return Status::OK;
}

//=============================== GaussianVdsp ===============================
GaussianVdsp::GaussianVdsp(TileManager tm, ExecuteMode mode) : VdspOp(tm, mode)
{
    do
    {
        xvTileManager *xv_tm = static_cast<xvTileManager*>(m_tm);
        if (DT_NULL == xv_tm)
        {
            AURA_XTENSA_LOG("xv_tm is null ptr");
            break;
        }

        if (ExecuteMode::TILE == m_mode)
        {
            DT_VOID *buffer = xvAllocateBuffer(xv_tm, sizeof(GaussianTile), XV_MEM_BANK_COLOR_ANY, 128);
            if (DT_NULL == buffer)
            {
                AURA_XTENSA_LOG("xvAllocateBuffer error");
                break;
            }

            m_impl = new(buffer) GaussianTile(tm);
            if (DT_NULL == m_impl)
            {
                AURA_XTENSA_LOG("m_impl is null ptr");
                break;
            }
        }
        else if (ExecuteMode::FRAME == m_mode)
        {
            DT_VOID *buffer = xvAllocateBuffer(xv_tm, sizeof(GaussianFrame), XV_MEM_BANK_COLOR_ANY, 128);
            if (DT_NULL == buffer)
            {
                AURA_XTENSA_LOG("xvAllocateBuffer error");
                break;
            }

            m_impl = new(buffer) GaussianFrame(tm);
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

Status GaussianVdsp::SetArgs(const Mat *src, Mat *dst, DT_S32 ksize, DT_F32 sigma, BorderType border_type, const Scalar &border_value)
{
    if ((DT_NULL == src) || (DT_NULL == dst))
    {
        AURA_XTENSA_LOG("src/dst is null ptr");
        return Status::ERROR;
    }

    if (ExecuteMode::FRAME == m_mode)
    {
        GaussianFrame *impl = static_cast<GaussianFrame*>(m_impl);
        if (DT_NULL == impl)
        {
            AURA_XTENSA_LOG("impl is null ptr");
            return Status::ERROR;
        }

        return impl->SetArgs(src, dst, ksize, sigma, border_type, border_value);
    }
    else
    {
        AURA_XTENSA_LOG("execute mode is not frame");
        return Status::ERROR;
    }
}

Status GaussianVdsp::SetArgs(const TileWrapper *src, TileWrapper *dst, DT_S32 ksize, DT_F32 sigma)
{
    if (ExecuteMode::TILE == m_mode)
    {
        GaussianTile *impl = static_cast<GaussianTile*>(m_impl);
        if (DT_NULL == impl)
        {
            AURA_XTENSA_LOG("impl is null ptr");
            return Status::ERROR;
        }

        return impl->SetArgs(src, dst, ksize, sigma);
    }
    else
    {
        AURA_XTENSA_LOG("execute mode is not tile");
        return Status::ERROR;
    }
}

AURA_VDSP_OP_CPP(Gaussian)

//============================ GaussianTile ============================
GaussianTile::GaussianTile(TileManager tm) : VdspOpTile(tm), m_ksize(0), m_sigma(0.f), m_elem_type(ElemType::INVALID), m_channel(0),
                                             m_xv_src_tile(DT_NULL), m_xv_dst_tile(DT_NULL), m_kernel(DT_NULL)
{}

Status GaussianTile::SetArgs(const TileWrapper *src, TileWrapper *dst, DT_S32 ksize, DT_F32 sigma)
{
    if (DT_FALSE == m_flag)
    {
        if (src->GetElemType() != dst->GetElemType())
        {
            AURA_XTENSA_LOG("src dst elem_type is not equal");
            return Status::ERROR;
        }

        if (src->GetChannel() != dst->GetChannel())
        {
            AURA_XTENSA_LOG("src dst channel is not equal");
            return Status::ERROR;
        }

        if (ksize != 3)
        {
            AURA_XTENSA_LOG("ksize only support 3");
            return Status::ERROR;
        }

        if (src->GetChannel() != 1)
        {
            AURA_XTENSA_LOG("channel only support 1");
            return Status::ERROR;
        }

        ElemType elem_type = src->GetElemType();
        if (elem_type != ElemType::U8 && elem_type != ElemType::U16)
        {
            AURA_XTENSA_LOG("elem_type only support u8/u16");
            return Status::ERROR;
        }

        m_ksize = ksize;
        m_sigma = sigma;
        m_elem_type = src->GetElemType();
        m_channel = src->GetChannel();

        m_xv_src_tile = src;
        m_xv_dst_tile = dst;

        m_flag = DT_TRUE;
    }
    else
    {
        m_xv_src_tile = src;
        m_xv_dst_tile = dst;
    }

    return Status::OK;
}

Status GaussianTile::PrepareKmat()
{
    xvTileManager *xv_tm = static_cast<xvTileManager*>(m_tm);
    if (DT_NULL == xv_tm)
    {
        AURA_XTENSA_LOG("xv_tm is null ptr");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    vector<DT_F32> kernel(m_ksize, 0);
    kernel = GetGaussianKernel(m_ksize, m_sigma);

#define GET_GAUSSIAN_KMAT(type)                                                             \
    constexpr DT_U32 Q = GaussianTraits<type>::Q;                                           \
                                                                                            \
    ret = GetGaussianKmat<type, Q>(xv_tm, m_ksize, kernel, m_kernel);                       \

    switch (m_elem_type)
    {
        case ElemType::U8:
        {
            GET_GAUSSIAN_KMAT(DT_U8)
            break;
        }

        case ElemType::U16:
        {
            GET_GAUSSIAN_KMAT(DT_U16)
            break;
        }

        case ElemType::S16:
        {
            GET_GAUSSIAN_KMAT(DT_S16)
            break;
        }

        case ElemType::U32:
        {
            GET_GAUSSIAN_KMAT(DT_U32)
            break;
        }

        case ElemType::S32:
        {
            GET_GAUSSIAN_KMAT(DT_S32)
            break;
        }

        case ElemType::F32:
        {
            ret = GetGaussianKmat(xv_tm, m_ksize, kernel, m_kernel);
            break;
        }

        default:
        {
            AURA_XTENSA_LOG("Unsupported source format");
            return Status::ERROR;
        }
    }

#undef GET_GAUSSIAN_KMAT

    AURA_STATUS_RETURN(ret);
}

Status GaussianTile::Initialize()
{
    if (DT_NULL == m_kernel)
    {
        // Prepare kmat
        if (PrepareKmat() != Status::OK)
        {
            AURA_XTENSA_LOG("PrepareKmat failed");
            return Status::ERROR;
        }
    }

    return Status::OK;
}

Status GaussianTile::DeInitialize()
{
    m_ksize       = 0;
    m_sigma       = 0.f;
    m_elem_type   = ElemType::INVALID;
    m_channel     = 0;
    m_xv_src_tile = DT_NULL;
    m_xv_dst_tile = DT_NULL;
    m_kernel      = DT_NULL;

    return Status::OK;
}

Status GaussianTile::Run()
{
    if (DT_NULL == m_xv_src_tile || DT_NULL == m_xv_dst_tile)
    {
        AURA_XTENSA_LOG("xv_src_tile/xv_dst_tile is null ptr");
        return Status::ERROR;
    }

    DT_S32 ret = AURA_XTENSA_ERROR;

    const xvTile *xv_src = static_cast<const xvTile*>(m_xv_src_tile->GetData());
    xvTile *xv_dst = static_cast<xvTile*>(m_xv_dst_tile->GetData());
    if (xv_src == DT_NULL || xv_dst == DT_NULL)
    {
        AURA_XTENSA_LOG("xv_src_tile/xv_dst_tile data is null ptr");
        return Status::ERROR;
    }

    switch (m_ksize)
    {
        case 3:
        {
            ret = Gaussian3x3Vdsp(xv_src, xv_dst, m_kernel, m_elem_type, m_channel);
            break;
        }

        default:
        {
            AURA_XTENSA_LOG("Unsupported ksize");
            return Status::ERROR;
        }
    }

    AURA_XTENSA_RETURN(ret);
}

//============================ GaussianFrame ============================
GaussianFrame::GaussianFrame(TileManager tm) : VdspOpFrame(tm, 2), m_ksize(0), m_sigma(0.f), m_gaussian_tile(DT_NULL)
{
    do
    {
        xvTileManager *xv_tm = static_cast<xvTileManager*>(m_tm);
        if (DT_NULL == xv_tm)
        {
            AURA_XTENSA_LOG("xv_tm is null ptr");
            break;
        }

        DT_VOID *buffer = xvAllocateBuffer(xv_tm, sizeof(GaussianTile), XV_MEM_BANK_COLOR_ANY, 128);
        if (DT_NULL == buffer)
        {
            AURA_XTENSA_LOG("xvAllocateBuffer failed");
            break;
        }

        m_gaussian_tile = new (buffer) GaussianTile(m_tm);
        if (DT_NULL == m_gaussian_tile)
        {
            AURA_XTENSA_LOG("m_gaussian_tile is null ptr");
            break;
        }
    } while(0);
}

Status GaussianFrame::SetArgs(const Mat *src, Mat *dst, DT_S32 ksize, DT_F32 sigma, BorderType border_type, const Scalar &border_value)
{
    xvTileManager *xv_tm = static_cast<xvTileManager*>(m_tm);
    if (DT_NULL == xv_tm)
    {
        AURA_XTENSA_LOG("xv_tm is null ptr");
        return Status::ERROR;
    }

    if (!(src->IsValid() && dst->IsValid()))
    {
        AURA_XTENSA_LOG("invalid src or dst");
        return Status::ERROR;
    }

    if (!src->IsEqual(*dst))
    {
        AURA_XTENSA_LOG("src and dst should have the same size");
        return Status::ERROR;
    }

    if ((src->GetSizes().m_height < ksize) || (src->GetSizes().m_width < ksize))
    {
        AURA_XTENSA_LOG("height/width must bigger than ksize");
        return Status::ERROR;
    }

    if (ksize != 3)
    {
        AURA_XTENSA_LOG("ksize only support 3");
        return Status::ERROR;
    }

    DT_S32 ch = src->GetSizes().m_channel;
    if (ch != 1)
    {
        AURA_XTENSA_LOG("channel only support 1");
        return Status::ERROR;
    }

    ElemType elem_type = src->GetElemType();
    if ((elem_type != ElemType::U8) && (elem_type != ElemType::U16))
    {
        AURA_XTENSA_LOG("elem_type only support u8/u16");
        return Status::ERROR;
    }

    m_ksize     = ksize;
    m_sigma     = sigma;
    m_src_sizes = 1;
    m_dst_sizes = 1;

    m_ref_tile = RefTileWrapper(m_tile_num, Point2i(0, 0), dst->GetSizes(), dst->GetElemType(), 1);
    if (!m_ref_tile.IsValid())
    {
        AURA_XTENSA_LOG("m_ref_tile is invalid");
        return Status::ERROR;
    }

    FrameWrapper src_frame(xv_tm, src, border_type, border_value);
    FrameWrapper dst_frame(xv_tm, dst, border_type, border_value);
    if ((!src_frame.IsValid()) || (!dst_frame.IsValid()))
    {
        AURA_XTENSA_LOG("src_frame/dst_frame is invalid");
        return Status::ERROR;
    }

    m_frames.push_back(src_frame);
    m_frames.push_back(dst_frame);
    m_elem_types.push_back(src->GetElemType());
    m_elem_types.push_back(dst->GetElemType());
    m_channels.push_back(src->GetSizes().m_channel);
    m_channels.push_back(dst->GetSizes().m_channel);

    return Status::OK;
}

Status GaussianFrame::DeInitialize()
{
    if (m_gaussian_tile != DT_NULL)
    {
        m_gaussian_tile->DeInitialize();
        m_gaussian_tile = DT_NULL;
    }

    m_ksize     = 0;
    m_sigma     = 0.f;
    m_src_sizes = 0;
    m_dst_sizes = 0;

    VdspOpFrame::DeInitialize();

    return Status::OK;
}

DT_VOID GaussianFrame::Prepare(xvTileManager *xv_tm, RefTile *xv_ref_tile, DT_VOID *obj, DT_VOID *tiles, DT_S32 flag)
{
    if ((DT_NULL == xv_tm) || (DT_NULL == obj) || (DT_NULL == tiles))
    {
        AURA_XTENSA_LOG("xv_tm/obj/tiles is null ptr!");
        return;
    }

    if (sizeof(RefTile) != sizeof(RefTileWrapper))
    {
        AURA_XTENSA_LOG("sizeof(RefTile) != sizeof(RefTileWrapper)\n");
        return;
    }

    RefTileWrapper *ref_tile      = reinterpret_cast<RefTileWrapper*>(xv_ref_tile);
    GaussianFrame *gaussian_frame = static_cast<GaussianFrame*>(obj);
    if ((DT_NULL == ref_tile) || (DT_NULL == gaussian_frame))
    {
        AURA_XTENSA_LOG("ref_tile/gaussian_frame is null ptr");
        return;
    }

    DT_S32 tile_num = gaussian_frame->m_tile_num;
    if ((gaussian_frame->m_frames.size() != tile_num) || (gaussian_frame->m_elem_types.size() != tile_num) || (gaussian_frame->m_channels.size() != tile_num))
    {
        AURA_XTENSA_LOG("frames/elem_types/channels size is not equal to m_tile_num\n");
        return;
    }

    Status ret       = Status::ERROR;
    DT_S32 src_sizes = gaussian_frame->m_src_sizes;
    DT_S32 dst_sizes = gaussian_frame->m_dst_sizes;
    xvTile *xv_tiles = static_cast<xvTile*>(tiles);

    for (DT_S32 i = 0; i < src_sizes; i++)
    {
        TileWrapper tile_src(static_cast<DT_VOID*>(xv_tiles + i), (gaussian_frame->m_elem_types)[i], (gaussian_frame->m_channels)[i]);
        Sizes sizes(gaussian_frame->m_ksize / 2, gaussian_frame->m_ksize / 2);
        ret = tile_src.Update(ref_tile->x, ref_tile->y, ref_tile->tile_width, ref_tile->tile_height, sizes);
        if (ret != Status::OK)
        {
            AURA_XTENSA_LOG("Update failed!\n");
            return;
        }

        ret = tile_src.Register(xv_tm, DT_NULL, gaussian_frame->m_frames[i], XV_INPUT_TILE, flag);
        if (ret != Status::OK)
        {
            AURA_XTENSA_LOG("Register failed!\n");
            return;
        }
    }

    for (DT_S32 i = src_sizes; i < src_sizes + dst_sizes; i++)
    {
        TileWrapper tile_dst(static_cast<DT_VOID*>(xv_tiles + i), gaussian_frame->m_elem_types[i], gaussian_frame->m_channels[i]);
        Sizes sizes(0, 0);
        ret = tile_dst.Update(ref_tile->x, ref_tile->y, ref_tile->tile_width, ref_tile->tile_height, sizes);
        if (ret != Status::OK)
        {
            AURA_XTENSA_LOG("Update failed!\n");
            return;
        }

        ret = tile_dst.Register(xv_tm, DT_NULL, gaussian_frame->m_frames[i], XV_OUTPUT_TILE, flag);
        if (ret != Status::OK)
        {
            AURA_XTENSA_LOG("Register failed!\n");
            return;
        }
    }

    return;
}

DT_S32 GaussianFrame::Execute(DT_VOID *obj, DT_VOID *tiles)
{
    if ((DT_NULL == obj) || (DT_NULL == tiles))
    {
        AURA_XTENSA_LOG("obj/tiles is null ptr");
        return AURA_XTENSA_ERROR;
    }

    GaussianFrame *gaussian_frame = static_cast<GaussianFrame*>(obj);
    if (DT_NULL == gaussian_frame)
    {
        AURA_XTENSA_LOG("gaussian_frame is null ptr");
        return AURA_XTENSA_ERROR;
    }

    if (DT_NULL == gaussian_frame->m_gaussian_tile)
    {
        AURA_XTENSA_LOG("m_gaussian_tile is null ptr");
        return AURA_XTENSA_ERROR;
    }

    xvTile *xv_src = static_cast<xvTile*>(tiles);
    xvTile *xv_dst = static_cast<xvTile*>(xv_src + 1);
    TileWrapper tile_src(xv_src, gaussian_frame->m_elem_types[0], gaussian_frame->m_channels[0]);
    TileWrapper tile_dst(xv_dst, gaussian_frame->m_elem_types[1], gaussian_frame->m_channels[1]);

    Status ret = gaussian_frame->m_gaussian_tile->SetArgs(&tile_src, &tile_dst, gaussian_frame->m_ksize, gaussian_frame->m_sigma);
    if (ret != Status::OK)
    {
        AURA_XTENSA_LOG("gaussian tile SetArgs failed");
        return AURA_XTENSA_ERROR;
    }

    ret = gaussian_frame->m_gaussian_tile->Initialize();
    if (ret != Status::OK)
    {
        AURA_XTENSA_LOG("gaussian tile Initialize failed");
        return AURA_XTENSA_ERROR;
    }

    ret = gaussian_frame->m_gaussian_tile->Run();
    if (ret != Status::OK)
    {
        AURA_XTENSA_LOG("gaussian tile Run failed");
        return AURA_XTENSA_ERROR;
    }

    return AURA_XTENSA_OK;
}

AURA_VDSP_OP_FRAME_CPP(Gaussian)

//============================ GetGaussianKernel ============================
vector<DT_F32> GetGaussianKernel(DT_S32 ksize, DT_F32 sigma)
{
    vector<DT_F32> kernel(ksize, 0);
    constexpr DT_S32 small_gaussian_size = 7;
    constexpr DT_F32 small_gaussian_tab[][small_gaussian_size] =
    {
        {1.f},
        {0.25f, 0.5f, 0.25f},
        {0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f},
        {0.03125f, 0.109375f, 0.21875f, 0.28125f, 0.21875f, 0.109375f, 0.03125f}
    };

    if ((ksize <= small_gaussian_size) && (sigma <= 0))
    {
        const DT_F32 *t_ptr = small_gaussian_tab[ksize >> 1];
        for (DT_S32 i = 0; i < ksize; i++)
        {
            kernel[i] = t_ptr[i];
        }

        return kernel;
    }

    vector<DT_F32> vec_kernel(ksize, 0);
    DT_F64 sigma_value = sigma > 0 ? static_cast<DT_F64>(sigma) : ((ksize - 1) * 0.5 - 1) * 0.3 + 0.8;
    DT_F64 sigma2 = -0.5 / (sigma_value * sigma_value);
    DT_F64 sum = 0;

    for (DT_S32 i = 0; i < ksize; i++)
    {
        DT_F64 x = i - (ksize - 1) * 0.5;
        vec_kernel[i] = static_cast<DT_F32>(Exp(sigma2 * x * x));
        sum += vec_kernel[i];
    }

    sum = 1.0 / sum;

    for (DT_S32 i = 0; i < ksize; i++)
    {
        kernel[i] = static_cast<DT_F32>(vec_kernel[i] * sum);
    }

    return kernel;
}

//============================ GaussianRpc ============================
Status GaussianRpc(TileManager xv_tm, XtensaRpcParam &rpc_param)
{
    if (DT_NULL == xv_tm)
    {
        AURA_XTENSA_LOG("xv_tm is null ptr");
        return Status::ERROR;
    }

    Mat src;
    Mat dst;
    DT_S32 ksize;
    DT_F32 sigma;
    BorderType border_type;
    Scalar border_value;
    GaussianInParamVdsp in_param(rpc_param);
    Status ret = in_param.Get(src, dst, ksize, sigma, border_type, border_value);
    if (ret != Status::OK)
    {
        AURA_XTENSA_LOG("Get rpc param failed!");
        return Status::ERROR;
    }

    GaussianVdsp gaussian(xv_tm);
    return OpCall(gaussian, &src, &dst, ksize, sigma, border_type, border_value);
}

AURA_XTENSA_RPC_FUNC_REGISTER("aura.ops.filter.Gaussian", GaussianRpc)

} // namespace xtensa
} // namespace aura