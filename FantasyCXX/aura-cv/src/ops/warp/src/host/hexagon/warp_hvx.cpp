#include "warp_impl.hpp"
#include "warp_comm.hpp"

namespace aura
{

static Status WarpAffineHvx(Context *ctx, const Mat &src, const Mat &matrix, Mat &dst, WarpType warp_type, InterpType interp_type, BorderType border_type,
                            const Scalar &border_value, std::string &profiling_string, const OpTarget &target)
{
    if (MI_NULL == ctx)
    {
        return Status::ERROR;
    }

    HexagonEngine *engine = ctx->GetHexagonEngine();
    if (MI_NULL == engine)
    {
        AURA_ADD_ERROR_STRING(ctx, "Hexagon engine is nullptr");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    MI_S32 mesh_h = (dst.GetSizes().m_height + 15) / 16 + 1;
    MI_S32 mesh_w = (dst.GetSizes().m_width + 15) / 16 + 1;
    MI_S32 stride = (mesh_w * 2 * ElemTypeSize(ElemType::S32) + 128) & (-64);

    Mat grid = Mat(ctx, ElemType::S32, aura::Sizes3(mesh_h, mesh_w, 2), AURA_MEM_DMA_BUF_HEAP, aura::Sizes(mesh_h, stride));
    if (!grid.IsValid())
    {
        AURA_ADD_ERROR_STRING(ctx, "grid is invalid");
        return Status::ERROR;
    }

    ret = InitMapGrid(ctx, matrix, grid, 16);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "InitMapGrid failed");
        return Status::ERROR;
    }

    HexagonRpcParam rpc_param(ctx);
    WarpInParam     in_param(ctx, rpc_param);

    ret = in_param.Set(src, grid, dst, warp_type, interp_type, border_type, border_value);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "Set failed");
        return Status::ERROR;
    }

    HexagonProfiling profiling;
    ret = engine->Run(AURA_OPS_WARP_PACKAGE_NAME, AURA_OPS_WARP_OP_NAME, rpc_param, &profiling);
    if (Status::OK == ret && MI_TRUE == target.m_data.hvx.profiling)
    {
        profiling_string = " " + HexagonProfilingToString(profiling);
    }

    AURA_RETURN(ctx, ret);
}

WarpHvx::WarpHvx(Context *ctx, WarpType warp_type, const OpTarget &target) : WarpImpl(ctx, warp_type, target)
{}

Status WarpHvx::SetArgs(const Array *src, const Array *matrix, Array *dst, InterpType interp_type,
                        BorderType border_type, const Scalar &border_value)
{
    if (WarpImpl::SetArgs(src, matrix, dst, interp_type, border_type, border_value) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "WarpImpl::Initialize failed");
        return Status::ERROR;
    }

    if ((src->GetArrayType() != ArrayType::MAT) || (matrix->GetArrayType() != ArrayType::MAT) || (dst->GetArrayType() != ArrayType::MAT))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src matrix dst must be mat type");
        return Status::ERROR;
    }

    if ((m_warp_type == WarpType::AFFINE && (matrix->GetSizes().m_height != 2 || matrix->GetSizes().m_width != 3)) ||
        (m_warp_type == WarpType::PERSPECTIVE && (matrix->GetSizes().m_height != 3 || matrix->GetSizes().m_width != 3)))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "the size of transform matrix is invalid");
        return Status::ERROR;
    }

    if (matrix->GetElemType() != ElemType::F64)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "the type of transform matrix should be f64");
        return Status::ERROR;
    }

    const Mat *src_mat    = dynamic_cast<const Mat*>(src);
    const Mat *matrix_mat = dynamic_cast<const Mat*>(matrix);
    Mat       *dst_mat    = dynamic_cast<Mat*>(dst);
    if ((MI_NULL == src_mat || (MI_NULL == matrix_mat)) || (MI_NULL == dst_mat))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src or matrix or dst is not mat");
        return Status::ERROR;
    }

    if ((src_mat->GetMemType() != AURA_MEM_DMA_BUF_HEAP) || (dst_mat->GetMemType() != AURA_MEM_DMA_BUF_HEAP))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "mat memory must be AURA_MEM_DMA_BUF_HEAP type");
        return Status::ERROR;
    }

    MI_S32 ch = src->GetSizes().m_channel;
    if (ch != 1)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "channel only support 1");
        return Status::ERROR;
    }

    ElemType elem_type = src->GetElemType();
    if (elem_type != ElemType::U8)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "elem_type only support u8");
        return Status::ERROR;
    }

    if (interp_type != InterpType::NEAREST && interp_type != InterpType::LINEAR)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "interp_type only support nearest/linear");
        return Status::ERROR;
    }

    if (border_type != BorderType::CONSTANT && border_type != BorderType::REPLICATE)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "border_type only support constant/replicate");
        return Status::ERROR;
    }

    return Status::OK;
}

Status WarpHvx::Run()
{
    const Mat *src    = dynamic_cast<const Mat*>(m_src);
    const Mat *matrix = dynamic_cast<const Mat*>(m_matrix);
    Mat       *dst    = dynamic_cast<Mat*>(m_dst);

    if ((MI_NULL == src) || (MI_NULL == matrix) || (MI_NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src or matrix or dst is nullptr");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    switch (m_warp_type)
    {
        case WarpType::AFFINE:
        {
            ret = WarpAffineHvx(m_ctx, *src, *matrix, *dst, m_warp_type, m_interp_type, m_border_type,
                                m_border_value, m_profiling_string, m_target);
            break;
        }

        case WarpType::PERSPECTIVE:
        {
            /* code */
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(m_ctx, "unsupported warp type");
        }
    }

    AURA_RETURN(m_ctx, ret);
}

std::string WarpHvx::ToString() const
{ 
    return WarpImpl::ToString() + m_profiling_string;
}

Status WarpAffineCoordHvx(Context *ctx, const Mat &matrix, Mat &map_xy, WarpType warp_type, const OpTarget &target)
{
    if (MI_NULL == ctx)
    {
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    std::string profiling_string;

    MI_S32 mesh_h = (map_xy.GetSizes().m_height + 15) / 16 + 1;
    MI_S32 mesh_w = (map_xy.GetSizes().m_width + 15) / 16 + 1;
    MI_S32 stride = (mesh_w * 2 * ElemTypeSize(ElemType::S32) + 128) & (-64);

    Mat grid = Mat(ctx, ElemType::S32, aura::Sizes3(mesh_h, mesh_w, 2), AURA_MEM_DMA_BUF_HEAP, aura::Sizes(mesh_h, stride));
    if (!grid.IsValid())
    {
        AURA_ADD_ERROR_STRING(ctx, "grid is invalid");
        return Status::ERROR;
    }

    ret = InitMapGrid(ctx, matrix, grid, 16);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "InitMapGrid failed");
        return Status::ERROR;
    }

    HexagonRpcParam  rpc_param(ctx);
    WarpCoordInParam in_param(ctx, rpc_param);

    ret = in_param.Set(grid, map_xy, warp_type);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "Set failed");
        return Status::ERROR;
    }

    HexagonProfiling profiling;
    HexagonEngine   *engine = ctx->GetHexagonEngine();

    ret = engine->Run(AURA_OPS_WARP_PACKAGE_NAME, AURA_OPS_WARP_COORD_OP_NAME, rpc_param, &profiling);
    if (Status::OK == ret && MI_TRUE == target.m_data.hvx.profiling)
    {
        profiling_string = " " + HexagonProfilingToString(profiling);
    }

    AURA_RETURN(ctx, ret);
}

Status WarpCoordHvx(Context *ctx, const Mat &matrix, Mat &map_xy, WarpType warp_type, const OpTarget &target)
{
    if (MI_NULL == ctx)
    {
        return Status::ERROR;
    }

    if ((map_xy.GetStrides().m_width & (AURA_HVLEN * 4 - 1)) != 0)
    {
        AURA_ADD_ERROR_STRING(ctx, "map_xy stride must align to AURA_HVLEN * 4");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    switch (warp_type)
    {
        case WarpType::AFFINE:
        {
            ret = WarpAffineCoordHvx(ctx, matrix, map_xy, warp_type, target);
            break;
        }

        case WarpType::PERSPECTIVE:
        {
            /* code */
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "unsupported warp type");
        }
    }

    AURA_RETURN(ctx, ret);
}

} // namespace aura
