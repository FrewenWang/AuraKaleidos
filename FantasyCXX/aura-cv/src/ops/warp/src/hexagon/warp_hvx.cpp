#include "aura/ops/warp/warp.hpp"
#include "warp_impl.hpp"
#include "warp_comm.hpp"

namespace aura
{

WarpHvx::WarpHvx(Context *ctx, WarpType warp_type, const OpTarget &target) : WarpImpl(ctx, warp_type, target)
{}

Status WarpHvx::SetArgs(const Array *src, const Array *matrix, Array *dst, InterpType interp_type,
                        BorderType border_type, const Scalar &border_value)
{
    if (WarpImpl::SetArgs(src, matrix, dst, interp_type, border_type, border_value) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "WarpImpl::SetArgs failed");
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
            ret = WarpAffineHvx(m_ctx, *src, *matrix, *dst, m_interp_type, m_border_type, m_border_value);
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
    return WarpImpl::ToString();
}

Status WarpRpc(Context *ctx, HexagonRpcParam &rpc_param)
{
    Mat        src;
    Mat        matrix;
    Mat        dst;
    WarpType   warp_type;
    InterpType interp_type;
    BorderType border_type;
    Scalar     border_value;

    WarpInParam in_param(ctx, rpc_param);

    Status ret = in_param.Get(src, matrix, dst, warp_type, interp_type, border_type, border_value);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get failed");
        return Status::ERROR;
    }

    ret = Status::ERROR;

    switch (warp_type)
    {
        case WarpType::AFFINE:
        {
            WarpAffine warp_affine(ctx, OpTarget::Hvx());
            ret = OpCall(ctx, warp_affine, &src, &matrix, &dst, interp_type, border_type, border_value);
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

Status WarpCoordRpc(Context *ctx, HexagonRpcParam &rpc_param)
{
    Mat        grid;
    Mat        map_xy;
    WarpType   warp_type;

    WarpCoordInParam in_param(ctx, rpc_param);

    Status ret = in_param.Get(grid, map_xy, warp_type);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get failed");
        return Status::ERROR;
    }

    ret = Status::ERROR;

    switch (warp_type)
    {
        case WarpType::AFFINE:
        {
            ret = WarpAffineCoordHvx(ctx, grid, map_xy);
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

AURA_HEXAGON_RPC_FUNC_REGISTER(AURA_OPS_WARP_PACKAGE_NAME, AURA_OPS_WARP_OP_NAME, WarpRpc);
AURA_HEXAGON_RPC_FUNC_REGISTER(AURA_OPS_WARP_PACKAGE_NAME, AURA_OPS_WARP_COORD_OP_NAME, WarpCoordRpc);

} // namespace aura