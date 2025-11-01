#include "aura/ops/resize/resize.hpp"
#include "resize_impl.hpp"
#include "resize_comm.hpp"

namespace aura
{

ResizeHvx::ResizeHvx(Context *ctx, const OpTarget &target) : ResizeImpl(ctx, target)
{}

Status ResizeHvx::SetArgs(const Array *src, Array *dst, InterpType type)
{
    if (ResizeImpl::SetArgs(src, dst, type) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "ResizeImpl::SetArgs failed");
        return Status::ERROR;
    }

    MI_S32 ch = src->GetSizes().m_channel;
    if (ch != 1 && ch != 2 && ch != 3)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "channel only support 1/2/3");
        return Status::ERROR;
    }

    ElemType elem_type = src->GetElemType();
    if (elem_type != ElemType::U8 && elem_type != ElemType::S8 && elem_type != ElemType::U16 && elem_type != ElemType::S16)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "elem_type only support u8/s8/u16/s16");
        return Status::ERROR;
    }

    return Status::OK;
}

Status ResizeHvx::Run()
{
    const Mat *src = dynamic_cast<const Mat*>(m_src);
    Mat *dst = dynamic_cast<Mat*>(m_dst);

    if ((MI_NULL == src) || (MI_NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src or dst is nullptr");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    switch (m_type)
    {
        case InterpType::NEAREST:
        {
            ret = ResizeNnHvx(m_ctx, *src, *dst);
            break;
        }

        case InterpType::LINEAR:
        {
            ret = ResizeBnHvx(m_ctx, *src, *dst);
            break;
        }

        case InterpType::CUBIC:
        {
            ret = ResizeCuHvx(m_ctx, *src, *dst);
            break;
        }

        case InterpType::AREA:
        {
            ret = ResizeAreaHvx(m_ctx, *src, *dst);
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(m_ctx, "interp type error");
            ret = Status::ERROR;
        }
    }

    AURA_RETURN(m_ctx, ret);
}

std::string ResizeHvx::ToString() const
{
    return ResizeImpl::ToString();
}

Status ResizeRpc(Context *ctx, HexagonRpcParam &rpc_param)
{
    Mat src;
    Mat dst;
    InterpType type;

    ResizeInParam in_param(ctx, rpc_param);
    Status ret = in_param.Get(src, dst, type);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get failed");
        return Status::ERROR;
    }

    Resize resize(ctx, OpTarget::Hvx());

    return OpCall(ctx, resize, &src, &dst, type);
}

AURA_HEXAGON_RPC_FUNC_REGISTER(AURA_OPS_RESIZE_PACKAGE_NAME, AURA_OPS_RESIZE_RESIZE_OP_NAME, ResizeRpc);

} // namespace aura
