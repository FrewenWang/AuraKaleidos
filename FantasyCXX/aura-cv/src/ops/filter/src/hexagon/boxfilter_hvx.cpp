#include "aura/ops/filter/boxfilter.hpp"
#include "boxfilter_impl.hpp"
#include "filter_comm.hpp"

namespace aura
{

BoxFilterHvx::BoxFilterHvx(Context *ctx, const OpTarget &target) : BoxFilterImpl(ctx, target)
{}

Status BoxFilterHvx::SetArgs(const Array *src, Array *dst, MI_S32 ksize,
                             BorderType border_type, const Scalar &border_value)
{
    if (BoxFilterImpl::SetArgs(src, dst, ksize, border_type, border_value) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "BoxFilterImpl::SetArgs failed");
        return Status::ERROR;
    }

    MI_S32 ch = src->GetSizes().m_channel;
    if (ch != 1 && ch != 2 && ch != 3)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "channel only support 1/2/3");
        return Status::ERROR;
    }

    ElemType elem_type = src->GetElemType();
    if (elem_type != ElemType::U8 && elem_type != ElemType::U16 && elem_type != ElemType::S16)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "elem_type only support u8/u16/s16");
        return Status::ERROR;
    }

    if ((ElemType::U8 == elem_type && ksize > 129) || (ElemType::U16 == elem_type && ksize > 65) ||
        (ElemType::S16 == elem_type && ksize > 65))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "U8 src ksize must less 129, U16/S16 src ksize must less 65");
        return Status::ERROR;
    }

    return Status::OK;
}

Status BoxFilterHvx::Run()
{
    const Mat *src = dynamic_cast<const Mat*>(m_src);
    Mat *dst = dynamic_cast<Mat*>(m_dst);

    if ((MI_NULL == src) || (MI_NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src or dst is nullptr");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    switch (m_ksize)
    {
        case 3:
        {
            ret = BoxFilter3x3Hvx(m_ctx, *src, *dst, m_border_type, m_border_value);
            break;
        }

        case 5:
        {
            ret = BoxFilter5x5Hvx(m_ctx, *src, *dst, m_border_type, m_border_value);
            break;
        }

        case 7:
        {
            ret = BoxFilter7x7Hvx(m_ctx, *src, *dst, m_border_type, m_border_value);
            break;
        }

        case 9:
        {
            ret = BoxFilter9x9Hvx(m_ctx, *src, *dst, m_border_type, m_border_value);
            break;
        }

        case 11:
        {
            ret = BoxFilter11x11Hvx(m_ctx, *src, *dst, m_border_type, m_border_value);
            break;
        }

        default:
        {
            ret = BoxFilterKxKHvx(m_ctx, *src, *dst, m_ksize, m_border_type, m_border_value);
            AURA_ADD_ERROR_STRING(m_ctx, "BoxFilterKxKHvx fail");
        }
    }

    AURA_RETURN(m_ctx, ret);
}

std::string BoxFilterHvx::ToString() const
{
    return BoxFilterImpl::ToString();
}

Status BoxFilterRpc(Context *ctx, HexagonRpcParam &rpc_param)
{
    Mat src;
    Mat dst;
    MI_S32 ksize;
    BorderType border_type;
    Scalar border_value;

    BoxFilterInParam in_param(ctx, rpc_param);
    Status ret = in_param.Get(src, dst, ksize, border_type, border_value);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get failed");
        return Status::ERROR;
    }

    BoxFilter boxfilter(ctx, OpTarget::Hvx());

    return OpCall(ctx, boxfilter, &src, &dst, ksize, border_type, border_value);
}

AURA_HEXAGON_RPC_FUNC_REGISTER(AURA_OPS_FILTER_PACKAGE_NAME, AURA_OPS_FILTER_BOXFILTER_OP_NAME, BoxFilterRpc);

} // namespace aura