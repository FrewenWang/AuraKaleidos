#include "aura/ops/filter/filter2d.hpp"
#include "filter2d_impl.hpp"
#include "filter_comm.hpp"

namespace aura
{

Filter2dHvx::Filter2dHvx(Context *ctx, const OpTarget &target) : Filter2dImpl(ctx, target)
{}

Status Filter2dHvx::SetArgs(const Array *src, Array *dst, const Array *kmat,
                            BorderType border_type, const Scalar &border_value)
{
    if (Filter2dImpl::SetArgs(src, dst, kmat, border_type, border_value) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Filter2dImpl::SetArgs failed");
        return Status::ERROR;
    }

    if (m_ksize != 3 && m_ksize != 5 && m_ksize != 7)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "ksize only support 3/5/7");
        return Status::ERROR;
    }

    DT_S32 ch = src->GetSizes().m_channel;
    if (ch != 1 && ch != 2 && ch != 3)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "channel only support 1/2/3");
        return Status::ERROR;
    }

    if (7 == m_ksize && 3 == ch)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "7x7 kernel size only supports channels 1 and 2");
        return Status::ERROR;
    }

    ElemType elem_type = src->GetElemType();
    if (dst->GetElemType() != elem_type || (elem_type != ElemType::U8 && elem_type != ElemType::U16 && elem_type != ElemType::S16))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "elem_type not support");
        return Status::ERROR;
    }

    return Status::OK;
}

Status Filter2dHvx::Run()
{
    const Mat *src = dynamic_cast<const Mat*>(m_src);
    Mat *dst = dynamic_cast<Mat*>(m_dst);
    const Mat *kmat = dynamic_cast<const Mat*>(m_kmat);

    if ((DT_NULL == src) || (DT_NULL == dst) || (DT_NULL == kmat))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src/dst/kmat is nullptr");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    std::vector<DT_S16> kdata(kmat->GetSizes().Total());
    for (DT_S32 y = 0; y < m_ksize; y++)
    {
        const DT_F32 *kernel = kmat->Ptr<DT_F32>(y);
        for (DT_S32 x = 0; x < m_ksize; x++)
        {
            kdata[y * m_ksize + x] = static_cast<DT_S16>(kernel[x] * 4096);
        }
    }

    switch (m_ksize)
    {
        case 3:
        {
            ret = Filter2d3x3Hvx(m_ctx, *src, *dst, kdata, m_border_type, m_border_value);
            break;
        }

        case 5:
        {
            ret = Filter2d5x5Hvx(m_ctx, *src, *dst, kdata, m_border_type, m_border_value);
            break;
        }

        case 7:
        {
            ret = Filter2d7x7Hvx(m_ctx, *src, *dst, kdata, m_border_type, m_border_value);
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(m_ctx, "unsupported ksize");
            return Status::ERROR;
        }
    }

    AURA_RETURN(m_ctx, ret);
}

std::string Filter2dHvx::ToString() const
{
    return Filter2dImpl::ToString();
}

Status Filter2dRpc(Context *ctx, HexagonRpcParam &rpc_param)
{
    Mat src;
    Mat dst;
    Mat kmat;
    BorderType border_type;
    Scalar border_value;

    Filter2dInParam in_param(ctx, rpc_param);
    Status ret = in_param.Get(src, dst, kmat, border_type, border_value);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get failed");
        return Status::ERROR;
    }

    Filter2d filter2d(ctx, OpTarget::Hvx());

    return OpCall(ctx, filter2d, &src, &dst, &kmat, border_type, border_value);
}

AURA_HEXAGON_RPC_FUNC_REGISTER(AURA_OPS_FILTER_PACKAGE_NAME, AURA_OPS_FILTER_FILTER2D_OP_NAME, Filter2dRpc);

} // namespace aura