#include "aura/ops/filter/laplacian.hpp"
#include "filter_comm.hpp"
#include "laplacian_impl.hpp"

namespace aura
{

LaplacianHvx::LaplacianHvx(Context *ctx, const OpTarget &target) : LaplacianImpl(ctx, target)
{}

Status LaplacianHvx::SetArgs(const Array *src, Array *dst, MI_S32 ksize,
                             BorderType border_type, const Scalar &border_value)
{
    if (LaplacianImpl::SetArgs(src, dst, ksize, border_type, border_value) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "LaplacianImpl::SetArgs failed");
        return Status::ERROR;
    }

    if ((ksize != 1) && (ksize != 3) && (ksize != 5) && (ksize != 7))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "ksize onlt support 1/3/5/7");
        return Status::ERROR;
    }

    MI_S32 ch = src->GetSizes().m_channel;
    if ((ch != 1) && (ch != 2) && (ch != 3))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "channel only support 1/2/3");
        return Status::ERROR;
    }

    ElemType src_elem_type = src->GetElemType();
    ElemType dst_elem_type = dst->GetElemType();
    if (!(((src_elem_type == ElemType::U8)  && (dst_elem_type == ElemType::S16)) ||
          ((src_elem_type == ElemType::U16) && (dst_elem_type == ElemType::U16)) ||
          ((src_elem_type == ElemType::S16) && (dst_elem_type == ElemType::S16))))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Unsupported src and dst elem_type config, only support U8S16/U16U16/S16S16");
        return Status::ERROR;
    }

    return Status::OK;
}

Status LaplacianHvx::Run()
{
    const Mat *src = dynamic_cast<const Mat*>(m_src);
    Mat       *dst = dynamic_cast<Mat*>(m_dst);

    if ((MI_NULL == src) || (MI_NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src or dst is nullptr");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    switch (m_ksize)
    {
        case 1:
        {
            ret = Laplacian1x1Hvx(m_ctx, *src, *dst, m_border_type, m_border_value);
            break;
        }

        case 3:
        {
            ret = Laplacian3x3Hvx(m_ctx, *src, *dst, m_border_type, m_border_value);
            break;
        }

        case 5:
        {
            ret = Laplacian5x5Hvx(m_ctx, *src, *dst, m_border_type, m_border_value);
            break;
        }

        case 7:
        {
            ret = Laplacian7x7Hvx(m_ctx, *src, *dst,  m_border_type, m_border_value);
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

std::string LaplacianHvx::ToString() const
{
    return LaplacianImpl::ToString();
}

Status LaplacianRpc(Context *ctx, HexagonRpcParam &rpc_param)
{
    Mat src;
    Mat dst;
    MI_S32 ksize;
    BorderType border_type;
    Scalar border_value;

    LaplacianInParam in_param(ctx, rpc_param);
    Status ret = in_param.Get(src, dst, ksize, border_type, border_value);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get failed");
        return Status::ERROR;
    }

    Laplacian laplacian(ctx, OpTarget::Hvx());

    return OpCall(ctx, laplacian, &src, &dst, ksize, border_type, border_value);
}

AURA_HEXAGON_RPC_FUNC_REGISTER(AURA_OPS_FILTER_PACKAGE_NAME, AURA_OPS_FILTER_LAPLACIAN_OP_NAME, LaplacianRpc);

} // namespace aura