#include "aura/ops/filter/sobel.hpp"
#include "filter_comm.hpp"
#include "sobel_impl.hpp"
#include "aura/runtime/logger.h"

namespace aura
{

SobelHvx::SobelHvx(Context *ctx, const OpTarget &target) : SobelImpl(ctx, target)
{}

Status SobelHvx::SetArgs(const Array *src, Array *dst, DT_S32 dx, DT_S32 dy, DT_S32 ksize, DT_F32 scale,
                         BorderType border_type, const Scalar &border_value)
{
    if (SobelImpl::SetArgs(src, dst, dx, dy, ksize, scale, border_type, border_value) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "SobelImpl::SetArgs failed");
        return Status::ERROR;
    }

    ElemType src_elem_type = src->GetElemType();
    ElemType dst_elem_type = dst->GetElemType();
    if (src_elem_type != ElemType::U8 || dst_elem_type != ElemType::S16)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "only support src elem_type u8, dst elem_type s16");
        return Status::ERROR;
    }

    if (m_ksize <= 0)
    {
        m_dx = m_dx > 0 ? m_dx : 3;
        m_dy = m_dy > 0 ? m_dy : 3;
        m_ksize = 3;
    }

    if ((m_dx > 0) && (m_dy > 0) && (m_ksize == 1))
    {
        m_ksize = 3;
    }

    if (m_ksize != 1 && m_ksize != 3 && m_ksize != 5)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "ksize is only suppose 1/3/5");
        return Status::ERROR;
    }

    DT_S32 ch = src->GetSizes().m_channel;

    if (ch != 1 && ch != 2 && ch != 3)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "channel is only suppose 1/2/3");
        return Status::ERROR;
    }

    return Status::OK;
}

Status SobelHvx::Run()
{
    const Mat *src = dynamic_cast<const Mat*>(m_src);
    Mat       *dst = dynamic_cast<Mat*>(m_dst);

    if ((DT_NULL == src) || (DT_NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src dst is null");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    switch (m_ksize)
    {
        case 1:
        {
            ret = Sobel1x1Hvx(m_ctx, *src, *dst, m_dx, m_dy, m_scale, m_border_type, m_border_value);
            break;
        }
        case 3:
        {
            ret = Sobel3x3Hvx(m_ctx, *src, *dst, m_dx, m_dy, m_scale, m_border_type, m_border_value);
            break;
        }
        case 5:
        {
            ret = Sobel5x5Hvx(m_ctx, *src, *dst, m_dx, m_dy, m_scale, m_border_type, m_border_value);
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

std::string SobelHvx::ToString() const
{
    return SobelImpl::ToString();
}

Status SobelRpc(Context *ctx, HexagonRpcParam &rpc_param)
{
    Mat src;
    Mat dst;
    DT_S32 dx;
    DT_S32 dy;
    DT_S32 ksize;
    DT_F32 scale;
    BorderType border_type;
    Scalar border_value;

    SobelInParam in_param(ctx, rpc_param);
    Status ret = in_param.Get(src, dst, dx, dy, ksize, scale, border_type, border_value);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get failed");
        return Status::ERROR;
    }

    Sobel sobel(ctx, OpTarget::Hvx());

    return OpCall(ctx, sobel, &src, &dst, dx, dy, ksize, scale, border_type, border_value);
}

AURA_HEXAGON_RPC_FUNC_REGISTER(AURA_OPS_FILTER_PACKAGE_NAME, AURA_OPS_FILTER_SOBEL_OP_NAME, SobelRpc);

} // namespace aura