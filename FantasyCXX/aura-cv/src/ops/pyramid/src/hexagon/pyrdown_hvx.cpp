#include "aura/ops/pyramid/pyrdown.hpp"
#include "pyrdown_impl.hpp"
#include "pyramid_comm.hpp"

namespace aura
{

PyrDownHvx::PyrDownHvx(Context *ctx, const OpTarget &target) : PyrDownImpl(ctx, target)
{}

Status PyrDownHvx::SetArgs(const Array *src, Array *dst, DT_S32 ksize, DT_F32 sigma,
                              BorderType border_type)
{
    if (PyrDownImpl::SetArgs(src, dst, ksize, sigma, border_type) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "PyrDownImpl::SetArgs failed");
        return Status::ERROR;
    }

    return Status::OK;
}

Status PyrDownHvx::Run()
{
    const Mat *src = dynamic_cast<const Mat*>(m_src);
    Mat *dst       = dynamic_cast<Mat*>(m_dst);

    if ((DT_NULL == src) || (DT_NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src or dst is nullptr");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    switch (m_ksize)
    {
        case 5:
        {
            ret = PyrDown5x5Hvx(m_ctx, *src, *dst, m_kmat, m_border_type);
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

std::string PyrDownHvx::ToString() const
{
    return PyrDownImpl::ToString();
}

Status PyrDownRpc(Context *ctx, HexagonRpcParam &rpc_param)
{
    Mat src;
    Mat dst;
    DT_S32 ksize;
    DT_F32 sigma;
    BorderType border_type;
    Scalar border_value;

    PyrDownInParam in_param(ctx, rpc_param);
    Status ret = in_param.Get(src, dst, ksize, sigma, border_type);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get failed");
        return Status::ERROR;
    }

    PyrDown pyrdown(ctx, OpTarget::Hvx());

    return OpCall(ctx, pyrdown, &src, &dst, ksize, sigma, border_type);
}

AURA_HEXAGON_RPC_FUNC_REGISTER(AURA_OPS_PYRAMID_PACKAGE_NAME, AURA_OPS_PYRAMID_PYRDOWN_OP_NAME, PyrDownRpc);

} // namespace aura