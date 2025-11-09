#include "aura/ops/pyramid/pyrup.hpp"
#include "pyrup_impl.hpp"
#include "pyramid_comm.hpp"

namespace aura
{

PyrUpHvx::PyrUpHvx(Context *ctx, const OpTarget &target) : PyrUpImpl(ctx, target)
{}

Status PyrUpHvx::SetArgs(const Array *src, Array *dst, DT_S32 ksize, DT_F32 sigma,
                            BorderType border_type)
{
    if (PyrUpImpl::SetArgs(src, dst, ksize, sigma, border_type) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "PyrUpImpl::SetArgs failed");
        return Status::ERROR;
    }

    return Status::OK;
}

Status PyrUpHvx::Run()
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
            ret = PyrUp5x5Hvx(m_ctx, *src, *dst, m_kmat, m_border_type);
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

std::string PyrUpHvx::ToString() const
{
    return PyrUpImpl::ToString();
}

Status PyrUpRpc(Context *ctx, HexagonRpcParam &rpc_param)
{
    Mat src;
    Mat dst;
    DT_S32 ksize;
    DT_F32 sigma;
    BorderType border_type;
    Scalar border_value;

    PyrUpInParam in_param(ctx, rpc_param);
    Status ret = in_param.Get(src, dst, ksize, sigma, border_type);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get failed");
        return Status::ERROR;
    }

    PyrUp pyrup(ctx, OpTarget::Hvx());

    return OpCall(ctx, pyrup, &src, &dst, ksize, sigma, border_type);
}

AURA_HEXAGON_RPC_FUNC_REGISTER(AURA_OPS_PYRAMID_PACKAGE_NAME, AURA_OPS_PYRAMID_PYRUP_OP_NAME, PyrUpRpc);

} // namespace aura