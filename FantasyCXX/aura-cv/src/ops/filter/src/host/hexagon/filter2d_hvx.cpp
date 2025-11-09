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

    if ((src->GetArrayType() != ArrayType::MAT) || (dst->GetArrayType() != ArrayType::MAT))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src dst must be mat type");
        return Status::ERROR;
    }

    if ((src->GetMemType() != AURA_MEM_DMA_BUF_HEAP) || (dst->GetMemType() != AURA_MEM_DMA_BUF_HEAP) ||
        (kmat->GetMemType() != AURA_MEM_DMA_BUF_HEAP))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "mat memory must be AURA_MEM_DMA_BUF_HEAP type");
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

    HexagonRpcParam rpc_param(m_ctx);
    Filter2dInParam in_param(m_ctx, rpc_param);
    ret = in_param.Set(*src, *dst, *kmat, m_border_type, m_border_value);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Set failed");
        return Status::ERROR;
    }

    HexagonProfiling profiling;
    HexagonEngine *engine = m_ctx->GetHexagonEngine();
    ret = engine->Run(AURA_OPS_FILTER_PACKAGE_NAME, AURA_OPS_FILTER_FILTER2D_OP_NAME, rpc_param, &profiling);

    if (Status::OK == ret && DT_TRUE == m_target.m_data.hvx.profiling)
    {
        m_profiling_string = " " + HexagonProfilingToString(profiling);
    }

    AURA_RETURN(m_ctx, ret);
}

std::string Filter2dHvx::ToString() const
{
    return Filter2dImpl::ToString() + m_profiling_string;
}

} // namespace aura