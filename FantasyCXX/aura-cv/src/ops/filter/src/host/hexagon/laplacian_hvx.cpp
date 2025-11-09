#include "laplacian_impl.hpp"
#include "filter_comm.hpp"

namespace aura
{

LaplacianHvx::LaplacianHvx(Context *ctx, const OpTarget &target) : LaplacianImpl(ctx, target)
{}

Status LaplacianHvx::SetArgs(const Array *src, Array *dst, DT_S32 ksize,
                             BorderType border_type,const Scalar &border_value)
{
    if (LaplacianImpl::SetArgs(src, dst, ksize, border_type, border_value) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "LaplacianImpl::SetArgs failed");
        return Status::ERROR;
    }

    if ((src->GetArrayType() != ArrayType::MAT) || (dst->GetArrayType() != ArrayType::MAT))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src dst must be mat type");
        return Status::ERROR;
    }

    if ((src->GetMemType() != AURA_MEM_DMA_BUF_HEAP) || (dst->GetMemType() != AURA_MEM_DMA_BUF_HEAP))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "mat memory must be AURA_MEM_DMA_BUF_HEAP type");
        return Status::ERROR;
    }

    if ((ksize != 1) && (ksize != 3) && (ksize != 5) && (ksize != 7))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "ksize only support 1/3/5/7");
        return Status::ERROR;
    }

    DT_S32 ch = src->GetSizes().m_channel;
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

    if ((DT_NULL == src) || (DT_NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src or dst is nullptr");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    HexagonRpcParam rpc_param(m_ctx);
    LaplacianInParam in_param(m_ctx, rpc_param);
    ret = in_param.Set(*src, *dst, m_ksize, m_border_type, m_border_value);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Set failed");
        return Status::ERROR;
    }

    HexagonProfiling profiling;
    HexagonEngine *engine = m_ctx->GetHexagonEngine();
    ret = engine->Run(AURA_OPS_FILTER_PACKAGE_NAME, AURA_OPS_FILTER_LAPLACIAN_OP_NAME, rpc_param, &profiling);

    if (Status::OK == ret && DT_TRUE == m_target.m_data.hvx.profiling)
    {
        m_profiling_string = " " + HexagonProfilingToString(profiling);
    }

    AURA_RETURN(m_ctx, ret);
}

std::string LaplacianHvx::ToString() const
{
    return LaplacianImpl::ToString() + m_profiling_string;
}

} // namespace aura