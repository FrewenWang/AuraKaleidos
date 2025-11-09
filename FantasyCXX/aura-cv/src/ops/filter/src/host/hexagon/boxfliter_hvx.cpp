#include "boxfilter_impl.hpp"
#include "filter_comm.hpp"

namespace aura
{

BoxFilterHvx::BoxFilterHvx(Context *ctx, const OpTarget &target) : BoxFilterImpl(ctx, target)
{}

Status BoxFilterHvx::SetArgs(const Array *src, Array *dst, DT_S32 ksize, BorderType border_type, const Scalar &border_value)
{
    if (BoxFilterImpl::SetArgs(src, dst, ksize, border_type, border_value) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "BoxFilterImpl::SetArgs failed");
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

    DT_S32 ch = src->GetSizes().m_channel;
    if (ch != 1 && ch != 2 && ch != 3)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "channel only support 1/2/3");
        return Status::ERROR;
    }

    return Status::OK;
}

Status BoxFilterHvx::Run()
{
    const Mat *src = dynamic_cast<const Mat*>(m_src);
    Mat *dst = dynamic_cast<Mat*>(m_dst);

    if ((DT_NULL == src) || (DT_NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src or dst is nullptr");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    HexagonRpcParam rpc_param(m_ctx);
    BoxFilterInParam in_param(m_ctx, rpc_param);
    ret = in_param.Set(*src, *dst, m_ksize, m_border_type, m_border_value);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Set failed");
        return Status::ERROR;
    }

    HexagonProfiling profiling;
    HexagonEngine *engine = m_ctx->GetHexagonEngine();
    ret = engine->Run(AURA_OPS_FILTER_PACKAGE_NAME, AURA_OPS_FILTER_BOXFILTER_OP_NAME, rpc_param, &profiling);

    if (Status::OK == ret && DT_TRUE == m_target.m_data.hvx.profiling)
    {
        m_profiling_string = " " + HexagonProfilingToString(profiling);
    }

    AURA_RETURN(m_ctx, ret);
}

std::string BoxFilterHvx::ToString() const
{
    return BoxFilterImpl::ToString() + m_profiling_string;
}

} // namespace mag