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

    if ((src->GetArrayType() != ArrayType::MAT) || (dst->GetArrayType() != ArrayType::MAT))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src dst must be mat type");
        return Status::ERROR;
    }

    ElemType src_elem_type = src->GetElemType();
    ElemType dst_elem_type = dst->GetElemType();
    if (src_elem_type != ElemType::U8 || dst_elem_type != ElemType::S16)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "only support src elem_type u8, dst elem_type s16");
        return Status::ERROR;
    }

    if ((src->GetMemType() != AURA_MEM_DMA_BUF_HEAP) || (dst->GetMemType() != AURA_MEM_DMA_BUF_HEAP))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "mat memory must be AURA_MEM_DMA_BUF_HEAP type");
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

Status SobelHvx::Run()
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
    SobelInParam in_param(m_ctx, rpc_param);
    ret = in_param.Set(*src, *dst, m_dx, m_dy, m_ksize, m_scale, m_border_type, m_border_value);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Set failed");
        return Status::ERROR;
    }

    HexagonProfiling profiling;
    HexagonEngine *engine = m_ctx->GetHexagonEngine();
    ret = engine->Run(AURA_OPS_FILTER_PACKAGE_NAME, AURA_OPS_FILTER_SOBEL_OP_NAME, rpc_param, &profiling);

    if (Status::OK == ret && DT_TRUE == m_target.m_data.hvx.profiling)
    {
        m_profiling_string = " " + HexagonProfilingToString(profiling);
    }

    AURA_RETURN(m_ctx, ret);
}

std::string SobelHvx::ToString() const
{
    return SobelImpl::ToString() + m_profiling_string;
}

} // namespace aura