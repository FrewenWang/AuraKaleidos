#include "median_impl.hpp"
#include "filter_comm.hpp"

namespace aura
{

MedianHvx::MedianHvx(Context *ctx, const OpTarget &target) : MedianImpl(ctx, target)
{}

Status MedianHvx::SetArgs(const Array *src, Array *dst, MI_S32 ksize)
{
    if (MedianImpl::SetArgs(src, dst, ksize) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "MedianImpl::Initialize failed");
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

    if (ksize != 3 && ksize != 5 && ksize != 7)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "ksize only support 3/5/7");
        return Status::ERROR;
    }

    MI_S32 ch = src->GetSizes().m_channel;
    if (ch != 1)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "channel only support 1");
        return Status::ERROR;
    }

    ElemType elem_type = src->GetElemType();
    if (elem_type != ElemType::S8 && elem_type != ElemType::U8 && elem_type != ElemType::S16 && elem_type != ElemType::U16)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "elem_type only support s8/u8/s16/u16");
        return Status::ERROR;
    }

    return Status::OK;
}

Status MedianHvx::Run()
{
    const Mat *src = dynamic_cast<const Mat*>(m_src);
    Mat *dst = dynamic_cast<Mat*>(m_dst);

    if ((MI_NULL == src) || (MI_NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src or dst is nullptr");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    HexagonRpcParam rpc_param(m_ctx);
    MedianInParam in_param(m_ctx, rpc_param);
    ret = in_param.Set(*src, *dst, m_ksize);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Set failed");
        return Status::ERROR;
    }

    HexagonProfiling profiling;
    HexagonEngine *engine = m_ctx->GetHexagonEngine();
    ret = engine->Run(AURA_OPS_FILTER_PACKAGE_NAME, AURA_OPS_FILTER_MEDIAN_OP_NAME, rpc_param, &profiling);

    if (Status::OK == ret && MI_TRUE == m_target.m_data.hvx.profiling)
    {
        m_profiling_string = " " + HexagonProfilingToString(profiling);
    }

    AURA_RETURN(m_ctx, ret);
}

std::string MedianHvx::ToString() const
{
    return MedianImpl::ToString() + m_profiling_string;
}

} // namespace aura