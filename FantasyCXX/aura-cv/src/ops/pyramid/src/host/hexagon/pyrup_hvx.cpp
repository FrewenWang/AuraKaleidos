#include "pyrup_impl.hpp"
#include "pyramid_comm.hpp"

namespace aura
{

PyrUpHvx::PyrUpHvx(Context *ctx, const OpTarget &target) : PyrUpImpl(ctx, target)
{}

Status PyrUpHvx::SetArgs(const Array *src, Array *dst, MI_S32 ksize, MI_F32 sigma,
                            BorderType border_type)
{
    if (PyrUpImpl::SetArgs(src, dst, ksize, sigma, border_type) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "PyrUpImpl::SetArgs failed");
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

    return Status::OK;
}

Status PyrUpHvx::Run()
{
    const Mat *src = dynamic_cast<const Mat*>(m_src);
    Mat *dst       = dynamic_cast<Mat*>(m_dst);

    if ((MI_NULL == src) || (MI_NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src or dst is nullptr");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    HexagonRpcParam rpc_param(m_ctx);
    PyrUpInParam in_param(m_ctx, rpc_param);
    ret = in_param.Set(*src, *dst, m_ksize, m_sigma, m_border_type);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Set failed");
        return Status::ERROR;
    }

    HexagonProfiling profiling;
    HexagonEngine *engine = m_ctx->GetHexagonEngine();
    ret = engine->Run(AURA_OPS_PYRAMID_PACKAGE_NAME, AURA_OPS_PYRAMID_PYRUP_OP_NAME, rpc_param, &profiling);

    if (Status::OK == ret && MI_TRUE == m_target.m_data.hvx.profiling)
    {
        m_profiling_string = " " + HexagonProfilingToString(profiling);
    }

    AURA_RETURN(m_ctx, ret);
}

std::string PyrUpHvx::ToString() const
{
    return PyrUpImpl::ToString() + m_profiling_string;
}

} // namespace aura