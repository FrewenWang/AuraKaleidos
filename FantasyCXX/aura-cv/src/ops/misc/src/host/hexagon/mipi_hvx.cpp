#include "mipi_impl.hpp"
#include "misc_comm.hpp"

namespace aura
{

MipiPackHvx::MipiPackHvx(Context *ctx, const OpTarget &target) : MipiPackImpl(ctx, target)
{}

Status MipiPackHvx::SetArgs(const Array *src, Array *dst)
{
    if (MipiPackImpl::SetArgs(src, dst) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "MipiPackImpl::SetArgs failed");
        return Status::ERROR;
    }

    if ((src->GetArrayType() != ArrayType::MAT) || (dst->GetArrayType() != ArrayType::MAT))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src dst must be mat type");
        return Status::ERROR;
    }

    return Status::OK;
}

Status MipiPackHvx::Run()
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
    MipiPackInParam in_param(m_ctx, rpc_param);
    ret = in_param.Set(*src, *dst);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Set param failed");
        return ret;
    }

    HexagonProfiling profiling;
    HexagonEngine *engine = m_ctx->GetHexagonEngine();
    ret = engine->Run(AURA_OPS_MISC_PACKAGE_NAME, AURA_OPS_MISC_MIPIPACK_OP_NAME, rpc_param, &profiling);

    if (Status::OK == ret && MI_TRUE == m_target.m_data.hvx.profiling)
    {
        m_profiling_string = " " + HexagonProfilingToString(profiling);
    }

    AURA_RETURN(m_ctx, ret);
}

std::string MipiPackHvx::ToString() const
{
    return MipiPackImpl::ToString() + m_profiling_string;
}

MipiUnPackHvx::MipiUnPackHvx(Context *ctx, const OpTarget &target) : MipiUnPackImpl(ctx, target)
{}

Status MipiUnPackHvx::SetArgs(const Array *src, Array *dst)
{
    if (MipiUnPackImpl::SetArgs(src, dst) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "MipiUnPackImpl::SetArgs failed");
        return Status::ERROR;
    }

    if ((src->GetArrayType() != ArrayType::MAT) || (dst->GetArrayType() != ArrayType::MAT))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src dst must be mat type");
        return Status::ERROR;
    }

    return Status::OK;
}

Status MipiUnPackHvx::Run()
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
    MipiUnPackInParam in_param(m_ctx, rpc_param);
    ret = in_param.Set(*src, *dst);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Set param failed");
        return ret;
    }

    HexagonProfiling profiling;
    HexagonEngine *engine = m_ctx->GetHexagonEngine();
    ret = engine->Run(AURA_OPS_MISC_PACKAGE_NAME, AURA_OPS_MISC_MIPIUNPACK_OP_NAME, rpc_param, &profiling);

    if (Status::OK == ret && MI_TRUE == m_target.m_data.hvx.profiling)
    {
        m_profiling_string = " " + HexagonProfilingToString(profiling);
    }

    AURA_RETURN(m_ctx, ret);
}

std::string MipiUnPackHvx::ToString() const
{
    return MipiUnPackImpl::ToString() + m_profiling_string;
}

} // namespace aura