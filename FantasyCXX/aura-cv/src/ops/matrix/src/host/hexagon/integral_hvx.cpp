#include "matrix_comm.hpp"
#include "integral_impl.hpp"
#include "aura/runtime/logger.h"

namespace aura
{

IntegralHvx::IntegralHvx(Context *ctx, const OpTarget &target) : IntegralImpl(ctx, target)
{}

Status IntegralHvx::SetArgs(const Array *src, Array *dst, Array *dst_sq)
{
    if (IntegralImpl::SetArgs(src, dst, dst_sq) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "IntegralImpl::SetArgs failed");
        return Status::ERROR;
    }

    if ((src->GetArrayType() != ArrayType::MAT))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src must be mat type");
        return Status::ERROR;
    }

    // dst must be non-null and mat type
    if (MI_NULL == dst)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "dst is nullptr");
        return Status::ERROR;
    }

    if (!dst->IsValid() || dst->GetArrayType() != ArrayType::MAT)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "dst is invalid or not mat type");
        return Status::ERROR;
    }

    // dst_sq must be null or invalid
    if (dst_sq && dst_sq->IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "only support Normal mode, dst_sq must be nullptr or invalid");
        return Status::ERROR;
    }

    MI_S32 ch = src->GetSizes().m_channel;
    if (ch != 1 && ch != 2)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "channel only support 1/2");
        return Status::ERROR;
    }

    return Status::OK;
}

Status IntegralHvx::Run()
{
    const Mat *src = dynamic_cast<const Mat*>(m_src);
    Mat *dst = dynamic_cast<Mat*>(m_dst);
    Mat *dst_sq = dynamic_cast<Mat*>(m_dst_sq);
    Mat dst0 = (dst != MI_NULL) ? *dst : Mat();
    Mat dst1 = (dst_sq != MI_NULL) ? *dst_sq : Mat();

    if (MI_NULL == src)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src or dst is nullptr");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    HexagonRpcParam rpc_param(m_ctx);
    IntegralInParam in_param(m_ctx, rpc_param);

    ret = in_param.Set(*src, dst0, dst1);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Set failed");
        return Status::ERROR;
    }

    HexagonProfiling profiling;
    HexagonEngine *engine = m_ctx->GetHexagonEngine();
    ret = engine->Run(AURA_OPS_MATRIX_PACKAGE_NAME, AURA_OPS_MATRIX_INTEGRAL_OP_NAME, rpc_param, &profiling);

    if (Status::OK == ret && MI_TRUE == m_target.m_data.hvx.profiling)
    {
        m_profiling_string = " " + HexagonProfilingToString(profiling);
    }

    AURA_RETURN(m_ctx, ret);
}

std::string IntegralHvx::ToString() const
{
    return IntegralImpl::ToString() + m_profiling_string;
}

} // namespace aura