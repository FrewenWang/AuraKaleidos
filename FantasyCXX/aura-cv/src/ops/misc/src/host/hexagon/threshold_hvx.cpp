#include "threshold_impl.hpp"
#include "misc_comm.hpp"

namespace aura
{

ThresholdHvx::ThresholdHvx(Context *ctx, const OpTarget &target) : ThresholdImpl(ctx, target)
{}

Status ThresholdHvx::SetArgs(const Array *src, Array *dst, DT_F32 thresh, DT_F32 max_val, DT_S32 type)
{
    if (ThresholdImpl::SetArgs(src, dst, thresh, max_val, type) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "ThresholdImpl::SetArgs failed");
        return Status::ERROR;
    }

    if ((src->GetArrayType() != ArrayType::MAT) || (dst->GetArrayType() != ArrayType::MAT))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src dst must be mat type");
        return Status::ERROR;
    }

    ElemType elem_type = src->GetElemType();
    if (elem_type != ElemType::U8 && elem_type != ElemType::U16 && elem_type != ElemType::S16)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "elem_type only support u8/u16/s16");
        return Status::ERROR;
    }

    return Status::OK;
}

Status ThresholdHvx::Run()
{
    const Mat *src = dynamic_cast<const Mat*>(m_src);
    Mat *dst = dynamic_cast<Mat*>(m_dst);

    if ((DT_NULL == src) || (DT_NULL == dst))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src or dst is nullptr");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    if ((1 == src->GetSizes().m_channel) && (ElemType::U8 == src->GetElemType()))
    {
        DT_S32 thresh = Floor(m_thresh);
        ret = ReCalcThresh(thresh);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "ReCalcThresh failed");
            return ret;
        }

        m_thresh = static_cast<DT_F32>(thresh);
    }

    HexagonRpcParam rpc_param(m_ctx);
    ThresholdInParam in_param(m_ctx, rpc_param);
    ret = in_param.Set(*src, *dst, m_thresh, m_max_val, m_type);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Set param failed");
        return ret;
    }

    HexagonProfiling profiling;
    HexagonEngine *engine = m_ctx->GetHexagonEngine();
    ret = engine->Run(AURA_OPS_MISC_PACKAGE_NAME, AURA_OPS_MISC_THRESHOLD_OP_NAME, rpc_param, &profiling);

    if (Status::OK == ret && DT_TRUE == m_target.m_data.hvx.profiling)
    {
        m_profiling_string = " " + HexagonProfilingToString(profiling);
    }

    AURA_RETURN(m_ctx, ret);
}

std::string ThresholdHvx::ToString() const
{
    return ThresholdImpl::ToString() + m_profiling_string;
}

} // namespace aura