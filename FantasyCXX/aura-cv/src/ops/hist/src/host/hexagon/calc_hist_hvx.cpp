#include "calc_hist_impl.hpp"
#include "hist_comm.hpp"

namespace aura
{

CalcHistHvx::CalcHistHvx(Context *ctx, const OpTarget &target) : CalcHistImpl(ctx, target)
{}

Status CalcHistHvx::SetArgs(const Array *src, DT_S32 channel, std::vector<DT_U32> &hist, DT_S32 hist_size, 
                            const Scalar &ranges, const Array *mask, DT_BOOL accumulate)
{
    Status ret = Status::ERROR;

    if (CalcHistImpl::SetArgs(src, channel, hist, hist_size, ranges, mask, accumulate) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "CalcHistImpl::SetArgs failed");
        return ret;
    }

    if (src->GetArrayType() != ArrayType::MAT)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src must be mat type");
        return ret;
    }

    if (src->GetElemType() != ElemType::U8)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src element type must be u8");
        return ret;
    }

    if (src->GetSizes().m_channel != 1)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "hist hvx only support channel 1");
        return ret;
    }

    if (0 != (src->GetStrides().m_width & 0x7F) && src->GetSizes().m_width != src->GetStrides().m_width)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src must be 128 aligned or width must be equal to stride");
        return ret;
    }

    if (mask->IsValid())
    {
        if (mask->GetStrides() != src->GetStrides())
        {
            AURA_ADD_ERROR_STRING(m_ctx, "mask stride must be equal to src stride");
            return ret;
        }
    }

    return Status::OK;
}

Status CalcHistHvx::Run()
{
    Status ret = Status::ERROR;

    const Mat *src = dynamic_cast<const Mat*>(m_src);
    const Mat *mask = dynamic_cast<const Mat*>(m_mask);

    if ((DT_NULL == src) || (DT_NULL == mask))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src or mask is nullptr");
        return ret;
    }

    HexagonRpcParam rpc_param(m_ctx, 2048);
    CalcHistInParam in_param(m_ctx, rpc_param);
    ret = in_param.Set(*src, m_channel, *m_hist, m_hist_size, m_ranges, *mask, m_accumulate);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Set failed");
        return ret;
    }

    HexagonProfiling profiling;
    HexagonEngine *engine = m_ctx->GetHexagonEngine();
    ret = engine->Run(AURA_OPS_HIST_PACKAGE_NAME, AURA_OPS_HIST_CALCHIST_OP_NAME, rpc_param, &profiling);

    if (Status::OK == ret)
    {
        if (DT_TRUE == m_target.m_data.hvx.profiling)
        {
            m_profiling_string = " " + HexagonProfilingToString(profiling);
        }

        CalcHistOutParam out_param(m_ctx, rpc_param);
        ret = out_param.Get(*m_hist);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "out_param get failed");
            return ret;
        }
    }

    AURA_RETURN(m_ctx, ret);
}

std::string CalcHistHvx::ToString() const
{
    return CalcHistImpl::ToString() + m_profiling_string;
}

} // namespace aura