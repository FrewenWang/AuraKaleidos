#include "matrix_comm.hpp"
#include "sum_impl.hpp"
#include "aura/runtime/logger.h"

namespace aura
{

SumHvx::SumHvx(Context *ctx, const OpTarget &target) : SumImpl(ctx, target)
{}

Status SumHvx::SetArgs(const Array *src, Scalar *result)
{
    if (SumImpl::SetArgs(src, result) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "SumImpl::SetArgs failed");
        return Status::ERROR;
    }

    if (src->GetArrayType() != ArrayType::MAT)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src must be mat type");
        return Status::ERROR;
    }

    if ((src->GetMemType() != AURA_MEM_DMA_BUF_HEAP))
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

    ElemType elem_type = src->GetElemType();
    if (elem_type != ElemType::U8 && elem_type != ElemType::S8 && elem_type != ElemType::U16 && elem_type != ElemType::S16)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "elem_type only support u8/s8/u16/s16");
        return Status::ERROR;
    }

    return Status::OK;
}

Status SumHvx::Run()
{
    const Mat *src = dynamic_cast<const Mat*>(m_src);

    if (DT_NULL == src)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src is nullptr");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    HexagonRpcParam rpc_param(m_ctx);
    SumInParam in_param(m_ctx, rpc_param);
    ret = in_param.Set(*src);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Set failed");
        return Status::ERROR;
    }

    HexagonProfiling profiling;
    HexagonEngine *engine = m_ctx->GetHexagonEngine();
    ret = engine->Run(AURA_OPS_MATRIX_PACKAGE_NAME, AURA_OPS_MATRIX_SUM_OP_NAME, rpc_param, &profiling);

    if (Status::OK == ret)
    {
        if (DT_TRUE == m_target.m_data.hvx.profiling)
        {
            m_profiling_string = " " + HexagonProfilingToString(profiling);
        }

        SumOutParam out_param(m_ctx, rpc_param);
        ret = out_param.Get(*m_result);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "out_param get failed");
            return Status::ERROR;
        }
    }

    AURA_RETURN(m_ctx, ret);
}

std::string SumHvx::ToString() const
{
    return SumImpl::ToString() + m_profiling_string;
}

MeanHvx::MeanHvx(Context *ctx, const OpTarget &target) : SumHvx(ctx, target)
{}

Status MeanHvx::Run()
{
    Status ret = SumHvx::Run();
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "SumHvx run failed.");
        return Status::ERROR;
    }

    const DT_S32 height = m_src->GetSizes().m_height;
    const DT_S32 width  = m_src->GetSizes().m_width;
    *m_result           = (*m_result) / static_cast<DT_F64>(height * width);

    AURA_RETURN(m_ctx, ret);
}

} // namespace aura