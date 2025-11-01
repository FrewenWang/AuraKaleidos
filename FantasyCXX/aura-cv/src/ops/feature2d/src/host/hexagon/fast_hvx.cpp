#include "fast_impl.hpp"
#include "feature2d_comm.hpp"

namespace aura
{

FastHvx::FastHvx(Context *ctx, const OpTarget &target) : FastImpl(ctx, target)
{}

Status FastHvx::SetArgs(const Array *src, std::vector<KeyPoint> &key_points, MI_S32 threshold,
                        MI_BOOL nonmax_suppression, FastDetectorType type, MI_U32 max_num_corners)
{
    if (FastImpl::SetArgs(src, key_points, threshold, nonmax_suppression, type, max_num_corners) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "FastImpl::SetArgs failed");
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

    MI_S32 height = src->GetSizes().m_height;
    if (height < 7)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src height only support > 7");
        return Status::ERROR;
    }

    if (type != FastDetectorType::FAST_9_16)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "fast feature hvx only support fast9");
        return Status::ERROR;
    }

    if (((MI_UPTR_T)(dynamic_cast<const Mat*>(src)->GetData()) & 127) && (src->GetStrides().m_width & 127))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src mat strides need align to 128");
        return Status::ERROR;
    }

    return Status::OK;
}

Status FastHvx::Run()
{
    const Mat *src = dynamic_cast<const Mat*>(m_src);

    if (MI_NULL == src)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src is nullptr");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    HexagonRpcParam rpc_param(m_ctx, 1024 + m_max_num_corners * sizeof(KeyPoint));
    FastInParam in_param(m_ctx, rpc_param);
    ret = in_param.Set(*src, m_threshold, m_nonmax_suppression, m_detector_type, m_max_num_corners);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Set failed");
        return Status::ERROR;
    }

    HexagonProfiling profiling;
    HexagonEngine *engine = m_ctx->GetHexagonEngine();
    ret = engine->Run(AURA_OPS_FEATURE2D_PACKAGE_NAME, AURA_OPS_FEATURE2D_FAST_OP_NAME, rpc_param, &profiling);

    if (Status::OK == ret)
    {
        if (MI_TRUE == m_target.m_data.hvx.profiling)
        {
            m_profiling_string = " " + HexagonProfilingToString(profiling);
        }

        FastOutParam out_param(m_ctx, rpc_param);
        ret = out_param.Get(*m_key_points);
        if (ret != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "Set failed");
            return Status::ERROR;
        }
    }

    AURA_RETURN(m_ctx, ret);
}

std::string FastHvx::ToString() const
{
    return FastImpl::ToString() + m_profiling_string;
}

} // namespace aura