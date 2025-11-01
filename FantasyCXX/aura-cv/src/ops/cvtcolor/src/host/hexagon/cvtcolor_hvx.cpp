#include "cvtcolor_impl.hpp"
#include "cvtcolor_comm.hpp"

namespace aura
{
CvtColorHvx::CvtColorHvx(Context *ctx, const OpTarget &target) : CvtColorImpl(ctx, target)
{}

Status CvtColorHvx::SetArgs(const std::vector<const Array*> &src, const std::vector<Array*> &dst, CvtColorType type)
{
    if (CvtColorImpl::SetArgs(src, dst, type) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "CvtColorImpl::SetArgs failed(HVX)");
        return Status::ERROR;
    }

    for (MI_U32 i = 0; i < src.size(); i++)
    {
        if (src[i]->GetArrayType() != ArrayType::MAT)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "src must be mat type");
            return Status::ERROR;
        }
    }

    for (MI_U32 i = 0; i < dst.size(); i++)
    {
        if (dst[i]->GetArrayType() != ArrayType::MAT)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "dst must be mat type");
            return Status::ERROR;
        }
    }

    return Status::OK;
}

Status CvtColorHvx::Run()
{
    std::vector<Mat> vec_src;
    std::vector<Mat> vec_dst;

    for (MI_U32 i = 0; i < m_src.size(); i++)
    {
        Mat *p_src = dynamic_cast<Mat*>(const_cast<Array*>(m_src[i]));
        vec_src.push_back(*p_src);
    }

    for (MI_U32 i = 0; i < m_dst.size(); i++)
    {
        Mat *p_dst = dynamic_cast<Mat*>(m_dst[i]);
        vec_dst.push_back(*p_dst);
    }

    if ((vec_src.size() <= 0) || (vec_dst.size() <= 0))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src dst is null");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    /// 传入这个RPC的param
    HexagonRpcParam rpc_param(m_ctx);
    CvtColorInParamHvx in_param(m_ctx, rpc_param);
    ret = in_param.Set(vec_src, vec_dst, m_type);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Set failed");
        return Status::ERROR;
    }

    HexagonProfiling profiling;
    HexagonEngine   *engine = m_ctx->GetHexagonEngine();
    ret                     = engine->Run(AURA_OPS_CVTCOLOR_PACKAGE_NAME, AURA_OPS_CVTCOLOR_OP_NAME, rpc_param, &profiling);

    if (Status::OK == ret && MI_TRUE == m_target.m_data.hvx.profiling)
    {
        m_profiling_string = " " + HexagonProfilingToString(profiling);
    }

    AURA_RETURN(m_ctx, ret);
}

std::string CvtColorHvx::ToString() const
{
    return CvtColorImpl::ToString() + m_profiling_string;
}

} // namespace aura