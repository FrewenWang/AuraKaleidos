#include "gaussian_impl.hpp"
#include "filter_comm.hpp"

namespace aura
{

GaussianVdsp::GaussianVdsp(Context *ctx, const OpTarget &target) : GaussianImpl(ctx, target)
{}

Status GaussianVdsp::SetArgs(const Array *src, Array *dst, MI_S32 ksize, MI_F32 sigma,
                             BorderType border_type, const Scalar &border_value)
{
    if (GaussianImpl::SetArgs(src, dst, ksize, sigma, border_type, border_value) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "GaussianImpl::SetArgs failed");
        return Status::ERROR;
    }

    if ((src->GetArrayType() != ArrayType::MAT) || (dst->GetArrayType() != ArrayType::MAT))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "src or dst must be mat type");
        return Status::ERROR;
    }

    if (m_ksize != 3)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "ksize only supports 3");
        return Status::ERROR;
    }

    MI_S32 ch = src->GetSizes().m_channel;
    if (ch != 1)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "channel only supports 1");
        return Status::ERROR;
    }

    ElemType src_elem_type = src->GetElemType();
    ElemType dst_elem_type = dst->GetElemType();
    if ((src_elem_type != ElemType::U8 || dst_elem_type != ElemType::U8))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "only support src elem_type u8, dst elem_type u8");
        return Status::ERROR;
    }

    return Status::OK;
}

Status GaussianVdsp::Initialize()
{
    if (GaussianImpl::Initialize() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "GaussianImpl::Initialize() failed");
        return Status::ERROR;
    }

    m_xtensa_src = XtensaMat::FromArray(m_ctx, *m_src);
    if (!m_xtensa_src.IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "m_xtensa_src is valid");
        return Status::ERROR;
    }

    if (m_xtensa_src.Sync(XtensaSyncType::WRITE) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "m_xtensa_src sync failed");
        return Status::ERROR;
    }

    m_xtensa_dst = XtensaMat::FromArray(m_ctx, *m_dst);
    if (!m_xtensa_dst.IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "m_xtensa_dst is valid");
        return Status::ERROR;
    }

    return Status::OK;
}

Status GaussianVdsp::DeInitialize()
{
    m_xtensa_src.Release();
    m_xtensa_dst.Release();

    if (GaussianImpl::DeInitialize() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "GaussianImpl::DeInitialize() failed");
        return Status::ERROR;
    }

    return Status::OK;
}

Status GaussianVdsp::Run()
{
    Status ret = Status::ERROR;

    XtensaRpcParam rpc_param(m_ctx);
    GaussianInParamVdsp in_param(m_ctx, rpc_param);
    if (in_param.Set(m_xtensa_src, m_xtensa_dst, m_ksize, m_sigma, m_border_type, m_border_value) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Set failed");
        return ret;
    }

    XtensaEngine *engine = m_ctx->GetXtensaEngine();
    if (MI_NULL == engine)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "GetXtensaEngine failed");
        return ret;
    }

    if (engine->Run(AURA_OPS_FILTER_PACKAGE_NAME, AURA_OPS_FILTER_GAUSSIAN_OP_NAME, rpc_param) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "engine Run failed");
        goto EXIT;
    }

    if (m_xtensa_dst.Sync(XtensaSyncType::READ) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "m_xtensa_dst sync failed");
        goto EXIT;
    }

    ret = Status::OK;
EXIT:
    AURA_RETURN(m_ctx, ret);
}

std::string GaussianVdsp::ToString() const
{
    return GaussianImpl::ToString();
}

} // namespace aura