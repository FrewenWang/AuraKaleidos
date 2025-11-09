#include "cvtcolor_impl.hpp"
#include "cvtcolor_comm.hpp"

namespace aura
{
CvtColorVdsp::CvtColorVdsp(Context *ctx, const OpTarget &target) : CvtColorImpl(ctx, target)
{}

Status CvtColorVdsp::SetArgs(const std::vector<const Array*> &src, const std::vector<Array*> &dst, CvtColorType type)
{
    if (CvtColorImpl::SetArgs(src, dst, type) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "CvtColorImpl::SetArgs failed(Vdsp)");
        return Status::ERROR;
    }

    for (DT_U32 i = 0; i < src.size(); i++)
    {
        if (src[i]->GetArrayType() != ArrayType::MAT)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "src must be mat type");
            return Status::ERROR;
        }
    }

    for (DT_U32 i = 0; i < dst.size(); i++)
    {
        if (dst[i]->GetArrayType() != ArrayType::MAT)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "dst must be mat type");
            return Status::ERROR;
        }
    }

    return Status::OK;
}

Status CvtColorVdsp::Initialize()
{
    if (CvtColorImpl::Initialize() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "CvtColorImpl::Initialize() failed");
        return Status::ERROR;
    }

    m_xtensa_src.clear();
    for (DT_U32 i = 0; i < m_src.size(); i++)
    {
        XtensaMat xtensa_src = XtensaMat::FromArray(m_ctx, *(m_src[i]));
        if (!xtensa_src.IsValid())
        {
            AURA_ADD_ERROR_STRING(m_ctx, "xtensa_src is valid");
            return Status::ERROR;
        }

        if (xtensa_src.Sync(XtensaSyncType::WRITE) != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "xtensa_src sync failed");
            return Status::ERROR;
        }
        m_xtensa_src.push_back(xtensa_src);
    }

    m_xtensa_dst.clear();
    for (DT_U32 i = 0; i < m_dst.size(); i++)
    {
        XtensaMat xtensa_dst = XtensaMat::FromArray(m_ctx, *(m_dst[i]));
        if (!xtensa_dst.IsValid())
        {
            AURA_ADD_ERROR_STRING(m_ctx, "xtensa_src is valid");
            return Status::ERROR;
        }
        m_xtensa_dst.push_back(xtensa_dst);
    }

    if ((m_xtensa_src.size() <= 0) || (m_xtensa_dst.size() <= 0))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "m_xtensa_src/m_xtensa_dst size must greater 0");
        return Status::ERROR;
    }

    return Status::OK;
}

Status CvtColorVdsp::DeInitialize()
{
    for (DT_U32 i = 0; i < m_xtensa_src.size(); i++)
    {
        m_xtensa_src[i].Release();
    }
    m_xtensa_src.clear();

    for (DT_U32 i = 0; i < m_xtensa_dst.size(); i++)
    {
        m_xtensa_dst[i].Release();
    }
    m_xtensa_dst.clear();

    if (CvtColorImpl::DeInitialize() != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "CvtColorImpl::DeInitialize() failed");
        return Status::ERROR;
    }

    return Status::OK;
}

Status CvtColorVdsp::Run()
{
    Status ret = Status::ERROR;

    XtensaRpcParam rpc_param(m_ctx);
    CvtColorInParamVdsp in_param(m_ctx, rpc_param);
    if (in_param.Set(m_xtensa_src, m_xtensa_dst, m_type) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Set failed");
        return Status::ERROR;
    }

    XtensaEngine *engine = m_ctx->GetXtensaEngine();
    if (DT_NULL == engine)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "GetXtensaEngine failed");
        return Status::ERROR;
    }

    if (engine->Run(AURA_OPS_CVTCOLOR_PACKAGE_NAME, AURA_OPS_CVTCOLOR_OP_NAME, rpc_param) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "engine Run failed");
        goto EXIT;
    }

    for (DT_U32 i = 0; i < m_dst.size(); i++)
    {
        if (m_xtensa_dst[i].Sync(XtensaSyncType::READ) != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "m_xtensa_dst sync failed");
            goto EXIT;
        }
    }

    ret = Status::OK;

EXIT:
    AURA_RETURN(m_ctx, ret);
}

std::string CvtColorVdsp::ToString() const
{
    return CvtColorImpl::ToString();
}

} // namespace aura