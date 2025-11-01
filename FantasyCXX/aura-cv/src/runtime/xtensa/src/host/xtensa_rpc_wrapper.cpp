#include "host/xtensa_library.hpp"
#include "host/xtensa_rpc_wrapper.hpp"

namespace aura
{

XtensaRpcWrapper::XtensaRpcWrapper(Context *ctx, AURA_VOID *handle, MI_U32 op_id, std::string &name, XtensaRpcParam *param) :
                                   m_ctx(ctx), m_handle(handle), m_op_id(op_id), m_param(param), m_xtensa_engine(MI_NULL)
{
    m_full_name_buffer = m_ctx->GetMemPool()->GetBuffer(AURA_ALLOC_PARAM(m_ctx, AURA_MEM_DMA_BUF_HEAP, name.size(), 0));

    m_xtensa_engine = m_ctx->GetXtensaEngine();
    if (MI_NULL == m_xtensa_engine)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "GetXtensaEngine failed, m_xtensa_engine is null ptr");
        return;
    }

    if (m_xtensa_engine->MapBuffer(m_full_name_buffer) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "m_full_name_buffer MapBuffer failed");
        return;
    }

    memcpy(m_full_name_buffer.m_data, name.c_str(), name.size());

    if (m_xtensa_engine->MapBuffer(param->m_rpc_param) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "m_buffer MapBuffer failed");
    }
}

XtensaRpcWrapper::~XtensaRpcWrapper()
{
    if (m_full_name_buffer.IsValid())
    {
        AURA_FREE(m_ctx, m_full_name_buffer.m_origin);
    }
}

Status XtensaRpcWrapper::Run()
{
    // cache sync
    m_xtensa_engine->CacheEnd(m_full_name_buffer.m_property);

    m_xtensa_engine->CacheEnd(m_param->m_rpc_param.m_property);

    XtensaRpcData rpc_data(m_xtensa_engine->GetDeviceAddr(m_full_name_buffer), m_full_name_buffer.m_size,
                           m_xtensa_engine->GetDeviceAddr(m_param->m_rpc_param), m_param->m_rpc_param.m_size);

    MI_U32 *in_data = NULL;
    MI_U32 in_data_len = 0;
    MI_S32 output = -1;
    MI_U32 output_size = sizeof(MI_S32);

    in_data = rpc_data.GetData(in_data_len);
    Status ret = Status::ERROR;

    if (vdsp_run_node(m_handle, m_op_id, 0, in_data, in_data_len, &output, output_size) != 0)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "vdsp_run_node failed");
        goto EXIT;
    }

    if (output != 0)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "result error");
        goto EXIT;
    }

    ret = Status::OK;

EXIT:
    if (m_xtensa_engine->UnmapBuffer(m_full_name_buffer) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "UnmapBuffer failed");
        ret = Status::ERROR;
    }

    if (m_xtensa_engine->UnmapBuffer(m_param->m_rpc_param) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "UnmapBuffer failed");
        ret = Status::ERROR;
    }

    AURA_RETURN(m_ctx, ret);
}

} // namespace aura