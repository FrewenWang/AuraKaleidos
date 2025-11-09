#ifndef AURA_RUNTIME_XTENSA_HOST_XTENSA_RPC_WRAPPER_HPP__
#define AURA_RUNTIME_XTENSA_HOST_XTENSA_RPC_WRAPPER_HPP__

#include "aura/runtime/core.h"
#include "aura/runtime/xtensa/host/rpc_param.hpp"
#include "aura/runtime/xtensa/host/xtensa_engine.hpp"

namespace aura
{

class XtensaRpcWrapper
{
public:
    XtensaRpcWrapper(Context *ctx, DT_VOID *handle, DT_U32 op_id, std::string &name, XtensaRpcParam *param);

    ~XtensaRpcWrapper();

    Status Run();

private:
    struct XtensaRpcData
    {
    public:
        XtensaRpcData(DT_U32 full_name_addr, DT_U32 full_name_len, DT_U32 rpc_param_addr, DT_U32 rpc_param_len)
        {
            m_data[0] = full_name_addr;
            m_data[1] = full_name_len;
            m_data[2] = rpc_param_addr;
            m_data[3] = rpc_param_len;
        }

        DT_U32* GetData(DT_U32 &len)
        {
            len = sizeof(m_data);
            return m_data;
        }

    private:
        DT_U32 m_data[4];
    };

    Context *m_ctx;
    void *m_handle;
    DT_U32 m_op_id;
    std::string m_name;
    XtensaRpcParam *m_param;
    XtensaEngine *m_xtensa_engine;
    Buffer m_full_name_buffer;
};
} // namespace aura

#endif // AURA_RUNTIME_XTENSA_HOST_XTENSA_RPC_WRAPPER_HPP__