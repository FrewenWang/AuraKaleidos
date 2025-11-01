#include "aura/runtime/xtensa/device/xtensa_runtime.hpp"
#include "aura/runtime/core/xtensa/comm.hpp"

#include "tileManager.h"
#include "xm_xrp_cmd_struct.h"

application_symbol_tray *g_symbol_tray = MI_NULL;

#if defined (__cplusplus)
extern "C"
{
#endif

class XtensaRpcData
{
public:
    XtensaRpcData(MI_U8 *data, MI_U32 len)
    {
        if (len == sizeof(m_data))
        {
            aura::xtensa::Memcpy(m_data, data, sizeof(m_data));
        }
        else
        {
            aura::xtensa::Memset(m_data, 0, sizeof(m_data));
        }
    }

    MI_CHAR* GetName(MI_U32 &len)
    {
        len = m_data[1];
        return reinterpret_cast<MI_CHAR*>(m_data[0]);
    }

    MI_U8* GetRpcParam(MI_U32 &len)
    {
        len = m_data[3];
        return reinterpret_cast<MI_U8*>(m_data[2]);
    }
    
private:
    MI_U32 m_data[4];
};

static xm_vdsp_pic_funcs pic_funcs;

AURA_VOID AuraExtensaRun(xrp_vdsp_cmd *msg)
{
    XtensaRpcData rpc_data(msg->in_data, msg->in_data_size);

    MI_S32 ret = -1;

    MI_CHAR *name = NULL;
    MI_U32 name_len = 0;

    MI_U8 *rpc_param = MI_NULL;
    MI_U32 rpc_param_len = 0;

    name = rpc_data.GetName(name_len);
    if (MI_NULL == name || 0 == name_len)
    {
        AURA_XTENSA_LOG("GetName failed, name=%p name_len=%zu\n", name, name_len);
        goto EXIT;
    }

    rpc_param = rpc_data.GetRpcParam(rpc_param_len);
    if (MI_NULL == rpc_param || 0 == rpc_param_len)
    {
        AURA_XTENSA_LOG("GetName failed, rpc_param=%p rpc_param_len=%zu\n", rpc_param, rpc_param_len);
        goto EXIT;
    }

    /*!< Invalidate the data cache for the operation name. */
    aura::xtensa::DCacheInvalidate(name, name_len);
    aura::xtensa::DCacheInvalidate(rpc_param, rpc_param_len);

    ret = aura::xtensa::VdspRpcCall(name, rpc_param, rpc_param_len);
    if (ret != 0)
    {
        AURA_XTENSA_LOG("VdspRpcCall failed\n");
    }

EXIT:
    msg->out_data_size = sizeof(ret);
    aura::xtensa::Memcpy(msg->out_data, &ret, msg->out_data_size);
}

AURA_VOID* _start(AURA_VOID* arg)
{
    g_symbol_tray = (application_symbol_tray*)arg;

    pic_funcs.func_num = 1;
    pic_funcs.func_cmd[0].function_handler = AuraExtensaRun;

    return (AURA_VOID*)&pic_funcs;
}

#if defined (__cplusplus)
}
#endif