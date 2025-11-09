#include "aura/runtime/xtensa/device/xtensa_runtime.hpp"
#include "xtensa_rpc_func_register.hpp"

#include "tileManager.h"
#include "tileManager_FIK_api.h"

extern application_symbol_tray *g_symbol_tray;

namespace aura
{
namespace xtensa
{

DT_S32 VdspRpcCall(const DT_CHAR *name, DT_U8 *param, DT_S32 param_len)
{
    if (DT_NULL == param)
    {
        AURA_XTENSA_LOG("input null\n");
        return AURA_XTENSA_ERROR;
    }

    DT_S32 ret = AURA_XTENSA_ERROR;

    XtensaRpcFuncRegister *ptr = DT_NULL;
    for (DT_U32 i = 0; i < (sizeof(g_rpc_func_map) / sizeof(g_rpc_func_map[0])); i++)
    {
        if (Strcmp(name, g_rpc_func_map[i].name) == 0)
        {
            ptr = (g_rpc_func_map + i);
            break;
        }
    }

    if ((DT_NULL == ptr) || (DT_NULL == ptr->func))
    {
        AURA_XTENSA_LOG("module(%s) do not support\n", name);
        return AURA_XTENSA_ERROR;
    }

    XtensaRpcParam rpc_param(param, param_len);
    ret = (DT_S32)ptr->func(g_symbol_tray->pTMObj, rpc_param);
    if (AURA_XTENSA_OK != ret)
    {
        AURA_XTENSA_LOG("ptr->call_func failed!\n");
        return ret;
    }

    return ret;
}

} // namespace xtensa
} // namespace aura