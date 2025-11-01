#include "aura/ops/core/xtensa/core.hpp"

namespace aura
{
namespace xtensa
{

using RpcParamInParam    = XtensaRpcParamType<string, map<MI_S32>>;
using RpcParamOutParam   = XtensaRpcParamType<MI_S32, vector<MI_S32>, map<MI_S32>>;

Status RpcParamTestRpc(TileManager xv_tm, XtensaRpcParam &rpc_param);

} // xtensa
} // aura