#include "aura/ops/core/xtensa/core.hpp"

namespace aura
{
namespace xtensa
{

using RpcParamInParam    = XtensaRpcParamType<string, map<DT_S32>>;
using RpcParamOutParam   = XtensaRpcParamType<DT_S32, vector<DT_S32>, map<DT_S32>>;

Status RpcParamTestRpc(TileManager xv_tm, XtensaRpcParam &rpc_param);

} // xtensa
} // aura