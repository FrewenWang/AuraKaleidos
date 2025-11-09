#include "aura/runtime/hexagon/rpc_param.hpp"

namespace aura
{

#define AURA_RUNTIME_PACKAGE_NAME              "aura.runtime.hexagon"
#define AURA_RUNTIME_INIT_CONTEXT_OP_NAME      "InitContext"
#define AURA_RUNTIME_SET_POWER_OP_NAME         "SetPower"
#define AURA_RUNTIME_GET_VERSION_OP_NAME       "GetVersion"
#define AURA_RUNTIME_GET_BACKTRACE_OP_NAME     "GetBacktrace"
#define AURA_RUNTIME_QUERY_PARAM_OP_NAME       "QueryParam"
#define AURA_RUNTIME_QUERY_HW_INFO_OP_NAME     "QueryHWInfo"
#define AURA_RUNTIME_QUERY_RT_INFO_OP_NAME     "QueryRTInfo"

using SetPowerInParam      = HexagonRpcParamType<HexagonPowerLevel, DT_BOOL, DT_U32>;
using InitContextInParam   = HexagonRpcParamType<LogOutput, LogLevel, std::string>;
using GetVersionOutParam   = HexagonRpcParamType<std::string>;
using GetBacktraceOutParam = HexagonRpcParamType<std::string>;
using QueryHexagonParam    = HexagonRpcParamType<std::string>;
using QueryHWInfoOutParam  = HexagonRpcParamType<DT_S32, DT_S32, DT_S32, Sequence<DT_S32>, Sequence<DT_S32>>;
using QueryRTInfoInParam   = HexagonRpcParamType<std::string>;
using QueryRTInfoFreqOutParam = HexagonRpcParamType<DT_F32>;
using QueryRTInfoVtcmOutParam = HexagonRpcParamType<DT_S32, DT_S32, Sequence<DT_S32>, Sequence<DT_S32>>;

} // namespace aura