#include <string>

#include "aura/runtime/hexagon/device/hexagon_runtime.hpp"
#include "aura_hexagon.h"

#include <unordered_map>

#include "AEEStdErr.h"
#include "HAP_perf.h"
#include "aura/runtime/core/defs.hpp"
#include "aura/runtime/core/types/built-in.hpp"
#include "aura/runtime/utils/logger.hpp"

namespace aura
{
class Context;

std::unordered_map<std::string, RpcFunc>& GetRpcFuncMap();

} // namespace aura

AEEResult aura_hexagon_open(const char *uri, remote_handle64 *h)
{
    AURA_UNUSED(uri);
    aura::Context *ctx = new aura::Context;
    if (DT_NULL == ctx)
    {
        return AEE_EFAILED;
    }

    *h = reinterpret_cast<remote_handle64>(ctx);

    return AEE_SUCCESS;
}

AEEResult aura_hexagon_close(remote_handle64 h)
{
    aura::Context *ctx = reinterpret_cast<aura::Context*>(h);
    if (DT_NULL == ctx)
    {
        return AEE_EFAILED;
    }

    delete ctx;
    return AEE_SUCCESS;
}

/**
 * 调用hexagon
 * @param h 远程的逻辑的回调句柄
 * @param name 对应的OP算子的名称。
 * @param name_len
 * @param mem
 * @param mem_len
 * @param param
 * @param param_len
 * @param profiling
 * @return
 */
AEEResult aura_hexagon_call(remote_handle64 h, const char *name, int name_len, const RpcMem *mem, int mem_len,
                            uint8 *param, int param_len, RpcProfiling *profiling)
{
    AURA_UNUSED(mem_len);

    aura::Context *ctx = reinterpret_cast<aura::Context*>(h);
    if (DT_NULL == ctx)
    {
        return AEE_EFAILED;
    }

    DT_U64 start_time = HAP_perf_get_time_us();
    DT_U64 start_cycs = HAP_perf_get_pcycles();

    std::string full_name(name, name_len);

    //// TODO 这个地方我没有看太懂。
    std::unordered_map<std::string, aura::RpcFunc>& func_map = aura::GetRpcFuncMap();
    if (func_map.find(full_name) == func_map.end())
    {
        AURA_ADD_ERROR_STRING(ctx, "invalid func name");
        return AEE_EFAILED;
    }

    aura::RpcFunc func = func_map[full_name];
    aura::HexagonRpcParam rpc_param(ctx, reinterpret_cast<const aura::HexagonRpcMem*>(mem), param, param_len);

    aura::Status ret = func(ctx, rpc_param);

    DT_U64 end_time = HAP_perf_get_time_us();
    DT_U64 end_cycs = HAP_perf_get_pcycles();

    DT_U64 cycles = end_cycs - start_cycs;
    profiling->status = static_cast<DT_U32>(ret);
    profiling->skel_time = end_time - start_time;
    profiling->clk_mhz = static_cast<DT_U64>(cycles / profiling->skel_time);

    if (aura::Status::ERROR == ret)
    {
        AURA_ADD_ERROR_STRING(ctx, "rpc func run failed");
        return AEE_EFAILED;
    }

    return AEE_SUCCESS;
}