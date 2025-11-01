#ifndef AURA_RUNTIME_HEXAGON_HOST_HEXAGON_ENGINE_IMPL_HPP__
#define AURA_RUNTIME_HEXAGON_HOST_HEXAGON_ENGINE_IMPL_HPP__

#include "aura/runtime/hexagon/host/hexagon_engine.hpp"
#include "aura_hexagon.h"
#include "aura/runtime/worker_pool.h"

namespace aura
{

class HexagonEngine::Impl
{
public:
    Impl(Context *ctx,
         MI_BOOL enable_hexagon,
         MI_BOOL unsigned_pd,
         const std::string &lib_prefix,
         MI_BOOL async_call,
         LogOutput ouput,
         LogLevel level,
         const std::string &file);
    ~Impl();

    Status SetPower(HexagonPowerLevel target_level, MI_BOOL enable_dcvs, MI_U32 client_id = 0);
    Status Run(const std::string &pack_name, const std::string &op_name, HexagonRpcParam &rpc_param, HexagonProfiling *profiling) const;
    std::string GetVersion() const;
    Status QueryHWInfo(HardwareInfo &info);
    Status QueryRTInfo(HexagonRTQueryType type, RealTimeInfo &info);
private:
    Status RegisterIonMem(HexagonRpcParam &rpc_param) const;
    Status UnregisterIonMem(HexagonRpcParam &rpc_param) const;
    std::string GetBacktrace() const;

    Context *m_ctx;
    MI_BOOL m_flag;
    MI_BOOL m_async_call;
    remote_handle64 m_handle;
    std::future<Status> m_init_token;
    std::shared_ptr<WorkerPool> m_wp;
    HardwareInfo m_hw_info;
};

} // namespace aura

#endif // AURA_RUNTIME_HEXAGON_HOST_HEXAGON_ENGINE_IMPL_HPP__