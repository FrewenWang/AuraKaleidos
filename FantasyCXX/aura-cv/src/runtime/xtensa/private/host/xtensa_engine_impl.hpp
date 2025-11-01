#ifndef AURA_RUNTIME_XTENSA_HOST_XTENSA_ENGINE_IMPL_HPP__
#define AURA_RUNTIME_XTENSA_HOST_XTENSA_ENGINE_IMPL_HPP__

#include "aura/runtime/xtensa/host/xtensa_engine.hpp"
#include <unordered_map>

#if defined(AURA_BUILD_XPLORER)
#  include "tileManager.h"
#  include "tileManager_FIK_api.h"
#else
#  include<future>
#endif // AURA_BUILD_XPLORER

namespace aura
{

class XtensaEngine::Impl
{
public:
    Impl(Context *ctx, MI_BOOL enable_xtensa, const std::string &pil_name, XtensaPriorityLevel priority);
    ~Impl();

    Status Run(const std::string &package, const std::string &module,
               XtensaRpcParam &rpc_param);

    Status CacheStart(MI_S32 fd);

    Status CacheEnd(MI_S32 fd);

    Status MapBuffer(const Buffer &buffer);

    Status UnmapBuffer(const Buffer &buffer);

    MI_U32 GetDeviceAddr(const Buffer &buffer);

    Status SetPower(XtensaPowerLevel level);

#if defined(AURA_BUILD_XPLORER)
    application_symbol_tray& GetSymbolTray();
    Status RegisterRpcFunc(PilRpcFunc rpc_func);
#endif // AURA_BUILD_XPLORER

private:
#if defined(AURA_BUILD_XPLORER)
    Status RegistTray();
#else
    Status LoadPil();
    Status UnloadPil();
#endif // AURA_BUILD_XPLORER

private:
#if !defined(AURA_BUILD_XPLORER)
    struct MemBlk
    {
        MemBlk(const Buffer &buffer, MI_U32 device_addr) : m_host_buffer(buffer), m_device_addr(device_addr)
        {}

        Buffer m_host_buffer;
        MI_U32 m_device_addr;
    };
#endif // AURA_BUILD_XTENSA

    Context *m_ctx;
    MI_BOOL m_flag;
    std::string m_pil_name;
    XtensaPriorityLevel m_priority;

#if defined(AURA_BUILD_XPLORER)
    PilRpcFunc m_func;
    application_symbol_tray m_symbol_tray;
    xvTileManager m_xv_tm;
#else
    MI_U32 m_op_id;
    AURA_VOID *m_handle;
    std::future<Status> m_init_token;
    std::shared_ptr<WorkerPool> m_wp;
    std::mutex m_lock;
    std::unordered_map<MI_UPTR_T, MemBlk> m_mblk_map;
#endif // AURA_BUILD_XPLORER
};

} // namespace aura
#endif // AURA_RUNTIME_XTENSA_HOST_XTENSA_ENGINE_IMPL_HPP__