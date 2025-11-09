#include "host/xtensa_engine_impl.hpp"
#include "aura/runtime/worker_pool/host/worker_pool.hpp"

#if !defined(AURA_BUILD_XPLORER)
#include "host/xtensa_library.hpp"
#include "host/xtensa_rpc_wrapper.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#else
#include "aura/runtime/xtensa/host/rpc_param.hpp"
#endif // AURA_BUILD_XPLORER

namespace aura
{

XtensaEngine::Impl::Impl(Context *ctx, DT_BOOL enable_xtensa, const std::string &pil_name, XtensaPriorityLevel priority)
                         : m_ctx(ctx), m_flag(DT_FALSE), m_pil_name(pil_name), m_priority(priority)
{
    if (enable_xtensa)
    {
#if defined(AURA_BUILD_XPLORER)
        AURA_UNUSED(m_priority);
#else
        m_handle = DT_NULL;
        m_wp.reset(new WorkerPool(ctx, AURA_TAG, CpuAffinity::ALL, CpuAffinity::ALL, 0, 1));

        auto init_func = [this]() -> Status
        {
            if (LoadPil() != Status::OK)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "LoadPil failed");
                return Status::ERROR;
            }

            m_flag = DT_TRUE;

            return Status::OK;
        };

        if (m_wp)
        {
            m_init_token = m_wp->AsyncRun(init_func);
        }
        else
        {
            AURA_ADD_ERROR_STRING(m_ctx, "m_wp null ptr");
        }
#endif // AURA_BUILD_XPLORER
    }
}

XtensaEngine::Impl::~Impl()
{
#if !defined(AURA_BUILD_XPLORER)
    if (m_init_token.valid())
    {
        m_init_token.wait();
    }
#endif // AURA_BUILD_XPLORER

    if (m_flag)
    {
#if !defined(AURA_BUILD_XPLORER)

        if (!m_mblk_map.empty())
        {
            AURA_LOGD(m_ctx, AURA_TAG, "****************** Unmap Blk info *******************\n");

            DT_S32 counter = 0;
            DT_S32 unmap_mem_size = 0;

            for (auto iter = m_mblk_map.begin(); iter != m_mblk_map.end(); ++iter)
            {
                DT_S32 type = iter->second.m_host_buffer.m_type;
                DT_S64 size = iter->second.m_host_buffer.m_size;

                AURA_LOGD(m_ctx, AURA_TAG, "* blk [%zu] - %p\n", counter, reinterpret_cast<DT_VOID*>(iter->first));
                AURA_LOGD(m_ctx, AURA_TAG, "*   fd: %zu\n", iter->second.m_host_buffer.m_property);
                AURA_LOGD(m_ctx, AURA_TAG, "*   mem type: %s\n", MemTypeToString(type).c_str());
                AURA_LOGD(m_ctx, AURA_TAG, "*   size: %zu byte\n", size);
                AURA_LOGD(m_ctx, AURA_TAG, "*   device_addr: %zu\n",  iter->second.m_device_addr);
                AURA_LOGD(m_ctx, AURA_TAG, "*\n");

                counter++;
                unmap_mem_size += size;


                if (vdsp_unmap_buffer(m_handle, iter->second.m_host_buffer.m_property) != 0)
                {
                    AURA_ADD_ERROR_STRING(m_ctx, "vdsp_unmap_buffer failed");
                }
            }

            AURA_LOGD(m_ctx, AURA_TAG, "***********************************************\n");

            AURA_LOGD(m_ctx, AURA_TAG, "* total unmap mem size: %.2f KB (%.4f MB)\n",
                      unmap_mem_size / 1024.f, unmap_mem_size / 1048576.f);

            m_mblk_map.clear();
        }

        if (UnloadPil() != Status::OK)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "UnloadPil failed");
        }
#endif // AURA_BUILD_XPLORER

        m_flag = DT_FALSE;
    }
}

#if defined(AURA_BUILD_XPLORER)
Status XtensaEngine::Impl::RegistTray()
{
    m_symbol_tray.pTMObj = &m_xv_tm;
    if (xvfInitTileManager(m_symbol_tray.pTMObj) != 0)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "XvfInitTileManager error");
        return Status::ERROR;
    }

    m_symbol_tray.tray_printf                                       = printf;
    m_symbol_tray.tray_vprintf                                      = vprintf;
    m_symbol_tray.tray_strcmp                                       = strcmp;
    m_symbol_tray.tray_memcpy                                       = memcpy;
    m_symbol_tray.tray_memset                                       = memset;
    m_symbol_tray.tray_strlen                                       = strlen;
    m_symbol_tray.tray_strcpy                                       = strcpy;
    m_symbol_tray.tray_memmove                                      = memmove;
    m_symbol_tray.tray_strstr                                       = strstr;
    m_symbol_tray.tray_modff                                        = modff;
    m_symbol_tray.tray_modf                                         = modf;
    m_symbol_tray.tray_fabsf                                        = fabsf;
    m_symbol_tray.tray_fabs                                         = fabs;
    m_symbol_tray.tray_sqrtf                                        = sqrtf;
    m_symbol_tray.tray_sqrt                                         = sqrt;
    m_symbol_tray.tray_expf                                         = expf;
    m_symbol_tray.tray_exp                                          = exp;
    m_symbol_tray.tray_exp2f                                        = exp2f;
    m_symbol_tray.tray_exp2                                         = exp2;
    m_symbol_tray.tray_logf                                         = logf;
    m_symbol_tray.tray_log                                          = log;
    m_symbol_tray.tray_log2f                                        = log2f;
    m_symbol_tray.tray_log2                                         = log2;
    m_symbol_tray.tray_log10f                                       = log10f;
    m_symbol_tray.tray_log10                                        = log10;
    m_symbol_tray.tray_powf                                         = powf;
    m_symbol_tray.tray_pow                                          = pow;
    m_symbol_tray.tray_sinf                                         = sinf;
    m_symbol_tray.tray_sin                                          = sin;
    m_symbol_tray.tray_cosf                                         = cosf;
    m_symbol_tray.tray_cos                                          = cos;
    m_symbol_tray.tray_tanf                                         = tanf;
    m_symbol_tray.tray_tan                                          = tan;
    m_symbol_tray.tray_asinf                                        = asinf;
    m_symbol_tray.tray_asin                                         = asin;
    m_symbol_tray.tray_acosf                                        = acosf;
    m_symbol_tray.tray_acos                                         = acos;
    m_symbol_tray.tray_atanf                                        = atanf;
    m_symbol_tray.tray_atan                                         = atan;
    m_symbol_tray.tray_atan2f                                       = atan2f;
    m_symbol_tray.tray_atan2                                        = atan2;
    m_symbol_tray.tray_xvSetupTile                                  = xvSetupTileHost;
    m_symbol_tray.tray_xvRegisterTile                               = xvRegisterTileHost;
    m_symbol_tray.tray_xvAllocateBuffer                             = xvAllocateBufferHost;
    m_symbol_tray.tray_xvFreeBuffer                                 = xvFreeBufferHost;
    m_symbol_tray.tray_xvCreateTile                                 = xvCreateTileHost;
    m_symbol_tray.tray_xvCheckTileReadyMultiChannel3D               = xvCheckTileReadyMultiChannel3DHost;
    m_symbol_tray.tray_xvCreateTile3D                               = xvCreateTile3DHost;
    m_symbol_tray.tray_xvInitIdmaMultiChannel4CH                    = xvInitIdmaMultiChannel4CHHost;
    m_symbol_tray.tray_xvInitTileManagerMultiChannel4CH             = xvInitTileManagerMultiChannel4CHHost;
    m_symbol_tray.tray_xvResetTileManager                           = xvResetTileManagerHost;
    m_symbol_tray.tray_xvFreeFrame                                  = xvFreeFrameHost;
    m_symbol_tray.tray_xvAddIdmaRequestMultiChannel_predicated_wide = xvAddIdmaRequestMultiChannel_predicated_wideHost;
    m_symbol_tray.tray_xvAddIdmaRequestMultiChannel_wide            = xvAddIdmaRequestMultiChannel_wideHost;
    m_symbol_tray.tray_xvAddIdmaRequestMultiChannel_wide3D          = xvAddIdmaRequestMultiChannel_wide3DHost;
    m_symbol_tray.tray_xvReqTileTransferInFastMultiChannel          = xvReqTileTransferInFastMultiChannelHost;
    m_symbol_tray.tray_xvReqTileTransferOutFastMultiChannel         = xvReqTileTransferOutFastMultiChannelHost;
    m_symbol_tray.tray_xvCheckForIdmaIndexMultiChannel              = xvCheckForIdmaIndexMultiChannelHost;
    m_symbol_tray.tray_xvSleepForTileMultiChannel                   = xvSleepForTileMultiChannelHost;
    m_symbol_tray.tray_xvWaitForiDMAMultiChannel                    = xvWaitForiDMAMultiChannelHost;
    m_symbol_tray.tray_xvSleepForiDMAMultiChannel                   = xvSleepForiDMAMultiChannelHost;
    m_symbol_tray.tray_xvWaitForTileFastMultiChannel                = xvWaitForTileFastMultiChannelHost;
    m_symbol_tray.tray_xvSleepForTileFastMultiChannel               = xvSleepForTileFastMultiChannelHost;
    m_symbol_tray.tray_xvSleepForTileMultiChannel3D                 = xvSleepForTileMultiChannel3DHost;
    m_symbol_tray.tray_xvReqTileTransferInMultiChannelPredicated    = xvReqTileTransferInMultiChannelPredicatedHost;
    m_symbol_tray.tray_xvReqTileTransferInMultiChannel              = xvReqTileTransferInMultiChannelHost;
    m_symbol_tray.tray_xvReqTileTransferOutMultiChannelPredicated   = xvReqTileTransferOutMultiChannelPredicatedHost;
    m_symbol_tray.tray_xvReqTileTransferInMultiChannel3D            = xvReqTileTransferInMultiChannel3DHost;
    m_symbol_tray.tray_xvReqTileTransferOutMultiChannel3D           = xvReqTileTransferOutMultiChannel3DHost;
    m_symbol_tray.tray_xvReqTileTransferOutMultiChannel             = xvReqTileTransferOutMultiChannelHost;
    m_symbol_tray.tray_idma_init_task                               = idma_init_task;
    m_symbol_tray.tray_idma_add_2d_desc64                           = idma_add_2d_desc64;
    m_symbol_tray.tray_idma_add_2d_desc64_wide                      = idma_add_2d_desc64_wide;
    m_symbol_tray.tray_idma_schedule_task                           = idma_schedule_task;
    m_symbol_tray.tray_idma_schedule_desc                           = idma_schedule_desc;
    m_symbol_tray.tray_idma_process_tasks                           = idma_process_tasks;
    m_symbol_tray.tray_idma_desc_done                               = idma_desc_done;

    return Status::OK;
}

application_symbol_tray& XtensaEngine::Impl::GetSymbolTray()
{
    return m_symbol_tray;
}

Status XtensaEngine::Impl::RegisterRpcFunc(PilRpcFunc rpc_func)
{
    if (DT_NULL == rpc_func)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "rpc_func null");
        return Status::ERROR;
    }

    m_func = rpc_func;
    Status ret = RegistTray();

    AURA_RETURN(m_ctx, ret);
}
#endif // AURA_BUILD_XPLORER

#if !defined(AURA_BUILD_XPLORER)
Status XtensaEngine::Impl::LoadPil()
{
    Status ret = Status::ERROR;

    DT_S32 fd = 0;
    DT_VOID *addr = DT_NULL;

    struct stat fstat;
    vdsp_init_para param;
    vdsp_init_response response;

    if ((fd = open(m_pil_name.c_str(), O_RDWR)) <= 0)
    {
        std::string info = "open " + m_pil_name + " failed";
        AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
        return ret;
    }

    if (stat(m_pil_name.c_str(), &fstat) < 0)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "stat failed");
        goto EXIT;
    }

    if (MAP_FAILED == (addr = mmap(DT_NULL, fstat.st_size, PROT_READ, MAP_SHARED, fd, 0)))
    {
        AURA_ADD_ERROR_STRING(m_ctx, "mmap failed");
        goto EXIT;
    }

    memset(&param, 0, sizeof(param));
    memset(&response, 0, sizeof(response));

    param.log_close                      = 1;
    param.priority                       = static_cast<DT_U32>(m_priority);
    param.time_out                       = 1800000;
    param.profiling                      = 1;
    param.is_custom                      = 1;
    param.custom_para.op_mirror_num      = 1;
    param.custom_para.op_func_num        = 1;
    param.custom_para.op_mirror_sizes[0] = fstat.st_size;
    param.custom_para.op_mirror_data[0]  = addr;
    param.custom_para.op_fun_entry[0]    = "AuraExtensaRun";

    if (vdsp_init(&m_handle, &param, &response) != 0)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "vdsp_init failed");
        ret = Status::ERROR;
        goto EXIT;
    }

    m_op_id = response.out_op_ids[0];

    ret = Status::OK;
EXIT:
    if (ret != Status::OK)
    {
        UnloadPil();
    }

    if (addr != DT_NULL)
    {
        munmap(addr, fstat.st_size);
    }

    if (fd > 0)
    {
        close(fd);
    }

    return ret;
}

Status XtensaEngine::Impl::UnloadPil()
{
    if (m_handle != DT_NULL)
    {
        if (vdsp_release(m_handle) != 0)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "vdsp_release failed");
            return Status::ERROR;
        }

        m_handle = DT_NULL;
    }

    return Status::OK;
}

Status XtensaEngine::Impl::Run(const std::string &package, const std::string &module, XtensaRpcParam &rpc_param)
{
    if (m_init_token.valid())
    {
        m_init_token.wait();
    }

    if (!m_flag)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "load pil is failed");
        return Status::ERROR;
    }

    std::string full_name = package + "." + module;

    XtensaRpcWrapper rpc_wrapper(m_ctx, m_handle, m_op_id, full_name, &rpc_param);
    return rpc_wrapper.Run();
}
#endif // AURA_BUILD_XPLORER


#if defined(AURA_BUILD_XPLORER)
Status XtensaEngine::Impl::Run(const std::string &package, const std::string &module, XtensaRpcParam &rpc_param)
{
    PilRpcFunc func = m_func;
    if (DT_NULL == func)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "function null");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    std::string full_name = package + "." + module;
    DT_S32 ret_func = func(full_name.c_str(), static_cast<DT_U8*>(rpc_param.m_rpc_param.m_origin), rpc_param.m_rpc_param.m_capacity);
    if (ret_func != 0)
    {
        ret = Status::ERROR;
        AURA_ADD_ERROR_STRING(m_ctx, "VdspRpcCall run failed");
        return Status::ERROR;
    }
    else
    {
        ret = Status::OK;
    }

    AURA_RETURN(m_ctx, ret);
}
#endif // AURA_BUILD_XPLORER

Status XtensaEngine::Impl::CacheStart(DT_S32 fd)
{
#if defined(AURA_BUILD_XPLORER)
    AURA_UNUSED(fd);
    return Status::OK;
#else

    if (vdsp_cache_start(m_handle, fd) != 0)
    {
        std::string info = "vdsp_cache_start failed, cache fd(" + std::to_string(fd) + ") failed";
        AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
        return Status::ERROR;
    }
    return Status::OK;
#endif
}

Status XtensaEngine::Impl::CacheEnd(DT_S32 fd)
{
#if defined(AURA_BUILD_XPLORER)
    AURA_UNUSED(fd);
    return Status::OK;
#else

    if (vdsp_cache_end(m_handle, fd) != 0)
    {
        std::string info = "vdsp_cache_end failed, cache fd(" + std::to_string(fd) + ") failed";
        AURA_ADD_ERROR_STRING(m_ctx, info.c_str());
        return Status::ERROR;
    }
    return Status::OK;
#endif
}

Status XtensaEngine::Impl::MapBuffer(const Buffer &buffer)
{
#if !defined(AURA_BUILD_XPLORER)
    if (m_init_token.valid())
    {
        m_init_token.wait();
    }
#endif // AURA_BUILD_XPLORER

    if (!buffer.IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "buffer is invalid");
        return Status::ERROR;
    }

#if !defined(AURA_BUILD_XPLORER)
    std::lock_guard<std::mutex> guard(m_lock);
    DT_UPTR_T host_addr = reinterpret_cast<DT_UPTR_T>(buffer.m_origin);
    DT_U32 device_addr = 0;

    if (!m_mblk_map.count(host_addr))
    {
        if (vdsp_map_buffer(m_handle, buffer.m_property, buffer.m_capacity, DT_FALSE, &device_addr) != 0)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "vdsp_map_buffer failed");
            return Status::ERROR;
        }

        m_mblk_map.emplace(host_addr, MemBlk(buffer, device_addr));
    }
#endif // AURA_BUILD_XPLORER

    return Status::OK;
}

Status XtensaEngine::Impl::UnmapBuffer(const Buffer &buffer)
{
#if !defined(AURA_BUILD_XPLORER)
    if (m_init_token.valid())
    {
        m_init_token.wait();
    }
#endif // AURA_BUILD_XPLORER

    if (!buffer.IsValid())
    {
        AURA_ADD_ERROR_STRING(m_ctx, "buffer is invalid");
        return Status::ERROR;
    }

#if !defined(AURA_BUILD_XPLORER)
    std::lock_guard<std::mutex> guard(m_lock);
    DT_UPTR_T host_addr = reinterpret_cast<DT_UPTR_T>(buffer.m_origin);

    if (m_mblk_map.count(host_addr))
    {
        if (vdsp_unmap_buffer(m_handle, buffer.m_property) != 0)
        {
            AURA_ADD_ERROR_STRING(m_ctx, "vdsp_unmap_buffer failed");
        }

        m_mblk_map.erase(host_addr);
    }
#endif // AURA_BUILD_XPLORER

    return Status::OK;
}

DT_U32 XtensaEngine::Impl::GetDeviceAddr(const Buffer &buffer)
{
    if (!buffer.IsValid())
    {
        return 0;
    }

#if !defined(AURA_BUILD_XPLORER)
    std::lock_guard<std::mutex> guard(m_lock);
    DT_UPTR_T host_addr = reinterpret_cast<DT_UPTR_T>(buffer.m_origin);

    if (m_mblk_map.count(host_addr))
    {
        return m_mblk_map.at(host_addr).m_device_addr;
    }
#else
    return reinterpret_cast<DT_U32>(buffer.m_origin);
#endif // AURA_BUILD_XPLORER

    return 0;
}

Status XtensaEngine::Impl::SetPower(XtensaPowerLevel level)
{
#if defined(AURA_BUILD_XPLORER)
    AURA_UNUSED(level);
#else
    if (m_init_token.valid())
    {
        m_init_token.wait();
    }

    if (!m_flag)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "load pil is failed");
        return Status::ERROR;
    }

    DT_U32 power_level = static_cast<DT_U32>(level);

    if (vdsp_set_power(m_handle, &power_level) != 0)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "vdsp_set_power failed");
        return Status::ERROR;
    }
#endif
    return Status::OK;
}

XtensaEngine::XtensaEngine(Context *ctx, DT_BOOL enable_xtensa, const std::string &pil_name, XtensaPriorityLevel priority)
{
    m_impl.reset(new XtensaEngine::Impl(ctx, enable_xtensa, pil_name, priority));
    if (!m_impl)
    {
        AURA_ADD_ERROR_STRING(ctx, "m_impl null ptr");
    }
}

XtensaEngine::~XtensaEngine()
{}

Status XtensaEngine::Run(const std::string &package, const std::string &module, XtensaRpcParam &rpc_param)
{
    if (m_impl)
    {
        return m_impl->Run(package, module, rpc_param);
    }
    return Status::ERROR;
}

Status XtensaEngine::CacheStart(DT_S32 fd)
{
    if (m_impl)
    {
        return m_impl->CacheStart(fd);
    }
    return Status::ERROR;
}

Status XtensaEngine::CacheEnd(DT_S32 fd)
{
    if (m_impl)
    {
        return m_impl->CacheEnd(fd);
    }
    return Status::ERROR;
}

Status XtensaEngine::MapBuffer(const Buffer &buffer)
{
    if (m_impl)
    {
        return m_impl->MapBuffer(buffer);
    }
    return Status::ERROR;
}

Status XtensaEngine::UnmapBuffer(const Buffer &buffer)
{
    if (m_impl)
    {
        return m_impl->UnmapBuffer(buffer);
    }
    return Status::ERROR;
}

DT_U32 XtensaEngine::GetDeviceAddr(const Buffer &buffer)
{
    if (m_impl)
    {
        return m_impl->GetDeviceAddr(buffer);
    }
    return 0;
}

Status XtensaEngine::SetPower(XtensaPowerLevel level)
{
    if (m_impl)
    {
        return m_impl->SetPower(level);
    }
    return Status::ERROR;
}

#if defined(AURA_BUILD_XPLORER)
application_symbol_tray& XtensaEngine::GetSymbolTray()
{
    return m_impl->GetSymbolTray();
}

Status XtensaEngine::RegisterRpcFunc(PilRpcFunc rpc_func)
{
    return m_impl->RegisterRpcFunc(rpc_func);
}
#endif // AURA_BUILD_XPLORER

} // namepsace aura