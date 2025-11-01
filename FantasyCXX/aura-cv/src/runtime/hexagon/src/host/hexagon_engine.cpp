#include "host/hexagon_engine_impl.hpp"
#include "hexagon_defs.hpp"

namespace aura
{

static MI_BOOL g_callback_registered = MI_FALSE;
static MI_S32 g_htp_status = -1;

// FIXME: The current 4.2 SDK doesn't support rpc_status feature in remote.h, and aura2.0's solution for handling restarts has not yet been finalized.
// So first comment and save it here xulei21@xiaomi.com

// static MI_S32 FastRpcNotifyFunction(AURA_VOID *context, MI_S32 domains, MI_S32 session, MI_S32 status)
// {
//     AURA_UNUSED(context);
//     AURA_UNUSED(domains);
//     AURA_UNUSED(session);

//     // Context *ctx = static_cast<Context *>(context);
//     // AURA_LOGI(ctx, AURA_TAG, "FastRpcNotifyFunction called with status: %d\n", status);

//     g_htp_status = status;

//     return 0;
// }

// static Status RegisterCallback(Context *ctx, AURA_VOID *callback_context)
// {
//     Status ret = Status::OK;

//     remote_dsp_capability data;
//     data.domain = CDSP_DOMAIN_ID;
//     data.attribute_ID = STATUS_NOTIFICATION_SUPPORT;

//     if (remote_handle_control(DSPRPC_GET_DSP_INFO, &data, sizeof(data)) != 0)
//     {
//         AURA_ADD_ERROR_STRING(ctx, "remote_handle_control failed.\n");
//         ret = Status::ERROR;
//     }
//     else
//     {
//         if (data.capability)
//         {
//             remote_rpc_notif_register_t data;
//             data.context = callback_context;
//             data.domain  = CDSP_DOMAIN_ID;
//             data.notifier_fn = FastRpcNotifyFunction;

//             if (remote_session_control(FASTRPC_REGISTER_STATUS_NOTIFICATIONS, &data, sizeof(data)) != 0)
//             {
//                 AURA_ADD_ERROR_STRING(ctx, "remote_session_control register callback failed.\n");
//                 ret = Status::ERROR;
//             }
//             else
//             {
//                 g_callback_registered = MI_TRUE;
//                 ret = Status::OK;
//             }
//         }
//         else
//         {
//             AURA_ADD_ERROR_STRING(ctx, "STATUS_NOTIFICATION_SUPPORT is false.\n");
//             ret = Status::ERROR;
//         }
//     }

//     AURA_RETURN(ctx, ret);
// }

static Status QueryHardwareInfo(Context *ctx, HardwareInfo &info)
{
    Status ret = Status::OK;

    // query dsp arch infos
    remote_dsp_capability data;
    data.domain = CDSP_DOMAIN_ID;
    data.attribute_ID = ARCH_VER;

    if (remote_handle_control(DSPRPC_GET_DSP_INFO, &data, sizeof(data)) != 0)
    {
        AURA_ADD_ERROR_STRING(ctx, "remote_handle_control failed.\n");
        ret = Status::ERROR;
    }
    else
    {
        info.arch_version = data.capability & 0xff;
    }

    // query dsp vtcm layout hvx units info
    HexagonRpcParam rpc_param(ctx);
    QueryHWInfoOutParam out_param(ctx, rpc_param);
    ret |= ctx->GetHexagonEngine()->Run(AURA_RUNTIME_PACKAGE_NAME, AURA_RUNTIME_QUERY_HW_INFO_OP_NAME, rpc_param);

    Sequence<MI_S32> seq_page_sizes{info.vtcm_layout.page_sizes, 16};
    Sequence<MI_S32> seq_page_count{info.vtcm_layout.page_count, 16};

    ret |= out_param.Get(info.num_hvx_units, info.vtcm_layout.total_vtcm_size, info.vtcm_layout.page_list_count, seq_page_sizes, seq_page_count);

    AURA_RETURN(ctx, ret);
}

static Status QueryRealTimeInfo(Context *ctx, HexagonRTQueryType type, RealTimeInfo &info)
{
    HexagonRpcParam rpc_param(ctx);

    QueryRTInfoInParam in_param(ctx, rpc_param);

    Status ret = Status::OK;

    switch (type)
    {
        case HexagonRTQueryType::CURRENT_FREQ:
        {
            QueryRTInfoFreqOutParam out_param(ctx, rpc_param);
            ret = in_param.Set("CurrentFreq");
            ret |= ctx->GetHexagonEngine()->Run(AURA_RUNTIME_PACKAGE_NAME, AURA_RUNTIME_QUERY_RT_INFO_OP_NAME, rpc_param);
            ret |= out_param.Get(info.cur_freq);
            break;
        }
        case HexagonRTQueryType::VTCM_INFO:
        {
            QueryRTInfoVtcmOutParam out_param(ctx, rpc_param);
            Sequence<MI_S32> seq_page_sizes{info.vtcm_layout.page_sizes, 16};
            Sequence<MI_S32> seq_page_count{info.vtcm_layout.page_count, 16};
            ret = in_param.Set("VtcmInfo");
            ret |= ctx->GetHexagonEngine()->Run(AURA_RUNTIME_PACKAGE_NAME, AURA_RUNTIME_QUERY_RT_INFO_OP_NAME, rpc_param);
            ret |= out_param.Get(info.vtcm_layout.total_vtcm_size, info.vtcm_layout.page_list_count, seq_page_sizes, seq_page_count);
            break;
        }
        case HexagonRTQueryType::HTP_STATUS:
        {
            // if (!g_callback_registered)
            // {
            //     AURA_ADD_ERROR_STRING(ctx, "FastRpc callback register failed or unsupported.\n");
            //     ret = Status::ERROR;
            // }
            // else
            // {
            //     info.user_pd_status = g_htp_status;
            //     ret = Status::OK;
            // }

            AURA_UNUSED(g_callback_registered);
            AURA_UNUSED(g_htp_status);

            AURA_ADD_ERROR_STRING(ctx, "This feature current is not supported.\n");
            ret = Status::ERROR;
            break;
        }
        default:
        {
            ret = Status::ERROR;
            AURA_ADD_ERROR_STRING(ctx, "Unsupported HexagonRTQueryType.");
            break;
        }
    }

    AURA_RETURN(ctx, ret);
}

HexagonEngine::Impl::Impl(Context *ctx, MI_BOOL enable_hexagon, MI_BOOL unsigned_pd, const std::string &lib_prefix,
                          MI_BOOL async_call, LogOutput ouput, LogLevel level, const std::string &file)
                          : m_ctx(ctx), m_flag(MI_FALSE), m_async_call(async_call), m_handle(0)
{
    if (enable_hexagon)
    {
        // if (RegisterCallback(m_ctx, static_cast<AURA_VOID*>(m_ctx)) != Status::OK)
        // {
        //     // callback_context is poniter to a data buffer for callback info.
        //     AURA_LOGE(m_ctx, AURA_TAG, "RegisterCallback failed.\n");
        // }
        m_wp.reset(new WorkerPool(ctx, AURA_TAG, CpuAffinity::ALL, CpuAffinity::ALL, 0, 1));

        /// 这个是整个HVX 引擎的初始化的过程， TODO 为什么要进行异步初始化
        auto init_func = [=]() -> Status
        {
            remote_dsp_capability data;
            data.domain = CDSP_DOMAIN_ID;
            data.attribute_ID = ARCH_VER;
            MI_S32 ret = remote_handle_control(DSPRPC_GET_DSP_INFO, &data, sizeof(data));
            if (0 == ret)
            {
                MI_S32 cdsp_version = data.capability & 0xff;
                if (cdsp_version >= 0x69) //0x69:8450
                {
                    remote_rpc_thread_params th_data;
                    th_data.domain = CDSP_DOMAIN_ID;
                    th_data.stack_size = 17 * 1024;
                    th_data.prio = -1;
                    ret = remote_session_control(FASTRPC_THREAD_PARAMS, static_cast<void*>(&th_data), sizeof(th_data));
                    if (ret != 0)
                    {
                        AURA_ADD_ERROR_STRING(m_ctx, "remote_session_control run failed");
                        return Status::ERROR;
                    }
                }
            }

            if (MI_TRUE == unsigned_pd)
            {
                remote_rpc_control_unsigned_module data;
                data.enable = 1;
                data.domain = CDSP_DOMAIN_ID;
                ret = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, static_cast<void*>(&data), sizeof(data));
                if (ret != 0)
                {
                    AURA_ADD_ERROR_STRING(m_ctx, "remote_session_control run failed");
                    return Status::ERROR;
                }
            }

            std::string url_domain = "file:///lib" + lib_prefix + "_skel.so?aura_hexagon_skel_handle_invoke&_modver=1.0&_dom=cdsp";
            ret = aura_hexagon_open(url_domain.c_str(), &m_handle);
            if (ret != 0)
            {
                AURA_ADD_ERROR_STRING(m_ctx, "aura_hexagon_open run failed");
                return Status::ERROR;
            }
            HexagonRpcParam rpc_param(m_ctx);
            InitContextInParam in_param(m_ctx, rpc_param);
            in_param.Set(ouput, level, file);

            std::string full_name = AURA_RUNTIME_PACKAGE_NAME + std::string(".") + AURA_RUNTIME_INIT_CONTEXT_OP_NAME;
            RpcProfiling profiling;

            if (RegisterIonMem(rpc_param) != Status::OK)
            {
                aura_hexagon_close(m_handle);
                m_handle = 0;
                AURA_ADD_ERROR_STRING(m_ctx, "RegisterIonMem failed");
                return Status::ERROR;
            }
            ////
            ret = aura_hexagon_call(m_handle, full_name.c_str(), full_name.size(),
                                    reinterpret_cast<RpcMem*>(rpc_param.m_rpc_mem.data()), rpc_param.m_rpc_mem.size(),
                                    static_cast<MI_U8*>(rpc_param.m_rpc_param.m_origin), rpc_param.m_rpc_param.m_size, &profiling);

            if (UnregisterIonMem(rpc_param) != Status::OK)
            {
                aura_hexagon_close(m_handle);
                m_handle = 0;
                AURA_ADD_ERROR_STRING(m_ctx, "UnregisterIonMem failed");
                return Status::ERROR;
            }

            if (ret != 0)
            {
                aura_hexagon_close(m_handle);
                m_handle = 0;
                AURA_ADD_ERROR_STRING(m_ctx, "aura_hexagon_call run failed");
                return Status::ERROR;
            }
            //// HVX 进行正常初始化完成之后，我们将这个变量设置为true
            m_flag = MI_TRUE;
            return Status::OK;
        };

        /// 使用work pool 进行初始化。进行HVX的异步初始化
        if (m_wp)
        {
            m_init_token = m_wp->AsyncRun(init_func);
        }
        else
        {
            AURA_ADD_ERROR_STRING(m_ctx, "m_wp null ptr");
        }
    }
}

HexagonEngine::Impl::~Impl()
{
    if (m_init_token.valid())
    {
        m_init_token.wait();
    }

    if (m_flag)
    {
        aura_hexagon_close(m_handle);
        m_flag = MI_FALSE;
        m_handle = 0;
    }
}

/**
 *
 * @param package   #define AURA_OPS_FILTER_PACKAGE_NAME               "aura.ops.filter"
 * @param module    #define AURA_OPS_FILTER_GAUSSIAN_OP_NAME          "Gaussian"
 * @param rpc_param   传递进来的高斯滤波的相关参数
 * @param profiling
 * @return
 */
Status HexagonEngine::Impl::Run(const std::string &package, const std::string &module, HexagonRpcParam &rpc_param, HexagonProfiling *profiling) const
{
    /// 获取HVX是否初始化完成。 TODO 这个地方是会阻塞住？？让出CPU权限？？等待放行？？
    if (m_init_token.valid())
    {
        m_init_token.wait();
    }
    /// 其实就是这个变量，判断HVX是否正常初始化完成
    if (!m_flag)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "invalid engine");
        return Status::ERROR;
    }

    ///
    auto run_func = [&]() -> Status
    {
        /// RPC通信的结果
        RpcProfiling rpc_prof;
        /// 注意： 局部变量数据，在进行声明完成之后，一定要记得赋予初值。
        memset(&rpc_prof, 0, sizeof(rpc_prof));
        /// 判断传入的fullname其实就是：aura.ops.filter.Gaussian 类似这样的情况。
        std::string full_name = package + "." + module;
        /// 调用HVX hexagon engine
        /// m_handle 是HVX运行处理的回调句柄。
        /// 传入的高斯滤波相关的数据
        /// TODO 所以这个aura_hexagon_call就是我们最重要的核心算法。这个函数的具体实现：在hexagon_core.cpp文件中
        MI_S32 ret = aura_hexagon_call(m_handle, full_name.c_str(), full_name.size(), reinterpret_cast<RpcMem*>(rpc_param.m_rpc_mem.data()), rpc_param.m_rpc_mem.size(),
                                       static_cast<MI_U8*>(rpc_param.m_rpc_param.m_origin), rpc_param.m_rpc_param.m_capacity, &rpc_prof);
        Status ret_status = static_cast<Status>(rpc_prof.status);
        if (ret != 0)
        {
            std::string backtrace = GetBacktrace();
            AURA_ADD_ERROR_STRING(m_ctx, backtrace.c_str());
            AURA_ADD_ERROR_STRING(m_ctx, "aura_hexagon_call run failed");
            return Status::ERROR;
        }
        else if (profiling != MI_NULL && Status::OK == ret_status)
        {
            memset(profiling, 0, sizeof(HexagonProfiling));
            profiling->skel_time = rpc_prof.skel_time;
            profiling->clk_mhz = rpc_prof.clk_mhz;
        }

        return ret_status;
    };

    //// TODO  这个到底是干嘛的？？？ 需要我们注册ION内存？？
    if (RegisterIonMem(rpc_param) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "RegisterIonMem failed");
        return Status::ERROR;
    }

    //// 判断这个函数是异步调用还是调用还是同步调用
    ///  如果是异步调用： ret = m_wp->AsyncRun(run_func).get();
    ///  如果是同步调用： ret = run_func();
    Time start_time = Time::Now();
    Status ret = Status::ERROR;
    if (m_async_call)
    {
        ret = m_wp->AsyncRun(run_func).get();
    }
    else
    {
        ret = run_func();
    }

    Time exe_time = Time::Now() - start_time;
    if (Status::OK == ret && profiling != MI_NULL)
    {
        profiling->rpc_time = exe_time.AsMicroSec() - profiling->skel_time;
    }

    /// 反注册ION mem
    if (UnregisterIonMem(rpc_param) != Status::OK)
    {
        ret = Status::ERROR;
        AURA_ADD_ERROR_STRING(m_ctx, "UnregisterIonMem failed");
    }
    /// 重置里面的Buffer数据
    rpc_param.ResetBuffer();

    AURA_RETURN(m_ctx, ret);
}

Status HexagonEngine::Impl::SetPower(HexagonPowerLevel target_level, MI_BOOL enable_dcvs, MI_U32 client_id)
{
    Status ret = Status::OK;

    HexagonRpcParam rpc_param(m_ctx);
    SetPowerInParam in_param(m_ctx, rpc_param);
    ret = in_param.Set(target_level, enable_dcvs, client_id);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Set failed");
        return ret;
    }
    ret |= Run(AURA_RUNTIME_PACKAGE_NAME, AURA_RUNTIME_SET_POWER_OP_NAME, rpc_param, MI_NULL);

    AURA_RETURN(m_ctx, ret);
}

std::string HexagonEngine::Impl::GetVersion() const
{
    HexagonRpcParam rpc_param(m_ctx, 4096);
    Status ret = Run(AURA_RUNTIME_PACKAGE_NAME, AURA_RUNTIME_GET_VERSION_OP_NAME, rpc_param, MI_NULL);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Run failed");
        return "INVALID";
    }
    GetVersionOutParam out_param(m_ctx, rpc_param);
    std::string version;
    ret |= out_param.Get(version);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Get failed");
        return "INVALID";
    }

    return version;
}

Status HexagonEngine::Impl::QueryHWInfo(HardwareInfo &info)
{
    static MI_BOOL call_once_flag = MI_FALSE;

    Status ret = Status::OK;

    if (!call_once_flag)
    {
        if (m_init_token.valid())
        {
            m_init_token.wait();
        }

        ret = QueryHardwareInfo(m_ctx, m_hw_info);

        call_once_flag = MI_TRUE;
    }

    info = m_hw_info;

    AURA_RETURN(m_ctx, ret);
}

Status HexagonEngine::Impl::QueryRTInfo(HexagonRTQueryType type, RealTimeInfo &info)
{
    Status ret = Status::OK;

    if (m_init_token.valid())
    {
        m_init_token.wait();
    }

    ret = QueryRealTimeInfo(m_ctx, type, info);

    AURA_RETURN(m_ctx, ret);
}

std::string HexagonEngine::Impl::GetBacktrace() const
{
    std::string backtrace;
    HexagonRpcParam rpc_param(m_ctx, 4096);
    std::string full_name = AURA_RUNTIME_PACKAGE_NAME + std::string(".") + AURA_RUNTIME_GET_BACKTRACE_OP_NAME;
    RpcProfiling rpc_prof;

    if (RegisterIonMem(rpc_param) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "RegisterIonMem failed");
        return std::string();
    }

    MI_S32 ret = aura_hexagon_call(m_handle, full_name.c_str(), full_name.size(), reinterpret_cast<RpcMem*>(rpc_param.m_rpc_mem.data()), rpc_param.m_rpc_mem.size(),
                                   static_cast<MI_U8*>(rpc_param.m_rpc_param.m_origin), rpc_param.m_rpc_param.m_capacity, &rpc_prof);

    if (UnregisterIonMem(rpc_param) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "UnregisterIonMem failed");
        return std::string();
    }

    if (ret != 0)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Run failed");
        return std::string();
    }

    GetBacktraceOutParam out_param(m_ctx, rpc_param);
    Status ret_status = out_param.Get(backtrace, MI_TRUE);
    if (ret_status != Status::OK)
    {
        AURA_ADD_ERROR_STRING(m_ctx, "Get failed");
        return std::string();
    }

    return backtrace;
}

Status HexagonEngine::Impl::RegisterIonMem(HexagonRpcParam &rpc_param) const
{
    Status ret = Status::ERROR;

    for (size_t i = 0; i < rpc_param.m_rpc_mem.size(); i++)
    {
        Buffer buffer = m_ctx->GetMemPool()->GetBuffer(rpc_param.m_rpc_mem[i].mem);
        if (buffer.IsValid() && (AURA_MEM_DMA_BUF_HEAP == buffer.m_type))
        {
            remote_register_buf_attr(buffer.m_origin, buffer.m_capacity, buffer.m_property, 0);
        }
        else
        {
            AURA_ADD_ERROR_STRING(m_ctx, "invalid buffer occur, note that extern buffer is not supported");
            goto EXIT;
        }
    }

    if (rpc_param.m_rpc_param.IsValid() && (AURA_MEM_DMA_BUF_HEAP == rpc_param.m_rpc_param.m_type))
    {
        remote_register_buf_attr(rpc_param.m_rpc_param.m_origin, rpc_param.m_rpc_param.m_capacity, rpc_param.m_rpc_param.m_property, 0);
    }
    else
    {
        AURA_ADD_ERROR_STRING(m_ctx, "rpc param store in an invalid buffer");
        goto EXIT;
    }

    ret = Status::OK;
EXIT:
    if (ret != Status::OK)
    {
        UnregisterIonMem(rpc_param);
    }
    return ret;
}

Status HexagonEngine::Impl::UnregisterIonMem(HexagonRpcParam &rpc_param) const
{
    for (size_t i = 0; i < rpc_param.m_rpc_mem.size(); i++)
    {
        Buffer buffer = m_ctx->GetMemPool()->GetBuffer(rpc_param.m_rpc_mem[i].mem);
        if (buffer.IsValid() && (AURA_MEM_DMA_BUF_HEAP == buffer.m_type))
        {
            remote_register_buf_attr(buffer.m_origin, buffer.m_capacity, -1, 0);
        }
        else
        {
            AURA_ADD_ERROR_STRING(m_ctx, "invalid buffer occur, note that extern buffer is not supported");
            return Status::ERROR;
        }
    }

    if (rpc_param.m_rpc_param.IsValid() && (AURA_MEM_DMA_BUF_HEAP == rpc_param.m_rpc_param.m_type))
    {
        remote_register_buf_attr(rpc_param.m_rpc_param.m_origin, rpc_param.m_rpc_param.m_capacity, -1, 0);
    }
    else
    {
        AURA_ADD_ERROR_STRING(m_ctx, "rpc param store in an invalid buffer");
        return Status::ERROR;
    }

    return Status::OK;
}

HexagonEngine::HexagonEngine(Context *ctx, MI_BOOL enable_hexagon, MI_BOOL unsigned_pd, const std::string &lib_prefix,
                             MI_BOOL async_call, LogOutput ouput, LogLevel level, const std::string &file)
{
    m_impl.reset(new HexagonEngine::Impl(ctx, enable_hexagon, unsigned_pd, lib_prefix, async_call, ouput, level, file));
    if (!m_impl)
    {
        AURA_ADD_ERROR_STRING(ctx, "m_impl null ptr");
    }
}

HexagonEngine::~HexagonEngine()
{
}

Status HexagonEngine::SetPower(HexagonPowerLevel target_level, MI_BOOL enable_dcvs, MI_U32 client_id)
{
    if (m_impl)
    {
        return m_impl->SetPower(target_level, enable_dcvs, client_id);
    }
    return Status::ERROR;
}

Status HexagonEngine::Run(const std::string &package, const std::string &module, HexagonRpcParam &rpc_param, HexagonProfiling *profiling) const
{
    if (m_impl)
    {
        return m_impl->Run(package, module, rpc_param, profiling);
    }
    return Status::ERROR;
}

std::string HexagonEngine::GetVersion() const
{
    if (m_impl)
    {
        return m_impl->GetVersion();
    }
    return "INVALID";
}

Status HexagonEngine::QueryHWInfo(HardwareInfo &info)
{
    if (m_impl)
    {
        return m_impl->QueryHWInfo(info);
    }
    return Status::ERROR;
}

Status HexagonEngine::QueryRTInfo(HexagonRTQueryType type, RealTimeInfo &info)
{
    if (m_impl)
    {
        return m_impl->QueryRTInfo(type, info);
    }
    return Status::ERROR;
}

} // namepsace aura