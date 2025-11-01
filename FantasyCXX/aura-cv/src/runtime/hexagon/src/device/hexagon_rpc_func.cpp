#include "aura/runtime/hexagon/device/hexagon_runtime.hpp"
#include "hexagon_defs.hpp"

#include "AEEStdDef.h"
#include "qurt_hvx.h"
#include "HAP_power.h"
#include "HAP_vtcm_mgr.h"
#include "HAP_compute_res.h"

namespace aura
{

Status InitContextRpc(Context *ctx, HexagonRpcParam &rpc_param)
{
    if (MI_NULL == ctx)
    {
        return Status::ERROR;
    }

    LogOutput ouput;
    LogLevel level;
    std::string file;

    InitContextInParam in_param(ctx, rpc_param);
    Status ret = in_param.Get(ouput, level, file);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "Get failed");
        return Status::ERROR;
    }

    ret = ctx->Initialize(ouput, level, file);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "ctx Initialize failed");
    }

    AURA_RETURN(ctx, ret);
}

Status SetPowerRpc(Context *ctx, HexagonRpcParam &rpc_param)
{
    if (MI_NULL == ctx)
    {
        return Status::ERROR;
    }

    HexagonPowerLevel target_level;
    MI_BOOL enable_dcvs;
    MI_U32 client_id;

    SetPowerInParam in_param(ctx, rpc_param);
    Status ret = in_param.Get(target_level, enable_dcvs, client_id);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "ctx Initialize failed");
        return Status::ERROR;
    }

    ret = SetPower(ctx, target_level, enable_dcvs, client_id);

    AURA_RETURN(ctx, ret);
}

Status GetVersionRpc(Context *ctx, HexagonRpcParam &rpc_param)
{
    if (MI_NULL == ctx)
    {
        return Status::ERROR;
    }

    GetVersionOutParam out_param(ctx, rpc_param);
    Status ret = out_param.Set(ctx->GetVersion());

    AURA_RETURN(ctx, ret);
}

Status GetBacktraceRpc(Context *ctx, HexagonRpcParam &rpc_param)
{
    if (MI_NULL == ctx)
    {
        return Status::ERROR;
    }

    MI_S32 max_str_size = rpc_param.m_rpc_param.m_size - sizeof(MI_S32);
    std::string backtrace = ctx->GetLogger()->GetErrorString();
    if (!backtrace.empty())
    {
        backtrace.pop_back();
    }
    if (static_cast<MI_S32>(backtrace.size()) > max_str_size)
    {
        backtrace = backtrace.substr(0, max_str_size);
    }

    GetBacktraceOutParam out_param(ctx, rpc_param);
    Status ret = out_param.Set(backtrace);

    AURA_RETURN(ctx, ret);
}

Status QueryHWInfoRpc(Context *ctx, HexagonRpcParam &rpc_param)
{
    if (MI_NULL == ctx)
    {
        return Status::ERROR;
    }

    QueryHWInfoOutParam out_param(ctx, rpc_param);

    MI_S32 num_hvx_units = 0;
    MI_S32 total_vtcm_size = 0;
    MI_S32 num_vtcm_pages = 0;
    MI_S32 vtcm_page_sizes[16] = {0};
    MI_S32 vtcm_page_count[16] = {0};

    Status ret = Status::OK;

    // Step1: get num of hvx units
    num_hvx_units = (qurt_hvx_get_units() >> 8) & 0xFF;

#if __HEXAGON_ARCH__ >= 66
    // Step2: allocate an 4K vtcm page
    compute_res_attr_t res_attr;
    HAP_compute_res_attr_init(&res_attr);
    HAP_compute_res_attr_set_serialize(&res_attr, 0);
    HAP_compute_res_attr_set_vtcm_param(&res_attr, 4 * 1024, 1); // 4KB single page
    unsigned int context_id = HAP_compute_res_acquire(&res_attr, 10000);

    if (0 == context_id)
    {
        AURA_LOGE(ctx, AURA_TAG, "HAP_compute_res_acquire failed.");
        ret = Status::ERROR;
    }

    // Step3: query vtcm layout info
    unsigned int total_blk_size = 0;
    unsigned int avail_blk_size = 0;
    compute_res_vtcm_page_t total_blk_layout;
    compute_res_vtcm_page_t avail_blk_layout;

    if (HAP_compute_res_query_VTCM(0, &total_blk_size, &total_blk_layout, &avail_blk_size, &avail_blk_layout) != 0)
    {
        AURA_LOGE(ctx, AURA_TAG, "HAP_compute_res_query_VTCM failed.");
        ret = Status::ERROR;
    }
    else
    {
        total_vtcm_size = total_blk_size / 1024;
        num_vtcm_pages  = total_blk_layout.page_list_len + avail_blk_layout.page_list_len;

        MI_U32 idx = 0;
        for (MI_U32 i = 0; i < total_blk_layout.page_list_len; ++i)
        {
            vtcm_page_sizes[idx] = total_blk_layout.page_list[i].page_size / 1024;
            vtcm_page_count[idx] = 1;
            idx++;
        }

        for (MI_U32 i = 0; i < avail_blk_layout.page_list_len; ++i)
        {
            vtcm_page_sizes[idx] = avail_blk_layout.page_list[i].page_size / 1024;
            vtcm_page_count[idx] = 1;
            idx++;
        }
    }

    if (context_id != 0)
    {
        HAP_compute_res_release(context_id);
    }
#else
    unsigned int page_size = 0;
    unsigned int page_count = 0;
    if (HAP_query_total_VTCM(&page_size, &page_count) != 0)
    {
        ret = Status::ERROR;
        AURA_ADD_ERROR_STRING(ctx, "HAP_query_total_VTCM failed.\n");
    }
    else
    {
        total_vtcm_size = (page_size * page_count) / 1024;
        num_vtcm_pages  = 1;
        vtcm_page_sizes[0] = page_size;
        vtcm_page_count[0] = 1;
    }
#endif // __HEXAGON_ARCH__ >= 66

    Sequence<MI_S32> seq_page_sizes{vtcm_page_sizes, 16};
    Sequence<MI_S32> seq_page_count{vtcm_page_count, 16};

    ret |= out_param.Set(num_hvx_units, total_vtcm_size, num_vtcm_pages, seq_page_sizes, seq_page_count);

    AURA_RETURN(ctx, ret);
}

Status QueryRTInfoRpc(Context *ctx, HexagonRpcParam &rpc_param)
{
    std::string param_key;
    QueryRTInfoInParam in_param(ctx, rpc_param);

    if (in_param.Get(param_key) != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "get param failed");
        return Status::ERROR;
    }

    Status ret = Status::OK;

    if ("CurrentFreq" == param_key)
    {
        QueryRTInfoFreqOutParam out_param(ctx, rpc_param);

        MI_S32 context_id = 0;
        HAP_power_response_t response;
        response.type = HAP_power_get_clk_Freq;

        if (HAP_power_get(&context_id, &response) != 0)
        {
            AURA_LOGE(ctx, AURA_TAG, "HAP_power_get failed.");
            ret = out_param.Set(0.0f);
        }
        else
        {
            ret = out_param.Set(response.clkFreqHz / 1e6f);
        }
    }
    else if("VtcmInfo" == param_key)
    {
        if (MI_NULL == ctx)
        {
            return Status::ERROR;
        }

        QueryRTInfoVtcmOutParam out_param(ctx, rpc_param);

        MI_S32 avail_vtcm_size = 0;
        MI_S32 num_vtcm_pages = 0;
        MI_S32 vtcm_page_sizes[16] = {0};
        MI_S32 vtcm_page_count[16] = {0};

#if __HEXAGON_ARCH__ >= 66
        unsigned int total_blk_size = 0;
        unsigned int avail_blk_size = 0;
        compute_res_vtcm_page_t total_blk_layout;
        compute_res_vtcm_page_t avail_blk_layout;

        if (HAP_compute_res_query_VTCM(0, &total_blk_size, &total_blk_layout, &avail_blk_size, &avail_blk_layout) != 0)
        {
            AURA_LOGE(ctx, AURA_TAG, "HAP_compute_res_query_VTCM failed.");
            ret = Status::ERROR;
        }
        else
        {
            avail_vtcm_size = total_blk_size / 1024;
            num_vtcm_pages  = avail_blk_layout.page_list_len;

            for (MI_U32 i = 0; i < avail_blk_layout.page_list_len; ++i)
            {
                vtcm_page_sizes[i] = avail_blk_layout.page_list[i].page_size / 1024;
                vtcm_page_count[i] = avail_blk_layout.page_list[i].num_pages;
            }
        }
#else
        unsigned int avail_block_size = 0;
        unsigned int max_page_size = 0;
        unsigned int num_pages = 0;
        if (HAP_query_avail_VTCM(&avail_block_size, &max_page_size, &num_pages) != 0)
        {
            ret = Status::ERROR;
            AURA_ADD_ERROR_STRING(ctx, "HAP_query_avail_VTCM failed.\n");
        }
        else
        {
            avail_vtcm_size = (avail_block_size) / 1024;
            num_vtcm_pages  = 1;
            vtcm_page_sizes[0] = max_page_size;
            vtcm_page_count[0] = num_pages;
        }
#endif // __HEXAGON_ARCH__ >= 66

        Sequence<MI_S32> seq_page_sizes{vtcm_page_sizes, 16};
        Sequence<MI_S32> seq_page_count{vtcm_page_count, 16};
        ret |= out_param.Set(avail_vtcm_size, num_vtcm_pages, seq_page_sizes, seq_page_count);
    }
    AURA_RETURN(ctx, ret);
}

AURA_HEXAGON_RPC_FUNC_REGISTER(AURA_RUNTIME_PACKAGE_NAME, AURA_RUNTIME_INIT_CONTEXT_OP_NAME, InitContextRpc);
AURA_HEXAGON_RPC_FUNC_REGISTER(AURA_RUNTIME_PACKAGE_NAME, AURA_RUNTIME_SET_POWER_OP_NAME, SetPowerRpc);
AURA_HEXAGON_RPC_FUNC_REGISTER(AURA_RUNTIME_PACKAGE_NAME, AURA_RUNTIME_GET_VERSION_OP_NAME, GetVersionRpc);
AURA_HEXAGON_RPC_FUNC_REGISTER(AURA_RUNTIME_PACKAGE_NAME, AURA_RUNTIME_GET_BACKTRACE_OP_NAME, GetBacktraceRpc);
AURA_HEXAGON_RPC_FUNC_REGISTER(AURA_RUNTIME_PACKAGE_NAME, AURA_RUNTIME_QUERY_HW_INFO_OP_NAME, QueryHWInfoRpc);
AURA_HEXAGON_RPC_FUNC_REGISTER(AURA_RUNTIME_PACKAGE_NAME, AURA_RUNTIME_QUERY_RT_INFO_OP_NAME, QueryRTInfoRpc);

} // namespace aura