#include "aura/runtime/hexagon/device/hexagon_runtime.hpp"

#include <unordered_map>

#include "AEEStdDef.h"
#include "HAP_power.h"

#define AURA_CLINET_ID  (0x6d616765)        // 'm'(0x6d) 'a'(0x61) 'g'(0x67) 'e'(0x65)

namespace aura
{

std::unordered_map<std::string, RpcFunc>& GetRpcFuncMap()
{
    static std::unordered_map<std::string, RpcFunc> rpc_func_map;
    return rpc_func_map;
}

/**
 * 这个地方就是进行远程的方法调用的注册
 * 调用方来自：AURA_HEXAGON_RPC_FUNC_REGISTER
 * @param name
 * @param func
 * @return
 */
RpcFuncRegister::RpcFuncRegister(const std::string &name, RpcFunc func)
{
    auto& rpc_func_map = GetRpcFuncMap();

    if (rpc_func_map.find(name) == rpc_func_map.end())
    {
        rpc_func_map[name] = func;
    }
}

Status SetPower(Context *ctx, HexagonPowerLevel target_level, DT_BOOL enable_dcvs, DT_U32 client_id)
{
    DT_U32 power_level = HAP_DCVS_VCORNER_DISABLE;
    DT_U32 min_corner = HAP_DCVS_VCORNER_DISABLE;

    DT_U64 power_client_id = (0 == client_id) ? AURA_CLINET_ID : client_id;

    switch (target_level)
    {
        case HexagonPowerLevel::DEFAULT:
        {
            return Status::OK;
        }

        case HexagonPowerLevel::STANDBY:
        {
            power_level = HAP_DCVS_VCORNER_SVS2;
            break;
        }

        case HexagonPowerLevel::LOW:
        {
            power_level = HAP_DCVS_VCORNER_SVSPLUS;
            min_corner = ((DT_FALSE == enable_dcvs) ? power_level : HAP_DCVS_VCORNER_SVS2);
            break;
        }

        case HexagonPowerLevel::NORMAL:
        {
            power_level = HAP_DCVS_VCORNER_NOMPLUS;
            min_corner = ((DT_FALSE == enable_dcvs) ? power_level : HAP_DCVS_VCORNER_SVS2);
            break;
        }

        case HexagonPowerLevel::TURBO:
        {
            if (__HEXAGON_ARCH__ == 66)
            {
                power_level = HAP_DCVS_VCORNER_TURBO_PLUS;
            }
            else
            {
                power_level = HAP_DCVS_VCORNER_TURBO;
            }

            min_corner = ((DT_FALSE == enable_dcvs) ? power_level : HAP_DCVS_VCORNER_SVS2);
            break;
        }

        default:
        {
            AURA_ADD_ERROR_STRING(ctx, "power_level is invalid");
            return Status::ERROR;
        }
    }

    if (HexagonPowerLevel::STANDBY == target_level)
    {
        HAP_power_request_t request;
        memset(&request, 0, sizeof(HAP_power_request_t)); //Important to clear the structure if only selected fields are updated.
        request.type = HAP_power_set_DCVS_v2;
        request.dcvs_v2.dcvs_option = HAP_DCVS_V2_POWER_SAVER_MODE;
        request.dcvs_v2.dcvs_enable = FALSE;

        DT_S32 ret = HAP_power_set((void *)power_client_id, &request);
        if (ret != AEE_SUCCESS)
        {
            AURA_ADD_ERROR_STRING(ctx, ("HAP_power_set failed, error(" + std::to_string(ret) + ")").c_str());
            return Status::ERROR;
        }
    }
    else
    {
        HAP_power_request_t request;
        qurt_arch_version_t qurt_version;
        DT_S32 ret = qurt_sysenv_get_arch_version(&qurt_version);
        if (ret != QURT_EOK)
        {
            AURA_ADD_ERROR_STRING(ctx, ("qurt_sysenv_get_arch_version failed, error(" + std::to_string(ret) + ")").c_str());
            return Status::ERROR;
        }

        DT_S32 cdsp_version = qurt_version.arch_version & 0xff;
        if (cdsp_version < 0x69) // sm8450:0x69  sm8350:0x68 sm8250:0x66
        {
            memset(&request, 0, sizeof(HAP_power_request_t)); //Remove all votes for NULL context 
            request.type = HAP_power_set_DCVS_v2;
            request.dcvs_v2.dcvs_enable = TRUE;
            request.dcvs_v2.dcvs_option = HAP_DCVS_V2_POWER_SAVER_MODE;
            request.dcvs_v2.latency = 100;
            DT_S32 ret = HAP_power_set(NULL, &request); // here must use null  For SM8450 and later, Passing to NULL context to HAP_power_set() API is no longer allowed
            if (ret != AEE_SUCCESS)
            {
                AURA_ADD_ERROR_STRING(ctx, ("HAP_power_set failed, error(" + std::to_string(ret) + ")").c_str());
                return Status::ERROR;
            }
        }

        // reference code from examples/common/benchmark_v65/src_dsp/benchmark_imp.c/benchmark_setClocks
        memset(&request, 0, sizeof(HAP_power_request_t));

        // Important to clear the structure if only selected fields are updated
        request.type = HAP_power_set_apptype;
        request.apptype = HAP_POWER_COMPUTE_CLIENT_CLASS;

        ret = HAP_power_set((void *)power_client_id, &request);
        if (ret != AEE_SUCCESS)
        {
            AURA_ADD_ERROR_STRING(ctx, ("HAP_power_set failed, error(" + std::to_string(ret) + ")").c_str());
            return Status::ERROR;
        }

        // Configure clocks & DCVS mode
        memset(&request, 0, sizeof(HAP_power_request_t));
        request.type = HAP_power_set_DCVS_v2;

        request.dcvs_v2.dcvs_enable = TRUE;
        request.dcvs_v2.dcvs_params.target_corner = static_cast<HAP_dcvs_voltage_corner_t>(power_level);
        request.dcvs_v2.dcvs_params.min_corner = static_cast<HAP_dcvs_voltage_corner_t>(min_corner);                    // min corner = target corner
        request.dcvs_v2.dcvs_params.max_corner = static_cast<HAP_dcvs_voltage_corner_t>(power_level);                   // max corner = target corner

        request.dcvs_v2.dcvs_option = HAP_DCVS_V2_PERFORMANCE_MODE;
        request.dcvs_v2.set_dcvs_params = TRUE;
        request.dcvs_v2.set_latency = TRUE;
        request.dcvs_v2.latency = 100;
        
        ret = HAP_power_set((void *)power_client_id, &request);
        if (ret != AEE_SUCCESS)
        {
            AURA_ADD_ERROR_STRING(ctx, ("HAP_power_set failed, error(" + std::to_string(ret) + ")").c_str());
            return Status::ERROR;
        }

        // vote for HVX power
        memset(&request, 0, sizeof(HAP_power_request_t));
        request.type = HAP_power_set_HVX;
        request.hvx.power_up = TRUE;

        ret = HAP_power_set((void *)power_client_id, &request);
        if (ret != AEE_SUCCESS)
        {
            AURA_ADD_ERROR_STRING(ctx, ("HAP_power_set failed, error(" + std::to_string(ret) + ")").c_str());
            return Status::ERROR;
        }
    }

    return Status::OK;
}

} // namepsace aura