#include "device/include/xtensa_rpc_func_test.hpp"

namespace aura
{
namespace xtensa
{

template<typename Tp>
void CheckResult(const Tp &val1, const Tp &val2)
{
    if (val1 != val2)
    {
        AURA_XTENSA_LOG("Values are different!\n");
    }
}

template<MI_S32 StrMaxSize = AURA_STRING_DEFAULT_MAX_SIZE>
void CheckResult(const string_<StrMaxSize> &str1, const string_<StrMaxSize> &str2)
{
    if (Strcmp(str1.c_str(), str2.c_str()) != 0)
    {
         AURA_XTENSA_LOG("Values are different str1:%s str2:%s!\n", str1.c_str(), str2.c_str());
    }
}

Status RpcParamTestRpc(TileManager tm, XtensaRpcParam &rpc_param)
{
    AURA_UNUSED(tm);

    string str;
    map<MI_S32> map_in;

    RpcParamInParam in_param(rpc_param);
    Status ret = in_param.Get(str, map_in);
    if (ret != Status::OK)
    {
        AURA_XTENSA_LOG("Get failed!\n");
        return Status::ERROR;
    }

    string ref_str = "rpc pram test";
    CheckResult(str, ref_str);

    map<MI_S32> ref_map_in = {{"aaa", 1}, {"bbb", 2}, {"ccc", 3}};

    for (auto it = map_in.begin(); it != map_in.end(); it++)
    {
        auto ref_it = ref_map_in.find(it->first);
        if (ref_it == ref_map_in.end())
        {
            AURA_XTENSA_LOG("Can not find the key: %s\n", it->first.c_str());
            continue;
        }

        CheckResult<MI_S32>(it->second, ref_it->second);
    }

    MI_S32 value = 999;
    vector<MI_S32> vec = {1, 2, 3};
    map<MI_S32> map_out = {{"ddd", 4}, {"eee", 5}, {"fff", 6}};

    RpcParamOutParam out_param(rpc_param);
    ret = out_param.Set(value, vec, map_out, MI_FALSE);
    if (ret != Status::OK)
    {
        AURA_XTENSA_LOG("Set failed!\n");
        return Status::ERROR;
    }
    return Status::OK;
}

AURA_XTENSA_RPC_FUNC_REGISTER("aura.runtime.xtensa.rpc_param_test", RpcParamTestRpc);

} // xtensa
} // aura