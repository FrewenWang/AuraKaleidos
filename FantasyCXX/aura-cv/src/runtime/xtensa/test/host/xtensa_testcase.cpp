#include "aura/runtime/xtensa.h"
#include "aura/tools/unit_test.h"

using namespace aura;

#define AURA_RUNTIME_PACKAGE_NAME                 "aura.runtime.xtensa"
#define AURA_RUNTIME_RPC_PARAM_OP_NAME            "rpc_param_test"

using RpcParamInParam    = XtensaRpcParamType<std::string, std::unordered_map<std::string, MI_S32>>;
using RpcParamOutParam   = XtensaRpcParamType<MI_S32, std::vector<MI_S32>, std::unordered_map<std::string, MI_S32>>;

template<typename Tp>
void CheckResult(const Tp &val1, const Tp &val2)
{
    if (val1 != val2)
    {
        std::cout << "CheckResult failed!  Values are different: " << val1 << " != " << val2 << std::endl;
    }
}

template<>
void CheckResult(const std::string &str1, const std::string &str2)
{
    if (strcmp(str1.c_str(), str2.c_str()) != 0)
    {
         std::cout << "CheckResult failed! Values are different: " << str1 << " != " << str2 << std::endl;
    }
}

NEW_TESTCASE(runtime_xtensa_rpc_param_test)
{
    Context *ctx = UnitTest::GetInstance()->GetContext();
    Status ret = Status::OK;

    XtensaEngine *xtensa_engine = ctx->GetXtensaEngine();
    XtensaRpcParam   rpc_param(ctx);
    RpcParamInParam  in_param(ctx, rpc_param);
    RpcParamOutParam out_param(ctx, rpc_param);

    std::string str("rpc pram test");
    std::unordered_map<std::string, MI_S32> map = {{"aaa", 1}, {"bbb", 2}, {"ccc", 3}};

    ret = in_param.Set(str, map);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "Set failed");
        return;
    }

    ret = xtensa_engine->Run(AURA_RUNTIME_PACKAGE_NAME, AURA_RUNTIME_RPC_PARAM_OP_NAME,
                             rpc_param);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "xtensa_engine Run failed");
        return;
    }

    MI_S32 value = 0;
    std::vector<MI_S32> vec;
    std::unordered_map<std::string, MI_S32> map_out;
    ret = out_param.Get(value, vec, map_out);
    if (ret != Status::OK)
    {
        AURA_ADD_ERROR_STRING(ctx, "Set failed");
        return;
    }

    MI_S32 ref_value = 999;
    CheckResult(ref_value, value);

    std::vector<MI_S32> ref_vec = {1, 2, 3};
    for (size_t i = 0; i < vec.size(); i++)
    {
        CheckResult(ref_vec[i], vec[i]);
    }

    std::map<std::string, MI_S32> ref_map_out = {{"ddd", 4}, {"eee", 5}, {"fff", 6}};
    for (auto it = map_out.begin(); it != map_out.end(); it++)
    {
        auto ref_it = ref_map_out.find(it->first);
        if (ref_it == ref_map_out.end())
        {
            std::cout << "Can not find the key: " << ref_it->first.c_str() << std::endl;
            continue;
        }

        CheckResult(it->second, ref_it->second);
    }

    AddTestResult(AURA_GET_TEST_STATUS(ret));
}
