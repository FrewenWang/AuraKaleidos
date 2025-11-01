#include "aura/tools/unit_test.h"
#include "aura/runtime/hexagon.h"

#include "hexagon_test.hpp"

using namespace aura;

static std::vector<std::string> ParseParamString(const std::string &param_str)
{
    std::vector<std::string> params;
    const MI_CHAR *cmd_str = param_str.data();

    size_t last_pos = 0;
    for (size_t pos = 0; pos < param_str.size(); pos++)
    {
        if ('/' == cmd_str[pos])
        {
            if ((pos != last_pos) && (pos != last_pos + 1))
            {
                params.emplace_back(cmd_str + last_pos, pos - last_pos);
                last_pos = pos + 1;
            }
        }
    }

    return params;
}

static Status ListTestCases(Context *ctx, HexagonRpcParam &rpc_param)
{
    AURA_LOGI(ctx, AURA_TAG, "RPC called: [ListTestCases]");

    std::vector<std::string> case_names;

    ListTestCasesInParam param(ctx, rpc_param);

    std::string cmd_str;
    param.Get(cmd_str);

    std::vector<std::string> keywords = ParseParamString(cmd_str);

    for (const auto &kw : keywords)
    {
        AURA_LOGI(ctx, AURA_TAG, "keywords: [%s]", kw.c_str());
    }

    if (keywords.empty())
    {
        case_names = UnitTest::GetInstance()->GetTestCases();
    }
    else
    {
        if ("all" == keywords[0])
        {
            case_names = UnitTest::GetInstance()->GetTestCases();
        }
        else
        {
            case_names = UnitTest::GetInstance()->GetTestCases(keywords);
        }
    }

    {
        std::stringstream str_stream;
        for (const auto &name : case_names)
        {
            str_stream << name << std::endl;
            AURA_LOGI(ctx, AURA_TAG, "Case: %s", name.c_str());
        }

        param.Set(str_stream.str());
    }

    return Status::OK;
}

static Status RunTestCases(Context *ctx, HexagonRpcParam &rpc_param)
{
    AURA_LOGI(ctx, AURA_TAG, "RPC called: [RunTestCases]");

    // Get case_names need to run
    RunTestCasesInParam param(ctx, rpc_param);
    std::string case_str;
    param.Get(case_str);

    std::vector<std::string> case_names = ParseParamString(case_str);

    for (const auto &name : case_names)
    {
        AURA_LOGI(ctx, AURA_TAG, "Case need to run: [%s]", name.c_str());
    }

    if (UnitTest::GetInstance()->Initialize(ctx, "./data/", "./", "txt", "hexagon_auto_test") != Status::OK)
    {
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    if (UnitTest::GetInstance()->Run(case_names) != Status::OK)
    {
        goto EXIT;
    }

    if (UnitTest::GetInstance()->Record() != Status::OK)
    {
        goto EXIT;
    }

    {
        std::string report_str = UnitTest::GetInstance()->GetReport()->GetBriefString();
        param.Set(report_str);
    }

    if (UnitTest::GetInstance()->Report() != Status::OK)
    {
        goto EXIT;
    }

    ret = Status::OK;

EXIT:
    UnitTest::GetInstance()->DeInitialize();

    return ret;
}

AURA_HEXAGON_RPC_FUNC_REGISTER(AURA_TEST_PACKAGE_NAME, AURA_TEST_LIST_TEST_CASES_OP_NAME, ListTestCases);
AURA_HEXAGON_RPC_FUNC_REGISTER(AURA_TEST_PACKAGE_NAME, AURA_TEST_RUN_TEST_CASES_OP_NAME,  RunTestCases);