#include "unit_test_config.hpp"
#if defined(AURA_ENABLE_HEXAGON)
#  include "hexagon/hexagon_test.hpp"
#endif // AURA_ENABLE_HEXAGON

#include "aura/runtime.h"
#include "aura/tools.h"

#include <string>
#include <iomanip>
#include <iostream>
#include <map>
#include <unordered_set>

static std::string g_logo_str = R"(
  __  __            _____  ______   _    _         _  _  _______          _
 |  \/  |    /\    / ____||  ____| | |  | |       (_)| ||__   __|        | |
 | \  / |   /  \  | |  __ | |__    | |  | | _ __   _ | |_  | |  ___  ___ | |_
 | |\/| |  / /\ \ | | |_ ||  __|   | |  | || '_ \ | || __| | | / _ \/ __|| __|
 | |  | | / ____ \| |__| || |____  | |__| || | | || || |_  | ||  __/\__ \| |_
 |_|  |_|/_/    \_\\_____||______|  \____/ |_| |_||_| \__| |_| \___||___/ \__|
)";

static std::string help_info = R"(
Usage:
    Usage: aura_test_main [options] [args]

Options:
    -h --help                                      Show help info for aura_test_main.
    -l --list                                      List all testcase names.
    -t --target     [host/hexagon]                 Command is used for special target. (default is host)
    -r --run        [case1,case2,...]              Run with one or multiple certain testcases. (all for all testcases)
    -f --filter     [keyword1,keyword2,...]        Run testcase with keywords. (testcase name must contains all keyword)
    -b --blacklist  [keyword1,keyword2,...]        Run testcase with blacklist keywords. (testcase name must not contains any keyword)
    -c --config     [name.json]                    Get default config info, if [-d/--dump] is provided then dump to file.
    -s --stress     [count]                        Run in stress mode, this option must use as the last args for -r/--run.
    -d --dump       [filename]                     Dump default config to file. (To get template config file)
    -o --output     [output_path]                  Set output path for log and report. (default is ./)

Example usage:

    Show help info:                 aura_test_main -h/--help

    List test_cases:
        List all testcases:         aura_test_main -l/--list [-t <hexagon/host>]

    Run test cases:
        Run single testcase:        aura_test_main -r/--run case_name  [-t <hexagon/host>] -c config.json
        Run multi testcases:        aura_test_main -r/--run case_name  [-t <hexagon/host>] -c config.json
        Run all testcases:          aura_test_main -r/--run case_name  [-t <hexagon/host>] -c config.json

        Run with keyword filter:    aura_test_main -f/--filter    [-t <hexagon/host>] keyword1,keyword2,...
        Run with blacklist filter:  aura_test_main -b/--blacklist [-t <hexagon/host>] keyword1,keyword2,...
        Run in stress test mode:    aura_test_main -r/--run       [...] -s/--stress [count]

    Get default config:
        Dump default config:        aura_test_main -d/--dump config.json

    Set output path:
        Set output path:            aura_test_main -r/--run [...] -o/--output output_path
)";

using namespace aura;

static const DT_S32 g_split_width = 86;
static const DT_S32 g_rpc_buffer_size = (1 << (3 + 10)); // 8KB

static DT_VOID PrintHeading(const std::string &heading)
{
  std::cout << std::endl << std::string(g_split_width, '=') << std::endl;
  std::cout << heading << std::endl;
  std::cout << std::string(g_split_width, '=') << std::endl;
}

static DT_VOID PrintSplitLine()
{
  std::cout << std::string(g_split_width, '=') << std::endl;
}

static DT_S32 PrintHelpInfo()
{
    PrintHeading("Aura UnitTest Help Info");
    std::cout << help_info <<std::endl;
    PrintSplitLine();
    return 0;
}

static std::vector<std::pair<std::string, std::string>> g_option_pair =
{
    {"-t", "--target"},
    {"-r", "--run"},
    {"-f", "--filter"},
    {"-b", "--blacklist"},
    {"-c", "--config"},
    {"-s", "--stress"},
    {"-d", "--dump"},
    {"-o", "--output"},
};

static std::string IsOptionInList(const std::string &option)
{
    for (const auto &pair : g_option_pair)
    {
        if (pair.first == option || pair.second == option)
        {
            return pair.second;
        }
    }

    return std::string();
}

// split string by delimitor
static std::vector<std::string> SplitString(const std::string &str, const std::string &delim)
{
    std::vector<std::string> result;
    size_t last = 0;
    size_t index = str.find_first_of(delim, last);

    while (index != std::string::npos)
    {
        result.emplace_back(str.substr(last, index - last));
        last = index + 1;
        index = str.find_first_of(delim, last);
    }

    if (last < str.size())
    {
        result.emplace_back(str.substr(last));
    }

    return result;
}

static std::vector<std::string> ListHostCaseNames()
{
    return UnitTest::GetInstance()->GetTestCases();
}

static std::vector<std::string> ListHexagonCaseNames(Context *ctx)
{
    std::vector<std::string> names;
#if defined(AURA_ENABLE_HEXAGON)
    HexagonEngine *hexagon_engine = ctx->GetHexagonEngine();

    if (DT_NULL == hexagon_engine)
    {
        AURA_LOGE(ctx, AURA_TAG, "Get hexagon_engine failed.");
        return names;
    }

    HexagonRpcParam rpc_param(ctx, g_rpc_buffer_size);
    ListTestCasesInParam list_cases_param(ctx, rpc_param);

    list_cases_param.Set("all/");

    if (hexagon_engine->Run(AURA_TEST_PACKAGE_NAME, AURA_TEST_LIST_TEST_CASES_OP_NAME, rpc_param) != Status::OK)
    {
        AURA_LOGI(ctx, AURA_TAG, "Hexagon engine call: ListTestCases faild.\n");
        return names;
    }

    // Get rpc return case names
    {
        std::stringstream str_stream;
        std::string res_str;
        list_cases_param.Get(res_str);

        str_stream << res_str;
        while (!str_stream.eof())
        {
            std::string case_name;
            str_stream >> case_name;
            if (!case_name.empty())
            {
                names.emplace_back(case_name);
            }
        }
    }
#else
     AURA_UNUSED(ctx);
     AURA_UNUSED(g_rpc_buffer_size);
     AURA_LOGE(ctx, AURA_TAG, "ListHexagonCaseNames for target[hexagon] failed, AURA_ENABLE_HEXAGON is OFF, please rebuild with AURA_ENABLE_HEXAGON=ON\n");
#endif // AURA_ENABLE_HEXAGON
     return names;
}

static std::vector<std::string> GetAllTestCases(Context *ctx, const std::string &target)
{
    std::vector<std::string> all_cases;
    if ("host" == target)
    {
        all_cases = ListHostCaseNames();
    }
    else if ("hexagon" == target)
    {
        all_cases = ListHexagonCaseNames(ctx);
    }
    else
    {
        AURA_LOGE(ctx, AURA_TAG, "GetAllTestCases with unsupported target type.\n");
    }

    return all_cases;
}

static DT_VOID ListTestCases(Context *ctx, const std::string &target)
{
    std::string info = "List all testcases for target: " + target;
    PrintHeading(info);

    std::vector<std::string> all_cases = GetAllTestCases(ctx, target);
    for (const auto &name : all_cases)
    {
        std::cout << name << std::endl;
    }

    PrintSplitLine();
    return;
}

static Status RunHostTestCases(Context *ctx, const UnitTestConfig &cfg, std::vector<std::string> &case_names)
{
    PrintHeading("Host cases need to run:");
    for (const auto &name : case_names)
    {
        std::cout << name << std::endl;
    }
    PrintSplitLine();

    cfg.PrintInfo();

    if (UnitTest::GetInstance()->Initialize(ctx, cfg.m_data_path, cfg.m_dump_path, cfg.m_report_type, cfg.m_report_name,
                                            cfg.m_stress_count, cfg.m_enable_mem_profiling) != Status::OK)
    {
        AURA_LOGE(ctx, AURA_TAG, "UnitTest Init failed.\n");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;

    if (UnitTest::GetInstance()->Run(case_names) != Status::OK)
    {
        AURA_LOGE(ctx, AURA_TAG, "UnitTest Run failed.\n");
        goto EXIT;
    }

    if (UnitTest::GetInstance()->Record() != Status::OK)
    {
        AURA_LOGE(ctx, AURA_TAG, "UnitTest Record failed.\n");
        goto EXIT;
    }

    if (UnitTest::GetInstance()->Report() != Status::OK)
    {
        AURA_LOGE(ctx, AURA_TAG, "UnitTest Report failed.\n");
        goto EXIT;
    }

    ret = Status::OK;

EXIT:
    UnitTest::GetInstance()->DeInitialize();
    return ret;
}

static Status RunHexagonTestCases(Context *ctx, UnitTestConfig &cfg, std::vector<std::string> &case_names)
{
#if defined(AURA_ENABLE_HEXAGON)
    AURA_UNUSED(cfg);

    PrintHeading("Hexagon cases need to run:");
    for (const auto &name : case_names)
    {
        std::cout << name << std::endl;
    }
    PrintSplitLine();

    if (case_names.empty())
    {
        AURA_LOGE(ctx, AURA_TAG, "RunHexagonTestCases with empty case_names.\n");
        return Status::ERROR;
    }

    HexagonEngine *hexagon_engine = ctx->GetHexagonEngine();

    if (DT_NULL == hexagon_engine)
    {
        AURA_LOGE(ctx, AURA_TAG, "Get hexagon_engine failed.");
        return Status::ERROR;
    }

    HexagonRpcParam rpc_param(ctx, g_rpc_buffer_size);
    RunTestCasesInParam param(ctx, rpc_param);

    std::string str_buffer;
    for (const auto &str : case_names)
    {
        str_buffer += str + std::string("/");
    }

    param.Set(str_buffer);

    if (hexagon_engine->Run(AURA_TEST_PACKAGE_NAME, AURA_TEST_RUN_TEST_CASES_OP_NAME, rpc_param) != Status::OK)
    {
        AURA_LOGI(ctx, AURA_TAG, "Hexagon engine call: RunTestCases faild.\n");
        return Status::ERROR;
    }

    param.Get(str_buffer);

    std::cout << str_buffer << std::endl;

    return Status::OK;
#else
    AURA_UNUSED(ctx);
    AURA_UNUSED(cfg);
    AURA_UNUSED(case_names);
    AURA_UNUSED(g_rpc_buffer_size);
    AURA_LOGE(ctx, AURA_TAG, "RunHexagonTestCases for target[hexagon] failed, AURA_ENABLE_HEXAGON is OFF, please rebuild with AURA_ENABLE_HEXAGON=ON\n");
    return Status::OK;
#endif // AURA_ENABLE_HEXAGON
}

static Status RunTestCases(Context *ctx, UnitTestConfig &cfg, const std::string &target, std::map<std::string, std::string> &option_map)
{
    //----------------------------
    //     Get Filtered Cases
    // ---------------------------
    std::vector<std::string> test_cases;
    if (option_map.count("run") > 0)
    {
        if (option_map["run"] == "all")
        {
            test_cases = GetAllTestCases(ctx, target);
        }
        else
        {
            test_cases = SplitString(option_map["run"], ",");
        }
    }
    else // set all testcases as default
    {
        test_cases = GetAllTestCases(ctx, target);
    }

    std::vector<std::string> filter_keywords;
    if (option_map.count("filter") > 0)
    {
        filter_keywords = SplitString(option_map["filter"], ",");
    }

    std::vector<std::string> blacklist_keywords;
    if (option_map.count("blacklist") > 0)
    {
        blacklist_keywords = SplitString(option_map["blacklist"], ",");
    }

    // lambda function to filter test cases
    auto filter_func = [&filter_keywords, &blacklist_keywords](const std::string &case_name) -> DT_BOOL
    {
        for (const auto &keyword : blacklist_keywords)
        {
            if (StringContains(case_name, keyword))
            {
                return DT_FALSE;
            }
        }

        for (const auto &keyword : filter_keywords)
        {
            if (StringContains(case_name, keyword))
            {
                return DT_TRUE;
            }
        }

        return filter_keywords.empty();
    };

    std::vector<std::string> test_cases_filtered;
    for (const auto &name : test_cases)
    {
        if (filter_func(name))
        {
            test_cases_filtered.emplace_back(name);
        }
    }

    //----------------------------
    //     Update Config
    // ---------------------------
    // terminal setting is priorior to config file
    if (option_map.count("stress") > 0)
    {
        cfg.m_stress_count = std::stoi(option_map["stress"]);
    }
    // if output path is set, then update report path
    if (option_map.count("output") > 0)
    {
        cfg.m_report_name = option_map["output"] + "/" + cfg.m_report_name;
    }

    //----------------------------
    //     Run Test Cases
    // ---------------------------
    if ("host" == target)
    {
        if (RunHostTestCases(ctx, cfg, test_cases_filtered) != Status::OK)
        {
            AURA_LOGE(ctx, AURA_TAG, "RunHostTestCases failed.\n");
            return Status::ERROR;
        }

        return Status::OK;
    }
    else if ("hexagon" == target)
    {
        if (RunHexagonTestCases(ctx, cfg, test_cases_filtered) != Status::OK)
        {
            AURA_LOGE(ctx, AURA_TAG, "RunHexagonTestCases failed.\n");
            return Status::ERROR;
        }

        return Status::OK;
    }
    else
    {
        AURA_LOGE(ctx, AURA_TAG, "RunTestCases with unsupported target type.\n");
        return Status::ERROR;
    }
}

static Status ParseArgs(DT_S32 argc, DT_CHAR *argv[], std::map<std::string, std::string> &option_map,
                        DT_BOOL &is_help_set, DT_BOOL &is_list_set)
{
    if (1 == argc)
    {
        return Status::ERROR;
    }

    for (DT_S32 i = 1; i < argc; ++i)
    {
        if (std::string("-h") == argv[i] || std::string("--help") == argv[i])
        {
            is_help_set = DT_TRUE;
            return Status::OK;
        }
        else if (std::string("-l") == argv[i] || std::string("--list") == argv[i])
        {
            is_list_set = DT_TRUE;
        }
        else if (i + 1 < argc)
        {
            std::string option = IsOptionInList(argv[i]);
            if (!option.empty())
            {
                // remove first two chars in option (e.g. --config -> config)
                option_map[option.substr(2)] = argv[i + 1];
                ++i;
            }
            else
            {
                return Status::ERROR;
            }
        }
        else
        {
            return Status::ERROR;
        }
    }

    return Status::OK;
}

static std::shared_ptr<Context> InitTestMainContext(const UnitTestConfig &cfg, const std::map<std::string, std::string> &option_map)
{
    std::string target = "host";
    if (option_map.count("target") > 0)
    {
        target = option_map.at("target");
    }

    std::string hexagon_lib_prefix = ("hexagon" == target) ? "aura_test_main" : "aura_hexagon";
    std::string pil_name = cfg.m_pil_path + "aura_xtensa_pil.so";

    // Create context for unit_test
    Config config;
    config.SetLog(cfg.m_log_output, cfg.m_log_level, cfg.m_log_file_name);
    config.SetWorkerPool("UnitTest", cfg.m_compute_affinity, cfg.m_async_affinity);
    config.SetCLConf(DT_TRUE, cfg.m_cache_bin_path, cfg.m_cache_bin_prefix);
    config.SetHexagonConf(DT_TRUE, DT_TRUE, hexagon_lib_prefix);
    config.SetNNConf(DT_TRUE);
    config.SetXtensaConf(DT_TRUE, pil_name);

    std::shared_ptr<Context> ctx = std::make_shared<Context>(config);
    if (DT_NULL == ctx)
    {
        std::cout << "Context create failed." << std::endl;
        return DT_NULL;
    }

    if (ctx->Initialize() != Status::OK)
    {
        std::cout << "Context Initialize failed." << std::endl;
        return DT_NULL;
    }

    return ctx;
}

DT_S32 main(DT_S32 argc, DT_CHAR *argv[])
{
    std::map<std::string, std::string> option_map;
    DT_BOOL is_help_set = DT_FALSE;
    DT_BOOL is_list_set = DT_FALSE;

    //----------------------
    //      wrong usage
    //----------------------
    if (ParseArgs(argc, argv, option_map, is_help_set, is_list_set) != Status::OK)
    {
        return PrintHelpInfo();
    }

    //----------------------
    //       help mode
    //----------------------
    if (is_help_set)
    {
        return PrintHelpInfo();
    }

    // bind big cpu
    if (SetCpuAffinity(CpuAffinity::BIG) != Status::OK)
    {
        std::cout << "SetCpuAffinity bind big cpu failed." << std::endl;
    }

    // init options
    UnitTestConfig unit_test_cfg;
    if (option_map.count("config") > 0)
    {
        if (unit_test_cfg.Load(option_map["config"]) != Status::OK)
        {
            std::cout << "Load config file: " << option_map["config"] << "failed." << std::endl;
            return -1;
        }
    }

    // init context
    std::shared_ptr<Context> ctx = InitTestMainContext(unit_test_cfg, option_map);
    if (DT_NULL == ctx)
    {
        std::cout << "InitTestMainContext failed." << std::endl;
        return -1;
    }

    //----------------------
    //       list mode
    //----------------------
    std::string target = option_map.count("target") > 0 ? option_map["target"] : "host";
    if (is_list_set)
    {
        ListTestCases(ctx.get(), target);
    }

    // --------------------
    //       run mode
    // --------------------
    if (option_map.count("run") > 0 || option_map.count("filter") > 0 || option_map.count("blacklist") > 0)
    {
        if (RunTestCases(ctx.get(), unit_test_cfg, target, option_map) != Status::OK)
        {
            std::cout << "RunTestCases failed." << std::endl;
            return -1;
        }
    }

    // --------------------
    //       dump mode
    // --------------------
    if (option_map.count("dump") > 0)
    {
        std::string output_path = option_map.count("output") > 0 ? option_map["output"] : "./";
        std::string output_file = output_path + option_map["dump"];

        unit_test_cfg.Save(output_file);
    }

    return 0;
}
