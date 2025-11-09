#include "unit_test_config.hpp"

#include "aura/runtime.h"
#include "aura/tools.h"
#include "aura/runtime/xtensa/device/xtensa_runtime.hpp"

#include <string>
#include <iomanip>
#include <iostream>
#include <unordered_set>

static std::string g_logo_str = R"(
  _     _  _                             _    _         _  _  _______          _
  \ \  / /| |                           | |  | |       (_)| ||__   __|        | |
   \ \/ / | |_   ___  _ __   ___   __   | |  | | _ __   _ | |_  | |  ___  ___ | |_
    |  |  | __| / _ \| '_ \ / __| / _ \ | |  | || '_ \ | || __| | | / _ \/ __|| __|
   / /\ \ | |_ |  __/| | | |\__ \| |_| || |__| || | | || || |_  | ||  __/\__ \| |_
  /_/  \_\ \__| \___||_| |_||___/ \__|_| \____/ |_| |_||_| \__| |_| \___||___/ \__|
)";

static std::string help_info = R"(
Usage:
    Usage: aura_test_main [options] [args]

Options:
    -h --help                                              Show help info for aura_test_main.
    -l --list   [keyword1 keyword2 ...]                    List all testcase names, keyword is optional for filter names.
    -r --run    [config.json] case1 case2                  Run testcase with names, config.json is optional.
    -r --run    [-f/--filter] [config.json] [kw1 kw2 ...]  Run testcase with keywords contained mode with & logic.
    -c --config [-d/--dump] [name.json]                    Get default config info, if [-d/--dump] is provided then dump to file.
    -s --stress [count]                                    Run in stress mode, this option must use as the last args for -r/--run.
    -t --target [profile]                                  Command is used for special target.

Example usage:

    Show help info:                 aura_test_main -h/--help

    List test_cases:
        List all testcases:         aura_test_main -l/--list [-t <profile>]
        List with keyword filter:   aura_test_main -l/--list [-t <profile>] keyword1 keyword2 ...

    Run test cases:
        Run single testcase:        aura_test_main -r/--run [-t <profile>] case_name
        Run multi testcases:        aura_test_main -r/--run [-t <profile>] case_name1 case_name2 ...
        Run all testcases:          aura_test_main -r/--run [-t <profile>] all
        Run with keyword filter:    aura_test_main -r/--run -f/--filter [-t <profile>] keyword1 keyword2 ...
        Run in stress test mode:    aura_test_main -r/--run [...] -s/--stress [count]

    Run with custom json config: add [xxx.json] behind Run test cases command.

    Get default config:
        Print default config:       aura_test_main -c/--config
        Dump default config:        aura_test_main -c/--config -d/--dump filename(without .json))";

using namespace aura;

application_symbol_tray *g_symbol_tray = DT_NULL;

static const DT_S32 g_split_width = 86;

static std::string ParseStringOptions(std::vector<std::string> &options, const std::string &key_short, const std::string &key_long,
                                      size_t offset, const std::string &default_value)
{
    std::string result;

    if (options.empty() || offset > 1)
    {
        return result;
    }

    for (size_t idx = 0; idx < options.size(); idx++)
    {
        if (StringContains(options[idx], key_short) || (StringContains(options[idx], key_long)))
        {
            if (0 == offset)
            {
                result = options[idx];
                options.erase(options.begin() + idx);
            }
            else
            {
                if (idx + offset < options.size())
                {
                    result = options[idx + offset];
                    options.erase(options.begin() + idx);
                    options.erase(options.begin() + idx);
                }
                else
                {
                    result = default_value;
                    options.erase(options.begin() + idx);
                }
            }

            break;
        }
    }

    return result;
}

static DT_S32 ParseIntOptions(std::vector<std::string> &options, const std::string &key_short, const std::string &key_long,
                              size_t offset, DT_S32 default_value)
{
    DT_S32 result = 0;

    if (options.empty() || offset > 1)
    {
        return result;
    }

    auto is_num_func = [](const char &c) { return std::isdigit(c); };

    for (size_t idx = 0; idx < options.size(); idx++)
    {
        if (StringContains(options[idx], key_short) || (StringContains(options[idx], key_long)))
        {
            if (idx + offset < options.size())
            {
                std::string arg_str = options[idx + offset];

                if (std::all_of(arg_str.begin(), arg_str.end(), is_num_func))
                {
                    result = std::stoi(arg_str);
                    options.erase(options.begin() + idx);
                    options.erase(options.begin() + idx);
                }
                else
                {
                    result = default_value;
                    options.erase(options.begin() + idx);
                }
            }
            else
            {
                result = default_value;
                options.erase(options.begin() + idx);
            }

            break;
        }
    }

    return result;
}

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

static std::vector<std::string> ListCaseNames(const std::vector<std::string> &keywords = std::vector<std::string>())
{
    std::vector<std::string> names;
    if (keywords.empty())
    {
        names = UnitTest::GetInstance()->GetTestCases();
    }
    else
    {
        names = UnitTest::GetInstance()->GetTestCases(keywords);
    }
    return names;
}

static Status ListTestCases(Context *ctx, const std::string &target, std::vector<std::string> &keywords)
{
    if ("profile" == target)
    {
        std::vector<std::string> profile_case_names = ListCaseNames(keywords);
        PrintHeading("Current profile cases are: ");
        for (const auto &name : profile_case_names)
        {
            std::cout << name << std::endl;
        }
    }
    else
    {
        AURA_LOGE(ctx, AURA_TAG, "ListTestCases with unsupported target type.\n");
        return Status::ERROR;
    }
    PrintSplitLine();
    return Status::OK;
}

static Status RunProfileTestCases(Context *ctx, UnitTestConfig &cfg, std::vector<std::string> &case_names)
{
    PrintHeading("Profile cases need to run:");
    for (const auto &name : case_names)
    {
        std::cout << name << std::endl;
    }
    PrintSplitLine();

    cfg.PrintInfo();

    if (UnitTest::GetInstance()->Initialize(ctx, cfg.m_data_path, cfg.m_dump_path, cfg.m_report_type, cfg.m_report_name, cfg.m_stress_count) != Status::OK)
    {
        AURA_LOGE(ctx, AURA_TAG, "UnitTest Init failed.\n");
        return Status::ERROR;
    }

    Status ret = Status::ERROR;
    XtensaEngine *xtensa_engine = ctx->GetXtensaEngine();
    if (DT_NULL == xtensa_engine)
    {
        AURA_LOGE(ctx, AURA_TAG, "xtensa_engine is null.\n");
        return Status::ERROR;
    }

    if (xtensa_engine->RegisterRpcFunc(xtensa::VdspRpcCall) != Status::OK)
    {
        AURA_LOGE(ctx, AURA_TAG, "RegistTray failed.\n");
        goto EXIT;
    }

    g_symbol_tray = &(xtensa_engine->GetSymbolTray());

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

    return Status::OK;
}

static Status RunTestCases(Context *ctx, UnitTestConfig &cfg, const std::string &target, std::vector<std::string> &options)
{
    if (options.empty())
    {
        std::cout << "Please input case names for run." << std::endl;
        ListTestCases(ctx, target, options);
        return Status::ERROR;
    }

    DT_S32 stress_count = ParseIntOptions(options, "-s", "--stress", 1, 10);

    if (stress_count > 0)
    {
        cfg.m_stress_count = stress_count;
    }

    if (options.empty())
    {
        std::cout << "Please input run case informations, -h/--help to see example." << std::endl;
        return Status::ERROR;
    }

    if ("profile" == target)
    {
        std::vector<std::string> profile_case_names;

        if (std::string("-f") == options.front() || std::string("--filter") == options.front())
        {
            options.erase(options.begin());
            profile_case_names = ListCaseNames(options);
        }
        else if (std::string("all") == options.front())
        {
            profile_case_names = ListCaseNames();
        }
        else
        {
            // Run case use test_case name directly
            profile_case_names = options;
        }

        if (RunProfileTestCases(ctx, cfg, profile_case_names) != Status::OK)
        {
            AURA_LOGE(ctx, AURA_TAG, "RunProfileTestCases failed.\n");
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

static Status GetConfigInfo(std::vector<std::string> &options, const UnitTestConfig &cfg)
{
    if (options.empty())
    {
        cfg.PrintInfo();
        return Status::OK;
    }

    if ("-d" == options.front() || "--dump" == options.front())
    {
        options.erase(options.begin());
        Status result;

        if (options.empty())
        {
            // -c -d without filename
            result = cfg.Save("./default.json");
        }
        else
        {
            result = cfg.Save(options.front() + ".json");
        }

        if (result != Status::OK)
        {
            std::cout << "UnitTest dump config failed." << std::endl;
            return result;
        }
    }
    else
    {
        cfg.PrintInfo();
    }

    return Status::OK;
}

DT_S32 main(DT_S32 argc, DT_CHAR *argv[])
{
    if (1 == argc)
    {
        return PrintHelpInfo();
    }

    const std::string command(argv[1]);
    std::vector<std::string> options;

    for (DT_S32 i = 2; i < argc; ++i)
    {
        options.push_back(argv[i]);
    }

    std::unordered_set<std::string> command_set
    {
        "-l", "--list",
        "-r", "--run",
    };

    UnitTestConfig unit_test_cfg;
    std::string cfg_file = ParseStringOptions(options, ".json", ".json", 0, "");

    if (!cfg_file.empty())
    {
        if (unit_test_cfg.Load(cfg_file) != Status::OK)
        {
            std::cout << "Load config file: " << cfg_file << "failed." << std::endl;
            return -1;
        }
    }

    if ("-h" == command || "--help" == command)
    {
        return PrintHelpInfo();
    }
    else if ("-c" == command || "--config" == command)
    {
        return (GetConfigInfo(options, unit_test_cfg) == Status::OK);
    }
    else if (command_set.count(command) > 0)
    {
        std::string target = ParseStringOptions(options, "-t", "--target", 1, "profile");

        if (target.empty())
        {
            target = "profile";
        }

        // Create context for unit_test
        Config config;
        config.SetLog(unit_test_cfg.m_log_output, unit_test_cfg.m_log_level, unit_test_cfg.m_log_file_name);

        std::shared_ptr<Context> ctx = std::make_shared<Context>(config);
        if (DT_NULL == ctx)
        {
            std::cout << "Context create failed." << std::endl;
            return -1;
        }

        if (ctx->Initialize() != Status::OK)
        {
            std::cout << "Context Initialize failed." << std::endl;
            return -1;
        }

        if ("-l" == command || "--list" == command)
        {
            if (ListTestCases(ctx.get(), target, options) != Status::OK)
            {
                AURA_LOGE(ctx.get(), AURA_TAG, "ListTestCases failed.\n");
                return -1;
            }
            else
            {
                return 0;
            }
        }
        else if ("-r" == command || "--run" == command)
        {
            if (RunTestCases(ctx.get(), unit_test_cfg, target, options) != Status::OK)
            {
                AURA_LOGE(ctx.get(), AURA_TAG, "RunTestCases failed.\n");
                return -1;
            }
            else
            {
                return 0;
            }
        }
    }
    else
    {
        return PrintHelpInfo();
    }

    return 0;
}
