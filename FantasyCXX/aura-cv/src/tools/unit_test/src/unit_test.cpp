#include "aura/tools/unit_test/unit_test.hpp"
#include "aura/tools/unit_test/test_utils.hpp"
#include "aura/tools/json.h"

#include <algorithm>
#include <iomanip>
#include <fstream>
#include <unordered_map>

#if defined(AURA_BUILD_WIN)
#  define GREEN   ""
#  define BLUE    ""
#  define RED     ""
#  define AURANTA ""
#  define RESET   ""
#else
#  define GREEN   "\033[1;32m"
#  define BLUE    "\033[1;34m"
#  define RED     "\033[1;31m"
#  define AURANTA "\033[1;35m"
#  define RESET   "\033[0m"
#endif

static const std::string report_title_str = R"(
**************************************************************************************
*      _____                       _    __      __       _                           *
*      |  __ \                     | |   \ \    / /      | |                         *
*      | |__) |___ _ __   ___  _ __| |_   \ \  / /__ _ __| |__   ___  ___  ___       *
*      |  _  // _ \ '_ \ / _ \| '__| __|   \ \/ / _ \ '__| '_ \ / _ \/ __|/ _ \      *
*      | | \ \  __/ |_) | (_) | |  | |_     \  /  __/ |  | |_) | (_) \__ \  __/      *
*      |_|  \_\___| .__/ \___/|_|   \__|     \/ \___|_|  |_.__/ \___/|___/\___|      *
*                  | |                                                               *
*                  |_|                                                               *
**************************************************************************************
)";


namespace aura
{

static std::string GetTestModeString()
{
    std::stringstream sstream;
    sstream << std::setiosflags(std::ios::left);
#if defined(AURA_RELEASE)
    sstream << "BuildType: " << "Release" << std::endl;
#else
    sstream << "BuildType: " << "Debug  " << std::endl;
#endif // AURA_RELEASE

#if defined(__has_feature)
#  if __has_feature(address_sanitizer)
    sstream << "Asan: " << "Enable" << std::endl;
#  else
    sstream << "Asan: " << "Disable" << std::endl;
#  endif // address_sanitizer

#elif defined(__SANITIZE_ADDRESS__)
    sstream << "Asan: " << "Enable" << std::endl;
#else
    sstream << "Asan: " << "Disable" << std::endl;
#endif // __has_feature

    if (UnitTest::GetInstance()->IsStressMode())
    {
        sstream << "Running Mode: " << "StressTest(Count: " << UnitTest::GetInstance()->GetStressCount() << ")" << std::endl;
    }
    else
    {
        sstream << "Running Mode: " << "Normal"<< std::endl;
    }

    return sstream.str();
}

UnitTest* UnitTest::GetInstance()
{
    static UnitTest instance;
    return &instance;
}

Status UnitTest::Initialize(Context *ctx, const std::string &data_path, const std::string &dump_path, const std::string &report_type,
                            const std::string &report_name, DT_S32 stress_count, DT_S32 enable_mem_profiling)
{
    m_ctx = ctx;
    m_data_path = data_path;
    m_dump_path = dump_path;

    if (report_type == "txt" && !report_name.empty())
    {
        m_report.reset(new UnitTestReportText(report_name));
    }
    else if (report_type == "json" && !report_name.empty())
    {
        m_report.reset(new UnitTestReportJson(report_name));
    }
    else
    {
        AURA_LOGE(ctx, AURA_TAG, "Unsupported report type or empty report_name.\n");
        return Status::ERROR;
    }

    m_stress_count         = stress_count;
    m_enable_mem_profiling = enable_mem_profiling;
    m_initialized          = DT_TRUE;

    return Status::OK;
}

Context* UnitTest::GetContext()
{
    return m_ctx;
}

std::shared_ptr<UnitTestReport> UnitTest::GetReport()
{
    return m_report;
}

std::shared_ptr<TestCase> UnitTest::RegisterTestCase(std::shared_ptr<TestCase> test_case, const std::string &module,
                                                     const std::string &interface, const std::string &impl)
{
    if (DT_NULL == test_case)
    {
        AURA_LOGE(m_ctx, AURA_TAG, "RegisterTestCase failed: test_case is nullptr.\n", module.c_str(), interface.c_str(), impl.c_str());
        return DT_NULL;
    }

    if (module.empty() && interface.empty() && impl.empty())
    {
        AURA_LOGE(m_ctx, AURA_TAG, "UnitTest::RegisterTestCase failed: module interface impl all empty.\n");
        return DT_NULL;
    }

    if (this->m_case_map.count(test_case->GetName()) > 0)
    {
        AURA_LOGE(m_ctx, AURA_TAG, "UnitTest::RegisterTestCase failed: duplicate case_name[%s].\n", test_case->GetName().c_str());
        return DT_NULL;
    }
    else
    {
        this->m_case_map.insert(std::make_pair(test_case->GetName(), test_case));
        return test_case;
    }
}

std::vector<std::string> UnitTest::GetTestCases(const std::string &keyword) const
{
    std::vector<std::string> result;
    if (this->m_case_map.empty())
    {
        return result;
    }

    result.reserve(this->m_case_map.size());

    if (keyword.empty())
    {
        for (const auto &it : this->m_case_map)
        {
            result.push_back(it.first);
        }

        std::sort(result.begin(), result.end());

        return result;
    }
    else
    {
        for (const auto &it : this->m_case_map)
        {
            if (StringContains(it.first, keyword))
            {
                result.push_back(it.first);
            }
        }

        std::sort(result.begin(), result.end());

        return result;
    }
}

std::vector<std::string> UnitTest::GetTestCases(const std::vector<std::string> &keywords) const
{
    if (keywords.empty())
    {
        return this->GetTestCases();
    }

    if (1 == keywords.size())
    {
        return this->GetTestCases(keywords.front());
    }

    std::vector<std::string> all_names = this->GetTestCases();

    std::unordered_map<std::string, DT_S32> cases_map;

    for (const auto &it : all_names)
    {
        cases_map[it] = 0;
    }

    for (const auto &str : keywords)
    {
        std::vector<std::string> names = this->GetTestCases(str);

        for (const auto &name : names)
        {
            ++cases_map[name];
        }
    }

    std::vector<std::string> result;
    if (cases_map.size() > 0)
    {
        result.reserve(cases_map.size());
    }

    for (const auto &name : cases_map)
    {
        if (static_cast<DT_S32>(keywords.size()) == name.second)
        {
            result.push_back(name.first);
        }
    }

    std::sort(result.begin(), result.end());

    return result;
}

Status UnitTest::Run(const std::string &case_name)
{
    if (!this->m_initialized || !this->m_ctx || case_name.empty())
    {
        AURA_LOGE(this->m_ctx, AURA_TAG, "UnitTest::Run failed, invalid param.");
        return Status::ERROR;
    }

    std::string seperator(case_name.size() + 16, '*');

    AURA_LOGI(this->m_ctx, AURA_TAG, "%s\n", seperator.c_str());
    AURA_LOGI(this->m_ctx, AURA_TAG, "        %s\n", case_name.c_str());
    AURA_LOGI(this->m_ctx, AURA_TAG, "%s\n", seperator.c_str());

    if (this->m_case_map.count(case_name) > 0)
    {
        auto &test_case = m_case_map.at(case_name);
        test_case->Run();
    }
    else
    {
        AURA_LOGI(this->m_ctx, AURA_TAG, "testcase: [%s] does not exist\n", case_name.c_str());
    }

    return Status::OK;
}

Status UnitTest::Run(const std::vector<std::string> &case_names)
{
    for (const auto &case_name : case_names)
    {
        if (this->Run(case_name) != Status::OK)
        {
            return Status::ERROR;
        }
    }

    return Status::OK;
}

Status UnitTest::Record(const std::string &info)
{
    if (!this->m_initialized || !this->m_ctx || DT_NULL == this->m_report)
    {
        return Status::ERROR;
    }

    for (auto &case_pair : this->m_case_map)
    {
        if (TestStatus::UNTESTED != case_pair.second->GetStatus())
        {
            this->m_report->Record(info, case_pair.second);
            case_pair.second->Clear();
            case_pair.second->SetStatus(TestStatus::UNTESTED);
        }
    }

    return Status::OK;
}

Status UnitTest::PrintReportBrief() const
{
    if (DT_NULL == this->m_report)
    {
        return Status::ERROR;
    }
    else
    {
        this->m_report->PrintBrief();
        return Status::OK;
    }
}

Status UnitTest::PrintReportVerbose() const
{
    if (DT_NULL == this->m_report)
    {
        return Status::ERROR;
    }
    else
    {
        this->m_report->PrintVerbose();
        return Status::OK;
    }
}

Status UnitTest::Report(const std::string &destination)
{
    if (this->PrintReportBrief() != Status::OK)
    {
        return Status::ERROR;
    }

    if (!destination.empty())
    {
        return this->m_report->Report(destination);
    }
    else
    {
        return this->m_report->Report("");
    }
}

DT_VOID UnitTestReport::Record(const std::string &info, std::shared_ptr<TestCase> test_case)
{
    std::string round_info = info.empty() ? "" : info;

    const std::string &case_name      = test_case->GetName();
    const std::string &case_module    = test_case->GetModule();
    const std::string &case_interface = test_case->GetInterface();
    const std::string &case_impl      = test_case->GetImpl();

    if (TestStatus::UNTESTED != test_case->GetStatus())
    {
        ReportResult result;
        result.status    = test_case->GetStatus();
        result.m_result  = test_case->GetResults();
        result.name      = case_name;
        result.module    = case_module;
        result.interface = case_interface;
        result.impl      = case_impl;

        this->m_result[round_info].push_back(result);
    }
}

DT_VOID UnitTestReport::Clear()
{
    this->m_result.clear();
}

std::string UnitTestReport::GetBriefString() const
{
    std::stringstream sstream;
    sstream << std::setiosflags(std::ios::left);

    sstream << "**********************************************************" << std::endl;
    sstream << "*" << std::endl;
    sstream << "*                UnitTest Report Brief" << std::endl;
    sstream << "*" << std::endl;
    sstream << "**********************************************************" << std::endl;
    sstream << GetTestModeString();
    for (const auto &unit_test_result : this->m_result)
    {
        DT_S32 num_tested = 0;
        DT_S32 num_passed = 0;
        DT_S32 num_failed = 0;

        for (const auto &case_result : unit_test_result.second)
        {
            ++num_tested;

            if (TestStatus::PASSED == case_result.status)
            {
                sstream << std::setw(50) << case_result.name << ": " << GREEN << "Passed" << RESET << std::endl;
                ++num_passed;
            }
            else
            {
                sstream << std::setw(50) << case_result.name << ": " << RED << "Failed" << RESET << std::endl;
                ++num_failed;
            }
        }

        sstream << RESET << "----------------------------------------------------------" << std::endl;
        sstream << RESET << "Round Info:   " << unit_test_result.first << RESET << std::endl;
        sstream << RESET << "Tested: " << num_tested << RESET << std::endl;
        sstream << GREEN << "Passed: " << num_passed << RESET << std::endl;
        sstream << RED   << "Failed: " << num_failed << RESET << std::endl;
        sstream << RESET << "----------------------------------------------------------" << std::endl;
    }

    return sstream.str();
}

DT_VOID UnitTestReport::PrintBrief() const
{
    std::cout << GetBriefString() << std::endl;
}

std::string UnitTestReportText::GetReportString(DT_BOOL with_color)
{
    const DT_S32 default_line_width = 86;

    std::stringstream report_stream;

    /// Add report header
    report_stream << report_title_str;
    report_stream << GetTestModeString();

    for (auto &unit_test_result : this->m_result)
    {
        auto cmp_func = [](const ReportResult &res_a, const ReportResult &res_b)
        {
            return res_a.name < res_b.name;
        };

        std::sort(unit_test_result.second.begin(), unit_test_result.second.end(), cmp_func);

        /// Add tag for multi-round test
        if (with_color)
        {
            report_stream << RESET << "Info:   " << unit_test_result.first << RESET << std::endl;
        }
        else
        {
            report_stream << "Info:   " << unit_test_result.first << std::endl;
        }

        /// Used for brief report
        DT_S32 num_tested = 0;
        DT_S32 num_passed = 0;
        DT_S32 num_failed = 0;
        std::stringstream brief_stream;
        brief_stream << std::setiosflags(std::ios::left);

        /// Use for verpose report
        std::stringstream sstream;
        std::string prev_module_name;
        std::string prev_interface_name;
        std::string prev_impl_name;

        for (const auto &case_result : unit_test_result.second)
        {
            ++num_tested;

            if (TestStatus::PASSED == case_result.status)
            {
                if (with_color)
                {
                    brief_stream << std::setw(50) << case_result.name << ": " << GREEN << "Passed" << RESET << std::endl;
                }
                else
                {
                    brief_stream << std::setw(50) << case_result.name << ": " << "Passed" << std::endl;
                }
                ++num_passed;
            }
            else
            {
                if (with_color)
                {
                    brief_stream << std::setw(50) << case_result.name << ": " << RED << "Failed" << RESET << std::endl;
                }
                else
                {
                    brief_stream << std::setw(50) << case_result.name << ": " << "Failed" << std::endl;
                }
                ++num_failed;
            }

            const auto &sub_cases      = case_result.m_result;
            const auto &module_name    = case_result.module;
            const auto &interface_name = case_result.interface;
            const auto &impl_name      = case_result.impl;

            if (module_name != prev_module_name)
            {
                prev_module_name    = module_name;
                prev_interface_name = interface_name;
                prev_impl_name      = impl_name;

                sstream << "module(" << prev_module_name << ")" << std::endl;
                sstream << "    " << "interface(" << prev_interface_name << ")" << std::endl;
                sstream << "        " << "impl(" << prev_impl_name << ")" << std::endl;
            }

            if ((module_name == prev_module_name) && (interface_name != prev_interface_name))
            {
                prev_interface_name = interface_name;
                prev_impl_name      = impl_name;

                sstream << "    " << "interface(" << prev_interface_name << ")" << std::endl;
                sstream << "        " << "impl(" << prev_impl_name << ")" << std::endl;
            }

            if ((module_name == prev_module_name) && (interface_name == prev_interface_name)
                && (impl_name != prev_impl_name))
            {
                prev_impl_name = impl_name;
                sstream << "        " << "impl(" << prev_impl_name << ")" << std::endl;
            }

            std::stringstream accu_stream;
            std::stringstream perf_stream;

            for (const auto &item : sub_cases)
            {
                if (TestStatus::UNTESTED != item.accu_status)
                {
                    accu_stream << "                " << item.GetAccuResStr() << std::endl;
                }

                if (TestStatus::UNTESTED != item.perf_status)
                {
                    perf_stream << "                " << item.GetPerfResStr() << std::endl;
                }
            }

            std::string accu_string = accu_stream.str();
            std::string perf_string = perf_stream.str();

            if (!accu_string.empty())
            {
                sstream << "            accu:" << std::endl;
                sstream << accu_string;
            }
            if (!perf_string.empty())
            {
                sstream << "            perf:" << std::endl;
                sstream << perf_string;
            }
        }

        if (with_color)
        {
            report_stream << std::string(default_line_width, '*') << std::endl;
            report_stream << "Brief Report:" << std::endl;
            report_stream << RESET << "Tested: " << num_tested << RESET << std::endl;
            report_stream << GREEN << "Passed: " << num_passed << RESET << std::endl;
            report_stream << RED   << "Failed: " << num_failed << RESET << std::endl;
            report_stream << brief_stream.str();
            report_stream << RESET << std::string(default_line_width, '*') << std::endl;
        }
        else
        {
            report_stream << std::string(default_line_width, '*') << std::endl;
            report_stream << "Brief Report:" << std::endl;
            report_stream << "Tested: " << num_tested << std::endl;
            report_stream << "Passed: " << num_passed << std::endl;
            report_stream << "Failed: " << num_failed << std::endl;
            report_stream << brief_stream.str();
            report_stream << std::string(default_line_width, '*') << std::endl;
        }

        report_stream << sstream.str();
    }

    return report_stream.str();
}

Status UnitTestReportText::Report(const std::string &destination)
{
    std::string file_name;

    if (!destination.empty() && destination != m_name)
    {
        file_name = destination;
    }
    else
    {
        file_name = m_name;
    }

    if (GetFileSuffixStr(file_name) != "txt")
    {
        file_name.append(".txt");
    }

    std::ofstream ofstream(file_name, std::ofstream::out);

    if (!ofstream.is_open())
    {
        return Status::ERROR;
    }

    ofstream << this->GetReportString();
    ofstream.close();

    return Status::OK;
}

DT_VOID UnitTestReportText::PrintVerbose()
{
    std::cout << this->GetReportString(DT_TRUE) << std::endl;
}

Status UnitTestReportJson::Report(const std::string &destination)
{
    std::string file_name;

    if (!destination.empty() && destination != m_name)
    {
        file_name = destination;
    }
    else
    {
        file_name = m_name;
    }

    if (GetFileSuffixStr(file_name) != "json")
    {
        file_name.append(".json");
    }

    std::ofstream ofstream(file_name, std::ofstream::out);

    if (!ofstream.is_open())
    {
        return Status::ERROR;
    }

    aura_json::json json_obj;

    for (const auto &run_result : this->m_result)
    {
        aura_json::json unit_result_map;

        for (const auto &case_result : run_result.second)
        {
            unit_result_map[case_result.name]["status"]    = TestStatusToString(case_result.status);
            unit_result_map[case_result.name]["module"]    = case_result.module;
            unit_result_map[case_result.name]["interface"] = case_result.interface;
            unit_result_map[case_result.name]["impl"]      = case_result.impl;

            for (const auto &result : case_result.m_result)
            {
                aura_json::json item;
                item["param"]              = result.param;
                item["input"]              = result.input;
                item["output"]             = result.output;
                item["accu_benchmark"]     = result.accu_benchmark;
                item["accu_status"]        = TestStatusToString(result.accu_status);
                item["accu_result"]        = result.accu_result;
                item["perf_status"]        = TestStatusToString(result.perf_status);

                for (const auto &perf_result : result.perf_result)
                {
                    item["perf_result"][perf_result.first]["avg"] = perf_result.second.avg_time;
                    item["perf_result"][perf_result.first]["min"] = perf_result.second.min_time;
                    item["perf_result"][perf_result.first]["max"] = perf_result.second.max_time;
                }

                unit_result_map[case_result.name]["result"].push_back(item);
            }
        }

        if (run_result.first.empty())
        {
            json_obj["default"] = unit_result_map;
        }
        else
        {
            json_obj[run_result.first] = unit_result_map;
        }
    }

    ofstream << json_obj.dump(4) << std::endl;
    ofstream.close();

    return Status::OK;
}

DT_VOID UnitTestReportJson::PrintVerbose()
{
    return;
}

} // namespace aura
