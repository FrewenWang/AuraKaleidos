#ifndef AURA_TOOLS_UNIT_TEST_UNIT_TEST_HPP__
#define AURA_TOOLS_UNIT_TEST_UNIT_TEST_HPP__

#include "aura/tools/unit_test/test_case.hpp"

#include <vector>
#include <string>
#include <memory>
#include <unordered_map>

/**
 * @defgroup tools Tools
 * @{
 *    @defgroup unit_test Unit Test
 * @}
 */

namespace aura
{
/**
 * @addtogroup unit_test
 * @{
 */

/**
 * @brief Class representing a unit test report.
 *
 * The `UnitTestReport` class provides a flexible framework for recording, summarizing, and reporting test results
 * across various test cases. Serving as a base class, it allows derived classes to implement specific reporting
 * mechanisms and formats. Test results include both a brief overview and detailed information for each test case,
 * offering comprehensive insights into the testing process.
 */
class AURA_EXPORTS UnitTestReport
{
public:
    /**
     * @brief Structure representing the result of a test case.
     */
    struct ReportResult
    {
        TestStatus status;                  /*!< The status of the test case. */
        std::string name;                   /*!< The name of the test case. */
        std::string module;                 /*!< The module associated with the test case. */
        std::string interface;              /*!< The interface associated with the test case. */
        std::string impl;                   /*!< The implementation associated with the test case. */
        std::vector<TestResult> m_result;   /*!< Vector to store detailed test results. */
    };

    /**
     * @brief Constructor for the UnitTestReport class.
     *
     * @param name The name of the unit test report.
     */
    UnitTestReport(const std::string &name) : m_name(name)
    {}

    /**
     * @brief Get a brief string representation of the unit test report.
     *
     * @return A string containing a brief summary of the unit test report.
     */
    std::string GetBriefString() const;

    /**
     * @brief Print a brief summary of the unit test report.
     */
    DT_VOID PrintBrief() const;

    /**
     * @brief Clear all recorded results in the unit test report.
     */
    DT_VOID Clear();

    /**
     * @brief Record the result of a test case in the unit test report.
     *
     * @param info Information related to the test case.
     * @param test_case The test case object to record.
     */
    DT_VOID Record(const std::string &info, std::shared_ptr<TestCase> test_case);

    /**
     * @brief Virtual destructor for the UnitTestReport class.
     */
    virtual ~UnitTestReport()
    {};

    /**
     * @brief Abstract function to generate a report for the unit test.
     *
     * @param target The platform on which this function runs.
     *
     * @return Status::OK if successful; otherwise, an appropriate error status.
     */
    virtual Status Report(const std::string &target) = 0;

    /**
     * @brief Abstract function to print a verbose version of the unit test report.
     */
    virtual DT_VOID PrintVerbose()                   = 0;

protected:
    std::string m_name;                                                  /*!< The name of the unit test report. */
    std::unordered_map<std::string, std::vector<ReportResult>> m_result; /*!< Map to store test case results. */

};

/**
 * @brief Class representing a text-based unit test report.
 *
 * The `UnitTestReportText` class provides a comprehensive interface for generating, printing, and obtaining
 * text-based reports for unit tests. It is designed to be inherited for customization and integration with
 * specific test frameworks or environments.
 */
class AURA_EXPORTS UnitTestReportText : public UnitTestReport
{
public:
    /**
     * @brief Constructor for UnitTestReportText.
     *
     * @param name The name of the text-based unit test report.
     */
    UnitTestReportText(const std::string &name) : UnitTestReport(name)
    {}

    /**
     * @brief Virtual destructor for UnitTestReportText.
     */
    ~UnitTestReportText()
    {}

    /**
     * @brief Generate a text-based report for the unit test.
     *
     * @param destination The destination for the report (default is an empty string).
     *
     * @return Status::OK if successful; otherwise, an appropriate error status.
     */
    Status Report(const std::string &destination = std::string()) override;

    /**
     * @brief Print a verbose version of the text-based unit test report.
     */
    DT_VOID PrintVerbose() override;

    /**
     * @brief Get a string representation of the text-based unit test report.
     *
     * @param with_color Flag indicating whether to include color information (default is DT_FALSE).
     *
     * @return A string containing the text-based unit test report.
     */
    std::string GetReportString(DT_BOOL with_color = DT_FALSE);
};

/**
 * @brief Class representing a text-based unit test report.
 *
 * The `UnitTestReportText` class extends the functionality of the base `UnitTestReport` class to specifically
 * generate and print text-based test reports. This class serves as a concrete implementation tailored for text
 * output, offering an easy-to-read and comprehensible representation of test results.
 */
class AURA_EXPORTS UnitTestReportJson : public UnitTestReport
{
public:
    /**
     * @brief Constructor for UnitTestReportJson.
     *
     * @param name The name of the JSON-based unit test report.
     */
    UnitTestReportJson(const std::string &name) : UnitTestReport(name)
    {}

    /**
     * @brief Virtual destructor for UnitTestReportJson.
     */
    ~UnitTestReportJson()
    {}

    /**
     * @brief Generate a JSON-based report for the unit test.
     *
     * @param destination The destination for the report (default is an empty string).
     *
     * @return Status::OK if successful; otherwise, an appropriate error status.
     */
    virtual Status Report(const std::string &destination = std::string()) override;

    /**
     * @brief Print a verbose version of the JSON-based unit test report.
     */
    virtual DT_VOID PrintVerbose() override;
};

/**
 * @brief Class representing a unit test.
 *
 * The `UnitTest` class encapsulates functionality for initializing, running, and reporting unit tests.
 * It serves as the primary interface for managing and executing test cases within the test framework.
 */
class AURA_EXPORTS UnitTest
{
public:
    /**
     * @brief Get an instance of the UnitTest class.
     *
     * @return A pointer to the UnitTest instance.
     */
    static UnitTest* GetInstance();

    /**
     * @brief Initialize the unit test environment.
     *
     * @param ctx The pointer to the Context object.
     * @param data_path The path to test data.
     * @param dump_path The dump path of test result.
     * @param report_type The type of report to generate (default is "txt").
     * @param report_name The name of the report (default is "auto_test").
     * @param stress_count The stress count for the unit test (default is 0).
     * @param enable_mem_profiling The memory profiling flag for the unit test (default is 0).
     *
     * @return Status::OK if successful; otherwise, an appropriate error status.
     */
    Status Initialize(Context *ctx, const std::string &data_path, const std::string &dump_path, const std::string &report_type = "txt",
                      const std::string &report_name = "auto_test", DT_S32 stress_count = 0, DT_S32 enable_mem_profiling = 0);

    /**
     * @brief Deinitialize the unit test environment.
     */
    DT_VOID DeInitialize()
    {
        m_initialized = DT_FALSE;
        m_ctx = DT_NULL;
        m_report.reset();
        m_case_map.clear();
    }

    /**
     * @brief Register a test case for the unit test.
     *
     * @param test_case The test case to register.
     * @param module The module associated with the test case.
     * @param interface The interface associated with the test case.
     * @param impl The implementation associated with the test case.
     *
     * @return A shared pointer to the registered test case.
     */
    std::shared_ptr<TestCase> RegisterTestCase(std::shared_ptr<TestCase> test_case, const std::string &module,
                                               const std::string &interface, const std::string &impl);

    /**
     * @brief Get the context associated with the unit test.
     *
     * @return A pointer to the Context instance.
     */
    Context* GetContext();

    /**
     * @brief Get the report associated with the unit test.
     *
     * @return A shared pointer to the UnitTestReport instance.
     */
    std::shared_ptr<UnitTestReport> GetReport();

    /**
     * @brief Get the names of test cases that match the specified keyword.
     *
     * @param keyword The keyword to match.
     *
     * @return A vector containing the names of test cases that match the keyword.
     */
    std::vector<std::string> GetTestCases(const std::string &keyword = std::string()) const;

    /**
     * @brief Get the names of test cases that match the specified keywords.
     *
     * @param keywords A vector of keywords to match.
     *
     * @return A vector containing the names of test cases that match the keywords.
     */
    std::vector<std::string> GetTestCases(const std::vector<std::string> &keywords) const;

    /**
     * @brief Get the path to test data.
     *
     * @return The path to test data.
     */
    std::string GetDataPath() const
    {
        return m_data_path;
    }

    /**
     * @brief Get dump path of test result.
     *
     * @return The dump path of test result.
     */
    std::string GetDumpPath() const
    {
        return m_dump_path;
    }

    /**
     * @brief Check if the unit test is in stress mode.
     *
     * @return DT_TRUE if the unit test is in stress mode, DT_FALSE otherwise.
     */
    DT_BOOL IsStressMode() const
    {
        return (m_stress_count > 0);
    }

    /**
     * @brief Get the stress count for the unit test.
     *
     * @return The stress count for the unit test.
     */
    DT_S32 GetStressCount() const
    {
        return m_stress_count;
    }

    /**
     * @brief Get the memory profiling flag for the unit test.
     *
     * @return DT_TRUE if the unit test is enable memory Profiling, DT_FALSE otherwise.
     */
    DT_BOOL IsMemProfiling() const
    {
        return (m_enable_mem_profiling > 0);
    }

    /**
     * @brief Run a specific test case.
     *
     * @param case_name The name of the test case to run.
     *
     * @return Status::OK if successful; otherwise, an appropriate error status.
     */
    Status Run(const std::string &case_name);

    /**
     * @brief Run multiple test cases.
     *
     * @param case_names A vector containing the names of test cases to run.
     *
     * @return Status::OK if successful; otherwise, an appropriate error status.
     */
    Status Run(const std::vector<std::string> &case_names);

    /**
     * @brief Record information related to the unit test.
     *
     * @param info Additional information to record (default is an empty string).
     *
     * @return Status::OK if successful; otherwise, an appropriate error status.
     */
    Status Record(const std::string &info = std::string());

    /**
     * @brief Print a brief version of the unit test report.
     *
     * @return Status::OK if successful; otherwise, an appropriate error status.
     */
    Status PrintReportBrief() const;

    /**
     * @brief Print a verbose version of the unit test report.
     *
     * @return Status::OK if successful; otherwise, an appropriate error status.
     */
    Status PrintReportVerbose() const;

    /**
     * @brief Generate and report the unit test.
     *
     * @param destination The destination for the report (default is an empty string).
     *
     * @return Status::OK if successful; otherwise, an appropriate error status.
     */
    Status Report(const std::string &destination = std::string());

private:
    /**
     * @brief Private constructor for UnitTest.
     */
    UnitTest() : m_initialized(DT_FALSE), m_ctx(DT_NULL), m_report(DT_NULL)
    {}

    /**
     * @brief Default destructor for UnitTest.
     */
    ~UnitTest() = default;

    /**
     * @brief Disabled copy constructor and copy assignment operator to prevent copying instances.
     */
    AURA_DISABLE_COPY_AND_ASSIGN(UnitTest);

private:
    DT_BOOL m_initialized;                                                  /*!< Flag indicating whether the unit test is initialized. */
    Context *m_ctx;                                                         /*!< Pointer to the Context instance. */
    std::string m_data_path;                                                /*!< The path to test data. */
    std::string m_dump_path;                                                /*!< The dump path of test result. */
    DT_S32 m_stress_count;                                                  /*!< The stress count for the unit test. */
    DT_S32 m_enable_mem_profiling;                                          /*!< The memory profiling flag for the unit test. */
    std::shared_ptr<UnitTestReport> m_report;                               /*!< Shared pointer to the UnitTestReport instance. */
    std::unordered_map<std::string, std::shared_ptr<TestCase>> m_case_map;  /*!< Map of test case names to test cases. */

};

/**
 * @brief Macro to generate a unique test case name based on module, interface, and implementation names.
 *
 * This macro concatenates the provided module name, interface name, and implementation name
 * to create a unique identifier for a test case. The resulting identifier is suffixed with "_test".
 *
 * @param module_name The name of the module associated with the test case.
 * @param interface_name The name of the interface associated with the test case.
 * @param impl_name The name of the implementation associated with the test case.
 *
 * @return A unique identifier for the test case.
 */
#define TESTCASE_NAME(module_name, interface_name, impl_name) module_name##_##interface_name##_##impl_name##_test

/**
 * @brief Macro to define and register a new test case class.
 *
 * This macro defines a new test case class derived from a specified parent class and registers an instance
 * of the test case with the UnitTest instance. It also provides a constructor and a Run method that need
 * to be implemented for each test case.
 *
 * @param module_name The module name associated with the test case.
 * @param interface_name The interface name associated with the test case.
 * @param impl_name The implementation name associated with the test case.
 * @param parent_class The parent class from which the new test case class is derived.
 */
#define TESTCASE_TEST(module_name, interface_name, impl_name, parent_class)                                             \
    class TESTCASE_NAME(module_name, interface_name, impl_name) : public parent_class                                   \
    {                                                                                                                   \
    public:                                                                                                             \
        TESTCASE_NAME(module_name, interface_name, impl_name)(                                                          \
            const std::string &mod_str, const std::string &intf_str,                                                    \
            const std::string &impl_str) : parent_class(mod_str, intf_str, impl_str)                                    \
        {};                                                                                                             \
        virtual ~TESTCASE_NAME(module_name, interface_name, impl_name)()                                                \
        {};                                                                                                             \
                                                                                                                        \
        virtual DT_VOID Run() override;                                                                                 \
                                                                                                                        \
    private:                                                                                                            \
        static const std::shared_ptr<aura::TestCase> m_case;                                                            \
    };                                                                                                                  \
    const std::shared_ptr<aura::TestCase> TESTCASE_NAME(module_name, interface_name, impl_name)::m_case                 \
        = aura::UnitTest::GetInstance()->RegisterTestCase(                                                              \
            std::shared_ptr<aura::TestCase>(                                                                            \
                new TESTCASE_NAME(module_name, interface_name, impl_name)(#module_name, #interface_name, #impl_name)),  \
                    #module_name, #interface_name, #impl_name);                                                         \
                                                                                                                        \
    DT_VOID TESTCASE_NAME(module_name, interface_name, impl_name)::Run()

/**
 * @brief Macro to create and register a new test case with a single argument (test_case_name).
 */
#define NEW_TESTCASE1(test_case_name)                                                                                   \
    TESTCASE_TEST(test_case_name, Undef, Undef, aura::TestCase)

/**
 * @brief Macro to handle cases where the NEW_TESTCASE macro is called with two arguments (module and impl).
 */
#define NEW_TESTCASE2(module, impl)                                                                                     \
    TESTCASE_TEST(module, impl, Undef, aura::TestCase)

/**
 * @brief Macro to create and register a new test case with three arguments (module, interface, and impl).
 */
#define NEW_TESTCASE3(module, interface, impl)                                                                          \
    TESTCASE_TEST(module, interface, impl, aura::TestCase)

/**
 * @brief Macro to select the appropriate NEW_TESTCASE variant based on the number of arguments provided.
 */
#define NEW_TESTCASE_GET_MACRO(_1, _2, _3, MACRO_NAME, ...) MACRO_NAME

/**
 * @brief Macro to create and register a new test case with a variable number of arguments.
 */
#define NEW_TESTCASE(...)                                                                                               \
    NEW_TESTCASE_GET_MACRO(__VA_ARGS__, NEW_TESTCASE3, NEW_TESTCASE2, NEW_TESTCASE1, ...)(__VA_ARGS__)

/**
 * @}
 */
} // namespace aura

#endif // AURA_TOOLS_UNIT_TEST_UNIT_TEST_HPP__
