#ifndef AURA_TOOLS_UNIT_TEST_TEST_CASE_HPP__
#define AURA_TOOLS_UNIT_TEST_TEST_CASE_HPP__

#include "aura/tools/unit_test/test_types.hpp"

#include <vector>
#include <memory>
#include <map>
#include <sstream>
#include <unordered_set>

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
 * @brief Enumeration defining the status of a test case.
 */
enum class TestStatus
{
    UNTESTED    = 0, /*!< Test case has not been executed. */
    PASSED,          /*!< Test case passed. */
    FAILED,          /*!< Test case failed. */
};

/**
 * @brief Logical AND operator for combining test statuses.
 *
 * @param s0 The first test status.
 * @param s1 The second test status.
 *
 * @return The combined test status.
 */
AURA_INLINE TestStatus operator && (const TestStatus &s0, const TestStatus &s1)
{
    if (TestStatus::FAILED == s0 || TestStatus::FAILED == s1)
    {
        return TestStatus::FAILED;
    }

    if (TestStatus::UNTESTED == s0 && TestStatus::UNTESTED == s1)
    {
        return TestStatus::UNTESTED;
    }

    return TestStatus::PASSED;
}

/**
 * @brief Logical OR operator for combining test statuses.
 *
 * @param s0 The first test status.
 * @param s1 The second test status.
 *
 * @return The combined test status.
 */
AURA_INLINE TestStatus operator||(const TestStatus &s0, const TestStatus &s1)
{
    if (TestStatus::PASSED == s0 || TestStatus::PASSED == s1)
    {
        return TestStatus::PASSED;
    }

    if (TestStatus::UNTESTED == s0 && TestStatus::UNTESTED == s1)
    {
        return TestStatus::UNTESTED;
    }

    return TestStatus::FAILED;
}

/**
 * @brief Output stream operator for converting TestStatus to a string.
 */
AURA_INLINE std::ostream& operator<<(std::ostream &os, const TestStatus &status)
{
    switch (status)
    {
        case TestStatus::PASSED:
        {
            os << "passed";
            break;
        }

        case TestStatus::FAILED:
        {
            os << "failed";
            break;
        }

        case TestStatus::UNTESTED:
        {
            os << "untested";
            break;
        }

        default:
        {
            os << "invalid";
            break;
        }
    }

    return os;
}

/**
 * @brief Convert TestStatus to a string.
 */
AURA_INLINE std::string TestStatusToString(const TestStatus &status)
{
    std::stringstream sstream;
    sstream << status;
    return sstream.str();
}

/**
 * @brief Structure representing the result of a test.
 *
 * This structure encapsulates the results of a test, including both performance and accuracy.
 * It provides methods to retrieve string representations of these results.
 */
struct AURA_EXPORTS TestResult
{
    /**
     * @brief Default constructor for TestResult.
     */
    TestResult() : perf_status(TestStatus::UNTESTED), accu_status(TestStatus::UNTESTED)
    {}

    /**
     * @brief Get the accuracy result as a string.
     *
     * @return The string representation of the accuracy result.
     */
    std::string GetAccuResStr() const
    {
        std::stringstream str_stream;
        str_stream << "status(" << this->accu_status << ") param(" << this->param
                   << ") input(" << this->input << ") output(" << this->output
                   << ") benchmark(" << this->accu_benchmark << ") result("
                   << accu_result << ")";

        return str_stream.str();
    }

    /**
     * @brief Get the performance result as a string.
     *
     * @return The string representation of the performance result.
     */
    std::string GetPerfResStr() const
    {
        std::stringstream str_stream;

        str_stream << "status(" << this->perf_status << ") param(" << this->param
                << ") input(" << this->input << ") output(" << this->output << ")";

        str_stream << " result(";
        std::string perf_result_str;

        for (auto it = perf_result.begin(); it != perf_result.end(); ++it)
        {
            perf_result_str += it->first + std::string("(") + it->second.ToString() + std::string(") ");
        }

        if (!perf_result_str.empty())
        {
            perf_result_str.pop_back();
        }

        str_stream << perf_result_str << ")";

        return str_stream.str();
    }

    std::string param;                            /*!< Test parameter. */
    std::string input;                            /*!< Test input data. */
    std::string output;                           /*!< Test output data. */

    TestStatus perf_status;                       /*!< Performance test status. */
    std::map<std::string, TestTime> perf_result;  /*!< Performance test result. */

    TestStatus  accu_status;                      /*!< Accumulated test status. */
    std::string accu_benchmark;                   /*!< Accumulated benchmark result. */
    std::string accu_result;                      /*!< Accumulated test result. */
};

/**
 * @brief Class representing a test case.
 *
 * The `TestCase` class provides a structure for defining and executing test cases. It encapsulates 
 * information about the test case, including module, interface, implementation, status, and results. 
 * Derived classes should implement the `Run` method to define the actual test logic.
 */
class AURA_EXPORTS TestCase
{
public:
    /**
     * @brief Constructor for TestCase.
     *
     * @param module The module associated with the test case.
     * @param interface The interface associated with the test case.
     * @param impl The implementation associated with the test case.
     */
    TestCase(const std::string &module, const std::string &interface, const std::string &impl) : m_status(TestStatus::UNTESTED)
    {
        this->SetName(module, interface, impl);
    }

    /**
     * @brief Virtual destructor for TestCase.
     */
    virtual ~TestCase()
    {}

    /**
     * @brief Deleted copy constructor and copy assignment operator to prevent copying instances.
     */
    TestCase(const TestCase &) = delete;
    TestCase& operator=(const TestCase &) = delete;

    /**
     * @brief Add a test result to the test case.
     *
     * @param status The status of the test result.
     * @param result The detailed result information (default is an empty result).
     */
    DT_VOID AddTestResult(const TestStatus &status, const TestResult &result = TestResult())
    {
        if (TestStatus::UNTESTED == status)
        {
            return;
        }

        /// one failed all failed
        if (TestStatus::FAILED == status)
        {
            this->SetStatus(TestStatus::FAILED);
        }

        /// passed ony change untested or passed
        if (TestStatus::PASSED == status && TestStatus::FAILED != this->GetStatus())
        {
            this->SetStatus(TestStatus::PASSED);
        }

        const std::string case_tag = result.input + result.output + result.param;
        if (m_existed_results.count(case_tag) > 0)
        {
            return;
        }
        else
        {
            m_existed_results.insert(case_tag);
            this->m_results.push_back(result);
        }
    }

    /**
     * @brief Pure virtual function for running the test case.
     *
     * This function should be implemented by derived classes to define the actual test logic.
     */
    virtual DT_VOID Run() = 0;

    /**
     * @brief Clear all test results associated with the test case.
     */
    DT_VOID Clear()
    {
        this->m_results.clear();
    }

    /**
     * @brief Get the status of the test case.
     *
     * @return The status of the test case (UNTESTED, PASSED, or FAILED).
     */
    TestStatus GetStatus() const
    {
        return this->m_status;
    }

    /**
     * @brief Set the status of the test case.
     *
     * @param status The status to set for the test case.
     */
    DT_VOID SetStatus(const TestStatus &status)
    {
        this->m_status = status;
    }

    /**
     * @brief Get all test results associated with the test case (const version).
     *
     * @return A const reference to a vector containing all test results.
     */
    const std::vector<TestResult>& GetResults() const
    {
        return this->m_results;
    }

    /**
     * @brief Get all test results associated with the test case.
     *
     * @return A reference to a vector containing all test results.
     */
    std::vector<TestResult>& GetResults()
    {
        return this->m_results;
    }

    /**
     * @brief Get the name of the module associated with the test case.
     *
     * @return The module name.
     */
    const std::string& GetModule() const
    {
        return this->m_module;
    }

    /**
     * @brief Get the name of the interface associated with the test case.
     *
     * @return The interface name.
     */
    const std::string& GetInterface() const
    {
        return this->m_interface;
    }

    /**
     * @brief Get the name of the test case.
     *
     * @return The test case name.
     */
    const std::string& GetName() const
    {
        return this->m_name;
    }

    /**
     * @brief Get the name of the implementation associated with the test case.
     *
     * @return The implementation name.
     */
    const std::string& GetImpl() const
    {
        return this->m_impl;
    }

    /**
     * @brief Set the module name for the test case.
     *
     * @param module The module name.
     */
    DT_VOID SetModule(const std::string &module)
    {
        this->m_module = module;
    }

    /**
     * @brief Set the interface name for the test case.
     *
     * @param interface The interface name.
     */
    DT_VOID SetInterface(const std::string &interface)
    {
        this->m_interface = interface;
    }

    /**
     * @brief Set the implementation name for the test case.
     *
     * @param impl The implementation name.
     */
    DT_VOID SetImpl(const std::string &impl)
    {
        this->m_impl = impl;
    }

    /**
     * @brief Set the name of the test case based on module, interface, and implementation.
     *
     * If the interface and implementation is "Undef," only the module name is considered.
     *
     * @param module The module name.
     * @param interface The interface name.
     * @param impl The implementation name.
     */
    DT_VOID SetName(const std::string &module, const std::string &interface, const std::string &impl)
    {
        if (std::string("Undef") == interface && std::string("Undef") == impl)
        {
            this->m_name      = module;
            this->m_module    = module;
            this->m_interface = std::string("");
            this->m_impl      = std::string("");
        }
        else
        {
            this->m_module    = module;
            this->m_interface = interface;
            this->m_impl      = impl;
            this->m_name      = module + "_" + interface + "_" + impl;
        }
    }

private:
    TestStatus  m_status;                               /*!< The status of the test case. */
    std::string m_name;                                 /*!< The name of the test case. */
    std::string m_module;                               /*!< The module associated with the test case. */
    std::string m_interface;                            /*!< The interface associated with the test case. */
    std::string m_impl;                                 /*!< The implementation associated with the test case. */

    std::vector<TestResult>         m_results;          /*!< Vector to store test results. */
    std::unordered_set<std::string> m_existed_results;  /*!< Set to store unique identifiers for each test case to avoid duplicates. */
};

/**
 * @}
 */
} // namespace aura

#endif // AURA_TOOLS_UNIT_TEST_TEST_CASE_HPP__
