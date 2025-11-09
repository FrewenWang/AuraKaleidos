#ifndef AURA_TOOLS_UNIT_TEST_TEST_UTILS_HPP__
#define AURA_TOOLS_UNIT_TEST_TEST_UTILS_HPP__

#include "aura/runtime/context.h"
#include "aura/runtime/logger.h"

#include <vector>
#include <string>
#include <sstream>
#include <chrono>
#include <algorithm>
#include <utility>

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
 * @brief Templated function to check equality of two values in a test.
 *
 * @tparam Tp Type of values to be compared.
 *
 * @param ctx The pointer to the Context object.
 * @param a First value.
 * @param b Second value.
 * @param info Additional information for the test.
 * @param file File name where the test is located.
 * @param func Function name where the test is located.
 * @param line Line number where the test is located.
 *
 * @return Status::OK if successful; otherwise, an appropriate error status.
 */
template <typename Tp>
Status TestCheckEQ(Context *ctx, const Tp &a, const Tp &b,
                   const DT_CHAR *info, const DT_CHAR *file, const DT_CHAR *func, DT_S32 line)
{
    if (a == b)
    {
        return Status::OK;
    }
    AURA_LOG(ctx, AURA_TAG, LogLevel::ERROR, ("[%s %s %d] " + std::string(info)).c_str(), file, func, line);
    return Status::ERROR;
}

/**
 * @brief Macro to check equality of two values in a test.
 */
#define AURA_CHECK_EQ(ctx, a, b, info)    TestCheckEQ(ctx, a, b, info, __FILE__, __FUNCTION__, __LINE__)

/**
 * @brief Checks if two values are not equal and logs an error message if they are equal.
 *
 * @tparam Tp The type of the values to be compared.
 *
 * @param ctx The pointer to the Context object.
 * @param a The first value for comparison.
 * @param b The second value for comparison.
 * @param info Additional information for the error message.
 * @param file The file where the check is performed.
 * @param func The function where the check is performed.
 * @param line The line number where the check is performed.
 *
 * @return Status::OK if successful; otherwise, an appropriate error status.
 */
template <typename Tp>
Status TestCheckIEQ(Context *ctx, const Tp &a, const Tp &b,
                    const DT_CHAR *info, const DT_CHAR *file, const DT_CHAR *func, DT_S32 line)
{
    if (a != b)
    {
        return Status::OK;
    }
    AURA_LOG(ctx, AURA_TAG, LogLevel::ERROR, ("[%s %s %d] " + std::string(info)).c_str(), file, func, line);
    return Status::ERROR;
}

/**
 * @brief Macro to check inequality of two values in a test.
 */
#define AURA_CHECK_IEQ(ctx, a, b, info)    TestCheckIEQ(ctx, a, b, info, __FILE__, __FUNCTION__, __LINE__)

/**
 * @brief Macro to get test status based on the returned status.
 */
#define AURA_GET_TEST_STATUS(ret)          ((ret) == Status::OK ? TestStatus::PASSED : TestStatus::FAILED)

/**
 * @brief Converts a string to lowercase.
 *
 * @param str The input string to be converted to lowercase.
 */
AURA_INLINE DT_VOID StringToLower(std::string &str)
{
    std::transform(str.begin(), str.end(), str.begin(), ::tolower);
}

/**
 * @brief Converts a string to uppercase.
 *
 * @param str The input string to be converted to uppercase.
 */
AURA_INLINE DT_VOID StringToUpper(std::string &str)
{
    std::transform(str.begin(), str.end(), str.begin(), ::toupper);
}

/**
 * @brief Checks if a string contains another substring.
 *
 * @param str The input string to be checked.
 * @param sub_str The substring to check for within the input string.
 *
 * @return True if the substring is found, otherwise false.
 */
AURA_INLINE DT_BOOL StringContains(const std::string &str, const std::string &sub_str)
{
    return str.find(sub_str) != std::string::npos;
}

/**
 * @brief Extracts the file suffix from a given file path.
 *
 * @param str The input file path.
 *
 * @return The file suffix, or an empty string if no suffix is found.
 */
AURA_INLINE std::string GetFileSuffixStr(const std::string &str)
{
    std::string suffix;
    size_t idx = str.find_last_of('.');

    if (idx != std::string::npos)
    {
        suffix = str.substr(idx + 1);
    }

    return suffix;
}

/**
 * @brief Executes a function multiple times, measures the execution time, and calculates statistics.
 *
 * @tparam F The type of the function to be executed.
 * @tparam Types Variadic template for function arguments.
 *
 * @param count The total number of executions.
 * @param warm_up The number of warm-up executions to exclude from statistics.
 * @param time_result Reference to a TestTime struct to store the execution time statistics.
 * @param func The function to be executed.
 * @param args The arguments for the function.
 *
 * @return Status::OK if successful; otherwise, an appropriate error status.
 */
template<typename F, typename ...Types>
AURA_INLINE Status Executor(DT_S32 count, DT_S32 warm_up, TestTime &time_result, const F func, Types&&... args)
{
    Status status = Status::ERROR;

    Time max_time(0);
    Time min_time(UINT32_MAX);
    Time sum_time(0);
    DT_S32 used_cnt = 0;

    for (DT_S32 i = 0; i < count; i++)
    {
        Time start_time = Time::Now();
        status          = func(args...);
        Time exe_time   = Time::Now() - start_time;

        if (i >= warm_up)
        {
            sum_time = sum_time + exe_time;
            max_time = Max(max_time, exe_time);
            min_time = Min(min_time, exe_time);
            used_cnt++;
        }

        if (status != Status::OK)
        {
            break;
        }
    }

    time_result.avg_time = sum_time.AsMilliSec() / used_cnt;
    time_result.max_time = max_time.AsMilliSec();
    time_result.min_time = min_time.AsMilliSec();

    return status;
}

/**
 * @}
 */
} // namespace aura

#endif // AURA_TOOLS_UNIT_TEST_TEST_UTILS_HPP__
