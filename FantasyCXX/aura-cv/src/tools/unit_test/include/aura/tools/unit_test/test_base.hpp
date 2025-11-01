#ifndef AURA_TOOLS_UNIT_TEST_TEST_BASE_HPP__
#define AURA_TOOLS_UNIT_TEST_TEST_BASE_HPP__

#include "aura/tools/unit_test/test_case.hpp"

#include <vector>
#include <tuple>

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
 * @brief Helper structure for calculating the size of a tuple table at a specific index.
 *
 * @tparam TupleTable Type of the tuple table.
 * @tparam N Index of the tuple element to calculate the size.
 */
template<typename TupleTable, MI_S32 N>
struct CalcSizeHelper
{
    /**
     * @brief Static member function to calculate the size at the given index.
     *
     * @param t The TupleTable.
     * @return The calculated size.
     */
    static MI_S32 CalcSize(const TupleTable &t)
    {
        return std::get<N - 1>(t).size() * CalcSizeHelper<TupleTable, N - 1>::CalcSize(t);
    }
};

/**
 * @brief Specialization of CalcSizeHelper for the base case (N=1).
 *
 * @tparam TupleTable Type of the tuple table.
 */
template<typename TupleTable>
struct CalcSizeHelper<TupleTable, 1>
{
    /**
     * @brief Static member function to calculate the size at the base case index.
     *
     * @param t The TupleTable.
     * @return The calculated size.
     */
    static MI_S32 CalcSize(const TupleTable &t)
    {
        return std::get<0>(t).size();
    }
};

/**
 * @brief Helper structure to get parameters from a tuple table.
 *
 * @tparam TupleTable Type of the tuple table.
 * @tparam Tuple Type of the tuple.
 * @tparam N Index of the tuple element to get the parameter.
 */
template<typename TupleTable, typename Tuple, MI_S32 N>
struct GetParamHelper
{
    /**
     * @brief Static member function to get and populate the parameter at the given index.
     *
     * @param param_table The TupleTable containing parameters.
     * @param param_tuple The Tuple to be populated.
     * @param idx The index of the parameter to be extracted.
     */
    static AURA_VOID GetParam(const TupleTable &param_table, Tuple &param_tuple, MI_S32 idx)
    {
        MI_S32 size    = CalcSizeHelper<TupleTable, N - 1>::CalcSize(param_table);
        MI_S32 vec_idx = idx / size;
        GetParamHelper<TupleTable, Tuple, N - 1>::GetParam(param_table, param_tuple, idx - vec_idx * size);
        std::get<N - 1>(param_tuple) = std::get<N - 1>(param_table)[vec_idx];
    }
};

/**
 * @brief Specialization of GetParamHelper for the base case (N=1).
 *
 * @tparam TupleTable Type of the tuple table.
 * @tparam Tuple Type of the tuple.
 */
template<typename TupleTable, typename Tuple>
struct GetParamHelper<TupleTable, Tuple, 1>
{
    /**
     * @brief Static member function to get and populate the parameter at the base case index.
     *
     * @param param_table The TupleTable containing parameters.
     * @param param_tuple The Tuple to be populated.
     * @param idx The index of the parameter to be extracted.
     */
    static AURA_VOID GetParam(const TupleTable &param_table, Tuple &param_tuple, MI_S32 idx)
    {
        std::get<0>(param_tuple) = std::get<0>(param_table)[idx];
    }
};

/**
 * @brief Template class for a generic test base with parameterized test cases.
 *
 * This class provides a foundation for creating parameterized test bases. It allows defining a set of test parameters
 * through a tuple table and initializing the test with these parameters.
 *
 * Subclasses can implement specific test cases by inheriting from this class and providing their custom logic for
 * the test scenarios.
 *
 * @tparam TupleTable Type of the tuple table containing test parameters.
 * @tparam Tuple Type of the tuple representing a single set of parameters.
 */
template<typename TupleTable, typename Tuple>
class TestBase
{
public:
    /**
     * @brief Constructor for TestBase.
     *
     * @param table The tuple table containing test parameters.
     */
    TestBase(const TupleTable &table) : m_params()
    {
        Initialize(table);
    }

    /**
     * @brief Destructor for TestBase.
     */
    virtual ~TestBase()
    {}

    /**
     * @brief Initialize the test base with parameters from the tuple table.
     *
     * @param table The tuple table containing test parameters.
     */
    AURA_VOID Initialize(const TupleTable &table)
    {
        // parse params and save to m_params
        MI_S32 size = CalcSizeHelper<TupleTable, std::tuple_size<TupleTable>::value>::CalcSize(table);
        this->m_params.resize(size);
        for (MI_S32 i = 0; i < size; i++)
        {
            GetParamHelper<TupleTable, Tuple, std::tuple_size<TupleTable>::value>::GetParam(table, m_params[i], i);
        }
    }

    /**
     * @brief Get the test parameters for a specific index.
     *
     * @param index The index of the test parameters.
     * 
     * @return The tuple containing test parameters.
     */
    Tuple& GetParam(MI_S32 index)
    {
        return this->m_params[index];
    }

    /**
     * @brief Check the validity of test parameters at a specific index.
     *
     * @param index The index of the test parameters.
     * 
     * @return Status::OK if successful; otherwise, an appropriate error status.
     */
    virtual Status CheckParam(MI_S32 index)
    {
        AURA_UNUSED(index);
        return Status::OK;
    }

    /**
     * @brief Run the test with different parameter values.
     *
     * @param test_case The test case instance.
     * @param loop_count The number of times to loop through the tests (default is 0, no looping).
     */
    AURA_VOID RunTest(TestCase *test_case, MI_S32 loop_count = 0)
    {
        for (decltype(m_params.size()) i = 0; i < m_params.size(); i++)
        {
            if (CheckParam(i) == Status::OK)
            {
                RunOne(i, test_case, 0);
            }
        }

        if (loop_count > 0)
        {
            for (decltype(m_params.size()) i = 0; i < m_params.size(); i++)
            {
                if (CheckParam(i) == Status::OK)
                {
                    RunOne(i, test_case, loop_count);
                }
            }
        }
    }

    /**
     * @brief Run the test with parameters at a specific index.
     *
     * @param index The index of the test parameters.
     * @param test_case The test case instance.
     * @param loop_count The number of times to loop through the tests (default is 0, no looping).
     * 
     * @return The result of the test case execution.
     */
    virtual MI_S32 RunOne(MI_S32 index, TestCase *test_case, MI_S32 loop_count = 0) = 0;

private:
    std::vector<Tuple> m_params;    /*!< Vector to store test parameters. */
};

/**
 * @}
 */
} // namespace aura

#endif // AURA_TOOLS_UNIT_TEST_TEST_BASE_HPP__
