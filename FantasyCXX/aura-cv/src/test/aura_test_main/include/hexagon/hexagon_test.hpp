#ifndef AURA_TEST_HEXAGON_TEST_HPP__
#define AURA_TEST_HEXAGON_TEST_HPP__

#include "aura/runtime/hexagon/rpc_param.hpp"

namespace aura
{

#define AURA_TEST_PACKAGE_NAME                "aura.test"
#define AURA_TEST_LIST_TEST_CASES_OP_NAME     "ListTestCases"
#define AURA_TEST_RUN_TEST_CASES_OP_NAME      "RunTestCases"

using ListTestCasesInParam = HexagonRpcParamType<std::string>;
using RunTestCasesInParam  = HexagonRpcParamType<std::string>;

} // namespace aura

#endif // AURA_TEST_HEXAGON_TEST_HPP__