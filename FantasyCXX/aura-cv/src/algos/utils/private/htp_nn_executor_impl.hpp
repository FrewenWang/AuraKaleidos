#ifndef AURA_ALGOS_UTILS_HTP_NN_EXECUTOR_IMPL_HPP__
#define AURA_ALGOS_UTILS_HTP_NN_EXECUTOR_IMPL_HPP__

#include "aura/runtime/core.h"

#if (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))
#  include "aura/runtime/hexagon.h"
#endif // (defined(AURA_ENABLE_HEXAGON) || defined(AURA_BUILD_HEXAGON))

#if defined(AURA_ENABLE_NN)
#define AURA_HTP_NN_EXCUTOR_PACKAGE_NAME    "aura.htp.nn.executor"
#define AURA_HTP_NN_EXCUTOR_INITIALIZE      "Initialize"
#define AURA_HTP_NN_EXCUTOR_DEINITIALIZE    "DeInitialize"
#define AURA_HTP_NN_EXCUTOR_TEST_RUN        "TestRun"
#endif // defined(AURA_ENABLE_NN)

#endif // AURA_ALGOS_UTILS_HTP_NN_EXECUTOR_IMPL_HPP__