#ifndef AURA_ALGOS_HTP_NN_EXECUTOR_H__
#define AURA_ALGOS_HTP_NN_EXECUTOR_H__

#include "aura/config.h"

#if (defined(AURA_ENABLE_NN) && defined(AURA_ENABLE_HEXAGON))
#  include "aura/algos/utils/host/htp_nn_executor.hpp"
#elif (defined(AURA_ENABLE_NN) && defined(AURA_BUILD_HEXAGON))
#  include "aura/algos/utils/hexagon/htp_nn_executor.hpp"
#endif

#endif // AURA_ALGOS_HTP_NN_EXECUTOR_H__