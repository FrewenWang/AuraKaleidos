#include "qnn/qnn_executor_impl_v2133.hpp"

namespace aura
{

QnnExecutorImplRegister g_qnn_executor_impl_v213_register("qnn.v2.13", QnnExecutorImplHelper<QnnExecutorImplV2>);

} // namespace aura