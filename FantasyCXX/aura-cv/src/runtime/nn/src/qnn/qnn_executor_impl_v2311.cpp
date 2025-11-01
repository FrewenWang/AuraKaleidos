#include "qnn/qnn_executor_impl_v2311.hpp"

namespace aura
{

QnnExecutorImplRegister g_qnn_executor_impl_v231_register("qnn.v2.31", QnnExecutorImplHelper<QnnExecutorImplV2>);

} // namespace aura