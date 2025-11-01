#include "qnn/qnn_executor_impl_v2170.hpp"

namespace aura
{

QnnExecutorImplRegister g_qnn_executor_impl_v217_register("qnn.v2.17", QnnExecutorImplHelper<QnnExecutorImplV2>);

} // namespace aura