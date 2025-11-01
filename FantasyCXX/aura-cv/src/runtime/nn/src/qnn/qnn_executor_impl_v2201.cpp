#include "qnn/qnn_executor_impl_v2201.hpp"

namespace aura
{

QnnExecutorImplRegister g_qnn_executor_impl_v220_register("qnn.v2.20", QnnExecutorImplHelper<QnnExecutorImplV2>);

} // namespace aura