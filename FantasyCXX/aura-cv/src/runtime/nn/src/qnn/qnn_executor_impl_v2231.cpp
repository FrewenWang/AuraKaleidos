#include "qnn/qnn_executor_impl_v2231.hpp"

namespace aura
{

QnnExecutorImplRegister g_qnn_executor_impl_v223_register("qnn.v2.23", QnnExecutorImplHelper<QnnExecutorImplV2>);

} // namespace aura