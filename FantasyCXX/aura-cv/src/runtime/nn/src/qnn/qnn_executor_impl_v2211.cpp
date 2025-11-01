#include "qnn/qnn_executor_impl_v2211.hpp"

namespace aura
{

QnnExecutorImplRegister g_qnn_executor_impl_v221_register("qnn.v2.21", QnnExecutorImplHelper<QnnExecutorImplV2>);

} // namespace aura