#include "qnn/qnn_executor_impl_v2221.hpp"

namespace aura
{

QnnExecutorImplRegister g_qnn_executor_impl_v222_register("qnn.v2.22", QnnExecutorImplHelper<QnnExecutorImplV2>);

} // namespace aura