#include "qnn/qnn_executor_impl_v2190.hpp"

namespace aura
{

QnnExecutorImplRegister g_qnn_executor_impl_v219_register("qnn.v2.19", QnnExecutorImplHelper<QnnExecutorImplV2>);

} // namespace aura