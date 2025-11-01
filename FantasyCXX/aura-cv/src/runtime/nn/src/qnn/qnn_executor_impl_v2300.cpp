#include "qnn/qnn_executor_impl_v2300.hpp"

namespace aura
{

QnnExecutorImplRegister g_qnn_executor_impl_v230_register("qnn.v2.30", QnnExecutorImplHelper<QnnExecutorImplV2>);

} // namespace aura