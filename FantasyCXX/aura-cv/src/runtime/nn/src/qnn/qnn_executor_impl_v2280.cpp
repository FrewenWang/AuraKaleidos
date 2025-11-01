#include "qnn/qnn_executor_impl_v2280.hpp"

namespace aura
{

QnnExecutorImplRegister g_qnn_executor_impl_v228_register("qnn.v2.28", QnnExecutorImplHelper<QnnExecutorImplV2>);

} // namespace aura