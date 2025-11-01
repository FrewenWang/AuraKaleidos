#include "qnn/qnn_executor_impl_v2241.hpp"

namespace aura
{

QnnExecutorImplRegister g_qnn_executor_impl_v224_register("qnn.v2.24", QnnExecutorImplHelper<QnnExecutorImplV2>);

} // namespace aura