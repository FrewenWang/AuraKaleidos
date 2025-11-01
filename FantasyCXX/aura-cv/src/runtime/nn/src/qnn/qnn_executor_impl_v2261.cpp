#include "qnn/qnn_executor_impl_v2261.hpp"

namespace aura
{

QnnExecutorImplRegister g_qnn_executor_impl_v226_register("qnn.v2.26", QnnExecutorImplHelper<QnnExecutorImplV2>);

} // namespace aura