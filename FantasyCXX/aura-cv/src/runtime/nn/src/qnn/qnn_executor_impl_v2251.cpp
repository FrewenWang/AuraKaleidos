#include "qnn/qnn_executor_impl_v2251.hpp"

namespace aura
{

QnnExecutorImplRegister g_qnn_executor_impl_v225_register("qnn.v2.25", QnnExecutorImplHelper<QnnExecutorImplV2>);

} // namespace aura