#include "qnn/qnn_executor_impl_v2290.hpp"

namespace aura
{

QnnExecutorImplRegister g_qnn_executor_impl_v229_register("qnn.v2.29", QnnExecutorImplHelper<QnnExecutorImplV2>);

} // namespace aura