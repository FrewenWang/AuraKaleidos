#include "xnn/xnn_executor_impl_v100.hpp"

namespace aura
{

XnnExecutorImplRegister g_xnn_executor_impl_v100_register("xnn.v0.5", XnnExecutorImplHelper<XnnExecutorImplVx>);

} // namespace aura