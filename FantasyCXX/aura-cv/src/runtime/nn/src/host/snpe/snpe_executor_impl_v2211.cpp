#include "snpe/snpe_executor_impl_v2211.hpp"

namespace aura
{

SnpeExecutorImplRegister g_snpe_executor_impl_v221_register("snpe.v2.21", SnpeExecutorImplHelper<SnpeExecutorImplV2>);

} // namespace aura