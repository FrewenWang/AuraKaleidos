#include "snpe/snpe_executor_impl_v2300.hpp"

namespace aura
{

SnpeExecutorImplRegister g_snpe_executor_impl_v230_register("snpe.v2.30", SnpeExecutorImplHelper<SnpeExecutorImplV2>);

} // namespace aura