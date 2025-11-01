#include "np/np_executor_impl_v7.hpp"

namespace aura
{

NpExecutorImplRegister g_np_executor_impl_v7_register("np.v7", NpExecutorImplHelper<NpExecutorImplVx>);

} // namespace aura