#include "np/np_executor_impl_v8.hpp"

namespace aura
{

NpExecutorImplRegister g_np_executor_impl_v8_register("np.v8", NpExecutorImplHelper<NpExecutorImplVx>);

} // namespace aura