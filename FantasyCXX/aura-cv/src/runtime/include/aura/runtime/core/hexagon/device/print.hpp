#ifndef AURA_RUNTIME_CORE_HEXAGON_DEVICE_PRINT_HPP__
#define AURA_RUNTIME_CORE_HEXAGON_DEVICE_PRINT_HPP__

#include "aura/runtime/core/types.h"
#include "aura/runtime/core/hexagon/comm.hpp"

#include <string>

#define FARF_HIGH        1
#include "HAP_farf.h"
#include "hexagon_types.h"

namespace aura
{

template <typename Tp>
AURA_INLINE DT_VOID Q6_vprint_V(HVX_Vector v, DT_S32 id = 0, DT_S32 num_per_line = 8)
{
    Tp *p = (Tp*)(&v);

    FARF(HIGH, "********* %d start ***********\n", id);

    std::string line;
    DT_S32 total_num = AURA_HVLEN / sizeof(Tp);
    for (DT_S32 i = 0; i < total_num; i++)
    {
        line += std::to_string(p[i]) + " | ";
        if ((i + 1) % num_per_line == 0)
        {
            FARF(HIGH, "%s\n", line.c_str());
            line.clear();
        }
    }

    FARF(HIGH, "********* %d end ***********\n", id);
}

} // namespace aura

#endif // AURA_RUNTIME_CORE_HEXAGON_DEVICE_PRINT_HPP__