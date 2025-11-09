#include "aura/runtime/xtensa/device/xtensa_utils.hpp"

#include "tileManager.h"
#include "tileManager_FIK_api.h"

namespace aura
{
namespace xtensa
{

DT_VOID* AllocateBuffer(TileManager tm, DT_S32 size, DT_S32 align)
{
    xvTileManager *xv_tm = static_cast<xvTileManager*>(tm);
    if (DT_NULL == xv_tm)
    {
        AURA_XTENSA_LOG("xv_tm is null ptr");
        return DT_NULL;
    }

    return xvAllocateBuffer(xv_tm, size, XV_MEM_BANK_COLOR_ANY, align);
}

DT_S32 BufferCheckPointSave(TileManager tm)
{
    xvTileManager *xv_tm = static_cast<xvTileManager*>(tm);
    if (DT_NULL == xv_tm)
    {
        AURA_XTENSA_LOG("xv_tm is NULL\n");
        return AURA_XTENSA_ERROR;
    }

    return xvBufferCheckPointSave(xv_tm);
}

Status BufferCheckPointRestore(TileManager tm, DT_S32 idx)
{
    xvTileManager *xv_tm = static_cast<xvTileManager*>(tm);
    if (DT_NULL == xv_tm)
    {
        AURA_XTENSA_LOG("xv_tm is NULL\n");
        return Status::ERROR;
    }

    DT_S32 ret = xvBufferCheckPointRestore(xv_tm, idx);
    if (ret != AURA_XTENSA_OK)
    {
        AURA_XTENSA_LOG("xvBufferCheckPointRestore error\n");
        return Status::ERROR;
    }

    return Status::OK;
}

} // namespace xtensa
} // namespace aura