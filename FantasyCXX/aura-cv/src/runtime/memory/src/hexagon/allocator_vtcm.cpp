#include "hexagon/allocator_vtcm.hpp"

#include "HAP_compute_res.h"
#include "HAP_vtcm_mgr.h"

namespace aura
{

Buffer AllocatorVtcm::Allocate(MI_S64 size, MI_S32 align)
{
    size = AURA_ALIGN(size, 128);

    AURA_VOID *addr = MI_NULL;
#if __HEXAGON_ARCH__ >= 66
    compute_res_attr_t res_attr;
    HAP_compute_res_attr_init(&res_attr);
    HAP_compute_res_attr_set_serialize(&res_attr, 0);
    HAP_compute_res_attr_set_vtcm_param(&res_attr, size, (align != 0));
    auto ctx_id = HAP_compute_res_acquire(&res_attr, 1000); // Timeout 1ms
    addr = (MI_U8 *)HAP_compute_res_attr_get_vtcm_ptr(&res_attr);

    if (!ctx_id)
    {
        return Buffer();
    }
    else if (!addr)
    {
        HAP_compute_res_release(ctx_id);
        return Buffer();
    }
    else
    {
        return Buffer(AURA_MEM_VTCM, size, size, addr, addr, ctx_id);
    }
#else
    addr = HAP_request_VTCM(size, (align != 0));
    if (!addr)
    {
        return Buffer();
    }
    else
    {
        return Buffer(AURA_MEM_VTCM, size, size, addr, addr, 0);
    }
#endif // __HEXAGON_ARCH__
}

AURA_VOID AllocatorVtcm::Free(Buffer &buffer)
{
#if __HEXAGON_ARCH__ >= 66
    if (AURA_MEM_VTCM == buffer.m_type && buffer.m_property != 0)
    {
        HAP_compute_res_release(buffer.m_property);
        buffer.Clear();
    }
#else
    if (AURA_MEM_VTCM == buffer.m_type && buffer.m_origin != 0)
    {
        HAP_release_VTCM(buffer.m_origin);
        buffer.Clear();
    }
#endif // __HEXAGON_ARCH__
}

Status AllocatorVtcm::Map(const Buffer &buffer)
{
    AURA_UNUSED(buffer);
    return Status::OK;
}

Status AllocatorVtcm::Unmap(const Buffer &buffer)
{
    AURA_UNUSED(buffer);
    return Status::OK;
}

} // namespace aura