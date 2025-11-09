#include "allocator_heap.hpp"

#include <stdio.h>

namespace aura
{

Buffer AllocatorHeap::Allocate(DT_S64 size, DT_S32 align)
{
    if (align <= 0)
    {
#if defined(AURA_BUILD_HEXAGON)
        align = 128;
#else
        align = 64;
#endif
    }

    DT_VOID **align_addr = DT_NULL;
    DT_VOID *raw_addr = calloc(size + sizeof(DT_VOID*) + align, 1);
    if (raw_addr != DT_NULL)
    {
        align_addr = (DT_VOID **)AURA_ALIGN((DT_UPTR_T)((DT_VOID **)raw_addr + 1), align);
        align_addr[-1] = raw_addr;
    }

    return Buffer(AURA_MEM_HEAP, size, size, align_addr, align_addr, 0);
}

DT_VOID AllocatorHeap::Free(Buffer &buffer)
{
    if (AURA_MEM_HEAP == buffer.m_type && buffer.m_origin != DT_NULL)
    {
        DT_VOID *raw_addr = ((DT_VOID **)buffer.m_origin)[-1];
        if (raw_addr != DT_NULL)
        {
            free(raw_addr);
        }
        buffer.Clear();
    }
}

Status AllocatorHeap::Map(const Buffer &buffer)
{
    AURA_UNUSED(buffer);
    return Status::OK;
}

Status AllocatorHeap::Unmap(const Buffer &buffer)
{
    AURA_UNUSED(buffer);
    return Status::OK;
}

} // namespace aura