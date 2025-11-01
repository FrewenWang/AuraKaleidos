#ifndef AURA_RUNTIME_MEMORY_ALLOCATOR_VTCM_HPP__
#define AURA_RUNTIME_MEMORY_ALLOCATOR_VTCM_HPP__

#include "aura/runtime/memory/allocator.hpp"

namespace aura
{

class AllocatorVtcm : public Allocator
{
public:
    AllocatorVtcm() : Allocator(AURA_MEM_VTCM, "vtcm")
    {};

    ~AllocatorVtcm()
    {};

    Buffer Allocate(MI_S64 size, MI_S32 align = 0) override; // align != 0 is for single page, size will be aligned to 128 by default
    AURA_VOID Free(Buffer &) override;

    Status Map(const Buffer &) override;
    Status Unmap(const Buffer &) override;
};

} // namespace aura

#endif // AURA_RUNTIME_MEMORY_ALLOCATOR_VTCM_HPP__