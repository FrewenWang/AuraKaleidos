#ifndef AURA_RUNTIME_MEMORY_ALLOCATOR_HEAP_HPP__
#define AURA_RUNTIME_MEMORY_ALLOCATOR_HEAP_HPP__

#include "aura/runtime/memory/allocator.hpp"

namespace aura
{

class AllocatorHeap : public Allocator
{
public:
    AllocatorHeap() : Allocator(AURA_MEM_HEAP, "heap")
    {};

    ~AllocatorHeap(AURA_VOID)
    {};

    Buffer Allocate(MI_S64 size, MI_S32 align = 0) override;
    AURA_VOID Free(Buffer &) override;

    Status Map(const Buffer &) override;
    Status Unmap(const Buffer &) override;
};

} // namespace aura

#endif // AURA_RUNTIME_MEMORY_ALLOCATOR_HEAP_HPP__