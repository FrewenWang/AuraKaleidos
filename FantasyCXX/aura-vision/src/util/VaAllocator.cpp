#include "vision/util/VaAllocator.h"
#include <cstdlib>

#ifdef BUILD_FASTCV
#include "fastcv.h"
#endif

namespace aura::vision {

void* VaAllocator::allocate(unsigned long len) {
    if (len <= 0) {
        return nullptr;
    }
    return malloc(len);
}

void VaAllocator::deallocate(void* ptr) {
    if (ptr) {
        free(ptr);
    }
}

int VaAllocator::align_size(int len) {
    return len;
}

unsigned long VaAllocator::align_size(unsigned long sz, int n) {
    return (sz + n - 1) & -n;
}
unsigned char *VaAllocator::allocateInFcv(unsigned long bytes, int byteAlignment) {
    if (bytes <= 0) {
        return nullptr;
    }
#if defined(BUILD_FASTCV)
    return (unsigned char *) fcvMemAlloc(bytes, byteAlignment);
#else
    return (unsigned char *) allocate(bytes);
#endif
}
void VaAllocator::deallocateInFcv(void *ptr) {
#if defined(BUILD_FASTCV)
    if (ptr) {
        fcvMemFree(ptr);
    }
#else
    deallocate(ptr);
#endif
}

} // namespace aura::vision