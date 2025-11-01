//
// Created by frewen on 22-10-12.
//

#include "../core/include/aura/cv/Allocator.h"
#include <cstdlib>

#ifdef BUILD_FAST_CV
#include "fastcv.h"
#endif
namespace aura::aura_cv {

void *Allocator::allocate(unsigned long len) {
    if (len <= 0) {
        return nullptr;
    }
    return malloc(len);
}

unsigned char *Allocator::allocateInFastCV(unsigned long bytes, int byteAlignment) {
    if (bytes <= 0) {
        return nullptr;
    }
#if defined(BUILD_FAST_CV)
    return (unsigned char *) fcvMemAlloc(bytes, byteAlignment);
#else
    return (unsigned char *) allocate(bytes);
#endif
}

void Allocator::deallocate(void *ptr) {
    if (ptr) {
        free(ptr);
    }
}

int Allocator::alignSize(int len) {
    return len;
}

unsigned long Allocator::alignSize(unsigned long size, int n) {
    return (size + n - 1) & -n;
}

void Allocator::deallocateInFastCV(void *ptr) {
    // TODO 添加对FastCV的支持
}


}
