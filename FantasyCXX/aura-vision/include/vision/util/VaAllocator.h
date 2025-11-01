#pragma once

namespace aura::vision {

class VaAllocator {
public:
    static void *allocate(unsigned long len);
    static void deallocate(void *ptr);
    static int align_size(int len);
    static unsigned long align_size(unsigned long len, int size);
    static unsigned char *allocateInFcv(unsigned long bytes, int byteAlignment);
    static void deallocateInFcv(void *ptr);
};

} // namespace vision
