//
// Created by LiWendong on 23-3-15.
//

#include <memory.h>

#include "Memory.h"
#include "OpConfig.h"

namespace aura::vision {
namespace op {

static void memcpy_neon(void *dst, const void *src, size_t &size) {
//    if (size & 63)
//        size = (size & -64) + 64;
//    asm volatile (
//        "NEONCopyPLD: \n"
//        " VLDM %[src]!,{d0-d7} \n"
//        " VSTM %[dst]!,{d0-d7} \n"
//        " SUBS %[size],%[size],#0x40 \n"
//        " BGT NEONCopyPLD \n"
//        : [dst]"+r"(dst), [src]"+r"(src), [size]"+r"(size) : : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "cc", "memory");
}

void Memory::memoryCopy(void *dst, const void *src, size_t &size) {
#if SUPPORT_NEON
    memcpy_neon(dst, src, size);
#else
    memcpy(dst, src, size);
#endif
}

} // namespace op
} // namespace aura::vision