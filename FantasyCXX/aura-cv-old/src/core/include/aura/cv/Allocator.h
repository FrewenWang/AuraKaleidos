//
// Created by frewen on 22-10-12.
//
#pragma once

namespace aura::cv {

class Allocator {
public:
    /**
     * 内存分配
     * @param len
     * @return
     */
    static void *allocate(unsigned long len);

    /**
     * 使用FastCV进行内存分配
     * @param bytes
     * @param byteAlignment
     * @return
     */
    static unsigned char *allocateInFastCV(unsigned long bytes, int byteAlignment);

    /**
     * 内存回收
     * @param ptr
     */
    static void deallocate(void *ptr);

    static int alignSize(int len);

    /**
     * 字节对其
     * @param size
     * @param n
     * @return
     */
    static unsigned long alignSize(unsigned long size, int n);

    static void deallocateInFastCV(void *ptr);
};

}

