//
// Created by frewen on 25-3-9.
//
#include <stdio.h>
#include <stdlib.h>

#define TEST_NUM    2000000000

long sum(unsigned char *a, unsigned long lenght) {
    long nSum = 0;
    for(unsigned long i = 0; i < lenght; i++)
        nSum += a[i];
    return nSum;
}

void Rand(unsigned char *a, unsigned long lenght) {
    for(unsigned long i = 0; i < lenght; i++)
        a[i] = 1;//rand() % 100;
}

void* start(void *arg) {
    /// 我们首先进行无符号char的分配，相当于分配了20亿长度的char数组
    unsigned char *pa = (unsigned char*)malloc(sizeof(unsigned char)*TEST_NUM);
    /// 让这个20亿数组的所有数据都为1
    Rand(pa, TEST_NUM);
    /// 计算这个20亿数据的数据进行计算叠加
    long ret = sum(pa, TEST_NUM);
    printf("ret=%ld\n", ret);
    /// 进行动态malloc的内存空间进行释放
    free(pa);

}

/**
 * 执行编译产出的脚本： gcc -O2 -std=c99 test_perf_sum.c -o test
 *  sudo perf stat ./test
 * @return
 */
int main() {
    start(NULL);
    return 0;
}


// sudo perf stat ./test
// (base) frewen@FreweniUbuntuStation:~/03.ProgramSpace/01.WorkSpace/AuraKaleidoScope/AuraKaleidoCXX/test/src/performance$ sudo perf stat ./test
// [sudo] password for frewen:
// ret=2000000000
//
//  Performance counter stats for './test':
//
//           1,087.18 msec task-clock                #    0.999 CPUs utilized   // task-clock：CPU 利用率，该值高，说明程序的多数时间花费在 CPU 计算上而非 IO。
//                  7      context-switches          #    6.439 /sec    // Context-switches：进程切换次数，记录了程序运行过程中发生了多少次进程切换，频繁的进程切换是应该避免的。
//                  1      cpu-migrations            #    0.920 /sec    // CPU-migrations：表示进程 t1 运行过程中发生了多少次 CPU 迁移，即被调度器从一个 CPU 转移到另外一个 CPU 上运行
//            488,335      page-faults               #  449.175 K/sec   // page-faults 内存页面交换
//      5,055,772,780      cpu_core/cycles/          #    4.650 G/sec       Cycles：处理器时钟，一条机器指令可能需要多个 cycles
//      <not counted>      cpu_atom/cycles/                                              (0.00%)
//     20,127,969,611      cpu_core/instructions/    #   18.514 G/sec
//      <not counted>      cpu_atom/instructions/                                        (0.00%)
//      4,371,963,100      cpu_core/branches/        #    4.021 G/sec
//      <not counted>      cpu_atom/branches/                                            (0.00%)
//            334,202      cpu_core/branch-misses/   #  307.402 K/sec                             //   Cache-misses: cache 失效的次数。 从输出我们可以看出这个程序是一个cpu密集型的程序。
//      <not counted>      cpu_atom/branch-misses/                                       (0.00%)
//
//        1.087899471 seconds time elapsed
//
//        0.715915000 seconds user
//        0.371956000 seconds sys

