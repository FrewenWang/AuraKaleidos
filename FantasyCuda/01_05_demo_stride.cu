//
// Created by frewen on 25-3-8.
//
// 代码参考：https://github.com/HuangCongQing/cuda-learning/blob/main/04stride_loop/stride.cu
// 跨步循环(数据量比block*threads 要多)
#include <stdio.h>
#include <stdlib.h> // cpu的malloc函数

void cpu(int* a, int N)
{
    for (int i = 0; i < N; i++)
    {
        a[i] = i;
    }
    printf("hello cpu\n");
}

// global 将在gpu上运行并可全局调用
__global__ void gpu(int* a, int N)
{
    int thread_i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x; // (block数量*每个block的线程数)=所有线程数===============================
    // 每次跨stride到下一个
    printf("gpu NOT process thread_i = %d,blockIdx.x = %d, blockDim.x= %d, stride=%d  \n", thread_i, blockIdx.x,
           blockDim.x, stride);

    // TODO 这个地方有疑问，前面所有的线程都调用完了。后面的循环内才进行调用？？
    for (int i = thread_i; i < N; i += stride)
    {
        printf("gpu process i = %d ,thread_i = %d, stride=%d  \n", i, thread_i, stride);
        a[i] *= 2; // 放大2倍
    }
}

// 验证
bool check(int* a, int N)
{
    for (int i = 0; i < N; i++)
    {
        if (a[i] != 2 * i) return false;
    }
    return true;
}


int main()
{
    const int N = 2 << 5; //二进制左移运算符。
    size_t size = N * sizeof(int);
    int* a; //取指针的地址&a
    cudaMallocManaged(&a, size); // 既可以被cpu使用也可以被gpu使用
    cpu(a, N);

    // gpu
    size_t threads = 256;
    size_t blocks = (N + threads - 1) / threads; // 算法竞赛向上取整  ceil也可
    /// 传入参数：blocks = （1，1，1）   threads = （256，1，1）
    gpu<<<blocks, threads>>>(a, N); // 每一个数都拥有一个线程
    cudaDeviceSynchronize();

    check(a, N) ? printf("Ok") : printf("Sorry, error");

    ///需要释放cuda分配的对应的空间大小
    cudaFree(a);
}
