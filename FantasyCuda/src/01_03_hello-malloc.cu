// 分配空间函数不能放在核函数里面,放在外面.
// cudaMallocManaged(&a, size); // 分配内存，既可以被cpu使用也可以被gpu使用
// cudaFree(a);  // 释放内存

#include <stdio.h>
#include <stdlib.h> // cpu的malloc函数

void cpu(int* a, int N)
{
    // 这个地方遍历是64位
    for (int i = 0; i < N; i++)
    {
        a[i] = i;
        printf("a[%d] = %d ,", i, i);
    }
    printf(" \n =============hello cpu====================== \n");
}


// 函数定义:定义一个数组,把数组内的值放大两倍.<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
// global 将在gpu上运行并可全局调用
__global__ void gpu(int* a, int N)
{
    int thread_i = blockIdx.x * blockDim.x + threadIdx.x; // 类似for循环了!
    // printf("blockIdx.x= %d ,", blockIdx.x);   // 因为我们只定义了一个grid网格块，所以这个blockIdx.x始终为0
    // printf("blockDim.x= %d ,", blockDim.x);   // 因为我们只定义了一个grid网格块，并且（256，1,1） 所以blockDim及时固定的256
    printf("gpu NOT process thread_i = %d, threadIdx.x=%d  \n",thread_i,threadIdx.x);
    if (thread_i < N)
    {
        // 数组大小是N，需要判断下。这个主要就是在GPU侧将指针里面的所有数据放大两倍
        a[thread_i] *= 2; // 放大2倍
        printf("gpu process thread_i = %d, stride=%d  \n",thread_i);
    }
}

// 验证
bool check(int* a, int N)
{
    for (int i = 0; i < N; i++)
    {
        // 判断放大两倍是否存在问题
        if (a[i] != 2 * i) return false;
    }
    return true;
}


int main()
{
    const int N = 2 << 5; //二进制左移运算符。2的5次方是64多少个数
    /// 记录分配的字节大小。 在64位和32位的机器上int都是占用4个字节。 所以这个地方分配的大小是256个字节。
    size_t size = N * sizeof(int); //内存 nBytes = N * sizeof(float);
    // 定义一个int型的指针，然后调用cudaMallocManaged
    int* a; //取指针的地址&a

    /// 这个cudaMallocManaged方法，是给a进行分配内存。这个方法既可以被CPU使用，也可以被GPU使用
    /// 如果注释掉，则会因为指针没有分配对象的内存空间报错
    cudaMallocManaged(&a, size); //  给a分配的内存.  cudaMallocManaged既可以被cpu使用也可以被gpu使用
    // cpu
    cpu(a, N);


    // gpu。 定义使用的线程数量
    size_t threads = 256; // 定义线程数量
    /// 定义使用的线程块的数量
    size_t blocks = (N + threads - 1) / threads; //blocks 希望每一个数都拥有一个线程  | 算法竞赛向上取整,等同于ceil函数

    /// 在调用时需要用`<<<grid, block>>>`来指定kernel核函数要执行的线程数量
    /// kernel_fun<<< grid, block >>>(prams...);
    /// 然后我们又调用了gpu这个kernal 函数
    /// 所以上面的blocks是1， 所以其实就是grid 是一个（1，1,1）
    /// threads ,所以其实是一个grid里面是（256，1,1）共有256个线程
    gpu<<<blocks, threads>>>(a, N); //

    /// 这一步主要的操作就是让上面的数据
    cudaDeviceSynchronize(); // 同步一下,不然报错!！！！！！！！！！！！！！！！！

    check(a, N) ? printf("查看是是否存在并发问题：Ok") : printf("抱歉，存在并发问题:Sorry");
    cudaFree(a);
}
