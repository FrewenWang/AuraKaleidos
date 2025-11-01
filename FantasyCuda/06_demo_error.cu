//
// Created by frewen on 25-3-8.
//
#include <cstdio>
#include <cassert>

void cpu(int* a, int N) {
    for (int i = 0; i < N; i++)
    {
        a[i] = i;
    }
    printf("hello cpu\n");
}

/**
 * global 将在gpu上运行并可全局调用
 **/
__global__ void gpu(int* a, int N) {
    int thread_i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x; // 所以线程数
    for (int i = thread_i; i < N; i += stride) {
        a[i] *= 2; // 放大2倍
    }
    printf("hello gpu\n");
}

/**
 * 验证
 * @param a
 * @param N
 * @return
 */
bool check(int* a, int N) {
    for (int i = 0; i < N; i++) {
        if (a[i] != 2 * i) {
            return false;
        }
    }
    return true;
}

// 处理错误的宏定义=============================================================
inline cudaError_t checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA runtime error: %s\n", cudaGetErrorString(result));
        assert(result==cudaSuccess);
    }
    return result;
}


int main() {
    const int N = 2 << 5; //二进制左移运算符。
    size_t size = N * sizeof(int);
    int* a; //取指针的地址&a

    /// 这个给int型的指针分配64个int长度的字节
    // cudaMallocManaged(&a, size); // 既可以被cpu使用也可以被gpu使用
    cudaError_t err;
    // // 处理error
    // err = cudaMallocManaged(&a, size); //=========================================================
    // if(err != cudaSuccess){
    //     printf("Error:  %s\n", cudaGetErrorString(err));
    // }
    // 宏定义函数
    checkCuda(cudaMallocManaged(&a, size));
    cpu(a, N);

    // gpu
    size_t threads = 256;
    size_t blocks = (N + threads - 1) / threads; // 算法竞赛向上取整  ceil也可


    // gpu<<<blocks, threads>>>(a, N); // 每一个数都拥有一个线程
    // 这个地方，我们故意出错，我们传入县城
    gpu<<<blocks, -1>>>(a, N);
    // 解决无返回值问题
    err = cudaGetLastError(); //==================================================================
    if (err != cudaSuccess) {
        printf("Error:  %s\n", cudaGetErrorString(err));
    }

    // 我们设置让等待cuda设备执行完毕之后。
    cudaDeviceSynchronize();

    check(a, N) ? printf("Ok") : printf("Sorry, error");
    cudaFree(a);
}
