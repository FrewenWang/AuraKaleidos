#include <stdio.h>


/**
* CPU所在位置称为为主机端（host）
* 而GPU所在位置称为设备端（device）
*/
/**
 * 在cpu上运行并可全局调用
 * 仅可以从host上调用，一般省略不写
 */
void cpuIgnoreHost() {
  printf("hello ===cpuIgnoreHost===\n");
}

__host__ void cpuHost() {
  printf("hello ===cpuHost===\n");
}


__device__ void gpuDevice() {
  // 通过一个表达式区分不同线程
  // blockIdx.x * blockDim.x + threadIdx.x;
  // 只希望第一个block的第一个线程去打印
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    printf("只希望第一个block的第一个线程去打印: \n");
    printf("hello ===gpuDevice===\n");
  }
}

/**
 * __global__：是标记这个函数device上执行
 * 并且是从host中调用。
 * 返回类型必须是void
 **/
__global__ void gpu() {
  // 通过一个表达式区分不同线程
  // blockIdx.x * blockDim.x + threadIdx.x;
  printf("hello ===blockIdx.x:%d \n", blockIdx.x);
  printf("hello ===threadIdx.x:%d \n", threadIdx.x);
  // 只希望第一个block的第一个线程去打印
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    printf("只希望第一个block的第一个线程去打印: \n");
    printf("hello ===gpu===\n");
  }
}


/**
 * @brief 代码参考：https://github.com/HuangCongQing/cuda-learning

 * @return int 
 */
int main() {
  cpuIgnoreHost();

  /// 核函数的调用
  ///
  gpu<<<2,3>>>(); // gpu配置 《block数 线程数》 不用for循环 O(n)的算法直接变成O(1)
  cudaDeviceSynchronize();
  cpuHost();
}
