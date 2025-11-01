//
// Created by Frewen.Wang on 2024/12/16.
//
#include <iostream>
#include <stdio.h>

// __global__ void hello_world(void) {
//   printf("GPU: Hello world!\n");
// }
/**
 * 代码参考：
 * @return
 */
int main() {
  int dev = 0;
  // 在 CUDA 编程中，cudaDeviceProp 是一个结构体（struct），用于描述 CUDA 设备的硬件属性和配置信息。
  // 通过查询这个对象，开发者可以获取当前 GPU 的关键参数（如计算能力、内存大小、核心数量等），从而编写与设备特性适配的高效代码。
  cudaDeviceProp devProp;
  cudaGetDeviceProperties(&devProp, dev);
  // GPU 型号名称（如 “NVIDIA GeForce RTX 3090”）
  std::cout << "使用GPU device " << dev << ": " << devProp.name << std::endl;
  // 全局内存（显存）总大小（字节）
  std::cout << "全局内存（显存）总大小（字节）：" << std::to_string(devProp.totalGlobalMem / 1024.0 / 1024) << " MB" << std::endl;
  std::cout << "每个线程块的共享内存大小：" << devProp.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
  // 每个线程块可用的寄存器数量
  std::cout << "每个线程块可用的寄存器数量：" << devProp.regsPerBlock << std::endl;
  // 一个线程束（Warp）包含的线程数（通常是 32）
  std::cout << "一个线程束（Warp）包含的线程数（通常是 32）：" << devProp.regsPerBlock << std::endl;
  // 每个线程块支持的最大线程数
  std::cout << "每个线程块的最大线程数：" << devProp.maxThreadsPerBlock << std::endl;

  // 流式多处理器（SM）的数量
  std::cout << "流式多处理器（SM）的数量：" << devProp.multiProcessorCount << std::endl;


  std::cout << "每个EM的最大线程数：" << devProp.maxThreadsPerMultiProcessor << std::endl;
  std::cout << "每个SM的最大线程束数：" << devProp.maxThreadsPerMultiProcessor / 32 << std::endl;

  // 使用GPU device 0: NVIDIA RTX A3000 12GB Laptop GPU
  // 全局内存（显存）总大小（字节）：12045.187500 MB
  // 每个线程块的共享内存大小：48 KB
  // 每个线程块可用的寄存器数量：65536
  // 一个线程束（Warp）包含的线程数（通常是 32）：65536
  // 每个线程块的最大线程数：1024
  // 流式多处理器（SM）的数量：32
  // 每个EM的最大线程数：1536
  // 每个SM的最大线程束数：48

  printf("=============CPU: Hello world!====================\n");
  // hello_world<<<1,10>>>();
  // cudaDeviceReset();//if no this line ,it can not output hello world from gpu
}
