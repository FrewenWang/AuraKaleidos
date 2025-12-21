//
// Created by Frewen.Wang on 2024/12/16.
//
#include<stdio.h>
#include<stdlib.h> // cpu的malloc函数
#include<iostream>

// 两个向量加法kernel，grid和block均为一维
/**
 * 函数内部计算了全局索引index，然后使用stride进行循环处理。
 * 详细解释threadIdx、blockIdx、blockDim这些内置变量的含义，以及grid和block的维度设置。
 * 为什么使用stride和循环，这可能涉及到网格跨步循环（grid-stride loop）的概念，用于处理超出线程总数的大数据量。
 *
 * __global__: 声明这是一个CUDA核函数，由CPU调用，在GPU执行。
 * 参数: x, y 是输入向量，z 是输出向量，n 是向量长度。
 **/
__global__ void add(float* x, float * y, float* z, int n)
{
  // 获取全局索引
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  // 步长
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
  {
    z[i] = x[i] + y[i];
  }
}


int main()
{
  int N = 1 << 20;  // N: 向量长度设置为 2^20（1048576），适合大规模并行计算。
  int nBytes = N * sizeof(float);  // 计算1048576*4 float是4个字节。

  // 申请host内存。 malloc: 在主机内存中为 x, y, z 分配空间。
  float *x, *y, *z;
  x = (float*)malloc(nBytes);
  y = (float*)malloc(nBytes);
  z = (float*)malloc(nBytes);

  // 初始化数据
  // 初始化: 将 x 和 y 分别初始化为 10.0 和 20.0，预期结果 z[i] = 30.0
  for (int i = 0; i < N; ++i)
  {
    x[i] = 10.0;
    y[i] = 20.0;
  }

  // 申请device内存. 进行分配设备内存。
  // cuda分配设备内存使用cudaMalloc函数。
  // cudaMalloc: 在设备（GPU）上分配内存，参数为指向设备指针的指针和字节数。
  //
  float *d_x, *d_y, *d_z;
  cudaMalloc((void**)&d_x, nBytes);
  cudaMalloc((void**)&d_y, nBytes);
  cudaMalloc((void**)&d_z, nBytes);

  // 将host数据拷贝到device
  cudaMemcpy((void*)d_x, (void*)x, nBytes, cudaMemcpyHostToDevice);
  cudaMemcpy((void*)d_y, (void*)y, nBytes, cudaMemcpyHostToDevice);
  // 定义kernel的执行配置
  dim3 blockSize(256);
  dim3 gridSize((N + blockSize.x - 1) / blockSize.x);
  // 执行kernel
  add << < gridSize, blockSize >> >(d_x, d_y, d_z, N);

  // 将device得到的结果拷贝到host
  cudaMemcpy((void*)z, (void*)d_z, nBytes, cudaMemcpyDeviceToHost);

  // 检查执行结果
  float maxError = 0.0;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(z[i] - 30.0));
  std::cout << "最大误差: " << maxError << std::endl;

  // 释放device内存
  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_z);
  // 释放host内存
  free(x);
  free(y);
  free(z);

  return 0;
}