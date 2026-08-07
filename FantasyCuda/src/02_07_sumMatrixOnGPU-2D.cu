#include <cstdio>
#include <time.h>
#include <sys/time.h>

void checkResult(float *hostRef, float *gpuRef, const int N)
{
    double eplison = 1.0E-5;
    int match = 1;
    for (int i = 0; i < N; i++)
    {
        if (abs(hostRef[i] - gpuRef[i]) > eplison)
        {
            match = 0;
            printf("do not match\n");
            break;
        }
    }

    if (match)
        printf("match!\n");
    return;
}

double cpuSecond()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

void initialData(float *ip, int size)
{
    time_t t;
    srand((unsigned int)time(&t));
    for (int i = 0; i < size; i++)
    {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}

void  sumMatrixOnCPU(float *A, float *B, float *C, const int nx, const int ny)
{
    float *ia = A;
    float *ib = B;
    float *ic = C;
    for (int iy = 0; iy < ny; iy++)
    {
        for (int ix = 0; ix < nx; ix++)
        {
            ic[ix] = ia[ix] + ib[ix];
        }
        ic += nx;
        ib += nx;
        ia += nx;
    }//
}

/**
 * @brief 
 * 
 * @param Mat_A 
 * @param Mat_B 
 * @param Mat_C 
 * @param nx 
 * @param ny 
 * @return __global__  __global__ 告诉 nvcc：这是设备端（GPU）函数，但由主机端（CPU）调用。返回值必须是 void。
 */
__global__ void sumMatrixOnGPU(float *Mat_A, float *Mat_B, float *Mat_C, const int nx, const int ny)
{
    /// 计算X方向上的索引，这两句把“线程层次坐标”映射到“矩阵元素坐标”。
    /// threadIdx.{x,y}	当前线程在 block 内的局部编号	0 … blockDim.{x,y}-1
    /// blockIdx.{x,y}	当前 block在 grid 内的编号	0 … gridDim.{x,y}-1
    /// blockDim.{x,y}	每个 block 的宽高	由主机端 <<<..., dim3(blockDim.x, blockDim.y)>>> 传入
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    /// 这个全局的索引号
    unsigned idx = iy * nx + ix;

    if (ix < nx && iy < ny) {
        Mat_C[idx] = Mat_A[idx] + Mat_B[idx];
    }
}

__global__ void sumMatrixOnGPU1D(float *Mat_A, float *Mat_B, float *Mat_C, const int nx, const int ny)
{
    /// 只需要计算对应x的坐标的
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    /// 
    if (ix < nx)
    {
        /// 这个是在这里面
        for (int iy = 0; iy < ny; iy++)
        {
            int idx = iy * nx + ix;
            Mat_C[idx] = Mat_A[idx] + Mat_B[idx];
        }
    }
}

int main()
{
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("Using device:%d, %s\n", dev, deviceProp.name);

    // set up data for matrix
    int nx = 1 << 14;
    int ny = 1 << 14;
    int nxy = nx * ny;
    int nBytes = nxy * (sizeof(float));
    printf("Matrix size: nx %d ny %d\n", nx, ny);

    // malloc host mem
    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef = (float *)malloc(nBytes);

    // initialize data at host side
    double iStart = cpuSecond();
    initialData(h_A, nxy);
    initialData(h_B, nxy);
    double iElaps = cpuSecond() - iStart;
    printf("initial host matrix:%f\n", iElaps);

    /// reset host ref and gpu ref
    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // add matrix at host side for result checks
    iStart = cpuSecond();
    sumMatrixOnCPU(h_A, h_B, hostRef, nx, ny);
    iElaps = cpuSecond() - iStart;
    printf("add matrix at host side:%f\n", iElaps);

    // malloc device global mem
    float *d_MatA, *d_MatB, *d_MatC;
    cudaMalloc((void **)&d_MatA, nBytes);
    cudaMalloc((void **)&d_MatB, nBytes);
    cudaMalloc((void **)&d_MatC, nBytes);

    // transfer data from host to device
    cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice);

    // invoke cuda kernel
    int dimx = 32;
    int dimy = 16;
    /// 定义线程块是32*16
    dim3 block(dimx, dimy);
    /// 这个表达式本身并不是“向上取整”的魔法，而是“向上取整”的一种技巧性实现。
    /// 只有当 被除数刚好是 block.x 的整数倍时，结果才会正好等于 nx / block.x；
    /// 否则，它会比 nx / block.x 大 1，从而间接实现了向上取整。
    /// grid(512,1025)
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    iStart = cpuSecond();
    sumMatrixOnGPU<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    /// 执行完毕，需要进行同步
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    printf("add matrix at device %f\n", iElaps);
    printf("sumMatrixOnGPU2D <<<(%d,%d), (%d,%d)>>>\n", grid.x, grid.y, block.x, block.y);

    // copy kernel result to host side
    /// 把计算的答案，从device copy 到 host
    cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost);

    // check device result
    // 
    checkResult(hostRef, gpuRef, nxy);

    // transfer data from host to device
    cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice);


    /// 
    dim3 block1D(32, 1);
    dim3 grid1D((nx + block1D.x - 1) / block1D.x, 1);

    iStart = cpuSecond();
    sumMatrixOnGPU1D<<<block1D, grid1D>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    printf("add matrix at device %f\n", iElaps);
    printf("sumMatrixOnGPU1D <<<(%d,%d), (%d,%d)>>>\n", grid1D.x, grid1D.y, block1D.x, block1D.y);

    // check device result
    checkResult(hostRef, gpuRef, nxy);

    // free device global memory
    free(hostRef);
    free(gpuRef);
    free(h_A);
    free(h_B);
    /// 
    cudaFree(d_MatA);
    cudaFree(d_MatB);
    cudaFree(d_MatC);

    // reset device
    cudaDeviceReset();


    /// 输出结果
    // Using device:0, NVIDIA RTX A3000 12GB Laptop GPU
    // Matrix size: nx 16384 ny 16384
    // initial host matrix:8.701033
    // add matrix at host side:0.462545
    // add matrix at device 0.052855
    // sumMatrixOnGPU2D <<<(512,1024), (32,16)>>>
    // match!
    // add matrix at device 0.011571
    // sumMatrixOnGPU1D <<<(512,1), (32,1)>>>
    // match!

    return 0;
}