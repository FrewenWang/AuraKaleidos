#include <cstdio>

int main(int argc, char **argv)
{
    int nElem = 1024;

    /// 我们共计有1024个元素，下面我们定义了1024个线程块，并且是是一维的线程块
    dim3 block(1024);
    printf("%d\n", block.x);  // 所以这个地方block.x是1024
    dim3 grid((nElem + block.x - 1) / block.x);
    // 每个线程块的网格是1x1x1的线程
    printf("grid.x: %d, grid.y: %d, grid.z: %d\n", grid.x, grid.y, grid.z);

    /// 1024个元素。512个线程块
    block.x = 512;
    grid.x = (nElem + block.x - 1) / block.x;
    printf("grid.x: %d, grid.y: %d, grid.z: %d\n", grid.x, grid.y, grid.z);

    block.x = 256;
    grid.x = (nElem + block.x - 1) / block.x;
    printf("grid.x: %d, grid.y: %d, grid.z: %d\n", grid.x, grid.y, grid.z);

    block.x = 128;
    grid.x = (nElem + block.x - 1) / block.x;
    printf("grid.x: %d, grid.y: %d, grid.z: %d\n", grid.x, grid.y, grid.z);

    cudaDeviceReset();
    return 0;
}