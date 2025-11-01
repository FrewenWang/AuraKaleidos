//
// Created by Frewen.Wang on 25-5-16.
//

#ifdef MAC
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define WIDTH    4095
#define HEIGHT   2161
#define ALIGNED_WIDTH ((WIDTH + 3) & ~3)  // 4字节对齐
#define WORKGROUP_SIZE 16

#define CHECK_ERROR(err)                                                               \
if (err != CL_SUCCESS)                                                                  \
{                                                                                       \
    fprintf(stderr, "OpenCL Error: %d @ %s:%d\n", err, __FILE__, __LINE__); exit(1);    \
    exit(err);                                                                              \
}

/*--------------------------- 公共函数 ---------------------------*/
// 反射101坐标计算
inline int reflect101(int pos, int max)
{
    if (pos < 0) return -pos - 1;
    if (pos >= max) return 2 * max - pos - 1;
    return pos;
}

// 生成3x3高斯核（归一化）
void generateGaussianKernel(float sigma, float kernel[9])
{
    float sum = 0.0f;
    for (int y = -1; y <= 1; y++)
    {
        for (int x = -1; x <= 1; x++)
        {
            float val                     = exp(-(x * x + y * y) / (2 * sigma * sigma));
            kernel[(y + 1) * 3 + (x + 1)] = val;
            sum += val;
        }
    }
    for (int i = 0; i < 9; i++) kernel[i] /= sum; // 归一化[4](@ref)
}

/*------------------------ OpenCL实现部分 ------------------------*/
const char *kernelSource = R"(
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_CLAMP_TO_EDGE |
                             CLK_FILTER_NEAREST;

__kernel void gaussian_filter(
    __read_only  image2d_t input,
    __write_only image2d_t output,
    __constant float* gauss_kernel)
{
    const int2 coord = (int2)(get_global_id(0), get_global_id(1));
    if(coord.x >= WIDTH || coord.y >= HEIGHT) return;
    if(coord.x == 0 && coord.y == 0) {
        float4 pixel = read_imagef(input, sampler, coord);
        printf("输入数据：%f，%f，%f, %f \n",pixel.x,pixel.y,pixel.z,pixel.w);
    }
    float4 sum = (float4)(0.0f);
    for(int dy=-1; dy<=1; dy++) {
        for(int dx=-1; dx<=1; dx++) {
            int2 pos = coord + (int2)(dx, dy);
            float4 pixel = read_imagef(input, sampler, pos);
            sum += pixel * gauss_kernel[(dy+1)*3 + (dx+1)];
        }
    }
    write_imagef(output, coord, sum); // 写入RGBA四通道[6](@ref)
})";

// 边界反射预处理（生成对齐的RGBA数据）
void preprocessReflection(unsigned char *src, unsigned char *dst)
{
    for (int y = 0; y < HEIGHT; y++)
    {
        for (int x = 0; x < ALIGNED_WIDTH; x++)
        {
            int srcX = reflect101(x, WIDTH);
            // 将灰度值填充到RGBA四个通道
            int dstIdx         = (y * ALIGNED_WIDTH + x) * 4;
            unsigned char gray = src[y * WIDTH + srcX];
            dst[dstIdx]        = gray; // R
            dst[dstIdx + 1]    = gray; // G
            dst[dstIdx + 2]    = gray; // B
            dst[dstIdx + 3]    = 255; // Alpha[7](@ref)
        }
    }
}

void openclGaussianFilter(unsigned char *input, unsigned char *output, float sigma)
{
    // OpenCL环境初始化
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_int err;

    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CHECK_ERROR(err);
    queue = clCreateCommandQueue(context, device, 0, &err);
    CHECK_ERROR(err);

    // 预处理反射边界
    unsigned char *alignedInput = (unsigned char *)malloc(ALIGNED_WIDTH * HEIGHT * 4);
    preprocessReflection(input, alignedInput);
    for (int y = 0; y < 20; y++)
    {
        printf("alignedInput[%d] = %d \n", y, alignedInput[y]);
    }

    // 创建Image对象（CL_RGBA格式）
    cl_image_format format = {CL_RGBA, CL_UNORM_INT8};
    cl_image_desc desc     = {
        CL_MEM_OBJECT_IMAGE2D,
        ALIGNED_WIDTH, HEIGHT, 1, 1,
        0, 0, 0, 0 // row_pitch=ALIGNED_WIDTH*4[2](@ref)
    };
    cl_mem inputImage = clCreateImage(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      &format, &desc, alignedInput, &err);
    CHECK_ERROR(err);


    cl_mem outputImage = clCreateImage(context, CL_MEM_WRITE_ONLY, &format, &desc, NULL, &err);
    CHECK_ERROR(err);

    // 传递高斯核数据
    float kernelData[9];
    generateGaussianKernel(sigma, kernelData);
    cl_mem kernelBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         sizeof(kernelData), kernelData, &err);
    CHECK_ERROR(err);

    // 编译内核
    // 编译内核（传递尺寸宏定义）
    char buildOptions[128];
    sprintf(buildOptions, "-D WIDTH=%d -D HEIGHT=%d", WIDTH, HEIGHT);
    program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, &err);
    CHECK_ERROR(err);
    err = clBuildProgram(program, 1, &device, buildOptions, NULL, NULL);
    if (err == CL_BUILD_PROGRAM_FAILURE)
    {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = (char *)malloc(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        printf("Build log:\n%s\n", log);
        free(log);
    }
    CHECK_ERROR(err);
    kernel = clCreateKernel(program, "gaussian_filter", &err);
    CHECK_ERROR(err);

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputImage);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputImage);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 2, sizeof(kernelBuffer), &kernelBuffer);
    CHECK_ERROR(err);

    // 执行内核
    size_t globalSize[2] = {ALIGNED_WIDTH, HEIGHT};
    size_t localSize[2]  = {WORKGROUP_SIZE, WORKGROUP_SIZE};
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalSize, localSize, 0, NULL, NULL);
    CHECK_ERROR(err);

    // 读取结果（提取R通道）
    unsigned char *alignedOutput = (unsigned char *)malloc(ALIGNED_WIDTH * HEIGHT * 4);
    size_t origin[3]             = {0, 0, 0}, region[3] = {ALIGNED_WIDTH, HEIGHT, 1};
    clEnqueueReadImage(queue, outputImage, CL_TRUE, origin, region,
                       ALIGNED_WIDTH * 4, 0, alignedOutput, 0, NULL, NULL);

    for (int y = 0; y < HEIGHT; y++)
    {
        for (int x = 0; x < WIDTH; x++)
        {
            int srcIdx            = (y * ALIGNED_WIDTH + x) * 4;
            output[y * WIDTH + x] = alignedOutput[srcIdx]; // 取R通道值
        }
    }

    // 资源释放
    free(alignedInput);
    free(alignedOutput);
    clReleaseMemObject(inputImage);
    clReleaseMemObject(outputImage);
    clReleaseMemObject(kernelBuffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}

/*-------------------------- C语言实现部分 --------------------------*/
void cpuGaussianFilter(unsigned char *input, unsigned char *output, float sigma)
{
    float kernel[9];
    generateGaussianKernel(sigma, kernel);

    for (int y = 0; y < HEIGHT; y++)
    {
        for (int x = 0; x < WIDTH; x++)
        {
            float sum = 0.0f;
            for (int dy = -1; dy <= 1; dy++)
            {
                for (int dx = -1; dx <= 1; dx++)
                {
                    int srcX = reflect101(x + dx, WIDTH);
                    int srcY = reflect101(y + dy, HEIGHT);
                    sum += input[srcY * WIDTH + srcX] * kernel[(dy + 1) * 3 + (dx + 1)];
                }
            }
            output[y * WIDTH + x] = static_cast<unsigned char>(sum + 0.5f); // 四舍五入[11](@ref)
        }
    }
}

/*-------------------------- 验证框架 --------------------------*/
int main()
{
    // 初始化测试数据
    unsigned char *input        = (unsigned char *)malloc(WIDTH * HEIGHT);
    unsigned char *openclOutput = (unsigned char *)malloc(WIDTH * HEIGHT);
    unsigned char *cpuOutput    = (unsigned char *)malloc(WIDTH * HEIGHT);

    // 生成梯度测试图像
    for (int y = 0; y < HEIGHT; y++)
    {
        for (int x = 0; x < WIDTH; x++)
        {
            input[y * WIDTH + x] = rand() % 256;
        }
    }
    for (int y = 0; y < 20; y++)
    {
        printf("input[%d] = %d \n", y, input[y]);
    }

    // 执行滤波（σ=1.5）
    float sigma   = 0.8f;
    clock_t start = clock();
    openclGaussianFilter(input, openclOutput, sigma);
    printf("OpenCL耗时: %.2fms\n", 1000.0 * (clock() - start) / CLOCKS_PER_SEC);
    for (int y = 0; y < 20; y++)
    {
        printf("openclOutput[%d] = %d \n", y, openclOutput[y]);
    }
    start = clock();
    cpuGaussianFilter(input, cpuOutput, sigma);
    for (int y = 0; y < 20; y++)
    {
        printf("cpuOutput[%d] = %d \n", y, cpuOutput[y]);
    }
    printf("CPU耗时: %.2fms\n", 1000.0 * (clock() - start) / CLOCKS_PER_SEC);

    // 结果验证
    int errorCount = 0;
    for (int i = 0; i < WIDTH * HEIGHT; i++)
    {
        if (abs(openclOutput[i] - cpuOutput[i]) > 1)
        {
            errorCount++;
        }
    }
    printf("差异像素占比: %.4f%%\n", errorCount * 100.0 / (WIDTH * HEIGHT));

    free(input);
    free(openclOutput);
    free(cpuOutput);
    return 0;
}
