//
// Created by wangzhijiang on 25-5-16.
//

/*------------------------- 高斯滤波原理说明 -------------------------*
1. 高斯核生成：根据σ计算3x3权重矩阵，权重按G(x,y)=exp(-(x²+y²)/(2σ²))生成[4](@ref)
2. 边界反射101：通过镜像扩展边界像素，例如索引-1对应像素1，索引W对应W-2[1](@ref)
3. OpenCL优化：使用CL_RGBA格式减少内存访问次数，利用图像对象硬件加速
4. 向量化处理：每个work-item处理4个像素，提高内存访问效率[2,3](@ref)
*-------------------------------------------------------------------*/

/***************************** 主机端代码 *****************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <CL/cl.h>

#define WIDTH   4095
#define HEIGHT  2161
#define WG_SIZE 128      // 根据网页3测试结果优化的最佳工作组大小[3](@ref)

// 生成3x3高斯核（支持任意σ）
void generate_gaussian_kernel(float kernel[9], float sigma)
{
    float sum            = 0.0f;
    const float sigma_sq = 2.0f * sigma * sigma;

    // 计算高斯权重
    for (int y = -1; y <= 1; y++)
    {
        for (int x = -1; x <= 1; x++)
        {
            float val                     = exp(-(x * x + y * y) / sigma_sq);
            kernel[(y + 1) * 3 + (x + 1)] = val;
            sum += val;
        }
    }

    // 归一化处理
    for (int i = 0; i < 9; i++)
        kernel[i] /= sum;
}

// 反射101边界扩展（网页1/4原理）[1,4](@ref)
void border_reflect101(unsigned char *src, unsigned char *dst,
                       int width, int height, int pad)
{
    for (int y = 0; y < height + 2; y++)
    {
        for (int x = 0; x < width + 2; x++)
        {
            int src_x = x - pad;
            int src_y = y - pad;

            // 垂直反射
            if (src_y < 0) src_y = -src_y;
            else if (src_y >= height) src_y = 2 * height - src_y - 1;

            // 水平反射
            if (src_x < 0) src_x = -src_x;
            else if (src_x >= width) src_x = 2 * width - src_x - 1;

            dst[y * (width + 2) + x] = src[src_y * width + src_x];
        }
    }
}

/*************************** OpenCL内核代码 ***************************/
const char *kernel_code = R"(
// 合法sampler声明（解决CL_INVALID_SAMPLER错误）[10](@ref)
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
                             CLK_ADDRESS_MIRRORED_REPEAT |  // 反射101模式
                             CLK_FILTER_NEAREST;

__kernel void gaussian_filter(__read_only image2d_t input,
                            __write_only image2d_t output,
                            __constant float* weights) {
    // 每个work-item处理4个像素（x维度步长4）
    const int2 coord = (int2)(get_global_id(0)*4, get_global_id(1));

    float4 sum = (float4)(0.0f);

    // 3x3卷积核遍历（网页2优化思想）[2](@ref)
    for(int dy=-1; dy<=1; dy++) {
        for(int dx=-1; dx<=1; dx++) {
            // 硬件自动处理边界反射
            int2 src_coord = coord + (int2)(dx, dy);
            uint4 pixel = read_imageui(input, sampler, src_coord);

            // 权重计算（支持任意σ）
            float weight = weights[(dy+1)*3 + (dx+1)];
            sum += convert_float4(pixel) * weight;
        }
    }

    // 写入四个处理后的像素（向量化优化）[3](@ref)
    write_imageui(output, coord, convert_uint4(sum));
}
)";

/*************************** GPU实现部分 ***************************/
double gpu_gaussian(unsigned char *src, unsigned char *dst,
                    int width, int height, float sigma)
{
    // OpenCL环境初始化
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_mem input, output;
    cl_event timing_event;

    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    queue   = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, NULL);

    // 图像预处理（网页9对齐要求）[9](@ref)
    unsigned char *padded = malloc((width + 2) * (height + 2));
    border_reflect101(src, padded, width, height, 1);

    // 转换为RGBA格式（64字节对齐优化）
    const int aligned_width = (width + 3) / 4 * 4; // 对齐到4的倍数
    const size_t row_pitch  = aligned_width * 4;    // 行间距=宽度*4字节
    unsigned char *rgba     = calloc(aligned_width * (height + 2), 4);
    for (int y = 0; y < height + 2; y++)
    {
        for (int x = 0; x < aligned_width; x++)
        {
            int src_x                           = (x < width) ? x : 2 * width - x - 1; // 右侧反射
            rgba[y * aligned_width * 4 + x * 4] = padded[y * (width + 2) + src_x];
        }
    }

    // 创建OpenCL图像对象（网页10格式说明）[10](@ref)
    cl_image_format fmt = {CL_RGBA, CL_UNORM_INT8};
    input               = clCreateImage2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                            &fmt, aligned_width, height + 2, row_pitch, rgba, NULL);
    output = clCreateImage2D(context, CL_MEM_WRITE_ONLY, &fmt,
                             aligned_width, height + 2, row_pitch, NULL, NULL);

    // 构建程序（增加错误检查）
    cl_int err;
    program = clCreateProgramWithSource(context, 1, &kernel_code, NULL, &err);
    err     = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = (char *)malloc(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        printf("Build error:\n%s\n", log);
        exit(1);
    }
    kernel = clCreateKernel(program, "gaussian_filter", NULL);

    // 设置内核参数（支持任意σ）
    float gauss_weights[9];
    generate_gaussian_kernel(gauss_weights, sigma);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &output);
    clSetKernelArg(kernel, 2, sizeof(gauss_weights), gauss_weights);

    // 执行内核（优化工作组配置）[3](@ref)
    size_t global_work[2] = {aligned_width / 4, height + 2}; // 每个work-item处理4像素
    size_t local_work[2]  = {WG_SIZE, 1};                // 根据网页3测试结果优化
    clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work, local_work,
                           0, NULL, &timing_event);

    // 读取结果（严格对齐）
    unsigned char *result = malloc(aligned_width * (height + 2) * 4);
    size_t origin[3]      = {0, 0, 0}, region[3] = {aligned_width, height + 2, 1};
    clEnqueueReadImage(queue, output, CL_TRUE, origin, region,
                       row_pitch, 0, result, 0, NULL, NULL);

    // 转换回灰度并裁剪边界
    for (int y = 1; y <= height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            dst[(y - 1) * width + x] = result[y * aligned_width * 4 + (x / 4) * 4 + x % 4];
        }
    }

    // 计算执行时间
    cl_ulong start, end;
    clGetEventProfilingInfo(timing_event, CL_PROFILING_COMMAND_START,
                            sizeof(start), &start, NULL);
    clGetEventProfilingInfo(timing_event, CL_PROFILING_COMMAND_END,
                            sizeof(end), &end, NULL);
    double gpu_time = (end - start) * 1e-6; // 转换为毫秒

    // 资源释放
    clReleaseMemObject(input);
    clReleaseMemObject(output);
    clReleaseKernel(kernel);
    /* ... 其他资源释放 ... */
    return gpu_time;
}

/*************************** CPU验证部分 ***************************/
void cpu_gaussian(unsigned char *src, unsigned char *dst,
                  int width, int height, float sigma)
{
    float kernel[9];
    generate_gaussian_kernel(kernel, sigma);

    unsigned char *padded = malloc((width + 2) * (height + 2));
    border_reflect101(src, padded, width, height, 1);

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            float sum = 0.0f;
            for (int dy = -1; dy <= 1; dy++)
            {
                for (int dx = -1; dx <= 1; dx++)
                {
                    int pos = (y + 1 + dy) * (width + 2) + (x + 1 + dx);
                    sum += padded[pos] * kernel[(dy + 1) * 3 + (dx + 1)];
                }
            }
            dst[y * width + x] = (unsigned char)fminf(fmaxf(sum, 0), 255);
        }
    }
    free(padded);
}

/*************************** 主函数对比 ***************************/
int main()
{
    // 初始化数据
    unsigned char *src = malloc(WIDTH * HEIGHT);  // 原始灰度数据
    unsigned char *cpu = malloc(WIDTH * HEIGHT);
    unsigned char *gpu = malloc(WIDTH * HEIGHT);

    // CPU执行（精确计时）
    clock_t t1 = clock();
    cpu_gaussian(src, cpu, WIDTH, HEIGHT, 1.5f);
    double cpu_time = (double)(clock() - t1) / CLOCKS_PER_SEC * 1000;

    // GPU执行
    double gpu_time = gpu_gaussian(src, gpu, WIDTH, HEIGHT, 1.5f);

    // 结果对比（允许1像素误差）
    int match      = 0;
    float max_diff = 0, avg_diff = 0;
    for (int i = 0; i < WIDTH * HEIGHT; i++)
    {
        float diff = fabs(cpu[i] - gpu[i]);
        avg_diff += diff;
        if (diff > max_diff) max_diff = diff;
        if (diff < 1.0f) match++;
    }
    avg_diff /= (WIDTH * HEIGHT);

    printf("\n性能对比:");
    printf("\n├─ CPU耗时: %.2f ms", cpu_time);
    printf("\n├─ GPU耗时: %.2f ms", gpu_time);
    printf("\n└─ 加速比: %.1f倍", cpu_time / gpu_time);

    printf("\n\n结果对比:");
    printf("\n├─ 最大差异: %.2f", max_diff);
    printf("\n├─ 平均差异: %.4f", avg_diff);
    printf("\n└─ 一致率: %.2f%%\n", match * 100.0 / (WIDTH * HEIGHT));

    free(src);
    free(cpu);
    free(gpu);
    return 0;
}
